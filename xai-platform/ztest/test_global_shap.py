#!/usr/bin/env python3
"""
Diagnostic script to test global SHAP explanation flow.
"""
import asyncio
import sys
from datetime import datetime
from bson import ObjectId

# Add backend to path
sys.path.insert(0, 'backend')

from app.db.mongo import connect_db, get_db, storage
from app.services.model_loader_service import ModelLoaderService
from app.workers.celery_app import celery_app
import json

async def test_global_shap():
    """Test global SHAP computation manually."""
    print("Connecting to database...")
    await connect_db()
    db = await get_db()

    # Find a model
    print("\nLooking for a model in database...")
    model = await db.models.find_one({"is_active": True})
    if not model:
        model = await db.models.find_one()

    if not model:
        print("ERROR: No models found in database!")
        return

    model_id = str(model["_id"])
    print(f"Found model: {model.get('name')} (ID: {model_id})")
    print(f"Model type: {model.get('model_type')}")
    print(f"Model family: {model.get('model_family')}")
    print(f"Feature schema: {model.get('feature_schema')}")

    # Check if background data exists
    bg_path = model.get("background_data_path")
    if not bg_path:
        print(f"WARNING: No background data path stored for model {model_id}")
        print("You need to upload background data via the frontend first.")
        return

    print(f"\nBackground data path: {bg_path}")

    # Try to download background data
    try:
        print("Attempting to download background data...")
        bg_bytes = await storage.download_file(bg_path)
        import pandas as pd
        bg_df = pd.read_csv(pd.io.common.BytesIO(bg_bytes))
        print(f"Background data loaded: {len(bg_df)} rows, {len(bg_df.columns)} columns")
        print(f"Columns: {list(bg_df.columns)}")
    except Exception as e:
        print(f"ERROR downloading/reading background data: {e}")
        return

    # Check for existing global SHAP explanation
    print("\nChecking for existing global SHAP explanation...")
    existing = await db.explanations.find_one({
        "model_id": model_id,
        "explanation_type": "global",
        "method": "shap"
    }, sort=[("created_at", -1)])

    if existing:
        print(f"Found existing explanation: {existing['_id']}")
        print(f"Created at: {existing.get('created_at')}")
        print(f"Feature names count: {len(existing.get('feature_names', []))}")
        print(f"SHAP values shape: {len(existing.get('shap_values', []))} rows")
    else:
        print("No existing global SHAP explanation found.")

    # Check Celery task
    print("\nChecking Celery connection...")
    try:
        inspect = celery_app.control.inspect()
        active = inspect.active()
        if active:
            print(f"Active workers: {active}")
        else:
            print("WARNING: No active workers found!")
    except Exception as e:
        print(f"ERROR checking Celery: {e}")

    # Optionally trigger a new computation
    print("\nDo you want to trigger a new global SHAP computation? (y/n): ", end="")
    if input().lower() == 'y':
        print("Sending task to Celery...")
        task = celery_app.send_task("compute_global_shap", args=[model_id, bg_path])
        print(f"Task ID: {task.id}")
        print("Check worker logs for progress.")

if __name__ == "__main__":
    asyncio.run(test_global_shap())
