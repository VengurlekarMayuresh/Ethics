#!/usr/bin/env python3
"""
Bulk update tool to fix incorrect feature_schema in existing models.

This script:
1. Fetches all models from the database
2. Downloads each model file from MinIO
3. Extracts the correct feature schema from the sklearn pipeline
4. Updates the model's feature_schema in MongoDB

Usage:
  cd backend
  python fix_feature_schemas.py
"""

import os
import sys
import joblib
import pandas as pd
import pymongo
from minio import Minio
from minio.error import S3Error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.config import settings

def extract_feature_schema_from_pipeline(pipeline):
    """
    Extract feature schema from a sklearn pipeline with ColumnTransformer.
    Returns a list of FeatureSchema dicts.
    """
    features = []

    # Find the preprocessor (ColumnTransformer)
    preprocessor = None
    if isinstance(pipeline, Pipeline):
        for step_name, step_obj in pipeline.steps:
            if isinstance(step_obj, ColumnTransformer):
                preprocessor = step_obj
                break

    if preprocessor is None:
        print("  WARNING: No ColumnTransformer found in pipeline")
        return features

    # Get raw feature names
    if hasattr(preprocessor, 'feature_names_in_'):
        raw_feature_names = preprocessor.feature_names_in_.tolist()
    else:
        print("  WARNING: No feature_names_in_ on preprocessor")
        return features

    # Build feature types map from transformers
    feature_types = {}  # name -> "categorical" or "numeric"
    if hasattr(preprocessor, 'transformers_'):
        for transformer_name, transformer_obj, cols in preprocessor.transformers_:
            transformer_class = transformer_obj.__class__.__name__
            if 'OneHotEncoder' in transformer_class:
                for col in cols:
                    feature_types[col] = "categorical"
            elif any(num_type in transformer_class for num_type in ['StandardScaler', 'MinMaxScaler', 'Normalizer', 'MaxAbsScaler']):
                for col in cols:
                    feature_types[col] = "numeric"
            else:
                # Unknown transformer, assume numeric
                for col in cols:
                    feature_types[col] = "numeric"

    # Extract options for categorical features from OneHotEncoder
    for name in raw_feature_names:
        feature_type = feature_types.get(name, "numeric")  # default to numeric if not found
        options = []

        if feature_type == "categorical":
            # Find the OneHotEncoder that handles this column
            if hasattr(preprocessor, 'transformers_'):
                for transformer_name, transformer_obj, cols in preprocessor.transformers_:
                    if name in cols and hasattr(transformer_obj, 'categories_'):
                        idx = list(cols).index(name)
                        if idx < len(transformer_obj.categories_):
                            raw_cats = transformer_obj.categories_[idx]
                            options = [str(cat) for cat in raw_cats]
                        break

        # Build FeatureSchema dict (matching ModelResponse format)
        feature = {
            "name": name,
            "type": feature_type,
            "options": options
        }
        features.append(feature)

    return features

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fix feature_schema in existing models')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated without making changes')
    parser.add_argument('--model-id', type=str, help='Only fix a specific model ID')
    args = parser.parse_args()

    print("=" * 80)
    print("BULK FEATURE SCHEMA FIX")
    if args.dry_run:
        print("MODE: DRY RUN (no changes will be made)")
    print("=" * 80)

    # Connect to MongoDB
    mongo_client = pymongo.MongoClient(settings.MONGODB_URL)
    db = mongo_client.get_default_database()
    models_collection = db.models

    # Connect to MinIO
    minio_client = Minio(
        endpoint=f"{settings.MINIO_ENDPOINT}:{settings.MINIO_PORT}",
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=False
    )
    bucket = settings.MINIO_BUCKET

    # Fetch models (optionally filtered by ID)
    query = {}
    if args.model_id:
        query["_id"] = pymongo.ObjectId(args.model_id) if args.model_id else None
    models = list(models_collection.find(query))
    print(f"Found {len(models)} models in database\n")

    if not args.dry_run:
        # Ask for confirmation (skip if running non-interactively)
        import sys
        if sys.stdin.isatty():
            print("This script will:")
            print("  - Download each model file from MinIO")
            print("  - Extract correct feature schema from sklearn pipelines")
            print("  - Update the database with fixed schemas")
            print()
            response = input("Do you want to proceed? (yes/no): ").strip().lower()
            if response != "yes":
                print("Aborted.")
                mongo_client.close()
                return
        else:
            print("Non-interactive mode detected - proceeding without confirmation")

    updated_count = 0
    error_count = 0
    skipped_count = 0

    for model in models:
        model_id = str(model["_id"])
        model_name = model.get("name", "Unnamed")
        file_path = model.get("file_path")
        framework = model.get("framework", "unknown")

        print(f"Processing: [{model_id}] {model_name} (framework: {framework})")

        if not file_path:
            print("  SKIPPED: No file_path\n")
            skipped_count += 1
            continue

        # Only process sklearn models for now
        if framework != "sklearn":
            print(f"  SKIPPED: Only sklearn models are supported (got {framework})\n")
            skipped_count += 1
            continue

        try:
            # Download model from MinIO
            response = minio_client.get_object(bucket, file_path)
            model_bytes = response.read()
            response.close()
            response.release_conn()

            # Load pipeline
            pipeline = joblib.load(io.BytesIO(model_bytes))

            # Extract feature schema
            new_schema = extract_feature_schema_from_pipeline(pipeline)

            if not new_schema:
                print(f"  WARNING: Could not extract schema (not a Pipeline with ColumnTransformer?)\n")
                skipped_count += 1
                continue

            # Compare with existing schema and merge (preserve min/max/mean if present)
            old_schema = model.get("feature_schema", [])
            old_schema_dict = {item["name"]: item for item in old_schema}

            merged_schema = []
            changes = []
            for new_feature in new_schema:
                name = new_feature["name"]
                old_feature = old_schema_dict.get(name, {})

                # Preserve min/max/mean from old schema if they exist
                merged_feature = {
                    "name": name,
                    "type": new_feature["type"],
                    "options": new_feature.get("options", []),
                    "min": old_feature.get("min"),
                    "max": old_feature.get("max"),
                    "mean": old_feature.get("mean"),
                }
                merged_schema.append(merged_feature)

                # Check if there's a meaningful change (type or options)
                if old_feature.get("type") != new_feature["type"] or old_feature.get("options") != new_feature.get("options"):
                    changes.append(f"{name}: {old_feature.get('type')}->{new_feature['type']}, options: {old_feature.get('options')}->{new_feature.get('options')}")

            # Update database if changes exist
            if changes:
                print(f"  CHANGES DETECTED ({len(changes)} features):")
                for change in changes:
                    print(f"    - {change}")
                if not args.dry_run:
                    result = models_collection.update_one(
                        {"_id": model["_id"]},
                        {"$set": {"feature_schema": merged_schema, "updated_at": datetime.utcnow()}}
                    )
                    if result.modified_count:
                        print("  UPDATED in database")
                        updated_count += 1
                    else:
                        print("  No update needed (already fixed)")
                else:
                    print("  (dry run: would update)")
            else:
                print("  Schema already correct")

        except Exception as e:
            import traceback
            print(f"  ERROR: {str(e)}")
            traceback.print_exc()
            error_count += 1

        print()

    print("=" * 80)
    print(f"SUMMARY: Updated={updated_count}, Skipped={skipped_count}, Errors={error_count}, Total={len(models)}")
    print("=" * 80)

    mongo_client.close()

if __name__ == "__main__":
    import io
    from datetime import datetime
    main()
