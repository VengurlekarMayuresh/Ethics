from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from app.models.model_meta import ModelResponse, FeatureSchema
from app.api.v1.auth import get_current_user, decode_token, oauth2_scheme
from app.db.mongo import get_db
from app.services.model_loader_service import ModelLoaderService
from datetime import datetime
from bson import ObjectId
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

router = APIRouter()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    db = await get_db()
    user = await db.users.find_one({"email": payload.get("sub")})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    user["_id"] = str(user["_id"])
    return user

@router.post("/", response_model=Dict[str, Any])
async def compare_models(
    model_ids: List[str],
    file: UploadFile = File(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Compare multiple models side-by-side.
    Returns feature importance comparison and prediction differences.
    """
    try:
        # Load evaluation dataset
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        results = []
        global_importance = []

        for model_id in model_ids:
            # Get model metadata
            db = await get_db()
            model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
            if not model_doc:
                continue  # Skip models not owned by user

            # Load model
            model_obj, framework = await ModelLoaderService.load_model(model_doc["file_path"])

            # Prepare data for prediction
            features = model_doc.get("feature_schema", [])
            feature_names = [f["name"] for f in features] if features else list(df.columns)
            X = df[feature_names]

            # Make predictions
            if framework == "sklearn":
                y_pred = model_obj.predict(X)
                if hasattr(model_obj, 'predict_proba'):
                    y_proba = model_obj.predict_proba(X)
                else:
                    y_proba = None
            elif framework == "xgboost":
                import xgboost as xgb
                dmatrix = xgb.DMatrix(X)
                y_pred = model_obj.predict(dmatrix)
                if hasattr(model_obj, 'predict_proba'):
                    y_proba = model_obj.predict_proba(dmatrix)
                else:
                    y_proba = None
            elif framework == "onnx":
                input_name = model_obj.get_inputs()[0].name
                y_pred = model_obj.run(None, {input_name: X.values.astype(np.float32)})[0]
                y_proba = None
            else:
                continue

            # Get model summary
            model_summary = await ModelLoaderService.get_model_summary(model_obj, framework)

            # Get feature importance (if available)
            feature_importance = []
            if framework == "sklearn":
                if hasattr(model_obj, 'feature_importances_'):
                    importances = model_obj.feature_importances_
                    feature_importance = [
                        {"feature": feature_names[i], "importance": float(importances[i])}
                        for i in range(len(feature_names))
                    ]
            elif framework == "xgboost":
                if hasattr(model_obj, 'feature_names'):
                    importances = model_obj.get_score(importance_type='gain')
                    feature_importance = [
                        {"feature": k, "importance": float(v)}
                        for k, v in importances.items()
                    ]

            # Add to results
            results.append({
                "model_id": model_id,
                "model_name": model_doc.get("name", "Unknown"),
                "task_type": model_summary.get("task_type", "unknown"),
                "feature_importance": feature_importance,
                "predictions": y_pred.tolist() if isinstance(y_pred, (np.ndarray, list)) else y_pred,
                "probabilities": y_proba.tolist() if y_proba is not None else None,
                "framework": framework
            })

            # Aggregate global importance
            if feature_importance:
                for fi in feature_importance:
                    existing = next((item for item in global_importance if item["feature"] == fi["feature"]), None)
                    if existing:
                        existing["importance"] += fi["importance"]
                    else:
                        global_importance.append(fi.copy())

        # Normalize global importance
        if global_importance:
            total_importance = sum(item["importance"] for item in global_importance)
            for item in global_importance:
                item["importance"] /= total_importance

        # Compare predictions across models
        prediction_comparison = []
        if len(results) > 1:
            for i in range(len(df)):
                row_comparison = {"row_index": i, "predictions": []}
                for result in results:
                    pred = result["predictions"]
                    if isinstance(pred, list):
                        row_comparison["predictions"].append(pred[i] if i < len(pred) else None)
                    else:
                        row_comparison["predictions"].append(pred)
                prediction_comparison.append(row_comparison)

        return {
            "models": results,
            "global_importance": sorted(global_importance, key=lambda x: x["importance"], reverse=True),
            "prediction_comparison": prediction_comparison,
            "dataset_size": len(df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{comparison_id}", response_model=Dict[str, Any])
async def get_comparison(
    comparison_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get comparison result by ID."""
    try:
        db = await get_db()
        comparison = await db.comparisons.find_one({"_id": ObjectId(comparison_id)})
        if not comparison:
            raise HTTPException(status_code=404, detail="Comparison not found")

        comparison["_id"] = str(comparison["_id"])
        return comparison

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))