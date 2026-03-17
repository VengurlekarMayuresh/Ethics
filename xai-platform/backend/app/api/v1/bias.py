from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from app.models.model_meta import ModelResponse, FeatureSchema
from app.api.v1.auth import get_current_user, decode_token, oauth2_scheme
from app.db.mongo import get_db
from app.workers.celery_app import celery_app
from app.services.model_loader_service import ModelLoaderService
from app.db.repositories.bias_repository import BiasRepository
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
    db = get_db()
    user = await db.users.find_one({"email": payload.get("sub")})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    user["_id"] = str(user["_id"])
    return user

@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_bias(
    model_id: str,
    protected_attribute: str,
    sensitive_attribute: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze model bias using fairness metrics.
    Requires evaluation dataset with protected attributes.
    """
    try:
        # Get model metadata
        db = get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Load evaluation dataset
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        # Check for required columns
        if protected_attribute not in df.columns:
            raise HTTPException(status_code=400, detail=f"Dataset missing protected attribute: {protected_attribute}")
        if sensitive_attribute not in df.columns:
            raise HTTPException(status_code=400, detail=f"Dataset missing sensitive attribute: {sensitive_attribute}")

        # Load model
        model_obj, framework = await ModelLoaderService.load_model(model_doc["file_path"])

        # Prepare data for prediction
        features = [col for col in df.columns if col not in [protected_attribute, sensitive_attribute]]
        X = df[features]
        y_true = df[protected_attribute]
        sensitive_values = df[sensitive_attribute]

        # Make predictions
        if framework == "sklearn":
            y_pred = model_obj.predict(X)
        elif framework == "xgboost":
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X)
            y_pred = model_obj.predict(dmatrix)
        elif framework == "onnx":
            input_name = model_obj.get_inputs()[0].name
            y_pred = model_obj.run(None, {input_name: X.values.astype(np.float32)})[0]
        else:
            raise HTTPException(status_code=500, detail="Prediction not implemented for this framework")

        # Compute bias metrics
        bias_metrics = compute_bias_metrics(y_true, y_pred, sensitive_values)

        # Store bias report
        bias_report = {
            "model_id": model_id,
            "user_id": current_user["_id"],
            "protected_attribute": protected_attribute,
            "sensitive_attribute": sensitive_attribute,
            "demographic_parity_diff": bias_metrics["demographic_parity_diff"],
            "equal_opportunity_diff": bias_metrics["equal_opportunity_diff"],
            "disparate_impact_ratio": bias_metrics["disparate_impact_ratio"],
            "group_metrics": bias_metrics["group_metrics"],
            "dataset_size": len(df),
            "created_at": datetime.utcnow()
        }

        bias_id = await BiasRepository.create(bias_report)

        return {
            "message": "Bias analysis complete",
            "bias_id": bias_id,
            "metrics": bias_metrics,
            "dataset_size": len(df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/{model_id}", response_model=List[Dict[str, Any]])
async def get_bias_reports(
    model_id: str,
    limit: int = 50,
    skip: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get bias reports for a model."""
    try:
        # Verify model belongs to user
        db = get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Get bias reports
        reports = await BiasRepository.get_by_model(model_id, limit, skip)
        return reports

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/compare", response_model=List[Dict[str, Any]])
async def compare_bias(
    model_ids: List[str],
    protected_attribute: str,
    sensitive_attribute: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Compare bias across multiple models.
    """
    try:
        # Load evaluation dataset
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        # Check for required columns
        if protected_attribute not in df.columns:
            raise HTTPException(status_code=400, detail=f"Dataset missing protected attribute: {protected_attribute}")
        if sensitive_attribute not in df.columns:
            raise HTTPException(status_code=400, detail=f"Dataset missing sensitive attribute: {sensitive_attribute}")

        results = []

        for model_id in model_ids:
            # Get model metadata
            db = get_db()
            model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
            if not model_doc:
                continue  # Skip models not owned by user

            # Load model
            model_obj, framework = await ModelLoaderService.load_model(model_doc["file_path"])

            # Prepare data for prediction
            features = [col for col in df.columns if col not in [protected_attribute, sensitive_attribute]]
            X = df[features]
            y_true = df[protected_attribute]
            sensitive_values = df[sensitive_attribute]

            # Make predictions
            if framework == "sklearn":
                y_pred = model_obj.predict(X)
            elif framework == "xgboost":
                import xgboost as xgb
                dmatrix = xgb.DMatrix(X)
                y_pred = model_obj.predict(dmatrix)
            elif framework == "onnx":
                input_name = model_obj.get_inputs()[0].name
                y_pred = model_obj.run(None, {input_name: X.values.astype(np.float32)})[0]
            else:
                continue

            # Compute bias metrics
            bias_metrics = compute_bias_metrics(y_true, y_pred, sensitive_values)

            results.append({
                "model_id": model_id,
                "model_name": model_doc.get("name", "Unknown"),
                "task_type": model_doc.get("task_type", "unknown"),
                "metrics": bias_metrics,
                "dataset_size": len(df)
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{model_id}", response_model=Dict[str, Any])
async def get_bias_metrics(
    model_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get aggregated bias metrics for a model."""
    try:
        # Verify model belongs to user
        db = get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Get all bias reports for this model
        reports = await BiasRepository.get_by_model(model_id)

        if not reports:
            raise HTTPException(status_code=404, detail="No bias reports found")

        # Aggregate metrics
        total_reports = len(reports)
        avg_metrics = {
            "demographic_parity_diff": 0,
            "equal_opportunity_diff": 0,
            "disparate_impact_ratio": 0
        }

        for report in reports:
            avg_metrics["demographic_parity_diff"] += report["demographic_parity_diff"]
            avg_metrics["equal_opportunity_diff"] += report["equal_opportunity_diff"]
            avg_metrics["disparate_impact_ratio"] += report["disparate_impact_ratio"]

        for key in avg_metrics:
            avg_metrics[key] /= total_reports

        return {
            "model_id": model_id,
            "total_reports": total_reports,
            "average_metrics": avg_metrics,
            "reports": reports
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def compute_bias_metrics(y_true, y_pred, sensitive_attribute):
    """
    Compute fairness metrics for bias analysis.
    Returns dictionary with demographic parity, equal opportunity, and disparate impact.
    """
    import numpy as np
    from collections import defaultdict

    try:
        groups = np.unique(sensitive_attribute)
        group_metrics = {}

        for group in groups:
            mask = sensitive_attribute == group
            group_metrics[str(group)] = {
                "positive_rate": float(np.mean(y_pred[mask])),
                "true_positive_rate": float(np.mean(y_pred[mask & (y_true == 1)])),
                "false_positive_rate": float(np.mean(y_pred[mask & (y_true == 0)])),
                "accuracy": float(np.mean(y_pred[mask] == y_true[mask]))
            }

        rates = [v["positive_rate"] for v in group_metrics.values()]
        tprs = [v["true_positive_rate"] for v in group_metrics.values()]

        min_rate = min(rates) if min(rates) > 0 else 1e-9
        min_tpr = min(tprs) if min(tprs) > 0 else 1e-9

        return {
            "demographic_parity_diff": max(rates) - min(rates),
            "equal_opportunity_diff": max(tprs) - min(tprs),
            "disparate_impact_ratio": min_rate / max(rates),
            "group_metrics": group_metrics
        }

    except Exception as e:
        return {
            "error": str(e),
            "demographic_parity_diff": None,
            "equal_opportunity_diff": None,
            "disparate_impact_ratio": None,
            "group_metrics": {}
        }