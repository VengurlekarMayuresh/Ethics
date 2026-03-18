from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from app.models.model_meta import ModelResponse, FeatureSchema
from app.api.v1.auth import get_current_user, decode_token, oauth2_scheme
from app.db.mongo import get_db
from app.workers.celery_app import celery_app
from app.services.model_loader_service import ModelLoaderService
from app.utils.file_handler import storage
from datetime import datetime
from bson import ObjectId
import json
from typing import Dict, Any, List, Optional
import pandas as pd

router = APIRouter()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    db = get_db()
    user = await db.users.find_one({"email": payload.get("sub")})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UUTHORIZED, detail="User not found")
    user["_id"] = str(user["_id"])
    return user

@router.post("/local/{model_id}")
async def request_local_explanation(
    model_id: str,
    file: UploadFile = File(None),
    input_data: str = Form(None),
    prediction_id: str = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Request SHAP explanation for a single prediction.
    This is an async operation - returns task_id immediately.
    """
    try:
        # Get model metadata
        db = get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Determine which prediction to use
        target_prediction_id = None

        if prediction_id:
            # Use provided prediction ID (must belong to user and model)
            pred = await db.predictions.find_one({
                "_id": ObjectId(prediction_id),
                "model_id": model_id,
                "user_id": current_user["_id"]
            })
            if not pred:
                raise HTTPException(status_code=400, detail="Prediction not found or does not belong to this model.")
            target_prediction_id = prediction_id
        elif input_data:
            # Create a new prediction entry
            input_dict = json.loads(input_data)
            prediction_doc = {
                "model_id": model_id,
                "user_id": current_user["_id"],
                "input_data": input_dict,
                "prediction": None,
                "probability": None,
                "latency_ms": None,
                "created_at": datetime.utcnow()
            }
            result = await db.predictions.insert_one(prediction_doc)
            target_prediction_id = str(result.inserted_id)
        else:
            # Get last prediction for this model
            last_pred = await db.predictions.find_one(
                {"model_id": model_id, "user_id": current_user["_id"]},
                sort=[("created_at", -1)]
            )
            if last_pred:
                target_prediction_id = str(last_pred["_id"])

        if not target_prediction_id:
            raise HTTPException(status_code=400, detail="No prediction found. Provide input_data, prediction_id, or run a prediction first.")

        # Trigger async SHAP computation
        task = celery_app.send_task("compute_shap_values", args=[target_prediction_id, model_id])
        celery_task_id = task.id

        # Update prediction with task_id
        await db.predictions.update_one(
            {"_id": ObjectId(target_prediction_id)},
            {"$set": {"explanation_task_id": celery_task_id}}
        )

        return {
            "message": "SHAP computation started",
            "task_id": celery_task_id,
            "prediction_id": target_prediction_id,
            "model_id": model_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sharp/{task_id}")
async def get_explanation_result(task_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get explanation result by task_id.
    Returns explanation if complete, or task status.
    """
    try:
        db = get_db()

        # Check task status in Celery
        task = celery_app.AsyncResult(task_id)
        if task.ready():
            result = task.get()
            if result.get("explanation_id"):
                explanation_id = result["explanation_id"]
                explanation = await db.explanations.find_one({"_id": ObjectId(explanation_id)})
                if explanation:
                    explanation["_id"] = str(explanation["_id"])
                    explanation["prediction_id"] = str(explanation["prediction_id"])
                    explanation["model_id"] = str(explanation["model_id"])
                    return {
                        "status": "complete",
                        "explanation": explanation
                    }
            return {"status": "complete", "result": result}
        else:
            return {
                "status": "pending",
                "task_state": task.state,
                "info": task.info if task.info else None
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/global/{model_id}")
async def request_global_explanation(
    model_id: str,
    background_data: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Request global SHAP explanation for a model.
    Requires background dataset for SHAP reference.
    """
    try:
        # Get model metadata
        db = get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Upload background data to storage
        contents = await background_data.read()
        bg_object_name = f"{current_user['_id']}/bg_{int(datetime.utcnow().timestamp())}_{background_data.filename}"
        storage.upload_file(contents, bg_object_name)

        # Store background data reference in model
        await db.models.update_one(
            {"_id": ObjectId(model_id)},
            {"$set": {"background_data_path": bg_object_name}}
        )

        # Trigger async global SHAP computation
        task = celery_app.send_task("compute_global_shap", args=[model_id, bg_object_name])
        task_id = task.id

        return {
            "message": "Global SHAP computation started",
            "task_id": task_id,
            "model_id": model_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/global/{model_id}/latest")
async def get_global_explanation(
    model_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get the latest global explanation for a model."""
    try:
        db = get_db()

        # Check model belongs to user
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Find latest global explanation for this model
        explanation = await db.explanations.find_one(
            {"model_id": model_id, "explanation_type": "global"},
            sort=[("created_at", -1)]
        )

        if explanation:
            explanation["_id"] = str(explanation["_id"])
            return explanation
        else:
            raise HTTPException(status_code=404, detail="No global explanation found. Please request one first.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# LIME Endpoints
@router.post("/lime/{model_id}")
async def request_lime_explanation(
    model_id: str,
    file: UploadFile = File(None),
    input_data: str = Form(None),
    prediction_id: str = Form(None),
    num_features: int = Form(10),
    current_user: dict = Depends(get_current_user)
):
    """
    Request LIME explanation for a single prediction.
    This is an async operation - returns task_id immediately.
    """
    try:
        # Get model metadata
        db = get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Determine which prediction to use
        target_prediction_id = None

        if prediction_id:
            # Use provided prediction ID
            pred = await db.predictions.find_one({
                "_id": ObjectId(prediction_id),
                "model_id": model_id,
                "user_id": current_user["_id"]
            })
            if not pred:
                raise HTTPException(status_code=400, detail="Prediction not found or does not belong to this model.")
            target_prediction_id = prediction_id
        elif input_data:
            # Create a new prediction entry
            input_dict = json.loads(input_data)
            prediction_doc = {
                "model_id": model_id,
                "user_id": current_user["_id"],
                "input_data": input_dict,
                "prediction": None,
                "probability": None,
                "latency_ms": None,
                "created_at": datetime.utcnow()
            }
            result = await db.predictions.insert_one(prediction_doc)
            target_prediction_id = str(result.inserted_id)
        else:
            # Get last prediction for this model
            last_pred = await db.predictions.find_one(
                {"model_id": model_id, "user_id": current_user["_id"]},
                sort=[("created_at", -1)]
            )
            if last_pred:
                target_prediction_id = str(last_pred["_id"])

        if not target_prediction_id:
            raise HTTPException(status_code=400, detail="No prediction found. Provide input_data, prediction_id, or run a prediction first.")

        # Trigger async LIME computation
        task = celery_app.send_task("compute_lime_values", args=[target_prediction_id, model_id, num_features])
        celery_task_id = task.id

        # Update prediction with task_id
        await db.predictions.update_one(
            {"_id": ObjectId(target_prediction_id)},
            {"$set": {"lime_task_id": celery_task_id}}
        )

        return {
            "message": "LIME computation started",
            "task_id": celery_task_id,
            "prediction_id": target_prediction_id,
            "model_id": model_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lime/{task_id}")
async def get_lime_result(task_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get LIME explanation result by task_id.
    """
    try:
        db = get_db()

        # Check task status in Celery
        task = celery_app.AsyncResult(task_id)
        if task.ready():
            result = task.get()
            if result.get("explanation_id"):
                explanation_id = result["explanation_id"]
                explanation = await db.explanations.find_one({"_id": ObjectId(explanation_id)})
                if explanation:
                    explanation["_id"] = str(explanation["_id"])
                    explanation["prediction_id"] = str(explanation["prediction_id"])
                    explanation["model_id"] = str(explanation["model_id"])
                    return {
                        "status": "complete",
                        "explanation": explanation
                    }
            return {"status": "complete", "result": result}
        else:
            return {
                "status": "pending",
                "task_state": task.state,
                "info": task.info if task.info else None
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lime/global/{model_id}")
async def request_global_lime(
    model_id: str,
    background_data: UploadFile = File(...),
    num_features: int = Form(10),
    current_user: dict = Depends(get_current_user)
):
    """
    Request global LIME explanation for a model.
    Requires background dataset.
    """
    try:
        # Get model metadata
        db = get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Upload background data
        contents = await background_data.read()
        bg_object_name = f"{current_user['_id']}/lime_bg_{int(datetime.utcnow().timestamp())}_{background_data.filename}"
        storage.upload_file(contents, bg_object_name)

        # Trigger async global LIME computation
        task = celery_app.send_task("compute_global_lime", args=[model_id, bg_object_name, num_features])
        task_id = task.id

        return {
            "message": "Global LIME computation started",
            "task_id": task_id,
            "model_id": model_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lime/global/{model_id}/latest")
async def get_global_lime(
    model_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get the latest global LIME explanation for a model."""
    try:
        db = get_db()

        # Check model belongs to user
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Find latest global LIME explanation
        explanation = await db.explanations.find_one(
            {"model_id": model_id, "method": "lime", "explanation_type": "global"},
            sort=[("created_at", -1)]
        )

        if explanation:
            explanation["_id"] = str(explanation["_id"])
            return explanation
        else:
            raise HTTPException(status_code=404, detail="No global LIME explanation found. Please request one first.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prediction/{prediction_id}")
async def get_explanation_by_prediction(
    prediction_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get the explanation (SHAP or LIME) for a specific prediction.
    Returns the latest explanation for that prediction.
    """
    try:
        db = get_db()

        # Verify prediction belongs to user
        prediction = await db.predictions.find_one({
            "_id": ObjectId(prediction_id),
            "user_id": current_user["_id"]
        })
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")

        # Find explanation for this prediction (any method, local)
        explanation = await db.explanations.find_one(
            {"prediction_id": prediction_id, "explanation_type": "local"},
            sort=[("created_at", -1)]
        )

        if explanation:
            explanation["_id"] = str(explanation["_id"])
            explanation["model_id"] = str(explanation["model_id"])
            explanation["prediction_id"] = str(explanation["prediction_id"])
            return explanation
        else:
            raise HTTPException(status_code=404, detail="No explanation found for this prediction. Request one first.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))