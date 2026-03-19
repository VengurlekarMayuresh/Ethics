from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from app.models.model_meta import ModelResponse, FeatureSchema
from app.api.v1.auth import get_current_user
from app.db.mongo import get_db
from app.services.prediction_service import ModelLoader, PredictionService
from app.utils.file_handler import storage
from app.utils.audit_logger import log_action, AuditActions
from datetime import datetime
from bson import ObjectId
import json
from typing import List, Dict, Any
import pandas as pd

router = APIRouter()

@router.post("/{model_id}", response_model=Dict[str, Any])
async def predict(
    request: Request,
    model_id: str,
    file: UploadFile = File(None),
    input_data: str = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Make a single prediction using the uploaded model.
    Accepts either file upload (CSV) or direct input data (JSON string).
    """
    try:
        # Get model metadata
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Load model
        model_obj, framework = await ModelLoader.load_model(model_doc["file_path"])

        # Prepare input data
        if file:
            # Handle CSV file upload
            contents = await file.read()
            df = pd.read_csv(pd.io.common.BytesIO(contents))
            input_dict = df.iloc[0].to_dict()  # Use first row for single prediction
        elif input_data:
            # Handle direct JSON input
            input_dict = json.loads(input_data)
        else:
            # Fallback for query parameters (backward compatibility/simple testing)
            query_params = dict(request.query_params)
            if query_params:
                input_dict = query_params
            else:
                raise HTTPException(status_code=400, detail="Either file, input_data or query parameters must be provided")

        # Validate input against feature schema
        feature_schema = [FeatureSchema(**schema) for schema in model_doc["feature_schema"]]
        validated_df = await PredictionService.validate_input(input_dict, feature_schema)

        # Make prediction
        prediction_result = await PredictionService.make_prediction(model_obj, framework, validated_df)

        # Format result
        formatted_result = PredictionService.format_prediction_result(prediction_result, input_dict)

        # Store prediction in database
        prediction_doc = {
            "model_id": model_id,
            "user_id": current_user["_id"],
            "input_data": input_dict,
            "prediction": formatted_result["prediction"],
            "probability": formatted_result["probabilities"],
            "latency_ms": 0,  # TODO: Add timing
            "created_at": datetime.utcnow()
        }

        result = await db.predictions.insert_one(prediction_doc)
        formatted_result["prediction_id"] = str(result.inserted_id)

        # Log audit event
        await log_action(
            user_id=current_user["_id"],
            action=AuditActions.PREDICTION_CREATE,
            resource_type="prediction",
            resource_id=str(result.inserted_id),
            details={"model_id": model_id, "model_name": model_doc.get("name")},
            request=request
        )

        return formatted_result

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in prediction API: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_id}/batch", response_model=List[Dict[str, Any]])
async def batch_predict(
    model_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Make batch predictions using CSV file upload.
    Returns predictions for each row in the CSV.
    """
    try:
        # Get model metadata
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Load model
        model_obj, framework = await ModelLoader.load_model(model_doc["file_path"])

        # Process CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        # Validate feature schema
        feature_schema = [FeatureSchema(**schema) for schema in model_doc["feature_schema"]]

        results = []
        for idx, row in df.iterrows():
            try:
                input_dict = row.to_dict()
                validated_df = await PredictionService.validate_input(input_dict, feature_schema)
                prediction_result = await PredictionService.make_prediction(model_obj, framework, validated_df)
                formatted_result = PredictionService.format_prediction_result(prediction_result, input_dict)
                formatted_result["row_index"] = int(idx)
                results.append(formatted_result)
            except Exception as e:
                results.append({
                    "row_index": int(idx),
                    "error": str(e),
                    "input_data": row.to_dict()
                })

        return results

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in prediction API: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_prediction_history(
    limit: int = 50,
    skip: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get prediction history for the current user."""
    try:
        db = await get_db()
        cursor = db.predictions.find(
            {"user_id": current_user["_id"]}
        ).sort("created_at", -1).skip(skip).limit(limit)

        predictions = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            doc["model_id"] = str(doc["model_id"])
            doc["user_id"] = str(doc["user_id"])
            predictions.append(doc)

        return predictions

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in prediction API: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{prediction_id}", response_model=Dict[str, Any])
async def get_prediction(
    prediction_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific prediction by ID."""
    try:
        db = await get_db()
        prediction = await db.predictions.find_one({
            "_id": ObjectId(prediction_id),
            "user_id": current_user["_id"]
        })

        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")

        prediction["_id"] = str(prediction["_id"])
        prediction["model_id"] = str(prediction["model_id"])
        prediction["user_id"] = str(prediction["user_id"])

        return prediction

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))