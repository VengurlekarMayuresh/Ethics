from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from app.models.model_meta import ModelResponse, FeatureSchema
from app.api.v1.auth import get_current_user
from app.db.mongo import get_db
from app.workers.celery_app import celery_app
from app.services.model_loader_service import ModelLoaderService
from app.utils.file_handler import storage
from app.utils.audit_logger import log_action, AuditActions
from app.websocket.manager import manager
from datetime import datetime
from bson import ObjectId
import json
import asyncio
from typing import Dict, Any, List, Optional
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

router = APIRouter()

@router.post("/local/{model_id}")
async def request_local_explanation(
    request: Request,
    model_id: str,
    file: UploadFile = File(None),
    input_data: str = Form(None),
    prediction_id: str = None,  # Can be passed as query param
    current_user: dict = Depends(get_current_user)
):
    """
    Request SHAP explanation for a single prediction.
    This is an async operation - returns task_id immediately.
    """
    try:
        # Get model metadata
        db = await get_db()
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

        # Log audit event
        await log_action(
            user_id=current_user["_id"],
            action=AuditActions.EXPLANATION_CREATE,
            resource_type="explanation",
            details={"task_id": celery_task_id, "prediction_id": target_prediction_id, "model_id": model_id, "method": "shap"},
            request=request
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
        db = await get_db()

        # Check task status in Celery
        task = celery_app.AsyncResult(task_id)
        if task.ready():
            result = task.get()
            explanation_data = None
            user_id = None

            if result.get("explanation_id"):
                explanation_id = result["explanation_id"]
                explanation = await db.explanations.find_one({"_id": ObjectId(explanation_id)})
                if explanation:
                    # Get model to find user_id
                    model = await db.models.find_one({"_id": ObjectId(explanation["model_id"])})
                    if model:
                        user_id = str(model["user_id"])

                    explanation["_id"] = str(explanation["_id"])
                    explanation["prediction_id"] = str(explanation["prediction_id"])
                    explanation["model_id"] = str(explanation["model_id"])
                    explanation_data = explanation

            response = {
                "status": "complete",
                "explanation": explanation_data if explanation_data else result
            }

            # Send WebSocket notification asynchronously (fire and forget)
            if user_id:
                asyncio.create_task(
                    manager.send_to_user({
                        "type": "explanation_complete",
                        "task_id": task_id,
                        "method": "shap",
                        "explanation_id": result.get("explanation_id")
                    }, user_id)
                )

            return response
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
    request: Request,
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
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Upload background data to storage
        contents = await background_data.read()
        bg_object_name = f"{current_user['_id']}/bg_{int(datetime.utcnow().timestamp())}_{background_data.filename}"
        await storage.upload_file(contents, bg_object_name)

        # Store background data reference in model
        await db.models.update_one(
            {"_id": ObjectId(model_id)},
            {"$set": {"background_data_path": bg_object_name}}
        )

        # Trigger async global SHAP computation
        task = celery_app.send_task("compute_global_shap", args=[model_id, bg_object_name])
        task_id = task.id

        # Log audit event
        await log_action(
            user_id=current_user["_id"],
            action=AuditActions.EXPLANATION_CREATE,
            resource_type="explanation",
            resource_id=model_id,
            details={"task_id": task_id, "type": "global", "method": "shap"},
            request=request
        )

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
        db = await get_db()

        # Check model belongs to user
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Find latest global explanation for this model
        explanation = await db.explanations.find_one(
            {"model_id": model_id, "explanation_type": "global", "method": "shap"},
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
    request: Request,
    model_id: str,
    file: UploadFile = File(None),
    input_data: str = Form(None),
    prediction_id: str = None,  # optional query param
    num_features: int = Form(10),
    current_user: dict = Depends(get_current_user)
):
    """
    Request LIME explanation for a single prediction.
    This is an async operation - returns task_id immediately.
    """
    try:
        # Get model metadata
        db = await get_db()
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

        # Log audit event
        await log_action(
            user_id=current_user["_id"],
            action=AuditActions.EXPLANATION_CREATE,
            resource_type="explanation",
            details={"task_id": celery_task_id, "prediction_id": target_prediction_id, "model_id": model_id, "method": "lime", "num_features": num_features},
            request=request
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
        db = await get_db()

        # Check task status in Celery
        task = celery_app.AsyncResult(task_id)
        if task.ready():
            result = task.get()
            explanation_data = None
            user_id = None

            if result.get("explanation_id"):
                explanation_id = result["explanation_id"]
                explanation = await db.explanations.find_one({"_id": ObjectId(explanation_id)})
                if explanation:
                    # Get model to find user_id
                    model = await db.models.find_one({"_id": ObjectId(explanation["model_id"])})
                    if model:
                        user_id = str(model["user_id"])

                    explanation["_id"] = str(explanation["_id"])
                    explanation["prediction_id"] = str(explanation["prediction_id"])
                    explanation["model_id"] = str(explanation["model_id"])
                    explanation_data = explanation

            response = {
                "status": "complete",
                "explanation": explanation_data if explanation_data else result
            }

            # Send WebSocket notification
            if user_id:
                asyncio.create_task(
                    manager.send_to_user({
                        "type": "explanation_complete",
                        "task_id": task_id,
                        "method": "lime",
                        "explanation_id": result.get("explanation_id")
                    }, user_id)
                )

            return response
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
    request: Request,
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
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Upload background data
        contents = await background_data.read()
        bg_object_name = f"{current_user['_id']}/lime_bg_{int(datetime.utcnow().timestamp())}_{background_data.filename}"
        await storage.upload_file(contents, bg_object_name)

        # Trigger async global LIME computation
        task = celery_app.send_task("compute_global_lime", args=[model_id, bg_object_name, num_features])
        task_id = task.id

        # Log audit event
        await log_action(
            user_id=current_user["_id"],
            action=AuditActions.EXPLANATION_CREATE,
            resource_type="explanation",
            resource_id=model_id,
            details={"task_id": task_id, "type": "global", "method": "lime", "num_features": num_features},
            request=request
        )

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
        db = await get_db()

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
    method: str = None,  # Optional filter: 'shap' or 'lime'
    current_user: dict = Depends(get_current_user)
):
    """
    Get the explanation (SHAP or LIME) for a specific prediction.
    Returns the latest explanation for that prediction.
    """
    try:
        db = await get_db()

        # Verify prediction belongs to user
        prediction = await db.predictions.find_one({
            "_id": ObjectId(prediction_id),
            "user_id": current_user["_id"]
        })
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")

        # Find explanation for this prediction (local and optionally filtered by method)
        prediction_id_filters = [{"prediction_id": prediction_id}]
        if ObjectId.is_valid(prediction_id):
            prediction_id_filters.append({"prediction_id": ObjectId(prediction_id)})

        filter_query = {"$or": prediction_id_filters, "explanation_type": "local"}
        if method in ('shap', 'lime'):
            filter_query["method"] = method

        explanation = await db.explanations.find_one(
            filter_query,
            sort=[("created_at", -1)]
        )

        if explanation:
            explanation["_id"] = str(explanation["_id"])
            explanation["model_id"] = str(explanation["model_id"])
            explanation["prediction_id"] = str(explanation["prediction_id"])
            return explanation
        else:
            # If explanation not found, check if a task is pending for the requested method
            task_id = None
            if method == 'shap':
                task_id = prediction.get("explanation_task_id")
            elif method == 'lime':
                task_id = prediction.get("lime_task_id")
            else:
                task_id = prediction.get("explanation_task_id") or prediction.get("lime_task_id")

            if task_id:
                task = celery_app.AsyncResult(task_id)
                if not task.ready():
                    return {
                        "status": "pending",
                        "task_id": task_id,
                        "task_state": task.state,
                        "info": task.info if task.info else None
                    }
                elif task.failed():
                    return {
                        "status": "failed",
                        "task_id": task_id,
                        "task_state": task.state,
                        "error": str(task.result) if task.result else "Task failed"
                    }
                elif task.state == "REVOKED":
                    return {
                        "status": "revoked",
                        "task_id": task_id,
                        "task_state": task.state,
                        "detail": "Computation was cancelled"
                    }

            raise HTTPException(status_code=404, detail="No explanation found for this prediction. Request one first.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export/{explanation_id}")
async def export_explanation(
    request: Request,
    explanation_id: str,
    format: str = "json",  # "json", "csv", "pdf"
    current_user: dict = Depends(get_current_user)
):
    """
    Export an explanation in various formats (JSON, CSV, PDF).
    """
    try:
        db = await get_db()

        # Find the explanation
        explanation = await db.explanations.find_one({"_id": ObjectId(explanation_id)})
        if not explanation:
            raise HTTPException(status_code=404, detail="Explanation not found")

        # Verify model belongs to user
        model_doc = await db.models.find_one({"_id": ObjectId(explanation["model_id"]), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=403, detail="Access denied")

        # Log audit event before returning
        await log_action(
            user_id=current_user["_id"],
            action=AuditActions.EXPLANATION_EXPORT,
            resource_type="explanation",
            resource_id=explanation_id,
            details={"format": format, "method": explanation.get("method", "unknown")},
            request=request
        )

        # Format the explanation data
        explanation_data = {
            "explanation_id": str(explanation["_id"]),
            "model_id": str(explanation["model_id"]),
            "model_name": model_doc.get("name", "Unknown"),
            "method": explanation.get("method", "shap"),
            "type": explanation.get("explanation_type", "local"),
            "created_at": explanation["created_at"].isoformat() if isinstance(explanation["created_at"], datetime) else explanation["created_at"],
            "data": explanation.get("explanation_data", {})
        }

        if format.lower() == "json":
            # Return JSON
            json_data = json.dumps(explanation_data, indent=2, default=str)
            return StreamingResponse(
                io.StringIO(json_data),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=explanation_{explanation_id}.json"}
            )

        elif format.lower() == "csv":
            # Convert to CSV (for feature importance)
            output = io.StringIO()
            if "feature_importance" in explanation.get("explanation_data", {}):
                df = pd.DataFrame(explanation["explanation_data"]["feature_importance"])
            elif "shap_values" in explanation.get("explanation_data", {}):
                # For SHAP values, create a simple table
                values = explanation["explanation_data"]["shap_values"]
                features = explanation["explanation_data"].get("feature_names", [])
                df = pd.DataFrame({
                    "feature": features if len(features) == len(values) else [f"Feature_{i}" for i in range(len(values))],
                    "value": values
                })
            else:
                # Generic flattening
                df = pd.DataFrame([explanation_data["data"]])

            df.to_csv(output, index=False)
            output.seek(0)
            return StreamingResponse(
                io.StringIO(output.getvalue()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=explanation_{explanation_id}.csv"}
            )

        elif format.lower() == "pdf":
            # Generate PDF report
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30
            )
            story.append(Paragraph(f"Explanation Report", title_style))
            story.append(Spacer(1, 12))

            # Metadata
            story.append(Paragraph(f"<b>Explanation ID:</b> {explanation_id}", styles['Normal']))
            story.append(Paragraph(f"<b>Model:</b> {model_doc.get('name', 'Unknown')}", styles['Normal']))
            story.append(Paragraph(f"<b>Method:</b> {explanation.get('method', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"<b>Type:</b> {explanation.get('explanation_type', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"<b>Generated:</b> {explanation_data['created_at']}", styles['Normal']))
            story.append(Spacer(1, 20))

            # Add explanation data as table if available
            data = explanation.get("explanation_data", {})
            if "feature_importance" in data:
                story.append(Paragraph("<b>Feature Importance</b>", styles['Heading2']))
                story.append(Spacer(1, 12))

                # Create table
                table_data = [["Feature", "Importance"]]
                for item in data["feature_importance"]:
                    if isinstance(item, dict):
                        feature = item.get("feature", str(item))
                        importance = item.get("importance", 0)
                    else:
                        feature = str(item)
                        importance = 0
                    table_data.append([feature, f"{importance:.4f}"])

                table = Table(table_data, colWidths=[300, 100])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)

            elif "shap_values" in data:
                story.append(Paragraph("<b>SHAP Values</b>", styles['Heading2']))
                story.append(Spacer(1, 12))

                shap_values = data["shap_values"]
                feature_names = data.get("feature_names", [f"Feature_{i}" for i in range(len(shap_values))])

                table_data = [["Feature", "SHAP Value"]]
                for fname, fvalue in zip(feature_names, shap_values):
                    table_data.append([fname, f"{fvalue:.4f}"])

                table = Table(table_data, colWidths=[300, 100])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)

            doc.build(story)
            buffer.seek(0)

            return StreamingResponse(
                buffer,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=explanation_{explanation_id}.pdf"}
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}. Supported formats: json, csv, pdf")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dependence/{model_id}")
async def get_shap_dependence(
    model_id: str,
    feature: str,
    background_data: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Get SHAP dependence data for a specific feature.
    Returns data points for scatter plot with SHAP values vs feature values.
    Requires background dataset to compute SHAP values.
    """
    try:
        db = await get_db()

        # Verify model belongs to user
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Read background data
        contents = await background_data.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        # Check if feature exists in the dataset
        if feature not in df.columns:
            raise HTTPException(status_code=400, detail=f"Feature '{feature}' not found in dataset")

        # Load model
        model_obj, framework = await ModelLoaderService.load_model(model_doc["file_path"])

        # Prepare features (exclude the target if it's in there? Actually just use all except the feature column itself for prediction)
        # For SHAP dependence, we need the entire feature matrix
        feature_columns = [col for col in df.columns if col != feature]
        X = df[feature_columns]

        # Make predictions to use as target? Actually we don't need y for dependence
        # We need the full feature matrix including the feature itself
        X_full = df.copy()

        # Compute SHAP values for background data
        import shap
        import numpy as np

        if framework in ["sklearn", "xgboost", "lightgbm"]:
            try:
                from sklearn.pipeline import Pipeline
                is_pipeline = isinstance(model_obj, Pipeline)
            except Exception:
                is_pipeline = False

            if is_pipeline:
                # For pipelines, preprocess the background data to numeric space
                preprocessor = None
                for step_name, step_obj in model_obj.steps:
                    if hasattr(step_obj, 'transform'):
                        preprocessor = step_obj
                        break

                if preprocessor is not None:
                    X_processed = preprocessor.transform(X_full)
                    if hasattr(X_processed, 'toarray'):
                        X_processed = X_processed.toarray()
                    X_numeric = np.asarray(X_processed, dtype=float)

                    # Background subset for SHAP (use preprocessed numeric)
                    bg = X_numeric[:min(100, len(X_numeric))]

                    # Use final estimator for prediction on preprocessed data
                    final_estimator = model_obj.steps[-1][1]
                    if hasattr(final_estimator, 'predict_proba'):
                        predict_fn = final_estimator.predict_proba
                    else:
                        predict_fn = final_estimator.predict

                    np.random.seed(42)  # Ensure deterministic sampling
                    explainer = shap.KernelExplainer(predict_fn, bg)
                    shap_values = explainer.shap_values(X_numeric)
                else:
                    # No preprocessor, fall back to raw numeric
                    predict_fn = model_obj.predict_proba if hasattr(model_obj, "predict_proba") else model_obj.predict
                    np.random.seed(42)  # Ensure deterministic sampling
                    explainer = shap.KernelExplainer(
                        lambda x: predict_fn(pd.DataFrame(x, columns=X_full.columns)),
                        X_full.iloc[:min(100, len(X_full))].values
                    )
                    shap_values = explainer.shap_values(X_full.values)
            else:
                try:
                    explainer = shap.TreeExplainer(model_obj)
                    shap_values = explainer.shap_values(X_full)
                except Exception:
                    # Some wrapped estimators under sklearn framework are not
                    # compatible with TreeExplainer; use model-agnostic fallback.
                    predict_fn = model_obj.predict_proba if hasattr(model_obj, "predict_proba") else model_obj.predict
                    np.random.seed(42)  # Ensure deterministic sampling
                    explainer = shap.KernelExplainer(
                        lambda x: predict_fn(pd.DataFrame(x, columns=X_full.columns)),
                        X_full.iloc[:min(100, len(X_full))].values
                    )
                    shap_values = explainer.shap_values(X_full.values)

            # For classification, shap_values might be a list (one per class)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
        elif framework == "onnx":
            # For ONNX, use KernelExplainer with a small subset due to complexity
            # Or use a simpler approach
            from shap.maskers import Independent
            np.random.seed(42)  # Ensure deterministic sampling
            explainer = shap.KernelExplainer(
                lambda x: model_obj.run(None, {model_obj.get_inputs()[0].name: x.astype(np.float32)})[0],
                X_full.iloc[:min(100, len(X_full))]  # Use subset for background
            )
            shap_values = explainer.shap_values(X_full)
        else:
            # Fallback to KernelExplainer for unknown frameworks
            np.random.seed(42)  # Ensure deterministic sampling
            explainer = shap.KernelExplainer(
                lambda x: model_obj.predict(pd.DataFrame(x, columns=X_full.columns)),
                X_full.iloc[:min(100, len(X_full))]
            )
            shap_values = explainer.shap_values(X_full)

        # Extract SHAP values for the requested feature
        if isinstance(shap_values, list):
            # For multi-class, take mean absolute value across classes
            shap_vals = np.abs(np.stack(shap_values)).mean(axis=0)
            # Find feature index in the original column order
            feature_index = list(X_full.columns).index(feature)
            shap_feature_values = shap_vals[:, feature_index] if shap_vals.ndim > 1 else shap_vals
        else:
            feature_index = list(X_full.columns).index(feature)
            shap_feature_values = shap_values[:, feature_index] if shap_values.ndim > 1 else shap_values

        # Prepare response data
        response_data = {
            "feature": feature,
            "x_values": df[feature].tolist(),
            "shap_values": shap_feature_values.tolist(),
            "x_label": feature,
            "y_label": "SHAP value",
            "n_samples": len(df),
            "method": "shap_dependence"
        }

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
