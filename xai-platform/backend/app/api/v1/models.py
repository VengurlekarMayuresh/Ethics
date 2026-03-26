from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from app.models.model_meta import ModelResponse, FeatureSchema
from app.api.v1.auth import get_current_user
from app.db.mongo import get_db
from app.services.model_loader_service import ModelLoaderService
from app.utils.file_handler import storage
from app.utils.audit_logger import log_action, AuditActions
from datetime import datetime
from bson import ObjectId
import json
from typing import List, Optional

router = APIRouter()

@router.post("/upload", response_model=ModelResponse)
async def upload_model(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    framework: str = Form(...),
    task_type: Optional[str] = Form(None),  # Auto-detected if not provided
    feature_schema: str = Form("[]"),  # JSON string, optional (auto-generated if empty)
    file: UploadFile = File(...),
    background_data: UploadFile = File(None),  # Optional CSV dataset
    current_user: dict = Depends(get_current_user)
):
    try:
        manual_schema = json.loads(feature_schema) if feature_schema.strip() else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid feature_schema JSON")

    # Read model file content
    model_file_content = await file.read()
    if not model_file_content:
        raise HTTPException(status_code=400, detail="Model file is empty")

    # Load model for analysis (without storing yet)
    try:
        model_obj, detected_framework = await ModelLoaderService.load_model_from_bytes(
            model_file_content, file.filename
        )
        # Use detected framework from file extension
        framework_to_use = detected_framework

        # Attempt to detect task_type from model
        model_info = await ModelLoaderService.get_model_info(model_obj, framework_to_use)
        detected_task_type = model_info.get("task_type", "unknown")

        # Prefer auto-detected task_type; fall back to provided if detection fails
        if detected_task_type != "unknown":
            task_type_to_use = detected_task_type
        elif task_type:
            task_type_to_use = task_type
        else:
            raise HTTPException(
                status_code=400,
                detail="Could not automatically determine task type (classification/regression). Please provide it manually."
            )
    except Exception as e:
        msg = str(e)
        if not msg.lower().startswith("failed to load model"):
            msg = f"Failed to load model: {msg}"
        raise HTTPException(status_code=400, detail=msg)

    # Analyze dataset if provided
    dataset_analysis = None
    background_data_path = None
    if background_data:
        dataset_bytes = await background_data.read()
        if dataset_bytes:
            try:
                dataset_analysis = await ModelLoaderService.analyze_dataset(dataset_bytes)
                # Upload dataset to storage
                bg_object_name = f"{current_user['_id']}/background_{int(datetime.utcnow().timestamp())}_{background_data.filename}"
                await storage.upload_file(dataset_bytes, bg_object_name)
                background_data_path = bg_object_name
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to analyze dataset: {str(e)}")

    # Generate or use feature schema
    if not manual_schema:
        # Auto-generate from model and optionally dataset
        feature_schema_objs = await ModelLoaderService.generate_feature_schema(
            model_obj, framework_to_use, dataset_analysis
        )
        # Convert to dict list for storage
        schema_to_store = [fs.dict() for fs in feature_schema_objs]
    else:
        # Use manually provided schema (override auto-generation)
        schema_to_store = manual_schema

    # Ensure we have at least some features; otherwise model unusable
    if not schema_to_store:
        raise HTTPException(
            status_code=400,
            detail="Could not detect feature names. Please provide a feature schema manually or upload a dataset alongside the model."
        )

    # Upload model file to storage
    model_object_name = f"{current_user['_id']}/{int(datetime.utcnow().timestamp())}_{file.filename}"
    await storage.upload_file(model_file_content, model_object_name)

    # Optionally detect model category (could store for explainability)
    try:
        model_category = await ModelLoaderService.detect_model_category(model_obj, framework_to_use)
    except:
        model_category = "unknown"

    # Save metadata to DB
    db = await get_db()
    model_doc = {
        "user_id": current_user["_id"],
        "name": name,
        "description": description,
        "framework": framework_to_use,
        "task_type": task_type_to_use,
        "feature_schema": schema_to_store,
        "file_path": model_object_name,
        "background_data_path": background_data_path,
        "protected_attributes": [],
        "tags": [],
        "version": "1.0",
        "metrics": {},
        "model_category": model_category,  # Store for explainability
        # New: specific estimator details
        "model_type": model_info["estimator_info"]["estimator_name"],
        "model_family": model_info["estimator_info"]["estimator_family"],
        "is_tree_based": model_info["estimator_info"]["is_tree_based"],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    result = await db.models.insert_one(model_doc)
    model_doc["_id"] = str(result.inserted_id)

    # Log audit event
    await log_action(
        user_id=current_user["_id"],
        action=AuditActions.MODEL_UPLOAD,
        resource_type="model",
        resource_id=str(result.inserted_id),
        details={"name": name, "framework": framework_to_use, "task_type": task_type, "auto_features": not manual_schema},
        request=request
    )

    return model_doc

@router.get("/", response_model=List[ModelResponse])
async def list_models(current_user: dict = Depends(get_current_user)):
    db = await get_db()
    cursor = db.models.find({"user_id": current_user["_id"]}).sort("created_at", -1)
    models = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        models.append(doc)
    return models

@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    db = await get_db()
    model = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model["_id"] = str(model["_id"])

    # Log audit event (viewing model details)
    await log_action(
        user_id=current_user["_id"],
        action=AuditActions.MODEL_VIEW,
        resource_type="model",
        resource_id=model_id,
        details={"model_name": model.get("name")},
        request=request
    )

    return model

@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    db = await get_db()
    model = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    model_name = model.get("name")

    if model.get("file_path"):
        try:
            await storage.delete_file(model["file_path"])
        except Exception as e:
            print(f"Error deleting file from MinIO: {e}")

    await db.models.delete_one({"_id": ObjectId(model_id)})

    # Log audit event
    await log_action(
        user_id=current_user["_id"],
        action=AuditActions.MODEL_DELETE,
        resource_type="model",
        resource_id=model_id,
        details={"model_name": model_name},
        request=request
    )

    return {"status": "success", "message": "Model deleted"}
