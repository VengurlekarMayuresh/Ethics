from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from app.models.model_meta import ModelCreate, ModelResponse, FeatureSchema
from app.api.v1.auth import get_current_user
from app.db.mongo import get_db
from app.utils.file_handler import storage
from app.utils.audit_logger import log_action, AuditActions
from datetime import datetime
from bson import ObjectId
import json
from typing import List

router = APIRouter()

@router.post("/upload", response_model=ModelResponse)
async def upload_model(
    name: str = Form(...),
    description: str = Form(""),
    framework: str = Form(...),
    task_type: str = Form(...),
    feature_schema: str = Form("[]"), # JSON string
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        schema = json.loads(feature_schema)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid feature_schema JSON")

    # Read and upload file to MinIO
    file_content = await file.read()
    object_name = f"{current_user['_id']}/{int(datetime.utcnow().timestamp())}_{file.filename}"
    storage.upload_file(file_content, object_name)

    # Save metadata to DB
    db = get_db()
    model_doc = {
        "user_id": current_user["_id"],
        "name": name,
        "description": description,
        "framework": framework,
        "task_type": task_type,
        "feature_schema": schema,
        "file_path": object_name,
        "protected_attributes": [],
        "tags": [],
        "version": "1.0",
        "metrics": {},
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
        details={"name": name, "framework": framework, "task_type": task_type},
        request=request
    )

    return model_doc

@router.get("/", response_model=List[ModelResponse])
async def list_models(current_user: dict = Depends(get_current_user)):
    db = get_db()
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
    db = get_db()
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
    db = get_db()
    model = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    model_name = model.get("name")

    if model.get("file_path"):
        try:
            storage.delete_file(model["file_path"])
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
