import os
import sys

def patch_explanations_py():
    path = "backend/app/api/v1/explanations.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "@router.post(\"/interpretml/{model_id}\")" in content:
        print("Endpoints already added to explanations.py")
        return

    routes = """

# ==========================================
# InterpretML Endpoints
# ==========================================
@router.post("/interpretml/{model_id}")
async def request_interpretml_explanation(
    request: Request,
    model_id: str,
    prediction_id: str = None,
    num_features: int = 10,
    file: UploadFile = File(None),
    input_data: str = Form(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        target_prediction_id = None
        if prediction_id:
            pred = await db.predictions.find_one({"_id": ObjectId(prediction_id), "model_id": model_id, "user_id": current_user["_id"]})
            if not pred:
                raise HTTPException(status_code=400, detail="Prediction not found")
            target_prediction_id = prediction_id
        elif input_data:
            input_dict = json.loads(input_data)
            prediction_doc = {
                "model_id": model_id, "user_id": current_user["_id"], "input_data": input_dict,
                "prediction": None, "probability": None, "latency_ms": None, "created_at": datetime.utcnow()
            }
            result = await db.predictions.insert_one(prediction_doc)
            target_prediction_id = str(result.inserted_id)
        else:
            last_pred = await db.predictions.find_one({"model_id": model_id, "user_id": current_user["_id"]}, sort=[("created_at", -1)])
            if last_pred:
                target_prediction_id = str(last_pred["_id"])

        if not target_prediction_id:
            raise HTTPException(status_code=400, detail="No prediction found.")

        task = celery_app.send_task("compute_interpretml_values", args=[target_prediction_id, model_id, num_features])
        celery_task_id = task.id

        await db.predictions.update_one({"_id": ObjectId(target_prediction_id)}, {"$set": {"interpretml_task_id": celery_task_id}})
        await log_action(user_id=current_user["_id"], action=AuditActions.EXPLANATION_CREATE, resource_type="explanation", details={"task_id": celery_task_id, "prediction_id": target_prediction_id, "model_id": model_id, "method": "interpretml"}, request=request)
        return {"message": "InterpretML computation started", "task_id": celery_task_id, "prediction_id": target_prediction_id, "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/interpretml/{task_id}")
async def get_interpretml_result(task_id: str, current_user: dict = Depends(get_current_user)):
    try:
        db = await get_db()
        task = celery_app.AsyncResult(task_id)
        if task.ready():
            result = task.get()
            explanation_data = None
            if isinstance(result, dict) and result.get("explanation_id"):
                explanation = await db.explanations.find_one({"_id": ObjectId(result["explanation_id"])})
                if explanation:
                    explanation["_id"] = str(explanation["_id"])
                    explanation["prediction_id"] = str(explanation["prediction_id"])
                    explanation["model_id"] = str(explanation["model_id"])
                    explanation_data = explanation
            return {"status": "complete", "explanation": explanation_data if explanation_data else result}
        else:
            return {"status": "pending", "task_state": task.state, "info": task.info if task.info else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interpretml/global/{model_id}")
async def request_global_interpretml(
    request: Request,
    model_id: str,
    background_data: UploadFile = File(...),
    num_features: int = Form(10),
    current_user: dict = Depends(get_current_user)
):
    try:
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        contents = await background_data.read()
        bg_object_name = f"{current_user['_id']}/interpretml_bg_{int(datetime.utcnow().timestamp())}_{background_data.filename}"
        await storage.upload_file(contents, bg_object_name)

        task = celery_app.send_task("compute_global_interpretml", args=[model_id, bg_object_name, num_features])
        return {"message": "Global InterpretML computation started", "task_id": task.id, "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/interpretml/global/{model_id}/latest")
async def get_global_interpretml(model_id: str, current_user: dict = Depends(get_current_user)):
    try:
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        query = {"method": "interpretml", "explanation_type": "global"}
        if ObjectId.is_valid(model_id):
            query["model_id"] = {"$in": [model_id, ObjectId(model_id)]}
        else:
            query["model_id"] = model_id

        explanation = await db.explanations.find_one(query, sort=[("created_at", -1)])
        if explanation:
            explanation["_id"] = str(explanation["_id"])
            return explanation
        else:
            raise HTTPException(status_code=404, detail="No global InterpretML explanation found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Alibi Endpoints
# ==========================================
@router.post("/alibi/{model_id}")
async def request_alibi_explanation(
    request: Request,
    model_id: str,
    prediction_id: str = None,
    num_features: int = 10,
    file: UploadFile = File(None),
    input_data: str = Form(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        target_prediction_id = None
        if prediction_id:
            pred = await db.predictions.find_one({"_id": ObjectId(prediction_id), "model_id": model_id, "user_id": current_user["_id"]})
            if not pred:
                raise HTTPException(status_code=400, detail="Prediction not found")
            target_prediction_id = prediction_id
        elif input_data:
            input_dict = json.loads(input_data)
            prediction_doc = {
                "model_id": model_id, "user_id": current_user["_id"], "input_data": input_dict,
                "prediction": None, "probability": None, "latency_ms": None, "created_at": datetime.utcnow()
            }
            result = await db.predictions.insert_one(prediction_doc)
            target_prediction_id = str(result.inserted_id)
        else:
            last_pred = await db.predictions.find_one({"model_id": model_id, "user_id": current_user["_id"]}, sort=[("created_at", -1)])
            if last_pred:
                target_prediction_id = str(last_pred["_id"])

        if not target_prediction_id:
            raise HTTPException(status_code=400, detail="No prediction found.")

        task = celery_app.send_task("compute_alibi_values", args=[target_prediction_id, model_id, num_features])
        celery_task_id = task.id

        await db.predictions.update_one({"_id": ObjectId(target_prediction_id)}, {"$set": {"alibi_task_id": celery_task_id}})
        return {"message": "Alibi computation started", "task_id": celery_task_id, "prediction_id": target_prediction_id, "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alibi/{task_id}")
async def get_alibi_result(task_id: str, current_user: dict = Depends(get_current_user)):
    try:
        db = await get_db()
        task = celery_app.AsyncResult(task_id)
        if task.ready():
            result = task.get()
            explanation_data = None
            if isinstance(result, dict) and result.get("explanation_id"):
                explanation = await db.explanations.find_one({"_id": ObjectId(result["explanation_id"])})
                if explanation:
                    explanation["_id"] = str(explanation["_id"])
                    explanation["prediction_id"] = str(explanation["prediction_id"])
                    explanation["model_id"] = str(explanation["model_id"])
                    explanation_data = explanation
            return {"status": "complete", "explanation": explanation_data if explanation_data else result}
        else:
            return {"status": "pending", "task_state": task.state, "info": task.info if task.info else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alibi/global/{model_id}")
async def request_global_alibi(
    request: Request,
    model_id: str,
    background_data: UploadFile = File(...),
    num_features: int = Form(10),
    current_user: dict = Depends(get_current_user)
):
    try:
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        contents = await background_data.read()
        bg_object_name = f"{current_user['_id']}/alibi_bg_{int(datetime.utcnow().timestamp())}_{background_data.filename}"
        await storage.upload_file(contents, bg_object_name)

        task = celery_app.send_task("compute_global_alibi", args=[model_id, bg_object_name, num_features])
        return {"message": "Global Alibi computation started", "task_id": task.id, "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alibi/global/{model_id}/latest")
async def get_global_alibi(model_id: str, current_user: dict = Depends(get_current_user)):
    try:
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        query = {"method": "alibi", "explanation_type": "global"}
        if ObjectId.is_valid(model_id):
            query["model_id"] = {"$in": [model_id, ObjectId(model_id)]}
        else:
            query["model_id"] = model_id

        explanation = await db.explanations.find_one(query, sort=[("created_at", -1)])
        if explanation:
            explanation["_id"] = str(explanation["_id"])
            return explanation
        else:
            raise HTTPException(status_code=404, detail="No global Alibi explanation found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# AIX360 Endpoints
# ==========================================
@router.post("/aix360/{model_id}")
async def request_aix360_explanation(
    request: Request,
    model_id: str,
    prediction_id: str = None,
    num_features: int = 10,
    file: UploadFile = File(None),
    input_data: str = Form(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        target_prediction_id = None
        if prediction_id:
            pred = await db.predictions.find_one({"_id": ObjectId(prediction_id), "model_id": model_id, "user_id": current_user["_id"]})
            if not pred:
                raise HTTPException(status_code=400, detail="Prediction not found")
            target_prediction_id = prediction_id
        elif input_data:
            input_dict = json.loads(input_data)
            prediction_doc = {
                "model_id": model_id, "user_id": current_user["_id"], "input_data": input_dict,
                "prediction": None, "probability": None, "latency_ms": None, "created_at": datetime.utcnow()
            }
            result = await db.predictions.insert_one(prediction_doc)
            target_prediction_id = str(result.inserted_id)
        else:
            last_pred = await db.predictions.find_one({"model_id": model_id, "user_id": current_user["_id"]}, sort=[("created_at", -1)])
            if last_pred:
                target_prediction_id = str(last_pred["_id"])

        if not target_prediction_id:
            raise HTTPException(status_code=400, detail="No prediction found.")

        task = celery_app.send_task("compute_aix360_values", args=[target_prediction_id, model_id, num_features])
        celery_task_id = task.id

        await db.predictions.update_one({"_id": ObjectId(target_prediction_id)}, {"$set": {"aix360_task_id": celery_task_id}})
        return {"message": "AIX360 computation started", "task_id": celery_task_id, "prediction_id": target_prediction_id, "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/aix360/{task_id}")
async def get_aix360_result(task_id: str, current_user: dict = Depends(get_current_user)):
    try:
        db = await get_db()
        task = celery_app.AsyncResult(task_id)
        if task.ready():
            result = task.get()
            explanation_data = None
            if isinstance(result, dict) and result.get("explanation_id"):
                explanation = await db.explanations.find_one({"_id": ObjectId(result["explanation_id"])})
                if explanation:
                    explanation["_id"] = str(explanation["_id"])
                    explanation["prediction_id"] = str(explanation["prediction_id"])
                    explanation["model_id"] = str(explanation["model_id"])
                    explanation_data = explanation
            return {"status": "complete", "explanation": explanation_data if explanation_data else result}
        else:
            return {"status": "pending", "task_state": task.state, "info": task.info if task.info else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/aix360/global/{model_id}")
async def request_global_aix360(
    request: Request,
    model_id: str,
    background_data: UploadFile = File(...),
    num_features: int = Form(10),
    current_user: dict = Depends(get_current_user)
):
    try:
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        contents = await background_data.read()
        bg_object_name = f"{current_user['_id']}/aix360_bg_{int(datetime.utcnow().timestamp())}_{background_data.filename}"
        await storage.upload_file(contents, bg_object_name)

        task = celery_app.send_task("compute_global_aix360", args=[model_id, bg_object_name, num_features])
        return {"message": "Global AIX360 computation started", "task_id": task.id, "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/aix360/global/{model_id}/latest")
async def get_global_aix360(model_id: str, current_user: dict = Depends(get_current_user)):
    try:
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        query = {"method": "aix360", "explanation_type": "global"}
        if ObjectId.is_valid(model_id):
            query["model_id"] = {"$in": [model_id, ObjectId(model_id)]}
        else:
            query["model_id"] = model_id

        explanation = await db.explanations.find_one(query, sort=[("created_at", -1)])
        if explanation:
            explanation["_id"] = str(explanation["_id"])
            return explanation
        else:
            raise HTTPException(status_code=404, detail="No global AIX360 explanation found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""

    with open(path, "a", encoding="utf-8") as f:
        f.write(routes)
    print("Patched explanations.py")


def patch_tasks_py():
    path = "backend/app/workers/tasks.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "@celery_app.task(bind=True, name=\"compute_interpretml_values\")" in content:
        print("Tasks already added to tasks.py")
        return

    # Create a generic task factory wrapper for these to avoid duplicating 200 lines each time
    tasks = """

# ==========================================
# New XAI Framework Tasks (InterpretML, Alibi, AIX360)
# ==========================================

def _run_xai_framework_local(self, prediction_id: str, model_id: str, method: str, num_features: int = 10):
    try:
        logger.info(f"[{method} local] START: prediction_id={prediction_id}, model_id={model_id}")
        self.update_state(state="PROGRESS", meta={"status": "loading model", "progress": 10})

        import asyncio
        async def async_task():
            from app.db.mongo import connect_db, get_db
            from app.services.model_loader_service import ModelLoaderService
            import pandas as pd
            import numpy as np
            from bson import ObjectId
            from datetime import datetime
            import traceback

            # Dynamic import with fallback
            try:
                if method == "interpretml":
                    from app.services.interpretml_service import interpretml_service as service
                elif method == "alibi":
                    from app.services.alibi_service import alibi_service as service
                elif method == "aix360":
                    from app.services.aix360_service import aix360_service as service
                else:
                    raise ValueError(f"Unknown method {method}")
            except ImportError as ie:
                logger.error(f"Framework import error for {method}: {ie}")
                return {"status": "failed", "error": f"{method.upper()} library not installed or import error. Detailed error: {ie}"}

            await connect_db()
            db = await get_db()

            prediction = await db.predictions.find_one({"_id": ObjectId(prediction_id)})
            model = await db.models.find_one({"_id": ObjectId(model_id)})
            if not prediction or not model:
                raise ValueError("Prediction or Model not found")

            model_obj, framework = await ModelLoaderService.load_model(model["file_path"])
            input_df = pd.DataFrame([prediction["input_data"]])
            
            # Simple dummy background dataset for exact format requirements
            training_df = input_df.copy()
            for _ in range(50):
                noisy = input_df.copy()
                for col in noisy.columns:
                    if pd.api.types.is_numeric_dtype(noisy[col]):
                        noisy[col] = noisy[col] + np.random.normal(0, max(float(noisy[col].iloc[0]) * 0.1, 1e-2))
                training_df = pd.concat([training_df, noisy], ignore_index=True)

            task_type = "classification" if model.get("task_type") in ["classification", "binary_classification"] else "regression"
            
            explainer = service.create_explainer(model_obj, framework, training_df, list(training_df.columns), mode=task_type)
            
            local_exp_data = service.explain_instance(
                explainer,
                model_obj,
                input_df.values[0],
                num_features=num_features
            )

            explanation_doc = {
                "prediction_id": prediction_id,
                "model_id": model_id,
                "method": method,
                "explanation_type": "local",
                "explanation_data": local_exp_data, # Use a generic payload
                "feature_names": list(training_df.columns),
                "task_id": self.request.id,
                "task_status": "complete",
                "created_at": datetime.utcnow()
            }

            result = await db.explanations.insert_one(explanation_doc)
            self.update_state(state="SUCCESS", meta={"status": "complete", "explanation_id": str(result.inserted_id), "progress": 100})
            return {"explanation_id": str(result.inserted_id), "status": "complete"}

        return asyncio.run(async_task())
    except Exception as e:
        import traceback
        logger.error(f"[{method} task failed] {traceback.format_exc()}")
        return {"status": "failed", "error": str(e)}

def _run_xai_framework_global(self, model_id: str, background_data_path: str, method: str, num_features: int = 10):
    try:
        logger.info(f"[{method} global] START: model_id={model_id}")
        self.update_state(state="PROGRESS", meta={"status": "loading model", "progress": 10})

        import asyncio
        async def async_task():
            from app.db.mongo import connect_db, get_db
            from app.services.model_loader_service import ModelLoaderService
            from app.utils.file_handler import storage
            import pandas as pd
            from bson import ObjectId
            from datetime import datetime

            # Dynamic import with fallback
            try:
                if method == "interpretml":
                    from app.services.interpretml_service import interpretml_service as service
                elif method == "alibi":
                    from app.services.alibi_service import alibi_service as service
                elif method == "aix360":
                    from app.services.aix360_service import aix360_service as service
                else:
                    raise ValueError(f"Unknown method {method}")
            except ImportError as ie:
                logger.error(f"Framework import error for {method}: {ie}")
                return {"status": "failed", "error": f"{method.upper()} library not installed or import error. Detailed error: {ie}"}

            await connect_db()
            db = await get_db()

            model = await db.models.find_one({"_id": ObjectId(model_id)})
            if not model:
                raise ValueError("Model not found")

            model_obj, framework = await ModelLoaderService.load_model(model["file_path"])
            
            # Load Background
            bg_bytes = await storage.download_file(background_data_path)
            training_df = pd.read_csv(importlib.util.find_spec('io') and __import__('io').BytesIO(bg_bytes))

            task_type = "classification" if model.get("task_type") in ["classification", "binary_classification"] else "regression"
            
            explainer = service.create_explainer(model_obj, framework, training_df, list(training_df.columns), mode=task_type)
            
            global_exp_data = service.explain_global(
                explainer,
                model_obj,
                training_df,
                num_features=num_features
            )

            explanation_doc = {
                "model_id": model_id,
                "method": method,
                "explanation_type": "global",
                "explanation_data": global_exp_data, # Generic payload
                "feature_names": list(training_df.columns),
                "task_id": self.request.id,
                "task_status": "complete",
                "created_at": datetime.utcnow()
            }

            result = await db.explanations.insert_one(explanation_doc)
            self.update_state(state="SUCCESS", meta={"status": "complete", "explanation_id": str(result.inserted_id), "progress": 100})
            return {"explanation_id": str(result.inserted_id), "status": "complete"}

        return asyncio.run(async_task())
    except Exception as e:
        import traceback
        logger.error(f"[{method} global task failed] {traceback.format_exc()}")
        return {"status": "failed", "error": str(e)}

@celery_app.task(bind=True, name="compute_interpretml_values")
def compute_interpretml_values(self, prediction_id: str, model_id: str, num_features: int = 10):
    return _run_xai_framework_local(self, prediction_id, model_id, "interpretml", num_features)

@celery_app.task(bind=True, name="compute_global_interpretml")
def compute_global_interpretml(self, model_id: str, background_data_path: str, num_features: int = 10):
    import importlib
    return _run_xai_framework_global(self, model_id, background_data_path, "interpretml", num_features)

@celery_app.task(bind=True, name="compute_alibi_values")
def compute_alibi_values(self, prediction_id: str, model_id: str, num_features: int = 10):
    return _run_xai_framework_local(self, prediction_id, model_id, "alibi", num_features)

@celery_app.task(bind=True, name="compute_global_alibi")
def compute_global_alibi(self, model_id: str, background_data_path: str, num_features: int = 10):
    import importlib
    return _run_xai_framework_global(self, model_id, background_data_path, "alibi", num_features)

@celery_app.task(bind=True, name="compute_aix360_values")
def compute_aix360_values(self, prediction_id: str, model_id: str, num_features: int = 10):
    return _run_xai_framework_local(self, prediction_id, model_id, "aix360", num_features)

@celery_app.task(bind=True, name="compute_global_aix360")
def compute_global_aix360(self, model_id: str, background_data_path: str, num_features: int = 10):
    import importlib
    return _run_xai_framework_global(self, model_id, background_data_path, "aix360", num_features)
"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(tasks)
    print("Patched tasks.py")

if __name__ == "__main__":
    patch_explanations_py()
    patch_tasks_py()

