from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.db.mongo import connect_db, close_db
from app.api.v1 import auth, models, predictions, explanations, bias, compare, api_keys, audit, notifications
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import time

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await close_db()

app = FastAPI(
    title="XAI Platform API",
    version="1.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    print(f"Path: {request.url.path} Method: {request.method} Status: {response.status_code} Duration: {duration:.4f}s")
    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error for path {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )

# Add CORS last so it wraps all responses, including middleware errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(predictions.router, prefix="/api/v1/predict", tags=["Predictions"])
app.include_router(explanations.router, prefix="/api/v1/explain", tags=["Explanations"])
app.include_router(bias.router, prefix="/api/v1/bias", tags=["Bias"])
app.include_router(compare.router, prefix="/api/v1/compare", tags=["Compare"])
app.include_router(api_keys.router, prefix="/api/v1/api-keys", tags=["API Key Management"])
app.include_router(audit.router, prefix="/api/v1/audit", tags=["Audit"])
app.include_router(notifications.router, tags=["Notifications"])
