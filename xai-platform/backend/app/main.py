from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.db.mongo import connect_db, close_db
from app.api.v1 import auth, models, predictions, explanations, bias, compare, api_keys
from app.middleware.rate_limit import RateLimitMiddleware

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.add_middleware(RateLimitMiddleware)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(predictions.router, prefix="/api/v1/predict", tags=["Predictions"])
app.include_router(explanations.router, prefix="/api/v1/explain", tags=["Explanations"])
app.include_router(bias.router, prefix="/api/v1/bias", tags=["Bias"])
app.include_router(compare.router, prefix="/api/v1/compare", tags=["Compare"])
app.include_router(api_keys.router, prefix="/api/v1/api-keys", tags=["API Keys"])
