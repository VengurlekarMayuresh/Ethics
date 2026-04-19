from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MONGODB_URL: str = "mongodb://localhost:27017/xai"
    REDIS_URL: str = "redis://localhost:6379"
    
    MINIO_ENDPOINT: str = "localhost"
    MINIO_PORT: int = 9000
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "xai-models"
    
    JWT_SECRET: str = "super-secret-jwt-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours for easier development
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    # Comma-separated module paths that define custom classes used in pickled models.
    # These classes are injected into __main__ during sklearn/joblib loading to support
    # models serialized from notebooks/scripts where classes were defined in __main__.
    PICKLE_CLASS_MODULES: str = "app.custom.loan_feature_engineer"
    
    OPENROUTER_API_KEY: str = ""


    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
