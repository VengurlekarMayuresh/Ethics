import boto3
from botocore.exceptions import ClientError
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
import io, asyncio
from typing import Any, Dict, List

class MinIOClient:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=f"http://{settings.MINIO_ENDPOINT}:{settings.MINIO_PORT}",
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            region_name='us-east-1'
        )
        self.bucket = settings.MINIO_BUCKET
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.s3_client.create_bucket(Bucket=self.bucket)

    async def upload_file(self, file_content: bytes, object_name: str) -> str:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.s3_client.upload_fileobj, io.BytesIO(file_content), self.bucket, object_name)
        return object_name

    async def download_file(self, object_name: str) -> bytes:
        loop = asyncio.get_event_loop()
        def _download():
            response = self.s3_client.get_object(Bucket=self.bucket, Key=object_name)
            return response['Body'].read()
        return await loop.run_in_executor(None, _download)

    async def delete_file(self, object_name: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.s3_client.delete_object, Bucket=self.bucket, Key=object_name)

storage = MinIOClient()

class MongoDB:
    client: AsyncIOMotorClient = None

db = MongoDB()

async def connect_db():
    print(f"Connecting to MongoDB at {settings.MONGODB_URL}")
    db.client = AsyncIOMotorClient(settings.MONGODB_URL)

    # Create collections if they don't exist
    database = db.client.get_default_database()
    existing_collections = await database.list_collection_names()
    
    collections = ["users", "models", "predictions", "explanations", "bias_reports", "audit_logs", "api_keys"]
    for coll in collections:
        if coll not in existing_collections:
            await database.create_collection(coll)
            
    # Add indexes
    await database.users.create_index("email", unique=True)
    await database.api_keys.create_index("key", unique=True)

async def close_db():
    if db.client:
        db.client.close()

async def get_db():
    return db.client.get_default_database()