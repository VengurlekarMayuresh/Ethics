import boto3
from botocore.exceptions import ClientError
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
import io
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

    def upload_file(self, file_content: bytes, object_name: str) -> str:
        self.s3_client.upload_fileobj(io.BytesIO(file_content), self.bucket, object_name)
        return object_name

    def download_file(self, object_name: str) -> bytes:
        response = self.s3_client.get_object(Bucket=self.bucket, Key=object_name)
        return response['Body'].read()

    def delete_file(self, object_name: str):
        self.s3_client.delete_object(Bucket=self.bucket, Key=object_name)

storage = MinIOClient()

class MongoDB:
    client: AsyncIOMotorClient = None

db = MongoDB()

async def connect_db():
    print(f"Connecting to MongoDB at {settings.MONGODB_URL}")
    db.client = AsyncIOMotorClient(settings.MONGODB_URL)

    # Create collections if they don't exist
    db.client.get_database().create_collection("users", validator={})
    db.client.get_database().create_collection("models", validator={})
    db.client.get_database().create_collection("predictions", validator={})
    db.client.get_database().create_collection("explanations", validator={})
    db.client.get_database().create_collection("bias_reports", validator={})
    db.client.get_database().create_collection("audit_logs", validator={})
    db.client.get_database().create_collection("api_keys", validator={})

async def close_db():
    if db.client:
        db.client.close()

async def get_db():
    return db.client.get_default_database()