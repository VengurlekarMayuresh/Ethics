import boto3
from botocore.exceptions import ClientError
from app.config import settings
import io, asyncio

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
