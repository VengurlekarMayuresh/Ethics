from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

class MongoDB:
    client: AsyncIOMotorClient = None

db = MongoDB()

async def connect_db():
    print(f"Connecting to MongoDB at {settings.MONGODB_URL}")
    db.client = AsyncIOMotorClient(settings.MONGODB_URL)
    
async def close_db():
    if db.client:
        db.client.close()

def get_db():
    return db.client.get_default_database()
