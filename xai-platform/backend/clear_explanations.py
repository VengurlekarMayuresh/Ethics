"""
One-off script to delete all LIME and SHAP explanation records from MongoDB.
Run from the backend directory:  python clear_explanations.py
"""
import asyncio
from app.db.mongo import connect_db, get_db

async def main():
    await connect_db()
    db = await get_db()
    result = await db.explanations.delete_many({})
    print(f"Deleted {result.deleted_count} explanation record(s).")

if __name__ == "__main__":
    asyncio.run(main())
