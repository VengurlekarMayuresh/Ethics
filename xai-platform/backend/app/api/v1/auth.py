from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.models.user import UserCreate, UserResponse, Token
from app.utils.auth import get_password_hash, verify_password, create_access_token, create_refresh_token, decode_token, oauth2_scheme
from app.db.mongo import get_db
from app.db.repositories.api_key_repository import APIKeyRepository
from datetime import datetime
from bson import ObjectId

router = APIRouter()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Get current user from JWT or API key.
    """
    # First try JWT
    payload = decode_token(token)
    if payload:
        db = get_db()
        user = await db.users.find_one({"email": payload.get("sub")})
        if user:
            user["_id"] = str(user["_id"])
            return user
        # If JWT valid but user not found, raise
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    # If not JWT, try API key
    api_key_data = await APIKeyRepository.verify(token)
    if api_key_data:
        db = get_db()
        user = await db.users.find_one({"_id": ObjectId(api_key_data["user_id"])})
        if user:
            user["_id"] = str(user["_id"])
            return user
        # API key valid but user not found (shouldn't happen)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    # Neither valid
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")

@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate):
    db = get_db()
    existing_user = await db.users.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    user_dict = user.model_dump()
    user_dict["hashed_password"] = get_password_hash(user_dict.pop("password"))
    user_dict["role"] = "user"
    user_dict["created_at"] = datetime.utcnow()
    
    result = await db.users.insert_one(user_dict)
    user_dict["_id"] = str(result.inserted_id)
    return user_dict

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db = get_db()
    user = await db.users.find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = create_access_token(data={"sub": user["email"]})
    refresh_token = create_refresh_token(data={"sub": user["email"]})
    
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    payload = decode_token(refresh_token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
        
    email = payload.get("sub")
    access_token = create_access_token(data={"sub": email})
    new_refresh_token = create_refresh_token(data={"sub": email})
    
    return {"access_token": access_token, "refresh_token": new_refresh_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user
