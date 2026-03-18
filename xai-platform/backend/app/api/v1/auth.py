from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from app.models.user import UserCreate, UserResponse, Token
from app.utils.auth import get_password_hash, verify_password, create_access_token, create_refresh_token, decode_token, oauth2_scheme
from app.db.mongo import get_db
from app.db.repositories.api_key_repository import APIKeyRepository
from app.utils.audit_logger import log_action, AuditActions
from datetime import datetime
from bson import ObjectId

router = APIRouter()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Get current user from JWT or API key.
    Returns user dict with additional fields:
      - _id: user ID
      - auth_method: "jwt" or "api_key"
      - api_key_scopes: List[str] if authenticated via API key
    """
    # First try JWT
    payload = decode_token(token)
    if payload:
        db = get_db()
        user = await db.users.find_one({"email": payload.get("sub")})
        if user:
            user["_id"] = str(user["_id"])
            user["auth_method"] = "jwt"
            return user
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    # If not JWT, try API key
    api_key_data = await APIKeyRepository.verify(token)
    if api_key_data:
        db = get_db()
        user = await db.users.find_one({"_id": ObjectId(api_key_data["user_id"])})
        if user:
            user["_id"] = str(user["_id"])
            user["auth_method"] = "api_key"
            user["api_key_scopes"] = api_key_data.get("scopes", [])
            return user
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")

@router.post("/register", response_model=UserResponse)
async def register(request: Request, user: UserCreate):
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

    # Log audit event
    await log_action(
        user_id=str(result.inserted_id),
        action=AuditActions.USER_REGISTER,
        resource_type="user",
        resource_id=str(result.inserted_id),
        details={"email": user.email},
        request=request
    )

    return user_dict

@router.post("/login", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
):
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

    # Log audit event
    await log_action(
        user_id=str(user["_id"]),
        action=AuditActions.USER_LOGIN,
        resource_type="user",
        resource_id=str(user["_id"]),
        request=request
    )

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


async def require_scope(required_scope: str, current_user: dict = Depends(get_current_user)):
    """
    Dependency to require a specific API key scope.
    For JWT users, always allowed (bypass scope check).
    For API key users, checks that the required scope is present.
    """
    if current_user.get("auth_method") == "api_key":
        scopes = current_user.get("api_key_scopes", [])
        if required_scope not in scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key does not have required scope: {required_scope}"
            )
    # For JWT, allow all (or implement additional role checks if needed)
    return True

