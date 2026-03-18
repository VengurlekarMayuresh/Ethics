import redis
import time
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import settings
from typing import Callable
import secrets

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using Redis.
    Limits:
    - Anonymous: 60 requests per minute per IP
    - Authenticated (JWT): 300 requests per minute per user ID
    - API Key: 500 requests per minute per API key
    """

    def __init__(self, app):
        super().__init__(app)
        self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        # Limits
        self.anonymous_limit = 60
        self.jwt_limit = 300
        self.api_key_limit = 500
        self.window = 60  # 1 minute window

    async def dispatch(self, request: Request, call_next: Callable):
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)

        # Identify client
        identifier = None
        limit = self.anonymous_limit

        # Check for API key in Authorization header (Bearer token could be JWT or API key)
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer "
            # Try to verify as JWT first (short JWT tokens are JWT, API keys are longer random strings)
            # We'll try to decode as JWT. If decode_token fails, treat as API key.
            from app.utils.auth import decode_token
            payload = decode_token(token)
            if payload:
                # JWT authenticated
                identifier = f"jwt:{payload.get('sub')}"
                limit = self.jwt_limit
            else:
                # Could be API key - verify using API key repository
                from app.db.repositories.api_key_repository import APIKeyRepository
                # First, get by prefix to avoid full scan? We'll just try verify directly which will scan by prefix internally.
                api_key_data = await APIKeyRepository.verify(token)
                if api_key_data:
                    identifier = f"apikey:{api_key_data['id']}"
                    limit = self.api_key_limit
        else:
            # No Bearer, use IP address
            client_host = request.client.host if request.client else "unknown"
            identifier = f"ip:{client_host}"

        if identifier:
            # Check rate limit
            key = f"ratelimit:{identifier}"
            current = self.redis_client.get(key)
            if current and int(current) >= limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(self.window)}
                )
            # Increment count, set expiry if first request
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.ttl(key)
            count, ttl = pipe.execute()
            if ttl == -1:
                self.redis_client.expire(key, self.window)
        else:
            # Should not happen but fallback
            pass

        response = await call_next(request)
        return response
