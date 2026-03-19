from jose import jwt
from datetime import datetime, timedelta

secret = "super-secret-jwt-key-change-in-production"
algo = "HS256"

data = {"sub": "test@example.com", "exp": datetime.utcnow() + timedelta(minutes=15)}
token = jwt.encode(data, secret, algorithm=algo)
print(f"Generated token: {token}")

try:
    decoded = jwt.decode(token, secret, algorithms=[algo])
    print(f"Decoded data: {decoded}")
    if decoded["sub"] == "test@example.com":
        print("Verification SUCCESS")
    else:
        print("Verification FAILED: wrong sub")
except Exception as e:
    print(f"Verification FAILED with error: {str(e)}")
