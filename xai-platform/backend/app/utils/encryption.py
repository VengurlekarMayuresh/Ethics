"""
Encryption utilities for at-rest data protection.
Uses Fernet (symmetric encryption) for PII encryption.
"""

import os
from cryptography.fernet import Fernet
from typing import Optional

# Load encryption key from environment
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
_fernet = None

def get_cipher() -> Fernet:
    """Get or initialize Fernet cipher."""
    global _fernet
    if _fernet is None:
        if not ENCRYPTION_KEY:
            raise ValueError("ENCRYPTION_KEY environment variable not set")
        # Ensure key is valid bytes
        key = ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY
        _fernet = Fernet(key)
    return _fernet

def encrypt(data: str) -> str:
    """Encrypt a string value."""
    if not data:
        return data
    cipher = get_cipher()
    return cipher.encrypt(data.encode()).decode()

def decrypt(encrypted_data: str) -> str:
    """Decrypt an encrypted string."""
    if not encrypted_data:
        return encrypted_data
    cipher = get_cipher()
    return cipher.decrypt(encrypted_data.encode()).decode()

def encrypt_dict(data: dict, fields: list) -> dict:
    """Encrypt specific fields in a dictionary."""
    encrypted = data.copy()
    for field in fields:
        if field in encrypted and encrypted[field]:
            encrypted[field] = encrypt(str(encrypted[field]))
    return encrypted

def decrypt_dict(data: dict, fields: list) -> dict:
    """Decrypt specific fields in a dictionary."""
    decrypted = data.copy()
    for field in fields:
        if field in decrypted and decrypted[field]:
            decrypted[field] = decrypt(str(decrypted[field]))
    return decrypted
