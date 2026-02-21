"""Authentication utilities: password hashing, JWT tokens, Fernet encryption."""
import os
import base64
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt
from cryptography.fernet import Fernet

# --- JWT config ---
SECRET_KEY = os.environ.get("SDGS_JWT_SECRET", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
REFRESH_TOKEN_EXPIRE_DAYS = 7


# --- Password hashing (using bcrypt directly) ---

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(user_id: int, username: str, encryption_key: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "username": username,
        "enc_key": encryption_key,
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "type": "refresh",
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# --- Fernet encryption for API keys ---

def derive_fernet_key(password: str, salt: bytes) -> str:
    """Derive a Fernet key from user password + salt using PBKDF2."""
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations=480_000, dklen=32)
    return base64.urlsafe_b64encode(dk).decode()


def generate_salt() -> bytes:
    return secrets.token_bytes(16)


def encrypt_value(plaintext: str, fernet_key: str) -> str:
    f = Fernet(fernet_key.encode())
    return f.encrypt(plaintext.encode()).decode()


def decrypt_value(ciphertext: str, fernet_key: str) -> str:
    f = Fernet(fernet_key.encode())
    return f.decrypt(ciphertext.encode()).decode()
