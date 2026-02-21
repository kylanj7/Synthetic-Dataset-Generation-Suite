"""Authentication endpoints: register, login, refresh."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    decode_token, derive_fernet_key, generate_salt,
)
from ..db.database import get_db
from ..db.models import User
from ..schemas import RegisterRequest, LoginRequest, TokenResponse, RefreshRequest

router = APIRouter()


@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if len(req.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters",
        )

    existing = db.query(User).filter(User.username == req.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        )

    salt = generate_salt()
    enc_key = derive_fernet_key(req.password, salt)

    user = User(
        username=req.username,
        password_hash=hash_password(req.password),
        encryption_key_salt=salt.hex(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return TokenResponse(
        access_token=create_access_token(user.id, user.username, enc_key),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    salt = bytes.fromhex(user.encryption_key_salt)
    enc_key = derive_fernet_key(req.password, salt)

    return TokenResponse(
        access_token=create_access_token(user.id, user.username, enc_key),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(req: RefreshRequest, db: Session = Depends(get_db)):
    payload = decode_token(req.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    user_id = int(payload["sub"])
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    # For refresh, we can't derive the encryption key without the password.
    # Return a token with empty enc_key — frontend should re-login for key ops.
    return TokenResponse(
        access_token=create_access_token(user.id, user.username, ""),
        refresh_token=create_refresh_token(user.id),
    )
