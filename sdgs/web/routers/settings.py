"""Settings API: API key CRUD and HuggingFace token management."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import encrypt_value, decrypt_value
from ..db.database import get_db
from ..db.models import ApiKey, User
from ..deps import CurrentUser, get_current_user
from ..schemas import ApiKeyInfo, SaveApiKeyRequest, HFTokenStatus, SaveHFTokenRequest

router = APIRouter()


def _mask_key(key: str) -> str:
    if len(key) <= 4:
        return "****"
    return "*" * (len(key) - 4) + key[-4:]


# --- API Keys ---

@router.get("/keys", response_model=list[ApiKeyInfo])
def list_api_keys(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    keys = db.query(ApiKey).filter(ApiKey.user_id == current_user.id).all()
    result = []
    for k in keys:
        try:
            decrypted = decrypt_value(k.encrypted_key, current_user.encryption_key)
            masked = _mask_key(decrypted)
        except Exception:
            masked = "****"
        result.append(ApiKeyInfo(
            provider_name=k.provider_name,
            masked_key=masked,
            updated_at=k.updated_at,
        ))
    return result


@router.put("/keys/{provider}")
def save_api_key(
    provider: str,
    req: SaveApiKeyRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.encryption_key:
        raise HTTPException(400, "Re-login required for key operations")

    encrypted = encrypt_value(req.api_key, current_user.encryption_key)

    existing = db.query(ApiKey).filter(
        ApiKey.user_id == current_user.id,
        ApiKey.provider_name == provider,
    ).first()

    if existing:
        existing.encrypted_key = encrypted
    else:
        db.add(ApiKey(
            user_id=current_user.id,
            provider_name=provider,
            encrypted_key=encrypted,
        ))

    db.commit()
    return {"status": "ok"}


@router.delete("/keys/{provider}")
def delete_api_key(
    provider: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    key = db.query(ApiKey).filter(
        ApiKey.user_id == current_user.id,
        ApiKey.provider_name == provider,
    ).first()

    if not key:
        raise HTTPException(404, "API key not found")

    db.delete(key)
    db.commit()
    return {"status": "ok"}


# --- HuggingFace Token ---

@router.get("/hf-token", response_model=HFTokenStatus)
def get_hf_token_status(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == current_user.id).first()
    return HFTokenStatus(configured=bool(user and user.hf_token))


@router.put("/hf-token")
def save_hf_token(
    req: SaveHFTokenRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.encryption_key:
        raise HTTPException(400, "Re-login required for key operations")

    user = db.query(User).filter(User.id == current_user.id).first()
    user.hf_token = encrypt_value(req.token, current_user.encryption_key)
    db.commit()
    return {"status": "ok"}


@router.delete("/hf-token")
def delete_hf_token(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == current_user.id).first()
    user.hf_token = None
    db.commit()
    return {"status": "ok"}
