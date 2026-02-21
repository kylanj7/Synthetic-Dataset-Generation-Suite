"""Provider listing API with per-user key status."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..db.models import ApiKey
from ..deps import CurrentUser, get_current_user
from ..schemas import ProviderInfo

router = APIRouter()


@router.get("/providers", response_model=list[ProviderInfo])
def get_providers(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from sdgs.providers import list_providers, load_provider_config

    # Get user's stored keys
    user_keys = {
        k.provider_name
        for k in db.query(ApiKey).filter(ApiKey.user_id == current_user.id).all()
    }

    result = []
    for name in list_providers():
        config = load_provider_config(name)
        result.append(ProviderInfo(
            name=name,
            default_model=config.get("default_model", ""),
            api_key_env=config.get("api_key_env"),
            has_key=name in user_keys,
        ))
    return result
