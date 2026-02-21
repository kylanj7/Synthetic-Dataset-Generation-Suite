"""Dataset API: CRUD, pipeline execution, samples, HuggingFace push."""
import os
import re
import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..auth import decrypt_value
from ..db.database import get_db
from ..db.models import Dataset, ApiKey, QAPair
from ..deps import CurrentUser, get_current_user
from ..schemas import (
    CreateDatasetRequest, DatasetResponse, DatasetListResponse,
    DatasetSamplesResponse, QAPairResponse,
    HFPushRequest, HFPushResponse,
)
from ..services.job_runner import submit_job, cancel_job
from ..services.dataset_service import run_dataset_pipeline

router = APIRouter()


def _resolve_api_key(
    provider: str | None,
    current_user: CurrentUser,
    db: Session,
) -> str | None:
    """Resolve API key: user's encrypted key -> env var -> None."""
    if provider and current_user.encryption_key:
        stored = db.query(ApiKey).filter(
            ApiKey.user_id == current_user.id,
            ApiKey.provider_name == provider,
        ).first()
        if stored:
            try:
                return decrypt_value(stored.encrypted_key, current_user.encryption_key)
            except Exception:
                pass

    # Environment variable fallback
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
    }
    if provider and provider in env_map:
        val = os.environ.get(env_map[provider])
        if val:
            return val

    return None


@router.post("", response_model=DatasetResponse)
def create_dataset(
    req: CreateDatasetRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    safe_topic = re.sub(r'[^a-zA-Z0-9_ -]', '', req.topic)[:80]
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"{safe_topic}_{ts}"

    ds = Dataset(
        user_id=current_user.id,
        name=name,
        topic=req.topic,
        status="pending",
        provider=req.provider,
        model=req.model,
        target_size=req.target_size,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    # Resolve API key
    api_key = _resolve_api_key(req.provider, current_user, db)

    # Submit pipeline to background executor
    submit_job(
        ds.id,
        run_dataset_pipeline,
        dataset_id=ds.id,
        topic=req.topic,
        provider=req.provider,
        model=req.model,
        api_key=api_key,
        target_size=req.target_size,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
    )

    return DatasetResponse.model_validate(ds)


@router.get("", response_model=DatasetListResponse)
def list_datasets(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Dataset).filter(Dataset.user_id == current_user.id)
    total = query.count()
    datasets = (
        query.order_by(Dataset.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    return DatasetListResponse(
        datasets=[DatasetResponse.model_validate(ds) for ds in datasets],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    dataset_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return DatasetResponse.model_validate(ds)


@router.post("/{dataset_id}/cancel")
def cancel_dataset(
    dataset_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")

    if cancel_job(dataset_id):
        return {"status": "cancelled"}
    raise HTTPException(400, "Cannot cancel — dataset is not running or already completed")


@router.get("/{dataset_id}/samples", response_model=DatasetSamplesResponse)
def get_dataset_samples(
    dataset_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    search: str | None = None,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")

    query = db.query(QAPair).filter(QAPair.dataset_id == dataset_id)

    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            QAPair.instruction.ilike(search_pattern) |
            QAPair.output.ilike(search_pattern)
        )

    total = query.count()
    samples = (
        query.order_by(QAPair.id)
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    return DatasetSamplesResponse(
        samples=[
            QAPairResponse(
                id=qa.id,
                instruction=qa.instruction,
                output=qa.output,
                is_valid=qa.is_valid,
                was_healed=qa.was_healed,
                source_title=qa.source_title,
                think_text=qa.think_text,
                answer_text=qa.answer_text,
            )
            for qa in samples
        ],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.post("/{dataset_id}/push-hf", response_model=HFPushResponse)
def push_to_huggingface(
    dataset_id: int,
    req: HFPushRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    if ds.status != "completed":
        raise HTTPException(400, "Dataset must be completed before pushing to HuggingFace")

    # Get HF token
    from ..db.models import User
    user = db.query(User).filter(User.id == current_user.id).first()
    if not user or not user.hf_token:
        raise HTTPException(400, "HuggingFace token not configured. Add it in Settings.")

    try:
        hf_token = decrypt_value(user.hf_token, current_user.encryption_key)
    except Exception:
        raise HTTPException(400, "Cannot decrypt HF token. Please re-login and try again.")

    from ..services.hf_service import push_dataset_to_hf
    repo_url = push_dataset_to_hf(
        hf_token=hf_token,
        repo_name=req.repo_name,
        dataset_path=ds.output_path or "",
        topic=ds.topic,
        pair_count=ds.actual_size,
        valid_count=ds.valid_count,
        provider=ds.provider,
        model=ds.model,
        description=req.description,
        private=req.private,
    )

    ds.hf_repo = req.repo_name
    db.commit()

    return HFPushResponse(repo_url=repo_url, hf_repo=req.repo_name)
