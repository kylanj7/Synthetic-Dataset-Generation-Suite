"""Dataset API: CRUD, pipeline execution, samples, HuggingFace push."""
import os
import re
import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..auth import decrypt_value
from ..db.database import get_db
from ..db.models import Dataset, ApiKey, QAPair, User
from ..deps import CurrentUser, get_current_user
from ..schemas import (
    CreateDatasetRequest, BatchCreateRequest, DatasetResponse, DatasetListResponse,
    DatasetSamplesResponse, QAPairResponse,
    HFPushRequest, HFPushResponse,
    CreateFromPapersRequest, ImportHFRequest,
)
from ..services.job_runner import submit_job, cancel_job
from ..services.dataset_service import run_dataset_pipeline, run_from_papers_pipeline, import_from_huggingface

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
        max_tokens=req.max_tokens,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    # Resolve API key
    api_key = _resolve_api_key(req.provider, current_user, db)

    # Resolve Semantic Scholar API key
    s2_api_key = None
    if current_user.encryption_key:
        user = db.query(User).filter(User.id == current_user.id).first()
        if user and user.s2_token:
            try:
                s2_api_key = decrypt_value(user.s2_token, current_user.encryption_key)
            except Exception:
                pass
    if not s2_api_key:
        s2_api_key = os.environ.get("S2_API_KEY")

    # Submit pipeline to background executor
    pipeline_kwargs = dict(
        dataset_id=ds.id,
        topic=req.topic,
        provider=req.provider,
        model=req.model,
        api_key=api_key,
        target_size=req.target_size,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
        s2_api_key=s2_api_key,
        max_tokens=req.max_tokens,
    )
    submit_job(ds.id, run_dataset_pipeline, **pipeline_kwargs)

    return DatasetResponse.model_validate(ds)


@router.post("/batch", response_model=list[DatasetResponse])
def create_batch_datasets(
    req: BatchCreateRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    results = []
    for item in req.datasets:
        safe_topic = re.sub(r'[^a-zA-Z0-9_ -]', '', item.topic)[:80]
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name = f"{safe_topic}_{ts}"

        ds = Dataset(
            user_id=current_user.id,
            name=name,
            topic=item.topic,
            status="pending",
            provider=item.provider,
            model=item.model,
            target_size=item.target_size,
            system_prompt=item.system_prompt,
            temperature=item.temperature,
            max_tokens=item.max_tokens,
        )
        db.add(ds)
        db.commit()
        db.refresh(ds)

        api_key = _resolve_api_key(item.provider, current_user, db)

        s2_api_key = None
        if current_user.encryption_key:
            user = db.query(User).filter(User.id == current_user.id).first()
            if user and user.s2_token:
                try:
                    s2_api_key = decrypt_value(user.s2_token, current_user.encryption_key)
                except Exception:
                    pass
        if not s2_api_key:
            s2_api_key = os.environ.get("S2_API_KEY")

        pipeline_kwargs = dict(
            dataset_id=ds.id,
            topic=item.topic,
            provider=item.provider,
            model=item.model,
            api_key=api_key,
            target_size=item.target_size,
            system_prompt=item.system_prompt,
            temperature=item.temperature,
            s2_api_key=s2_api_key,
            max_tokens=item.max_tokens,
        )
        submit_job(ds.id, run_dataset_pipeline, **pipeline_kwargs)
        results.append(DatasetResponse.model_validate(ds))

    return results


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


@router.post("/from-papers", response_model=DatasetResponse)
def create_dataset_from_papers(
    req: CreateFromPapersRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from ..db.models import Paper
    papers = db.query(Paper).filter(
        Paper.id.in_(req.paper_ids),
        Paper.user_id == current_user.id,
    ).all()
    if len(papers) != len(req.paper_ids):
        raise HTTPException(400, "Some paper IDs not found or not owned by you")

    # Build topic from paper titles
    titles = [p.title for p in papers[:3]]
    topic_label = ", ".join(t[:40] for t in titles)
    if len(papers) > 3:
        topic_label += f" (+{len(papers) - 3} more)"

    safe_topic = re.sub(r'[^a-zA-Z0-9_ -]', '', titles[0])[:50]
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"papers_{safe_topic}_{ts}"

    ds = Dataset(
        user_id=current_user.id,
        name=name,
        topic=f"Custom: {topic_label}",
        status="pending",
        provider=req.provider,
        model=req.model,
        target_size=len(papers) * 5,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    api_key = _resolve_api_key(req.provider, current_user, db)

    pipeline_kwargs = dict(
        dataset_id=ds.id,
        paper_ids=req.paper_ids,
        provider=req.provider,
        model=req.model,
        api_key=api_key,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    submit_job(ds.id, run_from_papers_pipeline, **pipeline_kwargs)

    return DatasetResponse.model_validate(ds)


@router.post("/import-hf", response_model=DatasetResponse)
def import_hf_dataset(
    req: ImportHFRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Derive name from repo_id
    repo_short = req.repo_id.split("/")[-1] if "/" in req.repo_id else req.repo_id
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"import_{repo_short}_{ts}"

    ds = Dataset(
        user_id=current_user.id,
        name=name,
        topic=f"Import: {req.repo_id}",
        status="pending",
        provider=None,
        model=None,
        target_size=0,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    # Get HF token
    hf_token = None
    from ..db.models import User as UserModel
    user = db.query(UserModel).filter(UserModel.id == current_user.id).first()
    if user and user.hf_token and current_user.encryption_key:
        try:
            hf_token = decrypt_value(user.hf_token, current_user.encryption_key)
        except Exception:
            pass
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    pipeline_kwargs = dict(
        dataset_id=ds.id,
        repo_id=req.repo_id,
        hf_token=hf_token,
        split=req.split,
    )
    submit_job(ds.id, import_from_huggingface, **pipeline_kwargs)

    return DatasetResponse.model_validate(ds)


@router.delete("/{dataset_id}/samples/{qa_id}")
def delete_qa_pair(
    dataset_id: int,
    qa_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")

    qa = db.query(QAPair).filter(
        QAPair.id == qa_id,
        QAPair.dataset_id == dataset_id,
    ).first()
    if not qa:
        raise HTTPException(404, "QA pair not found")

    # Update counters
    ds.actual_size = max(0, ds.actual_size - 1)
    if qa.is_valid:
        ds.valid_count = max(0, ds.valid_count - 1)
    else:
        ds.invalid_count = max(0, ds.invalid_count - 1)

    # Decrement source paper count
    if qa.paper_id:
        from ..db.models import Paper
        paper = db.query(Paper).filter(Paper.id == qa.paper_id).first()
        if paper:
            paper.qa_pair_count = max(0, paper.qa_pair_count - 1)

    db.delete(qa)
    db.commit()
    return {"status": "deleted"}


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

    if ds.status not in ("pending", "running"):
        raise HTTPException(400, "Dataset is not running")

    if cancel_job(dataset_id):
        return {"status": "cancelled"}

    # Job not in executor (stale/lost) — mark as cancelled directly
    ds.status = "cancelled"
    ds.completed_at = datetime.datetime.utcnow()
    db.commit()
    return {"status": "cancelled"}


@router.delete("/{dataset_id}")
def delete_dataset(
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

    if ds.status in ("pending", "running"):
        cancel_job(dataset_id)

    db.delete(ds)
    db.commit()
    return {"status": "deleted"}


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

    # Gather source papers for citations
    from ..db.models import Paper
    papers = db.query(Paper).filter(Paper.dataset_id == dataset_id).all()
    sources = []
    for p in papers:
        sources.append({
            "title": p.title,
            "authors": p.authors or [],
            "year": p.year,
            "url": p.url or "",
            "doi": p.doi or "",
            "source": p.source or "",
            "citation_count": p.citation_count or 0,
            "qa_pair_count": p.qa_pair_count or 0,
        })

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
        total_tokens=ds.total_tokens or 0,
        prompt_tokens=ds.prompt_tokens or 0,
        completion_tokens=ds.completion_tokens or 0,
        gpu_kwh=ds.gpu_kwh or 0.0,
        sources=sources,
    )

    ds.hf_repo = req.repo_name
    db.commit()

    return HFPushResponse(repo_url=repo_url, hf_repo=req.repo_name)
