"""Training & evaluation API: fine-tuning, judge evaluation, metrics."""
import datetime
import json
import os
import re
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..auth import decrypt_value
from ..config import BASE_DIR
from ..db.database import get_db, SessionLocal
from ..db.models import ApiKey, Dataset, TrainingRun, EvaluationRun, QAPair
from ..deps import CurrentUser, get_current_user
from ..engine.training_service import get_knobs, set_knob
from ..schemas import (
    StartTrainingRequest,
    TrainingRunResponse,
    TrainingRunListResponse,
    StartEvaluationRequest,
    EvaluationRunResponse,
    EvaluationRunListResponse,
    EvaluationDetailResponse,
    KnobsRequest,
    CorrectionRequest,
    MergeConvertRequest,
    MergeConvertResponse,
    PushModelRequest,
    PushModelResponse,
    ConfigInfo,
    ConfigListResponse,
    ArtifactEntry,
    ArtifactListResponse,
)
from ..services.job_runner import submit_training_job, cancel_training_job

router = APIRouter()


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _resolve_dataset_path(
    req: StartTrainingRequest,
    current_user: CurrentUser,
    db: Session,
) -> str:
    """Resolve a .jsonl path from the request, falling back through:
    1. Explicit ``dataset_path``
    2. ``Dataset.output_path`` from ``dataset_id``
    3. Most recent completed Dataset for this user
    If the resolved file doesn't exist, try building JSONL from QAPair DB
    records.
    """
    path: str | None = None
    dataset_id: int | None = req.dataset_id

    # 1. Explicit path
    if req.dataset_path:
        path = req.dataset_path

    # 2. From dataset_id
    if not path and dataset_id:
        ds = db.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.user_id == current_user.id,
        ).first()
        if not ds:
            raise HTTPException(404, "Dataset not found")
        path = ds.output_path

    # 3. Latest completed dataset
    if not path:
        ds = (
            db.query(Dataset)
            .filter(Dataset.user_id == current_user.id, Dataset.status == "completed")
            .order_by(Dataset.completed_at.desc())
            .first()
        )
        if ds:
            dataset_id = ds.id
            path = ds.output_path

    if not path:
        raise HTTPException(400, "No dataset path could be resolved. Provide dataset_id or dataset_path.")

    # If the file is missing, try building from DB
    if not Path(path).exists() and dataset_id:
        path = _build_jsonl_from_db(dataset_id, db)

    if not Path(path).exists():
        raise HTTPException(400, f"Dataset file not found: {path}")

    return path


def _build_jsonl_from_db(dataset_id: int, db: Session) -> str:
    """Build a .jsonl file from QAPair records in the database."""
    pairs = db.query(QAPair).filter(
        QAPair.dataset_id == dataset_id,
        QAPair.is_valid == True,  # noqa: E712
    ).all()

    if not pairs:
        raise HTTPException(400, "No valid QA pairs found in database for this dataset")

    out_dir = Path("data") / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dataset_{dataset_id}.jsonl"

    with open(out_path, "w") as f:
        for qa in pairs:
            f.write(json.dumps({"instruction": qa.instruction, "output": qa.output}) + "\n")

    return str(out_path)


def _load_yaml_configs(req: StartTrainingRequest):
    """Load YAML configs when config names are provided.

    Returns (model_config, training_config, dataset_config) — any may be None.
    """
    from ..engine.trainer import discover_configs, load_config

    model_config = None
    training_config = None
    dataset_config = None

    if req.model_config_name:
        configs = discover_configs("models")
        if req.model_config_name not in configs:
            raise HTTPException(
                400, f"Model config '{req.model_config_name}' not found. "
                f"Available: {list(configs.keys())}"
            )
        model_config = load_config(configs[req.model_config_name])

    if req.training_config_name:
        configs = discover_configs("training")
        if req.training_config_name not in configs:
            raise HTTPException(
                400, f"Training config '{req.training_config_name}' not found. "
                f"Available: {list(configs.keys())}"
            )
        training_config = load_config(configs[req.training_config_name])

    if req.dataset_config_name:
        configs = discover_configs("datasets")
        if req.dataset_config_name not in configs:
            raise HTTPException(
                400, f"Dataset config '{req.dataset_config_name}' not found. "
                f"Available: {list(configs.keys())}"
            )
        dataset_config = load_config(configs[req.dataset_config_name])

    return model_config, training_config, dataset_config


# -------------------------------------------------------------------------
# GET /configs/{config_type} — List available YAML configs
# -------------------------------------------------------------------------

@router.get("/configs/{config_type}", response_model=ConfigListResponse)
def list_configs(
    config_type: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    if config_type not in ("models", "datasets", "training"):
        raise HTTPException(400, "config_type must be 'models', 'datasets', or 'training'")

    from ..engine.trainer import discover_configs, load_config

    configs = discover_configs(config_type)
    items = []
    for name, path in configs.items():
        cfg = load_config(path)
        items.append(ConfigInfo(
            name=name,
            display_name=cfg.get("name", name),
            path=str(path),
        ))
    return ConfigListResponse(configs=items)


# -------------------------------------------------------------------------
# GET /artifacts — List adapters, GGUF files, checkpoints, merged models
# -------------------------------------------------------------------------

@router.get("/artifacts", response_model=ArtifactListResponse)
def list_artifacts(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    adapters: list[ArtifactEntry] = []
    gguf_files: list[ArtifactEntry] = []
    checkpoints: list[ArtifactEntry] = []
    merged_models: list[ArtifactEntry] = []
    seen_adapter_paths: set[str] = set()

    outputs_dir = BASE_DIR / "outputs"
    gguf_dir = BASE_DIR / "models" / "gguf"

    # Scan outputs/ for adapters, checkpoints, merged models
    if outputs_dir.is_dir():
        for run_dir in sorted(outputs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name

            # Adapters: subdir containing final_adapter/
            final_adapter = run_dir / "final_adapter"
            if final_adapter.is_dir():
                path = str(final_adapter)
                adapters.append(ArtifactEntry(path=path, label=run_name))
                seen_adapter_paths.add(path)

            # Adapters: subdir with adapter_config.json directly
            if (run_dir / "adapter_config.json").exists() and str(run_dir) not in seen_adapter_paths:
                path = str(run_dir)
                adapters.append(ArtifactEntry(path=path, label=run_name))
                seen_adapter_paths.add(path)

            # Checkpoints: checkpoint-* subdirs
            for ckpt in sorted(run_dir.glob("checkpoint-*")):
                if ckpt.is_dir():
                    try:
                        ckpt_num = ckpt.name.split("-", 1)[1]
                    except IndexError:
                        ckpt_num = ckpt.name
                    checkpoints.append(ArtifactEntry(
                        path=str(ckpt),
                        label=f"{run_name}/checkpoint-{ckpt_num}",
                    ))

            # Merged models: subdir containing a config.json but not adapter_config.json
            # (merged models are full model dirs, not LoRA adapters)
            if (run_dir / "merged").is_dir():
                merged_models.append(ArtifactEntry(
                    path=str(run_dir / "merged"),
                    label=f"{run_name}/merged",
                ))

    # Scan models/gguf/ for GGUF files
    if gguf_dir.is_dir():
        for gguf_file in sorted(gguf_dir.glob("*.gguf")):
            if gguf_file.is_file():
                gguf_files.append(ArtifactEntry(
                    path=str(gguf_file),
                    label=gguf_file.name,
                ))

    # Fallback: adapters from DB training runs not found on disk
    db_runs = db.query(TrainingRun).filter(
        TrainingRun.user_id == current_user.id,
        TrainingRun.adapter_path.isnot(None),
    ).all()
    for run in db_runs:
        if run.adapter_path and run.adapter_path not in seen_adapter_paths:
            if Path(run.adapter_path).exists():
                adapters.append(ArtifactEntry(
                    path=run.adapter_path,
                    label=run.run_name,
                ))
                seen_adapter_paths.add(run.adapter_path)

    return ArtifactListResponse(
        adapters=adapters,
        gguf_files=gguf_files,
        checkpoints=checkpoints,
        merged_models=merged_models,
    )


# -------------------------------------------------------------------------
# POST /start — Start fine-tuning
# -------------------------------------------------------------------------

@router.post("/start", response_model=TrainingRunResponse)
def start_training(
    req: StartTrainingRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Load YAML configs if config names provided
    yaml_model_config, yaml_training_config, yaml_dataset_config = _load_yaml_configs(req)

    # In config-driven mode with a HF dataset, dataset_path resolution is optional
    if yaml_dataset_config and not yaml_dataset_config.get("is_local", False):
        # HF dataset — no local path needed; use a placeholder
        dataset_path = req.dataset_path or "config-driven"
    else:
        dataset_path = _resolve_dataset_path(req, current_user, db)

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = f"train-{ts}"

    run = TrainingRun(
        user_id=current_user.id,
        dataset_id=req.dataset_id,
        run_name=run_name,
        status="pending",
        base_model=req.base_model,
        model_size=req.model_size,
        lora_rank=req.lora_rank,
        lora_alpha=req.lora_alpha,
        learning_rate=req.learning_rate,
        num_epochs=req.num_epochs,
        batch_size=req.batch_size,
        gradient_accumulation_steps=req.gradient_accumulation_steps,
        max_steps=req.max_steps,
        dataset_path=dataset_path,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Build config dicts — merge YAML configs with request params
    model_config = yaml_model_config or {
        "model_name": req.base_model,
        "size": req.model_size,
        "lora": {"r": req.lora_rank, "lora_alpha": req.lora_alpha},
    }
    training_config = yaml_training_config or {
        "learning_rate": req.learning_rate,
        "num_train_epochs": req.num_epochs,
        "per_device_train_batch_size": req.batch_size,
        "gradient_accumulation_steps": req.gradient_accumulation_steps,
        "max_steps": req.max_steps,
    }

    # Capture for closure
    _dataset_config = yaml_dataset_config
    _resume = req.resume_from_checkpoint

    from ..engine.training_service import train_with_metrics

    def _run(cancel_event=None):
        return train_with_metrics(
            run_id=run.id,
            dataset_path=dataset_path,
            model_config=model_config,
            training_config=training_config,
            dataset_config=_dataset_config,
            resume_from_checkpoint=_resume,
            cancel_event=cancel_event,
        )

    submit_training_job(run.id, _run, model_class="TrainingRun")

    return run


# -------------------------------------------------------------------------
# GET / — List training runs
# -------------------------------------------------------------------------

@router.get("", response_model=TrainingRunListResponse)
def list_training_runs(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(TrainingRun).filter(TrainingRun.user_id == current_user.id)
    total = q.count()
    runs = q.order_by(TrainingRun.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()
    return TrainingRunListResponse(runs=runs, total=total, page=page, per_page=per_page)


# -------------------------------------------------------------------------
# GET /{run_id} — Training run details
# -------------------------------------------------------------------------

@router.get("/{run_id}", response_model=TrainingRunResponse)
def get_training_run(
    run_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = db.query(TrainingRun).filter(
        TrainingRun.id == run_id,
        TrainingRun.user_id == current_user.id,
    ).first()
    if not run:
        raise HTTPException(404, "Training run not found")
    return run


# -------------------------------------------------------------------------
# POST /{run_id}/cancel — Cancel training
# -------------------------------------------------------------------------

@router.post("/{run_id}/cancel")
def cancel_training(
    run_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = db.query(TrainingRun).filter(
        TrainingRun.id == run_id,
        TrainingRun.user_id == current_user.id,
    ).first()
    if not run:
        raise HTTPException(404, "Training run not found")
    if run.status not in ("pending", "running"):
        raise HTTPException(400, f"Cannot cancel run with status '{run.status}'")

    cancelled = cancel_training_job(run_id, model_class="TrainingRun")
    return {"cancelled": cancelled}


# -------------------------------------------------------------------------
# POST /evaluate — Start evaluation
# -------------------------------------------------------------------------

@router.post("/evaluate", response_model=EvaluationRunResponse)
def start_evaluation(
    req: StartEvaluationRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Resolve model path
    model_path = req.model_path
    training_run: TrainingRun | None = None

    if not model_path and req.training_run_id:
        training_run = db.query(TrainingRun).filter(
            TrainingRun.id == req.training_run_id,
            TrainingRun.user_id == current_user.id,
        ).first()
        if not training_run:
            raise HTTPException(404, "Training run not found")
        if training_run.status != "completed":
            raise HTTPException(400, "Training run has not completed yet")
        model_path = training_run.adapter_path

    if not model_path:
        raise HTTPException(400, "Provide model_path or training_run_id")

    if not Path(model_path).exists():
        raise HTTPException(400, f"Model path not found: {model_path}")

    # Resolve test dataset path
    test_path = req.test_dataset_path
    if not test_path and training_run and training_run.dataset_path:
        test_path = training_run.dataset_path
    if not test_path:
        raise HTTPException(400, "Provide test_dataset_path or a training_run_id that has a dataset_path")
    if not Path(test_path).exists():
        raise HTTPException(400, f"Test dataset not found: {test_path}")

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = f"eval-{ts}"

    eval_run = EvaluationRun(
        user_id=current_user.id,
        training_run_id=req.training_run_id,
        run_name=run_name,
        status="pending",
        model_path=model_path,
        test_dataset_path=test_path,
        judge_model=req.judge_model,
        max_samples=req.max_samples,
    )
    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)

    def _run(cancel_event=None):
        from ..engine.evaluator import ModelEvaluator
        evaluator = ModelEvaluator(
            model_path=model_path,
            test_dataset_path=test_path,
            judge_model=req.judge_model,
            max_samples=req.max_samples,
            cancel_event=cancel_event,
        )
        return evaluator.run_evaluation()

    submit_training_job(eval_run.id, _run, model_class="EvaluationRun")

    return eval_run


# -------------------------------------------------------------------------
# GET /evaluations — List evaluation runs
# -------------------------------------------------------------------------

@router.get("/evaluations", response_model=EvaluationRunListResponse)
def list_evaluation_runs(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(EvaluationRun).filter(EvaluationRun.user_id == current_user.id)
    total = q.count()
    evals = q.order_by(EvaluationRun.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()
    return EvaluationRunListResponse(evaluations=evals, total=total, page=page, per_page=per_page)


# -------------------------------------------------------------------------
# GET /evaluations/{eval_id} — Evaluation detail with per-sample results
# -------------------------------------------------------------------------

@router.get("/evaluations/{eval_id}", response_model=EvaluationDetailResponse)
def get_evaluation_detail(
    eval_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    eval_run = db.query(EvaluationRun).filter(
        EvaluationRun.id == eval_id,
        EvaluationRun.user_id == current_user.id,
    ).first()
    if not eval_run:
        raise HTTPException(404, "Evaluation run not found")

    resp = EvaluationDetailResponse.model_validate(eval_run)
    resp.per_sample_results = eval_run.results_json
    resp.articles_log = eval_run.articles_json
    resp.correction_results = eval_run.correction_json
    return resp


# -------------------------------------------------------------------------
# POST /{run_id}/knobs — Adjust training hyperparameters mid-run
# -------------------------------------------------------------------------

@router.post("/{run_id}/knobs")
def update_knobs(
    run_id: int,
    req: KnobsRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = db.query(TrainingRun).filter(
        TrainingRun.id == run_id,
        TrainingRun.user_id == current_user.id,
    ).first()
    if not run:
        raise HTTPException(404, "Training run not found")
    if run.status != "running":
        raise HTTPException(400, f"Cannot adjust knobs on run with status '{run.status}'")

    knobs = get_knobs(run_id)
    if knobs is None:
        raise HTTPException(400, "No active knobs for this run (training may have finished)")

    if req.learning_rate is not None:
        set_knob(run_id, "learning_rate", req.learning_rate)

    return {"run_id": run_id, "knobs": get_knobs(run_id)}


# -------------------------------------------------------------------------
# POST /evaluate/{eval_id}/correct — Run Claude correction on failing samples
# -------------------------------------------------------------------------

@router.post("/evaluate/{eval_id}/correct")
def start_correction(
    eval_id: int,
    req: CorrectionRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    eval_run = db.query(EvaluationRun).filter(
        EvaluationRun.id == eval_id,
        EvaluationRun.user_id == current_user.id,
    ).first()
    if not eval_run:
        raise HTTPException(404, "Evaluation run not found")
    if eval_run.status != "completed":
        raise HTTPException(400, "Evaluation must be completed before correction")
    if not eval_run.results_json:
        raise HTTPException(400, "No per-sample results available for correction")

    # Resolve Anthropic API key: request param > stored ApiKey > env var
    api_key = req.api_key
    if not api_key:
        stored = db.query(ApiKey).filter(
            ApiKey.user_id == current_user.id,
            ApiKey.provider_name == "anthropic",
        ).first()
        if stored and current_user.encryption_key:
            try:
                api_key = decrypt_value(stored.encrypted_key, current_user.encryption_key)
            except Exception:
                pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(400, "No Anthropic API key found. Provide api_key, store one in settings, or set ANTHROPIC_API_KEY env var.")

    # Resolve dataset path from the associated training run
    dataset_path = None
    if eval_run.training_run_id:
        training_run = db.query(TrainingRun).filter(
            TrainingRun.id == eval_run.training_run_id,
        ).first()
        if training_run:
            dataset_path = training_run.dataset_path
    if not dataset_path:
        dataset_path = eval_run.test_dataset_path
    if not dataset_path or not Path(dataset_path).exists():
        raise HTTPException(400, "Cannot resolve dataset path for appending corrections")

    per_sample = eval_run.results_json
    articles_log = eval_run.articles_json or []

    # Capture eval_id for the daemon thread closure
    _eval_id = eval_id
    _score_threshold = req.score_threshold
    _model = req.model

    def _run_correction():
        from ..engine.correction_agent import CorrectionAgent
        agent = CorrectionAgent(
            api_key=api_key,
            score_threshold=_score_threshold,
            model=_model,
        )
        summary = agent.run_correction(
            eval_id=_eval_id,
            per_sample_results=per_sample,
            dataset_path=dataset_path,
            articles_log=articles_log,
        )
        # Store results in DB
        sess = SessionLocal()
        try:
            row = sess.query(EvaluationRun).filter(EvaluationRun.id == _eval_id).first()
            if row:
                row.correction_json = summary
                sess.commit()
        finally:
            sess.close()

    thread = threading.Thread(target=_run_correction, daemon=True)
    thread.start()

    return {"eval_id": eval_id, "status": "correction_started"}


# -------------------------------------------------------------------------
# POST /convert — LoRA merge + GGUF conversion
# -------------------------------------------------------------------------

@router.post("/convert", response_model=MergeConvertResponse)
def start_convert(
    req: MergeConvertRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    if not Path(req.adapter_path).exists():
        raise HTTPException(400, f"Adapter path not found: {req.adapter_path}")

    from ..engine.merge_convert import merge_and_convert

    gguf_path = merge_and_convert(
        adapter_path=req.adapter_path,
        base_model=req.base_model,
        quant_method=req.quant_method,
        output_name=req.output_name,
        keep_merged=req.keep_merged,
    )

    return MergeConvertResponse(gguf_path=gguf_path)


# -------------------------------------------------------------------------
# POST /push — Upload model to HuggingFace Hub
# -------------------------------------------------------------------------

@router.post("/push", response_model=PushModelResponse)
def push_to_hf(
    req: PushModelRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Resolve HF token from stored settings or env
    token = os.environ.get("HF_TOKEN")
    if not token:
        stored = db.query(ApiKey).filter(
            ApiKey.user_id == current_user.id,
            ApiKey.provider_name == "huggingface",
        ).first()
        if stored and current_user.encryption_key:
            try:
                token = decrypt_value(stored.encrypted_key, current_user.encryption_key)
            except Exception:
                pass
    if not token:
        raise HTTPException(400, "No HuggingFace token found. Set HF_TOKEN env var or store in settings.")

    from ..engine.push_hf import push_gguf, push_merged

    if req.gguf_path:
        if not Path(req.gguf_path).exists():
            raise HTTPException(400, f"GGUF file not found: {req.gguf_path}")
        url = push_gguf(
            gguf_path=req.gguf_path,
            repo_id=req.repo_id,
            private=req.private,
            token=token,
            base_model=req.base_model,
            description=req.description,
            dataset=req.dataset,
            author=req.author,
        )
    elif req.merged_model_dir:
        if not Path(req.merged_model_dir).exists():
            raise HTTPException(400, f"Model dir not found: {req.merged_model_dir}")
        url = push_merged(
            model_dir=req.merged_model_dir,
            repo_id=req.repo_id,
            private=req.private,
            token=token,
            base_model=req.base_model,
            description=req.description,
            dataset=req.dataset,
            author=req.author,
        )
    else:
        raise HTTPException(400, "Provide gguf_path or merged_model_dir")

    return PushModelResponse(repo_url=url)
