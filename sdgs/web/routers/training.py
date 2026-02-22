"""Training & evaluation API: fine-tuning, judge evaluation, metrics."""
import datetime
import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..db.models import Dataset, TrainingRun, EvaluationRun, QAPair
from ..deps import CurrentUser, get_current_user
from ..schemas import (
    StartTrainingRequest,
    TrainingRunResponse,
    TrainingRunListResponse,
    StartEvaluationRequest,
    EvaluationRunResponse,
    EvaluationRunListResponse,
    EvaluationDetailResponse,
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


# -------------------------------------------------------------------------
# POST /start — Start fine-tuning
# -------------------------------------------------------------------------

@router.post("/start", response_model=TrainingRunResponse)
def start_training(
    req: StartTrainingRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
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

    # Build config dicts for the trainer
    model_config = {
        "model_name": req.base_model,
        "size": req.model_size,
        "lora": {"r": req.lora_rank, "lora_alpha": req.lora_alpha},
    }
    training_config = {
        "learning_rate": req.learning_rate,
        "num_train_epochs": req.num_epochs,
        "per_device_train_batch_size": req.batch_size,
        "gradient_accumulation_steps": req.gradient_accumulation_steps,
        "max_steps": req.max_steps,
    }

    def _run(cancel_event=None):
        from ..engine.trainer import QwenTrainer
        trainer = QwenTrainer(
            dataset_path=dataset_path,
            model_config=model_config,
            training_config=training_config,
            cancel_event=cancel_event,
        )
        return trainer.run_full_pipeline()

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
    return resp
