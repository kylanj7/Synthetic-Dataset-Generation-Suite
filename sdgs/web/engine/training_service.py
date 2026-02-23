"""Orchestration layer wiring QwenTrainer to the broadcast queue."""
import threading
from typing import Any, Dict, Optional

from ..services.broadcast import BroadcastType, enqueue_broadcast

# Module-level registry of live knob dicts keyed by run_id.
_active_knobs: Dict[int, Dict[str, Any]] = {}
_knobs_lock = threading.Lock()


def get_knobs(run_id: int) -> Optional[Dict[str, Any]]:
    with _knobs_lock:
        return _active_knobs.get(run_id)


def set_knob(run_id: int, key: str, value: Any) -> None:
    with _knobs_lock:
        knobs = _active_knobs.get(run_id)
        if knobs is not None:
            knobs[key] = value


def train_with_metrics(
    run_id: int,
    dataset_path: str,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    dataset_config: Optional[Dict[str, Any]] = None,
    resume_from_checkpoint: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Run a full training pipeline with live metric broadcasting.

    Designed to be called from _run_training_job (sync, in thread pool).
    """
    from .trainer import QwenTrainer

    knobs: Dict[str, Any] = {}
    with _knobs_lock:
        _active_knobs[run_id] = knobs

    def on_metric(msg: Dict[str, Any]) -> None:
        step = msg.get("step", 0)
        max_steps = msg.get("max_steps", 0)
        loss = msg.get("loss")
        lr = msg.get("learning_rate")
        pct = round(step / max_steps * 100, 1) if max_steps else 0

        parts = [f"step {step}/{max_steps} ({pct}%)"]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        msg["monologue"] = " | ".join(parts)

        enqueue_broadcast(BroadcastType.TRAINING, run_id, msg)

    try:
        trainer = QwenTrainer(
            dataset_path=dataset_path,
            model_config=model_config,
            training_config=training_config,
            dataset_config=dataset_config,
            cancel_event=cancel_event,
            on_metric=on_metric,
            knobs=knobs,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        result = trainer.run_full_pipeline()

        enqueue_broadcast(BroadcastType.TRAINING, run_id, {
            "type": "complete",
            "monologue": "Training finished.",
            **{k: v for k, v in result.items() if k in (
                "final_loss", "total_steps", "training_runtime_seconds",
            )},
        })

        return result
    finally:
        with _knobs_lock:
            _active_knobs.pop(run_id, None)
