"""Fine-tuning and evaluation engine for SDGS.

Imports are lazy so the web app can start without torch/transformers/peft
installed.  The heavy dependencies are only loaded when the classes are
actually instantiated (i.e. when a training or evaluation job runs).
"""


def __getattr__(name: str):
    if name == "QwenTrainer":
        from .trainer import QwenTrainer
        return QwenTrainer
    if name == "ModelEvaluator":
        from .evaluator import ModelEvaluator
        return ModelEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["QwenTrainer", "ModelEvaluator"]
