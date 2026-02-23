"""QwenTrainer — configuration-driven fine-tuning engine.

Refactored from Qwen-Fine-Tuning-Lab/train.py.  All interactive prompts,
argparse, and wandb integration have been stripped.  The trainer is driven
entirely by dicts passed at construction time.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer


# ---------------------------------------------------------------------------
# MetricsCallback — streams step-level metrics + applies live knob changes
# ---------------------------------------------------------------------------

class MetricsCallback(TrainerCallback):
    """Emits training metrics via a callable and applies live knob adjustments."""

    def __init__(
        self,
        on_metric: Callable[[Dict[str, Any]], None],
        knobs: Optional[Dict[str, Any]] = None,
    ):
        self._on_metric = on_metric
        self._knobs = knobs or {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        self._on_metric({
            "type": "metric",
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else 0,
            "max_steps": state.max_steps,
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
        })

    def on_step_end(self, args, state, control, **kwargs):
        new_lr = self._knobs.get("learning_rate")
        if new_lr is None:
            return
        model = kwargs.get("model")
        if model is None:
            return
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(new_lr)
        self._knobs.pop("learning_rate", None)

# ---------------------------------------------------------------------------
# Defaults (matching Qwen 2.5 14B)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "model_name": "Qwen/Qwen2.5-14B-Instruct",
    "size": "14B",
    "max_seq_length": 2048,
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
    },
    "lora": {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
    "tokenizer": {
        "trust_remote_code": True,
        "padding_side": "right",
    },
}

DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "num_train_epochs": 1,
    "max_steps": -1,
    "learning_rate": 5e-5,
    "max_grad_norm": 1.0,
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "seed": 3407,
    "save_strategy": "steps",
    "save_steps": 25,
    "save_total_limit": 2,
    "eval_strategy": "no",
    "output_dir": "outputs",
}


class QwenTrainer:
    """Dict-driven Qwen fine-tuning pipeline."""

    def __init__(
        self,
        dataset_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[threading.Event] = None,
        on_metric: Optional[Callable[[Dict[str, Any]], None]] = None,
        knobs: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_path = dataset_path
        self.model_config = {**DEFAULT_MODEL_CONFIG, **(model_config or {})}
        self.training_config = {**DEFAULT_TRAINING_CONFIG, **(training_config or {})}
        self.cancel_event = cancel_event
        self.on_metric = on_metric
        self.knobs = knobs or {}

        self.model = None
        self.tokenizer = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self._run_name = f"sdgs-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # ------------------------------------------------------------------
    # Cancellation helper
    # ------------------------------------------------------------------

    def _check_cancel(self):
        if self.cancel_event and self.cancel_event.is_set():
            raise InterruptedError("Training cancelled by user")

    # ------------------------------------------------------------------
    # load_model  (from train.py:329-403)
    # ------------------------------------------------------------------

    def load_model(self):
        """Load model + tokenizer with BitsAndBytes 4-bit quantisation and LoRA."""
        self._check_cancel()

        mcfg = self.model_config
        model_name = mcfg["model_name"]
        quant_cfg = mcfg.get("quantization", {})
        lora_cfg = mcfg.get("lora", {})
        tok_cfg = mcfg.get("tokenizer", {})

        print(f"Loading model: {model_name}")

        compute_dtype = (
            torch.bfloat16
            if quant_cfg.get("bnb_4bit_compute_dtype") == "bfloat16"
            else torch.float16
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.get("load_in_4bit", True),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=tok_cfg.get("trust_remote_code", True),
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = tok_cfg.get("padding_side", "right")
        print("Tokenizer loaded.")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Model loaded.")

        self.model = prepare_model_for_kbit_training(self.model)

        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 16),
            lora_dropout=lora_cfg.get("lora_dropout", 0),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # prepare_dataset  (from train.py:494-564)
    # ------------------------------------------------------------------

    def prepare_dataset(self) -> Tuple[int, int, int]:
        """Load JSONL with ``{instruction, output}`` format, apply Qwen chat
        template, and create 80/10/10 splits.

        Returns (train_count, val_count, test_count).
        """
        self._check_cancel()

        print(f"Loading dataset: {self.dataset_path}")
        dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        print(f"Dataset loaded: {len(dataset)} examples")

        # Apply Qwen chat template
        def _format(examples):
            texts = []
            for inst, out in zip(examples["instruction"], examples["output"]):
                text = (
                    f"<|im_start|>user\n{inst}<|im_end|>\n"
                    f"<|im_start|>assistant\n{out}<|im_end|>"
                )
                texts.append(text)
            return {"text": texts}

        dataset = dataset.map(_format, batched=True)

        # 80/10/10 split
        seed = self.training_config.get("seed", 3407)
        split1 = dataset.train_test_split(test_size=0.2, seed=seed)
        self.train_dataset = split1["train"]
        split2 = split1["test"].train_test_split(test_size=0.5, seed=seed)
        self.val_dataset = split2["train"]
        self.test_dataset = split2["test"]

        counts = (len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))
        print(f"Split: train={counts[0]}, val={counts[1]}, test={counts[2]}")
        return counts

    # ------------------------------------------------------------------
    # train  (from train.py:571-643)
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Create SFTTrainer, run training, and return stats dict."""
        self._check_cancel()

        tcfg = self.training_config
        auto_precision = tcfg.get("auto_precision", True)
        if auto_precision:
            use_bf16 = torch.cuda.is_bf16_supported()
            use_fp16 = not use_bf16
        else:
            use_bf16 = tcfg.get("bf16", False)
            use_fp16 = tcfg.get("fp16", False)

        output_dir = Path(tcfg.get("output_dir", "outputs")) / self._run_name

        training_args = TrainingArguments(
            per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 4),
            warmup_steps=tcfg.get("warmup_steps", 10),
            num_train_epochs=tcfg.get("num_train_epochs", 1),
            max_steps=tcfg.get("max_steps", -1),
            learning_rate=tcfg.get("learning_rate", 5e-5),
            max_grad_norm=tcfg.get("max_grad_norm", 1.0),
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=tcfg.get("logging_steps", 1),
            optim=tcfg.get("optim", "adamw_8bit"),
            weight_decay=tcfg.get("weight_decay", 0.01),
            lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
            seed=tcfg.get("seed", 3407),
            output_dir=str(output_dir),
            save_strategy=tcfg.get("save_strategy", "steps"),
            save_steps=tcfg.get("save_steps", 25),
            save_total_limit=tcfg.get("save_total_limit", 2),
            eval_strategy=tcfg.get("eval_strategy", "no"),
            report_to="none",
        )

        callbacks = []
        if self.on_metric:
            callbacks.append(MetricsCallback(self.on_metric, self.knobs))

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            args=training_args,
            callbacks=callbacks or None,
        )

        print("Starting training...")
        trainer_stats = trainer.train()
        print("Training completed.")

        metrics = trainer_stats.metrics if hasattr(trainer_stats, "metrics") else {}
        stats = {
            "global_step": getattr(trainer_stats, "global_step", 0),
            "training_loss": getattr(trainer_stats, "training_loss", 0.0),
            "train_runtime": metrics.get("train_runtime", 0.0),
            "output_dir": str(output_dir),
        }
        return stats

    # ------------------------------------------------------------------
    # save_adapter
    # ------------------------------------------------------------------

    def save_adapter(self, output_dir: Optional[str] = None) -> str:
        """Save LoRA adapter + tokenizer.  Returns the adapter path."""
        base = Path(output_dir) if output_dir else Path(self.training_config.get("output_dir", "outputs")) / self._run_name
        adapter_path = base / "final_adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))
        print(f"LoRA adapter saved to: {adapter_path}")
        return str(adapter_path)

    # ------------------------------------------------------------------
    # run_full_pipeline  — convenience entry-point for job_runner
    # ------------------------------------------------------------------

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Load → prepare → train → save.  Returns a results dict."""
        self.load_model()
        train_count, val_count, test_count = self.prepare_dataset()
        stats = self.train()
        adapter_path = self.save_adapter(stats.get("output_dir"))

        return {
            "run_name": self._run_name,
            "adapter_path": adapter_path,
            "output_dir": stats.get("output_dir", ""),
            "final_loss": stats.get("training_loss", 0.0),
            "total_steps": stats.get("global_step", 0),
            "training_runtime_seconds": stats.get("train_runtime", 0.0),
            "train_samples": train_count,
            "val_samples": val_count,
            "test_samples": test_count,
        }
