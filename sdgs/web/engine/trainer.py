"""QwenTrainer — configuration-driven fine-tuning engine.

Ported from Qwen-Fine-Tuning-Lab/train.py.  All interactive prompts,
argparse, and wandb integration have been stripped.  The trainer is driven
entirely by dicts passed at construction time.

Supports:
- YAML config discovery and loading (models, datasets, training)
- Run naming with atomic index counter
- Markdown training log
- Config-driven dataset formatting (MCQ, context fields, simple Q&A)
- HuggingFace and local JSONL dataset loading
- Config-driven train/val/test splits
- Resume from checkpoint
- Run metadata saving
- Live metric streaming via MetricsCallback
- Live knob adjustments (learning rate)
"""

import json
import threading
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
# Configuration Discovery and Loading
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).parent / "configs"


def discover_configs(config_type: str) -> Dict[str, Path]:
    """Discover available YAML config files of a given type.

    Args:
        config_type: One of 'models', 'datasets', or 'training'

    Returns:
        Dict mapping config name (stem) to full Path
    """
    config_dir = CONFIGS_DIR / config_type
    configs = {}
    if config_dir.exists():
        for yaml_file in sorted(config_dir.glob("*.yaml")):
            configs[yaml_file.stem] = yaml_file
    return configs


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Training Log and Run Naming
# ---------------------------------------------------------------------------

TRAINING_LOG_FILE = Path(__file__).parent / "training_log.md"
TRAINING_INDEX_FILE = Path(__file__).parent / ".training_index.json"


def get_next_training_index() -> int:
    """Get the next training index and atomically increment the counter."""
    if TRAINING_INDEX_FILE.exists():
        with open(TRAINING_INDEX_FILE, "r") as f:
            data = json.load(f)
            index = data.get("next_index", 1)
    else:
        index = 1

    with open(TRAINING_INDEX_FILE, "w") as f:
        json.dump({"next_index": index + 1}, f)

    return index


def generate_run_name(
    model_config: Dict, dataset_config: Dict
) -> Tuple[str, int, str]:
    """Generate a unique run name: {model}-{size}-{dataset}-{date}-{index}.

    Returns:
        Tuple of (run_name, index, timestamp)
    """
    model_name = (
        model_config.get("name", "model").lower().replace(" ", "-").replace(".", "")
    )
    model_size = model_config.get("size", "").lower()
    dataset_name = (
        dataset_config.get("name", "dataset").lower().replace(" ", "-")
    )

    timestamp = datetime.now().strftime("%Y%m%d")
    index = get_next_training_index()
    run_name = f"{model_name}-{model_size}-{dataset_name}-{timestamp}-{index:03d}"

    return run_name, index, timestamp


def log_training_run(
    run_name: str,
    index: int,
    model_config: Dict,
    dataset_config: Dict,
    training_config: Dict,
    output_dir: Path,
    trainer_stats: Any = None,
) -> None:
    """Append a training run entry to training_log.md."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not TRAINING_LOG_FILE.exists():
        header = "# Training Log\n\nThis file tracks all fine-tuning runs.\n\n---\n\n"
        with open(TRAINING_LOG_FILE, "w") as f:
            f.write(header)

    entry = f"""
## Run #{index:03d}: {run_name}

**Date:** {timestamp}

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Name | {model_config.get('name', 'N/A')} |
| HuggingFace ID | `{model_config.get('model_name', 'N/A')}` |
| Size | {model_config.get('size', 'N/A')} |
| Max Seq Length | {model_config.get('max_seq_length', 'N/A')} |
| LoRA Rank (r) | {model_config.get('lora', {}).get('r', 'N/A')} |
| LoRA Alpha | {model_config.get('lora', {}).get('lora_alpha', 'N/A')} |
| Quantization | {model_config.get('quantization', {}).get('bnb_4bit_quant_type', 'N/A')} |

### Dataset Configuration
| Parameter | Value |
|-----------|-------|
| Name | {dataset_config.get('name', 'N/A')} |
| HuggingFace ID | `{dataset_config.get('dataset_name', 'N/A')}` |
| Domain | {dataset_config.get('domain', 'N/A')} |
| Train/Val/Test | {dataset_config.get('train_val_test_split', {}).get('train', 'N/A')} / {dataset_config.get('train_val_test_split', {}).get('val', 'N/A')} / {dataset_config.get('train_val_test_split', {}).get('test', 'N/A')} |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Batch Size | {training_config.get('per_device_train_batch_size', 'N/A')} |
| Gradient Accumulation | {training_config.get('gradient_accumulation_steps', 'N/A')} |
| Effective Batch Size | {training_config.get('per_device_train_batch_size', 1) * training_config.get('gradient_accumulation_steps', 1)} |
| Learning Rate | {training_config.get('learning_rate', 'N/A')} |
| LR Scheduler | {training_config.get('lr_scheduler_type', 'N/A')} |
| Warmup Steps | {training_config.get('warmup_steps', 'N/A')} |
| Epochs | {training_config.get('num_train_epochs', 'N/A')} |
| Max Steps | {training_config.get('max_steps', 'N/A')} |
| Optimizer | {training_config.get('optim', 'N/A')} |
| Weight Decay | {training_config.get('weight_decay', 'N/A')} |

### Output
| Item | Path |
|------|------|
| Output Directory | `{output_dir}` |
| LoRA Adapter | `{output_dir}/final_adapter` |
"""

    if trainer_stats:
        loss_val = getattr(trainer_stats, "training_loss", None)
        loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"
        entry += f"""
### Training Results
| Metric | Value |
|--------|-------|
| Total Steps | {getattr(trainer_stats, 'global_step', 'N/A')} |
| Training Loss | {loss_str} |
| Training Runtime | {getattr(trainer_stats, 'metrics', {}).get('train_runtime', 'N/A')} |
"""

    entry += "\n---\n"

    with open(TRAINING_LOG_FILE, "a") as f:
        f.write(entry)

    print(f"Training run logged to: {TRAINING_LOG_FILE}")


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
        self._on_metric(
            {
                "type": "metric",
                "step": state.global_step,
                "epoch": round(state.epoch, 4) if state.epoch else 0,
                "max_steps": state.max_steps,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
                "grad_norm": logs.get("grad_norm"),
            }
        )

    def on_step_end(self, args, state, control, **kwargs):
        new_lr = self._knobs.get("learning_rate")
        if new_lr is None:
            return
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(new_lr)
        self._knobs.pop("learning_rate", None)


# ---------------------------------------------------------------------------
# Dataset Formatting
# ---------------------------------------------------------------------------


def create_formatting_function(dataset_config: Dict):
    """Create a batched formatting function for the dataset.

    Handles:
    - MCQ datasets (options + correct_option -> answer letter)
    - Datasets with context fields
    - Simple Q&A datasets (no context)
    """
    fields = dataset_config.get("fields", {})
    prompt_template = dataset_config.get("prompt_template", "")
    context_format = dataset_config.get("context_format")

    instruction_field = fields.get("instruction", "instruction")
    response_field = fields.get("response", "response")
    context_fields = fields.get("context_fields", [])
    options = fields.get("options", [])
    correct_option = fields.get("correct_option")

    def formatting_func(examples):
        texts = []
        num_examples = len(examples[instruction_field])

        for i in range(num_examples):
            instruction = examples[instruction_field][i]
            response = examples[response_field][i]

            # MCQ datasets (e.g. MedMCQA)
            if options and correct_option:
                opt_values = [examples[opt][i] for opt in options]
                correct_idx = examples[correct_option][i]
                if isinstance(correct_idx, int) and 0 <= correct_idx < len(opt_values):
                    answer_letter = chr(65 + correct_idx)
                    response = f"{answer_letter}) {opt_values[correct_idx]}"

                if context_fields:
                    context_values = {
                        field: examples[field][i]
                        for field in context_fields
                        if field in examples
                    }
                    context = (
                        context_format.format(**context_values) if context_format else ""
                    )
                else:
                    context = ""

                text = prompt_template.format(
                    instruction,
                    opt_values[0] if len(opt_values) > 0 else "",
                    opt_values[1] if len(opt_values) > 1 else "",
                    opt_values[2] if len(opt_values) > 2 else "",
                    opt_values[3] if len(opt_values) > 3 else "",
                    context,
                    response,
                )

            # Datasets with context fields
            elif context_fields and context_format:
                context_values = {}
                for field in context_fields:
                    if field in examples:
                        context_values[field] = examples[field][i]
                try:
                    context = context_format.format(**context_values)
                except KeyError:
                    context = ""
                text = prompt_template.format(instruction, context, response)

            # Simple Q&A (no context)
            else:
                text = prompt_template.format(instruction, response)

            texts.append(text)

        return {"text": texts}

    return formatting_func


# ---------------------------------------------------------------------------
# Defaults (matching Qwen 2.5 14B — used when no config is provided)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "name": "Qwen 2.5 14B Instruct",
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
        "r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
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
    "warmup_steps": 100,
    "num_train_epochs": 3,
    "max_steps": -1,
    "learning_rate": 1e-5,
    "max_grad_norm": 0.3,
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.1,
    "lr_scheduler_type": "cosine",
    "seed": 3407,
    "save_strategy": "steps",
    "save_steps": 25,
    "save_total_limit": 3,
    "eval_strategy": "steps",
    "output_dir": "outputs",
}


class QwenTrainer:
    """Dict-driven Qwen fine-tuning pipeline with YAML config support."""

    def __init__(
        self,
        dataset_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        dataset_config: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[threading.Event] = None,
        on_metric: Optional[Callable[[Dict[str, Any]], None]] = None,
        knobs: Optional[Dict[str, Any]] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        self.dataset_path = dataset_path
        self.model_config = {**DEFAULT_MODEL_CONFIG, **(model_config or {})}
        self.training_config = {**DEFAULT_TRAINING_CONFIG, **(training_config or {})}
        self.dataset_config = dataset_config  # None = simple JSONL mode
        self.cancel_event = cancel_event
        self.on_metric = on_metric
        self.knobs = knobs or {}
        self.resume_from_checkpoint = resume_from_checkpoint

        self.model = None
        self.tokenizer = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # Generate run name from configs if available
        if self.dataset_config:
            self._run_name, self._run_index, self._run_timestamp = generate_run_name(
                self.model_config, self.dataset_config
            )
        else:
            self._run_name = f"sdgs-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self._run_index = 0
            self._run_timestamp = datetime.now().strftime("%Y%m%d")

    # ------------------------------------------------------------------
    # Cancellation helper
    # ------------------------------------------------------------------

    def _check_cancel(self):
        if self.cancel_event and self.cancel_event.is_set():
            raise InterruptedError("Training cancelled by user")

    # ------------------------------------------------------------------
    # load_model
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
            bnb_4bit_use_double_quant=quant_cfg.get(
                "bnb_4bit_use_double_quant", True
            ),
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
            target_modules=lora_cfg.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # prepare_dataset — config-driven or simple JSONL fallback
    # ------------------------------------------------------------------

    def prepare_dataset(self) -> Tuple[int, int, int]:
        """Load and prepare dataset with train/val/test splits.

        When ``dataset_config`` is provided, uses config-driven loading:
        - HuggingFace dataset or local JSONL
        - Custom field mapping and formatting
        - Config-driven split ratios

        When no ``dataset_config``, falls back to simple JSONL loading with
        Qwen chat template and 80/10/10 splits (backward compatible).

        Returns (train_count, val_count, test_count).
        """
        self._check_cancel()

        if self.dataset_config:
            return self._prepare_dataset_from_config()
        else:
            return self._prepare_dataset_simple()

    def _prepare_dataset_from_config(self) -> Tuple[int, int, int]:
        """Config-driven dataset loading (ported from lab)."""
        dcfg = self.dataset_config
        dataset_name = dcfg["dataset_name"]
        split = dcfg.get("split", "train")
        split_config = dcfg.get("train_val_test_split")

        print(f"Loading dataset: {dataset_name}")

        # Load the dataset
        is_local = dcfg.get("is_local", False)
        if is_local:
            dataset = load_dataset("json", data_files=dataset_name, split="train")
        else:
            dataset = load_dataset(dataset_name, split=split)
        print(f"Dataset loaded: {len(dataset)} examples")

        # Apply formatting
        formatting_func = create_formatting_function(dcfg)
        dataset = dataset.map(formatting_func, batched=True)
        print("Dataset formatted.")

        # Split
        if split_config:
            train_ratio = split_config.get("train", 0.6)
            val_ratio = split_config.get("val", 0.2)
            test_ratio = split_config.get("test", 0.2)
            seed = split_config.get("seed", 3407)

            temp_test_size = val_ratio + test_ratio
            split1 = dataset.train_test_split(test_size=temp_test_size, seed=seed)
            self.train_dataset = split1["train"]

            if test_ratio > 0:
                val_test_ratio = test_ratio / temp_test_size
                split2 = split1["test"].train_test_split(
                    test_size=val_test_ratio, seed=seed
                )
                self.val_dataset = split2["train"]
                self.test_dataset = split2["test"]
            else:
                self.val_dataset = split1["test"]

            print(
                f"Split: train={len(self.train_dataset)}, "
                f"val={len(self.val_dataset) if self.val_dataset else 0}, "
                f"test={len(self.test_dataset) if self.test_dataset else 0}"
            )

            # Save test dataset for later evaluation
            if self.test_dataset:
                test_save_path = Path("outputs") / "test_dataset"
                test_save_path.parent.mkdir(parents=True, exist_ok=True)
                self.test_dataset.save_to_disk(str(test_save_path))
                print(f"Test dataset saved to: {test_save_path}")
        else:
            self.train_dataset = dataset
            print("Using full dataset for training (no split)")

        counts = (
            len(self.train_dataset),
            len(self.val_dataset) if self.val_dataset else 0,
            len(self.test_dataset) if self.test_dataset else 0,
        )
        return counts

    def _prepare_dataset_simple(self) -> Tuple[int, int, int]:
        """Simple JSONL loading with Qwen chat template (backward compatible)."""
        print(f"Loading dataset: {self.dataset_path}")
        dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        print(f"Dataset loaded: {len(dataset)} examples")

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

        seed = self.training_config.get("seed", 3407)
        split1 = dataset.train_test_split(test_size=0.2, seed=seed)
        self.train_dataset = split1["train"]
        split2 = split1["test"].train_test_split(test_size=0.5, seed=seed)
        self.val_dataset = split2["train"]
        self.test_dataset = split2["test"]

        counts = (
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )
        print(f"Split: train={counts[0]}, val={counts[1]}, test={counts[2]}")
        return counts

    # ------------------------------------------------------------------
    # train
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

        # Conditional eval_steps and save_steps (only when strategy is "steps")
        save_strategy = tcfg.get("save_strategy", "steps")
        eval_strategy = tcfg.get("eval_strategy", "no")
        save_steps = (
            tcfg.get("save_steps", 25) if save_strategy == "steps" else None
        )
        eval_steps = (
            tcfg.get("eval_steps", 25) if eval_strategy == "steps" else None
        )

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
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=tcfg.get("save_total_limit"),
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            push_to_hub=tcfg.get("push_to_hub", False),
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
        trainer_stats = trainer.train(
            resume_from_checkpoint=self.resume_from_checkpoint
        )
        print("Training completed.")

        metrics = trainer_stats.metrics if hasattr(trainer_stats, "metrics") else {}
        stats = {
            "global_step": getattr(trainer_stats, "global_step", 0),
            "training_loss": getattr(trainer_stats, "training_loss", 0.0),
            "train_runtime": metrics.get("train_runtime", 0.0),
            "output_dir": str(output_dir),
            "_trainer_stats": trainer_stats,
        }
        return stats

    # ------------------------------------------------------------------
    # save_adapter
    # ------------------------------------------------------------------

    def save_adapter(self, output_dir: Optional[str] = None) -> str:
        """Save LoRA adapter + tokenizer.  Returns the adapter path."""
        base = (
            Path(output_dir)
            if output_dir
            else Path(self.training_config.get("output_dir", "outputs"))
            / self._run_name
        )
        adapter_path = base / "final_adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))
        print(f"LoRA adapter saved to: {adapter_path}")
        return str(adapter_path)

    # ------------------------------------------------------------------
    # save_run_metadata
    # ------------------------------------------------------------------

    def save_run_metadata(self, output_dir: str) -> None:
        """Save run_metadata.json for GGUF conversion and logging."""
        metadata = {
            "run_name": self._run_name,
            "run_index": self._run_index,
            "base_model": self.model_config.get("model_name"),
            "model_size": self.model_config.get("size"),
            "dataset": (
                self.dataset_config.get("name")
                if self.dataset_config
                else Path(self.dataset_path).stem
            ),
        }
        meta_path = Path(output_dir) / "run_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Run metadata saved to: {meta_path}")

    # ------------------------------------------------------------------
    # run_full_pipeline — convenience entry-point for job_runner
    # ------------------------------------------------------------------

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Load -> prepare -> train -> save.  Returns a results dict."""
        self.load_model()
        train_count, val_count, test_count = self.prepare_dataset()
        stats = self.train()
        adapter_path = self.save_adapter(stats.get("output_dir"))

        output_dir = stats.get("output_dir", "")

        # Save run metadata
        self.save_run_metadata(output_dir)

        # Log training run if we have dataset_config (config-driven mode)
        if self.dataset_config:
            log_training_run(
                run_name=self._run_name,
                index=self._run_index,
                model_config=self.model_config,
                dataset_config=self.dataset_config,
                training_config=self.training_config,
                output_dir=Path(output_dir),
                trainer_stats=stats.get("_trainer_stats"),
            )

        return {
            "run_name": self._run_name,
            "adapter_path": adapter_path,
            "output_dir": output_dir,
            "final_loss": stats.get("training_loss", 0.0),
            "total_steps": stats.get("global_step", 0),
            "training_runtime_seconds": stats.get("train_runtime", 0.0),
            "train_samples": train_count,
            "val_samples": val_count,
            "test_samples": test_count,
        }
