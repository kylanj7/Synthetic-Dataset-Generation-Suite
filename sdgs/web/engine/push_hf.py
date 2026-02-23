"""HuggingFace Hub Upload.

Ported from Qwen-Fine-Tuning-Lab/push_to_hf.py.
All interactive/argparse stripped; programmatic API only.

Usage:
    from sdgs.web.engine.push_hf import push_gguf, push_merged

    push_gguf("models/gguf/model-q4_k_m.gguf", "username/my-model")
    push_merged("_temp_merged_model/", "username/my-model-full")
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder


def _create_model_card(
    repo_id: str,
    base_model: str = "Qwen/Qwen2.5-14B-Instruct",
    gguf_name: Optional[str] = None,
    has_merged: bool = False,
    description: str = "",
    dataset: str = "",
    author: str = "",
) -> str:
    """Generate a README.md model card."""
    uploads = []
    if gguf_name:
        uploads.append(
            f"- **GGUF**: `{gguf_name}` - Quantized for efficient inference with llama.cpp"
        )
    if has_merged:
        uploads.append(
            "- **HuggingFace Format**: Full merged model compatible with `transformers`"
        )
    uploads_text = "\n".join(uploads)

    repo_name = repo_id.split("/")[-1]
    safe_name = repo_name.lower().replace("-", "_")

    return f"""---
language:
- en
license: apache-2.0
tags:
- qwen2.5
- fine-tuned
- lora
base_model: {base_model}
---

# {repo_name}

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model})
using LoRA (Low-Rank Adaptation).

## Model Description

{description or f"Fine-tuned {base_model} model."}

## Available Formats

{uploads_text}

## Usage

### Using GGUF (with llama.cpp, Ollama, LM Studio, etc.)

```bash
huggingface-cli download {repo_id} {gguf_name or "model.gguf"}
./llama.cpp/build/bin/llama-cli -m {gguf_name or "model.gguf"} -p "Your prompt here"
```

### Using HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

## Training Details

- **Base Model**: {base_model}
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: {dataset or "Custom dataset"}

## Limitations

This model inherits the limitations of the base model and may have
additional domain-specific limitations due to the fine-tuning dataset.

## Citation

```bibtex
@misc{{{safe_name},
  author = {{{author or "Author"}}},
  title = {{{repo_name}}},
  year = {{2026}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## License

Apache 2.0
"""


def push_gguf(
    gguf_path: str,
    repo_id: str,
    private: bool = True,
    token: Optional[str] = None,
    base_model: str = "Qwen/Qwen2.5-14B-Instruct",
    description: str = "",
    dataset: str = "",
    author: str = "",
) -> str:
    """Upload a GGUF file to HuggingFace Hub.

    Args:
        gguf_path: Path to the GGUF file
        repo_id: HuggingFace repo ID (e.g. "username/model-name")
        private: Make repository private
        token: HF API token (falls back to HF_TOKEN env var)
        base_model: Base model name for the model card
        description: Model description
        dataset: Dataset name for the model card
        author: Author name for citation

    Returns:
        Repository URL
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("No HuggingFace token. Pass token= or set HF_TOKEN env var.")

    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    gguf_name = os.path.basename(gguf_path)

    print(f"Uploading GGUF to {repo_id}...")

    # Create repo
    api = HfApi(token=token)
    repo_url = create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        exist_ok=True,
        repo_type="model",
    )
    print(f"Repository ready: {repo_url}")

    # Upload model card
    model_card = _create_model_card(
        repo_id=repo_id,
        base_model=base_model,
        gguf_name=gguf_name,
        description=description,
        dataset=dataset,
        author=author,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(model_card)
        readme_path = f.name

    try:
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
    finally:
        os.unlink(readme_path)

    # Upload GGUF
    file_size_gb = os.path.getsize(gguf_path) / (1024**3)
    print(f"Uploading GGUF: {gguf_name} ({file_size_gb:.2f} GB)")
    upload_file(
        path_or_fileobj=gguf_path,
        path_in_repo=gguf_name,
        repo_id=repo_id,
        token=token,
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Upload complete: {url}")
    return url


def push_merged(
    model_dir: str,
    repo_id: str,
    private: bool = True,
    token: Optional[str] = None,
    base_model: str = "Qwen/Qwen2.5-14B-Instruct",
    description: str = "",
    dataset: str = "",
    author: str = "",
) -> str:
    """Upload a merged model directory to HuggingFace Hub.

    Args:
        model_dir: Path to the merged model directory
        repo_id: HuggingFace repo ID
        private: Make repository private
        token: HF API token
        base_model: Base model name for the model card
        description: Model description
        dataset: Dataset name for the model card
        author: Author name

    Returns:
        Repository URL
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("No HuggingFace token. Pass token= or set HF_TOKEN env var.")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Uploading merged model to {repo_id}...")

    # Create repo
    api = HfApi(token=token)
    repo_url = create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        exist_ok=True,
        repo_type="model",
    )
    print(f"Repository ready: {repo_url}")

    # Upload model card
    model_card = _create_model_card(
        repo_id=repo_id,
        base_model=base_model,
        has_merged=True,
        description=description,
        dataset=dataset,
        author=author,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(model_card)
        readme_path = f.name

    try:
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
    finally:
        os.unlink(readme_path)

    # Upload model directory
    total_size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, dirs, files in os.walk(model_dir)
        for f in files
    )
    print(f"Uploading merged model ({total_size / 1024**3:.2f} GB)...")

    upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        token=token,
        ignore_patterns=["*.gguf", "outputs/*", "__pycache__/*", ".git/*"],
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Upload complete: {url}")
    return url
