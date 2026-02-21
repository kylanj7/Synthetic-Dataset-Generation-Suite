"""HuggingFace Hub integration: push datasets to HF."""
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def push_dataset_to_hf(
    hf_token: str,
    repo_name: str,
    dataset_path: str,
    topic: str,
    pair_count: int,
    valid_count: int,
    provider: str | None,
    model: str | None,
    description: str = "",
    private: bool = True,
) -> str:
    """Upload a JSONL dataset + README card to HuggingFace Hub.

    Returns the repo URL.
    """
    api = HfApi(token=hf_token)

    # Create or get the repo
    api.create_repo(
        repo_id=repo_name,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    # Upload the JSONL file
    file_path = Path(dataset_path)
    if file_path.exists():
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo="data/train.jsonl",
            repo_id=repo_name,
            repo_type="dataset",
        )

    # Generate and upload README card
    readme = _generate_readme(
        repo_name=repo_name,
        topic=topic,
        pair_count=pair_count,
        valid_count=valid_count,
        provider=provider,
        model=model,
        description=description,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme)
        readme_path = f.name

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
    )

    Path(readme_path).unlink(missing_ok=True)

    return f"https://huggingface.co/datasets/{repo_name}"


def _generate_readme(
    repo_name: str,
    topic: str,
    pair_count: int,
    valid_count: int,
    provider: str | None,
    model: str | None,
    description: str = "",
) -> str:
    """Generate a HuggingFace dataset card README."""
    model_info = f"{provider or 'unknown'}/{model or 'default'}"
    desc_section = f"\n{description}\n" if description else ""

    return f"""---
language:
- en
task_categories:
- question-answering
tags:
- synthetic
- sdgs
- {topic.replace(' ', '-').lower()}
size_categories:
- {'1K<n<10K' if pair_count >= 1000 else 'n<1K'}
---

# {repo_name.split('/')[-1]}

Synthetic Q&A dataset on **{topic}**, generated with SDGS (Synthetic Dataset Generation Suite).
{desc_section}
## Dataset Details

| Metric | Value |
|--------|-------|
| Topic | {topic} |
| Total Q&A Pairs | {pair_count} |
| Valid Pairs | {valid_count} |
| Provider/Model | {model_info} |

## Format

Each entry in `data/train.jsonl` contains:
- `instruction`: The question
- `output`: The answer (may include `<think>` and `<answer>` tags)
- `is_valid`: Whether the pair passed validation
- `was_healed`: Whether the pair was auto-corrected
- `source_title`: Title of the source paper

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{repo_name}")
```

## Generated with

[SDGS — Synthetic Dataset Generation Suite](https://github.com/kylanj7/Synthetic-Dataset-Generation-Suite)
"""
