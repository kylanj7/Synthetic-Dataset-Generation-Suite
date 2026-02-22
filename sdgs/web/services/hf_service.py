"""HuggingFace Hub integration: push datasets to HF."""
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def push_dataset_to_hf(
    hf_token: str,
    repo_name: str,
    qa_pairs: list[dict],
    topic: str,
    pair_count: int,
    valid_count: int,
    provider: str | None,
    model: str | None,
    description: str = "",
    private: bool = True,
    total_tokens: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    gpu_kwh: float = 0.0,
    sources: list[dict] | None = None,
) -> str:
    """Upload a JSONL dataset + README card to HuggingFace Hub.

    Returns the repo URL.
    """
    if not qa_pairs:
        raise ValueError("No Q&A pairs to upload — dataset is empty.")

    api = HfApi(token=hf_token)

    # Create or get the repo
    api.create_repo(
        repo_id=repo_name,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    # Write QA pairs to a temp JSONL and upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        jsonl_path = f.name

    api.upload_file(
        path_or_fileobj=jsonl_path,
        path_in_repo="data/train.jsonl",
        repo_id=repo_name,
        repo_type="dataset",
    )
    Path(jsonl_path).unlink(missing_ok=True)

    # Generate and upload README card
    readme = _generate_readme(
        repo_name=repo_name,
        topic=topic,
        pair_count=pair_count,
        valid_count=valid_count,
        provider=provider,
        model=model,
        description=description,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        gpu_kwh=gpu_kwh,
        sources=sources or [],
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
    total_tokens: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    gpu_kwh: float = 0.0,
    sources: list[dict] | None = None,
) -> str:
    """Generate a HuggingFace dataset card README."""
    model_info = f"{provider or 'unknown'}/{model or 'default'}"
    desc_section = f"\n{description}\n" if description else ""
    sources = sources or []

    # Build sources/citations section
    sources_section = ""
    if sources:
        sources_section = "\n## Sources\n\n"
        sources_section += f"This dataset was generated from **{len(sources)}** scholarly papers:\n\n"
        sources_section += "| # | Title | Authors | Year | Source | QA Pairs |\n"
        sources_section += "|---|-------|---------|------|--------|----------|\n"
        for i, s in enumerate(sources, 1):
            title = s.get("title", "Unknown")
            authors = ", ".join(s.get("authors", [])[:3])
            if len(s.get("authors", [])) > 3:
                authors += " et al."
            year = s.get("year") or "N/A"
            url = s.get("url", "")
            source_name = s.get("source", "")
            qa_count = s.get("qa_pair_count", 0)
            title_cell = f"[{title}]({url})" if url else title
            sources_section += f"| {i} | {title_cell} | {authors} | {year} | {source_name} | {qa_count} |\n"

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

## Generation Cost

| Metric | Value |
|--------|-------|
| Prompt Tokens | {prompt_tokens:,} |
| Completion Tokens | {completion_tokens:,} |
| Total Tokens | {total_tokens:,} |
| GPU Energy | {gpu_kwh:.4f} kWh |
{sources_section}
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
