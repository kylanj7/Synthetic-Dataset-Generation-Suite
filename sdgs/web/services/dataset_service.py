"""Unified dataset pipeline orchestrator.

Replaces the separate scrape/generate/extract/filter services with a single
pipeline: calculate papers needed → scrape → auto-filter → store results.
"""
import json
import math
import os
import re
import time
from pathlib import Path

from sdgs.scrape import run_scrape, generate_qa_for_paper
from sdgs.filter import filter_dataset

from ..config import DATA_DIR


def run_dataset_pipeline(
    dataset_id: int,
    topic: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    target_size: int,
    system_prompt: str | None,
    temperature: float,
    s2_api_key: str | None = None,
    max_tokens: int | None = None,
):
    """Full pipeline: calculate papers → scrape → filter → return stats.

    This runs inside a ThreadPoolExecutor with stdout captured for SSE.
    """
    # 1. Calculate paper count from target size (~3-4 QA pairs per paper)
    max_papers = max(10, math.ceil(target_size / 3))
    top_n = max(5, max_papers // 4)

    # 2. Set up output paths
    safe_topic = re.sub(r'[^a-zA-Z0-9_-]', '_', topic)[:50]
    output_path = str(DATA_DIR / f"{safe_topic}_{dataset_id}.jsonl")
    filtered_path = str(DATA_DIR / f"{safe_topic}_{dataset_id}_filtered.jsonl")

    # 2b. Set Semantic Scholar API key if available
    if s2_api_key:
        os.environ["S2_API_KEY"] = s2_api_key

    # 3. Run scrape (papers + Q&A generation)
    run_scrape(
        topic=topic,
        provider=provider,
        model=model,
        api_key=api_key,
        task_name="paper_qa",
        max_papers=max_papers,
        top_n=top_n,
        output_path=output_path,
        collect_only=False,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 4. Auto-filter the output
    raw_path = Path(output_path)
    if raw_path.exists() and raw_path.stat().st_size > 0:
        filter_dataset(
            output_path,
            output_file=filtered_path,
            strict=True,
            heal=True,
        )
    else:
        filtered_path = output_path

    # 5. Generate citations metadata file
    citations_path = str(DATA_DIR / f"{safe_topic}_{dataset_id}_citations.json")
    _write_citations_file(output_path, filtered_path, citations_path, topic)

    return {
        "output_path": output_path,
        "filtered_path": filtered_path,
        "citations_path": citations_path,
    }


def _write_citations_file(
    output_path: str,
    filtered_path: str,
    citations_path: str,
    topic: str,
):
    """Write a JSON file listing all papers cited by the dataset."""
    # Collect paper metadata from the raw JSONL (has all papers, not just filtered)
    seen_papers = {}
    for path in [output_path, filtered_path]:
        p = Path(path)
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                source_id = entry.get("source_paper_id") or entry.get("source_paper", "")
                if source_id and source_id not in seen_papers:
                    seen_papers[source_id] = {
                        "paper_id": source_id,
                        "title": entry.get("source_title", "Unknown"),
                        "authors": entry.get("authors", []),
                        "abstract": entry.get("abstract", ""),
                        "year": entry.get("year"),
                        "doi": entry.get("doi"),
                        "url": entry.get("url", ""),
                        "source": entry.get("source", ""),
                        "citation_count": entry.get("citation_count", 0),
                    }

    citations = {
        "topic": topic,
        "total_papers": len(seen_papers),
        "papers": list(seen_papers.values()),
    }

    with open(citations_path, "w") as f:
        json.dump(citations, f, indent=2)

    print(f"Citations metadata saved: {len(seen_papers)} papers → {citations_path}")


def parse_dataset_results(output_path: str, filtered_path: str) -> dict:
    """Parse the generated JSONL files to extract stats for the Dataset record."""
    stats = {
        "actual_size": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "healed_count": 0,
        "papers": [],
        "qa_pairs": [],
    }

    # Parse filtered output (primary dataset)
    filtered = Path(filtered_path)
    raw = Path(output_path)

    target = filtered if filtered.exists() and filtered.stat().st_size > 0 else raw
    if not target.exists():
        return stats

    pairs = []
    with open(target) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    stats["actual_size"] = len(pairs)

    # Count valid/invalid/healed
    for pair in pairs:
        is_valid = pair.get("is_valid", pair.get("valid", True))
        was_healed = pair.get("was_healed", False)
        if was_healed:
            stats["healed_count"] += 1
        if is_valid:
            stats["valid_count"] += 1
        else:
            stats["invalid_count"] += 1

    # Extract paper info and QA pairs
    seen_papers = {}
    for pair in pairs:
        source_id = pair.get("source_paper_id") or pair.get("source_paper", "")
        if source_id and source_id not in seen_papers:
            paper_info = {
                "paper_id": source_id,
                "title": pair.get("source_title", "Unknown"),
                "authors": pair.get("authors", []),
                "abstract": pair.get("abstract", ""),
                "year": pair.get("year"),
                "doi": pair.get("doi"),
                "url": pair.get("url", ""),
                "source": pair.get("source", ""),
                "citation_count": pair.get("citation_count", 0),
            }
            if pair.get("pdf_path"):
                paper_info["pdf_path"] = pair["pdf_path"]
            seen_papers[source_id] = paper_info

        # Extract think/answer from output
        output_text = pair.get("output", "")
        think_text = ""
        answer_text = output_text
        think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
        if think_match:
            think_text = think_match.group(1).strip()
        answer_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()

        stats["qa_pairs"].append({
            "instruction": pair.get("instruction", ""),
            "output": output_text,
            "is_valid": pair.get("is_valid", pair.get("valid", True)),
            "was_healed": pair.get("was_healed", False),
            "source_paper_id": source_id,
            "source_title": pair.get("source_title", ""),
            "think_text": think_text,
            "answer_text": answer_text,
        })

    stats["papers"] = list(seen_papers.values())
    return stats


def run_from_papers_pipeline(
    dataset_id: int,
    paper_ids: list[int],
    provider: str | None,
    model: str | None,
    api_key: str | None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
):
    """Generate Q&A from pre-existing papers (no scraping step)."""
    from ..db.database import SessionLocal
    from ..db.models import Paper

    db = SessionLocal()
    try:
        papers_db = db.query(Paper).filter(Paper.id.in_(paper_ids)).all()
    finally:
        db.close()

    if not papers_db:
        raise RuntimeError("No papers found for the given IDs")

    # Build paper dicts
    paper_dicts = []
    for p in papers_db:
        d = {
            "title": p.title,
            "authors": p.authors or [],
            "abstract": p.abstract or "",
            "paper_id": p.paper_id or str(p.id),
            "year": p.year,
            "doi": p.doi,
            "url": p.url or "",
            "source": p.source or "",
            "citation_count": p.citation_count or 0,
        }
        # If paper has a PDF on disk, extract text from it
        if p.pdf_path and Path(p.pdf_path).is_file():
            try:
                import pymupdf
                doc = pymupdf.open(p.pdf_path)
                full_text = "\n".join(page.get_text() for page in doc)
                doc.close()
                if full_text.strip():
                    d["full_text"] = full_text
                    print(f"  Extracted {len(full_text):,} chars from PDF: {p.title[:50]}...")
            except Exception as e:
                print(f"  PDF extraction failed for {p.title[:50]}: {e}")

        if p.pdf_path:
            d["pdf_path"] = p.pdf_path
        paper_dicts.append(d)

    papers_with_content = [p for p in paper_dicts if p.get("full_text") or p.get("abstract")]
    if not papers_with_content:
        raise RuntimeError("No papers with extractable content")

    print(f"Generating Q&A from {len(papers_with_content)} papers...")
    print("=" * 60)

    # Setup LLM client
    from sdgs.providers import get_client
    client, model_name, extra_params = get_client(provider, model=model, api_key=api_key)

    # Load task config
    task_config_path = Path(__file__).parent.parent.parent.parent / "configs" / "tasks" / "paper_qa.yaml"
    import yaml
    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)

    if system_prompt:
        task_config["generation"]["system_prompt"] = system_prompt
    if temperature is not None:
        task_config["generation"]["temperature"] = temperature

    effective_max_tokens = max_tokens if max_tokens is not None else 4096

    # Generate Q&A
    from sdgs.scrape import TokenTracker, GPUTracker
    token_tracker = TokenTracker()
    gpu_tracker = GPUTracker()
    gpu_tracker.start()

    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', paper_dicts[0]["title"])[:50]
    output_path = str(DATA_DIR / f"papers_{safe_name}_{dataset_id}.jsonl")
    filtered_path = str(DATA_DIR / f"papers_{safe_name}_{dataset_id}_filtered.jsonl")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for i, paper in enumerate(papers_with_content):
        title_short = paper["title"][:60]
        content_type = "full text" if paper.get("full_text") else "abstract"
        print(f"\n[{i+1}/{len(papers_with_content)}] {title_short}... ({content_type})")

        pairs = generate_qa_for_paper(
            client, model_name, extra_params, task_config, paper, token_tracker,
            max_tokens=effective_max_tokens,
        )

        if pairs:
            all_pairs.extend(pairs)
            print(f"  → {len(pairs)} Q&A pairs generated")
        else:
            print(f"  → No Q&A pairs generated")

    # Write output
    with open(output_file, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nTotal: {len(all_pairs)} Q&A pairs → {output_path}")

    gpu_kwh = gpu_tracker.stop()
    print(f"\nPrompt tokens: {token_tracker.prompt_tokens:,}")
    print(f"Completion tokens: {token_tracker.completion_tokens:,}")
    print(f"Total tokens: {token_tracker.total_tokens:,}")
    if gpu_kwh:
        print(f"Total energy: {gpu_kwh:.6f} kWh")

    # Filter
    raw = Path(output_path)
    if raw.exists() and raw.stat().st_size > 0:
        filter_dataset(output_path, output_file=filtered_path, strict=True, heal=True)
    else:
        filtered_path = output_path

    citations_path = str(DATA_DIR / f"papers_{safe_name}_{dataset_id}_citations.json")
    _write_citations_file(output_path, filtered_path, citations_path, "Custom selection")

    return {
        "output_path": output_path,
        "filtered_path": filtered_path,
        "citations_path": citations_path,
    }


def import_from_huggingface(
    dataset_id: int,
    repo_id: str,
    hf_token: str | None = None,
    split: str | None = None,
):
    """Import a dataset from HuggingFace Hub and convert to our JSONL format."""
    print(f"Importing dataset from HuggingFace: {repo_id}")
    if split:
        print(f"  Split: {split}")

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("The 'datasets' library is required. Install with: pip install datasets")

    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    if split:
        kwargs["split"] = split

    print("Downloading dataset...")
    try:
        ds = load_dataset(repo_id, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{repo_id}': {e}")

    # If we got a DatasetDict (no split specified), pick the best split
    from datasets import DatasetDict
    if isinstance(ds, DatasetDict):
        if split and split in ds:
            ds = ds[split]
        elif "train" in ds:
            ds = ds["train"]
            print(f"  Using 'train' split ({len(ds)} rows)")
        else:
            first_split = list(ds.keys())[0]
            ds = ds[first_split]
            print(f"  Using '{first_split}' split ({len(ds)} rows)")

    print(f"  Downloaded {len(ds)} rows")

    # Column mapping
    columns = set(ds.column_names)
    instruction_col = None
    output_col = None

    for candidate in ["instruction", "input", "question", "prompt"]:
        if candidate in columns:
            instruction_col = candidate
            break

    for candidate in ["output", "response", "answer", "completion"]:
        if candidate in columns:
            output_col = candidate
            break

    # Fallback: single text column
    if not instruction_col and "text" in columns:
        instruction_col = "text"

    if not instruction_col:
        raise RuntimeError(
            f"Could not map columns. Available: {sorted(columns)}. "
            "Expected one of: instruction, input, question, prompt, text"
        )

    print(f"  Mapping: {instruction_col} → instruction, {output_col or '(empty)'} → output")

    # Convert to JSONL
    repo_short = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', repo_short)[:50]
    output_path = str(DATA_DIR / f"import_{safe_name}_{dataset_id}.jsonl")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for row in ds:
            instruction = str(row.get(instruction_col, ""))
            output_text = str(row.get(output_col, "")) if output_col else ""

            if not instruction.strip():
                continue

            entry = {
                "instruction": instruction,
                "output": output_text,
                "source": f"huggingface:{repo_id}",
                "is_valid": True,
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    print(f"\nImported {count} pairs → {output_path}")

    return {
        "output_path": output_path,
        "filtered_path": output_path,
    }
