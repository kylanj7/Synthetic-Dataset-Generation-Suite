"""Unified dataset pipeline orchestrator.

Replaces the separate scrape/generate/extract/filter services with a single
pipeline: calculate papers needed → scrape → auto-filter → store results.
"""
import json
import math
import os
import re
from pathlib import Path

from sdgs.scrape import run_scrape
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
            seen_papers[source_id] = {
                "paper_id": source_id,
                "title": pair.get("source_title", "Unknown"),
                "authors": pair.get("authors", []),
                "abstract": pair.get("abstract", ""),
                "year": pair.get("year"),
                "url": pair.get("url", ""),
                "citation_count": pair.get("citation_count", 0),
            }

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
