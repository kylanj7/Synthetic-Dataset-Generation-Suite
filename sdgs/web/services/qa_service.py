"""Service wrapper for sdgs.qa dataset inspection."""
import json
import re
from pathlib import Path

from sdgs.qa import get_stats


def load_jsonl_samples(path: str, offset: int = 0, limit: int = 20, search: str | None = None) -> tuple[list[dict], int]:
    """Load paginated samples from a JSONL file with optional search."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if search:
                text = (record.get("instruction", "") + " " + record.get("output", "")).lower()
                if search.lower() not in text:
                    continue

            samples.append(record)

    total = len(samples)
    page_samples = samples[offset:offset + limit]

    # Extract think/answer text for each sample
    for s in page_samples:
        output = s.get("output", "")
        think_match = re.search(r"<think>(.*?)</think>", output, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        s["think_text"] = think_match.group(1).strip() if think_match else None
        s["answer_text"] = answer_match.group(1).strip() if answer_match else None
        s["is_valid"] = s.get("valid", True)
        s["was_healed"] = s.get("healed", False)

    return page_samples, total


def get_dataset_stats(path: str) -> dict:
    """Get statistics for a JSONL dataset."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not samples:
        return {
            "total": 0, "valid": 0, "invalid": 0, "healed": 0,
            "avg_instruction_len": 0, "avg_output_len": 0, "topics": {},
        }

    stats = get_stats(samples)

    valid = sum(1 for s in samples if s.get("valid", True))
    healed = sum(1 for s in samples if s.get("healed", False))
    avg_instr = sum(len(s.get("instruction", "")) for s in samples) / len(samples)
    avg_out = sum(len(s.get("output", "")) for s in samples) / len(samples)

    return {
        "total": len(samples),
        "valid": valid,
        "invalid": len(samples) - valid,
        "healed": healed,
        "avg_instruction_len": round(avg_instr, 1),
        "avg_output_len": round(avg_out, 1),
        "topics": stats.get("topics", {}),
    }


def list_datasets(data_dir: str) -> list[dict]:
    """List all JSONL files in the data directory."""
    result = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return result

    for f in sorted(data_path.glob("*.jsonl")):
        line_count = 0
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    line_count += 1

        result.append({
            "name": f.stem,
            "path": str(f),
            "size_bytes": f.stat().st_size,
            "line_count": line_count,
            "modified": f.stat().st_mtime,
        })

    return result
