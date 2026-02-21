"""Generalized data extraction from various sources."""
import json
from pathlib import Path


def extract_data(
    task_config: dict,
    output_path: str,
    sample_size: int | None = None,
) -> int:
    """
    Extract Q&A pairs from the configured source in a task config.

    Supports source types: huggingface, local_json, local_jsonl.

    Returns:
        Number of examples extracted.
    """
    source = task_config["source"]
    source_type = source["type"]
    fields = source["fields"]
    metadata_fields = fields.get("metadata", [])

    if source_type == "huggingface":
        data = _extract_huggingface(source, fields, metadata_fields)
    elif source_type == "local_json":
        data = _extract_local_json(source, fields, metadata_fields)
    elif source_type == "local_jsonl":
        data = _extract_local_jsonl(source, fields, metadata_fields)
    else:
        raise ValueError(f"Unknown source type: '{source_type}'")

    if sample_size and sample_size < len(data):
        data = data[:sample_size]

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Extracted {len(data)} examples to {output_path}")
    return len(data)


def _extract_huggingface(source: dict, fields: dict, metadata_fields: list) -> list[dict]:
    """Extract from a HuggingFace dataset."""
    from datasets import load_dataset

    dataset_name = source["dataset"]
    split = source.get("split", "train")

    print(f"Loading dataset from HuggingFace: {dataset_name} ({split})...")
    ds = load_dataset(dataset_name, split=split)
    print(f"Total examples in source: {len(ds)}")

    output = []
    for item in ds:
        entry = {
            "question": item[fields["question"]],
            "answer": item[fields["answer"]],
        }
        for meta_field in metadata_fields:
            entry[meta_field] = item.get(meta_field, "")
        output.append(entry)
    return output


def _extract_local_json(source: dict, fields: dict, metadata_fields: list) -> list[dict]:
    """Extract from a local JSON file (list of objects)."""
    path = source["path"]
    print(f"Loading from local JSON: {path}")
    with open(path) as f:
        raw = json.load(f)

    output = []
    for item in raw:
        entry = {
            "question": item[fields["question"]],
            "answer": item[fields["answer"]],
        }
        for meta_field in metadata_fields:
            entry[meta_field] = item.get(meta_field, "")
        output.append(entry)
    return output


def _extract_local_jsonl(source: dict, fields: dict, metadata_fields: list) -> list[dict]:
    """Extract from a local JSONL file."""
    path = source["path"]
    print(f"Loading from local JSONL: {path}")

    output = []
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            entry = {
                "question": item[fields["question"]],
                "answer": item[fields["answer"]],
            }
            for meta_field in metadata_fields:
                entry[meta_field] = item.get(meta_field, "")
            output.append(entry)
    return output
