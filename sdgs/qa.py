"""QA inspection and statistics for reasoning datasets."""
import json
import random
import re
from collections import Counter
from pathlib import Path


def get_stats(samples: list, topics: list[str] | None = None) -> dict:
    """Calculate dataset statistics with configurable topic keywords."""
    if topics is None:
        topics = []

    stats = {
        "total_samples": len(samples),
        "avg_instruction_len": 0,
        "avg_output_len": 0,
        "avg_think_len": 0,
        "avg_answer_len": 0,
        "has_latex_pct": 0,
        "healed_count": 0,
        "topics": Counter(),
    }

    think_lens = []
    answer_lens = []
    latex_count = 0

    for s in samples:
        instruction = s.get("instruction", "")
        output = s.get("output", "")

        stats["avg_instruction_len"] += len(instruction)
        stats["avg_output_len"] += len(output)

        if s.get("healed"):
            stats["healed_count"] += 1

        think_match = re.search(r"<think>(.*?)</think>", output, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL | re.IGNORECASE)

        if think_match:
            think_lens.append(len(think_match.group(1)))
        if answer_match:
            answer_lens.append(len(answer_match.group(1)))

        if re.search(r"\$[^$]+\$|\\\[.*?\\\]", output, re.DOTALL):
            latex_count += 1

        for kw in topics:
            if kw in instruction.lower():
                stats["topics"][kw] += 1

    n = len(samples)
    stats["avg_instruction_len"] = stats["avg_instruction_len"] // n if n else 0
    stats["avg_output_len"] = stats["avg_output_len"] // n if n else 0
    stats["avg_think_len"] = sum(think_lens) // len(think_lens) if think_lens else 0
    stats["avg_answer_len"] = sum(answer_lens) // len(answer_lens) if answer_lens else 0
    stats["has_latex_pct"] = round(100 * latex_count / n, 1) if n else 0

    return stats


def display_sample(idx: int, sample: dict, truncate: int = 500):
    """Display a single sample in readable format."""
    print(f"\n{'='*70}")
    print(f"SAMPLE {idx}")
    print("=" * 70)

    instruction = sample.get("instruction", "N/A")
    output = sample.get("output", "N/A")

    print(f"\n[INSTRUCTION] ({len(instruction)} chars)")
    print("-" * 40)
    print(instruction[:truncate] + ("..." if len(instruction) > truncate else ""))

    think_match = re.search(r"<think>(.*?)</think>", output, re.DOTALL | re.IGNORECASE)
    if think_match:
        think = think_match.group(1).strip()
        print(f"\n[REASONING] ({len(think)} chars)")
        print("-" * 40)
        print(think[:truncate] + ("..." if len(think) > truncate else ""))

    answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
        print(f"\n[ANSWER] ({len(answer)} chars)")
        print("-" * 40)
        print(answer[:300])

    if sample.get("healed"):
        print(f"\n[HEALED] Methods: {sample.get('heal_methods', [])}")


def run_qa(
    dataset_path: str,
    num_samples: int = 5,
    use_random: bool = False,
    stats_only: bool = False,
    offset: int = 0,
    topics: list[str] | None = None,
):
    """Run QA inspection on a dataset."""
    samples = []
    with open(dataset_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    print(f"Loaded {len(samples)} samples from {dataset_path}")

    stats = get_stats(samples, topics=topics)
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Total samples:        {stats['total_samples']}")
    if stats["healed_count"] > 0:
        print(f"Healed samples:       {stats['healed_count']} ({100*stats['healed_count']/len(samples):.1f}%)")
    print(f"Avg instruction len:  {stats['avg_instruction_len']} chars")
    print(f"Avg output len:       {stats['avg_output_len']} chars")
    print(f"Avg reasoning len:    {stats['avg_think_len']} chars")
    print(f"Avg answer len:       {stats['avg_answer_len']} chars")
    print(f"Has LaTeX:            {stats['has_latex_pct']}%")

    if stats["topics"]:
        print(f"\nTop topics:")
        for topic, count in stats["topics"].most_common(10):
            print(f"  {topic}: {count} ({100*count/len(samples):.1f}%)")

    if stats_only:
        return

    if use_random:
        indices = random.sample(range(len(samples)), min(num_samples, len(samples)))
    else:
        indices = list(range(offset, min(offset + num_samples, len(samples))))

    for idx in indices:
        display_sample(idx, samples[idx])

    print(f"\n{'='*70}")
    print("QA CHECKLIST")
    print("=" * 70)
    print("[ ] Reasoning is coherent and step-by-step")
    print("[ ] Formatting is correct (tags, LaTeX if applicable)")
    print("[ ] Answer matches the reasoning conclusion")
    print("[ ] No hallucinated content")
    print("[ ] Appropriate depth for the question complexity")
