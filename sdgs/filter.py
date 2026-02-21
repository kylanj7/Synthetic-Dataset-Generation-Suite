"""Filter and validate reasoning dataset for SFT quality."""
import json
from pathlib import Path

from .validate import (
    heal_sample,
    validate_latex,
    validate_reasoning_quality,
    validate_tags,
)


def filter_dataset(
    input_file: str,
    output_file: str | None = None,
    strict: bool = True,
    heal: bool = True,
    validation_rules: dict | None = None,
):
    """
    Filter a JSONL dataset, attempting to heal broken samples before discarding.

    Args:
        input_file: Path to input JSONL.
        output_file: Path to output JSONL (default: input_filtered.jsonl).
        strict: If True, discard samples with any issues. If False, only discard critical.
        heal: If True, attempt to heal broken samples before validation.
        validation_rules: Optional validation config from a task YAML.
    """
    input_path = Path(input_file)
    if output_file is None:
        output_file = str(input_path.parent / (input_path.stem + "_filtered.jsonl"))

    validation_rules = validation_rules or {}
    check_latex = validation_rules.get("check_latex_consistency", True)

    stats = {
        "total": 0,
        "passed_original": 0,
        "passed_healed": 0,
        "failed_tags": 0,
        "failed_latex": 0,
        "failed_quality": 0,
        "heal_attempts": 0,
        "heal_success": 0,
        "issues": [],
    }

    valid_samples = []

    print(f"Filtering: {input_file}")
    print(f"Mode: {'Strict' if strict else 'Lenient'} | Healing: {'ON' if heal else 'OFF'}")
    print("=" * 60)

    with open(input_file) as f:
        for line_num, line in enumerate(f, 1):
            stats["total"] += 1

            try:
                sample = json.loads(line.strip())
            except json.JSONDecodeError:
                stats["issues"].append((line_num, "Invalid JSON"))
                continue

            output = sample.get("output", "")

            # HEALING PHASE
            healed = False
            heal_methods = []
            if heal:
                tags_ok, _ = validate_tags(output)
                if not tags_ok:
                    stats["heal_attempts"] += 1
                    output, healed, heal_methods = heal_sample(output)
                    if healed:
                        stats["heal_success"] += 1
                        sample["output"] = output
                        sample["healed"] = True
                        sample["heal_methods"] = heal_methods

            # VALIDATION PHASE
            all_issues = []
            critical_fail = False

            tags_ok, tag_issues = validate_tags(output)
            if not tags_ok:
                stats["failed_tags"] += 1
                all_issues.extend(tag_issues)
                if any("</answer>" in issue for issue in tag_issues):
                    critical_fail = True

            if check_latex:
                plain_text_patterns = validation_rules.get("plain_text_patterns")
                latex_ok, latex_issues = validate_latex(output, plain_text_patterns)
                if not latex_ok:
                    stats["failed_latex"] += 1
                    all_issues.extend(latex_issues)

            quality_ok, quality_issues = validate_reasoning_quality(output, validation_rules)
            if not quality_ok:
                stats["failed_quality"] += 1
                all_issues.extend(quality_issues)

            # Decide whether to keep
            if strict:
                keep = len(all_issues) == 0
            else:
                keep = not critical_fail

            if keep:
                if healed:
                    stats["passed_healed"] += 1
                else:
                    stats["passed_original"] += 1
                valid_samples.append(sample)
            else:
                stats["issues"].append((line_num, all_issues, healed, heal_methods))
                if stats["total"] <= 20:
                    heal_info = f" (healed: {heal_methods})" if healed else ""
                    print(f"[{line_num}] REJECTED{heal_info}: {all_issues[0] if all_issues else 'Unknown'}")

    with open(output_file, "w") as f:
        for sample in valid_samples:
            f.write(json.dumps(sample) + "\n")

    total_passed = stats["passed_original"] + stats["passed_healed"]

    print("=" * 60)
    print("FILTER RESULTS")
    print("=" * 60)
    print(f"Total samples:     {stats['total']}")
    print(f"Passed (total):    {total_passed} ({100*total_passed/max(1,stats['total']):.1f}%)")
    print(f"  - Original:      {stats['passed_original']}")
    print(f"  - Healed:        {stats['passed_healed']}")
    if heal:
        print(f"\nHealing stats:")
        print(f"  - Attempts:      {stats['heal_attempts']}")
        print(f"  - Successful:    {stats['heal_success']} ({100*stats['heal_success']/max(1,stats['heal_attempts']):.1f}%)")
    print(f"\nRejected:          {stats['total'] - total_passed}")
    print(f"  - Tag issues:    {stats['failed_tags']}")
    print(f"  - LaTeX issues:  {stats['failed_latex']}")
    print(f"  - Quality issues:{stats['failed_quality']}")
    print(f"\nFiltered output:   {output_file}")

    if stats["issues"][:5]:
        print(f"\nSample rejection details (first 5):")
        for item in stats["issues"][:5]:
            if len(item) == 4:
                line_num, issues, was_healed, methods = item
                heal_info = f" [healed: {methods}]" if was_healed else ""
                print(f"  Line {line_num}{heal_info}: {issues[:2]}...")
            else:
                line_num, issues = item
                print(f"  Line {line_num}: {issues}")

    return stats
