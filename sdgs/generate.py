"""Core generation logic — provider-agnostic reasoning dataset generation."""
import json
import re
import time
from pathlib import Path

import openai

from .tracker import GPUTracker, TokenTracker
from .validate import validate_output


def _count_existing(output_path: str) -> int:
    """Count existing lines in output file for resume support."""
    p = Path(output_path)
    if not p.exists():
        return 0
    with open(p) as f:
        return sum(1 for _ in f)


def generate_single(
    client: openai.OpenAI,
    model: str,
    extra_params: dict,
    system_prompt: str,
    user_msg: str,
    temperature: float,
    validation_rules: dict,
    attempt: int = 1,
    token_tracker: TokenTracker | None = None,
    max_tokens: int = 4096,
) -> tuple[str | None, bool]:
    """
    Generate a single reasoning response and validate it.

    Returns:
        (content, is_valid)
    """
    # Separate internal rate_limit_delay from API params
    api_params = {k: v for k, v in extra_params.items() if not k.startswith("_")}

    try:
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if api_params:
            kwargs["extra_body"] = api_params

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        if token_tracker and hasattr(response, "usage"):
            token_tracker.update(response.usage)

        is_valid, validation_msg = validate_output(content, validation_rules)
        if is_valid:
            return content, True
        else:
            print(f"  [Attempt {attempt}] Invalid: {validation_msg}")
            return content, False

    except Exception as e:
        print(f"  [Attempt {attempt}] Error: {e}")
        return None, False


def run_generation(
    client: openai.OpenAI,
    model: str,
    extra_params: dict,
    task_config: dict,
    input_data: list[dict],
    output_path: str,
    resume: bool = True,
    max_tokens: int = 4096,
):
    """
    Run full dataset generation.

    Args:
        client: OpenAI-compatible client.
        model: Model name.
        extra_params: Provider-specific params (num_ctx, etc.).
        task_config: Task YAML config dict.
        input_data: List of {"question": ..., "answer": ...} dicts.
        output_path: Path to output JSONL.
        resume: If True, skip already-generated entries.
    """
    gen_config = task_config["generation"]
    system_prompt = gen_config["system_prompt"]
    user_template = gen_config["user_prompt_template"]
    temperature = gen_config.get("temperature", 0.2)
    max_retries = gen_config.get("max_retries", 3)
    validation_rules = task_config.get("validation", {})
    rate_delay = extra_params.get("_rate_limit_delay", 0)

    # Resume support
    skip_count = 0
    if resume:
        skip_count = _count_existing(output_path)
        if skip_count > 0:
            print(f"Resuming: skipping {skip_count} already-generated entries.")
            input_data = input_data[skip_count:]

    total = len(input_data) + skip_count
    print(f"Processing {len(input_data)} examples (total: {total})...")
    print(f"Model: {model} | Temperature: {temperature} | Max retries: {max_retries}")
    print(f"Output: {output_path}")
    print("=" * 60)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    token_tracker = TokenTracker()
    gpu_tracker = GPUTracker()
    gpu_tracker.start()

    stats = {"processed": 0, "valid": 0, "invalid_kept": 0, "skipped": 0, "total_retries": 0}

    with open(output_path, "a") as out_f:
        for i, entry in enumerate(input_data):
            q = entry["question"]
            a = entry["answer"]
            idx = i + skip_count + 1

            print(f"\n[{idx}/{total}] {q[:60]}...")

            user_msg = user_template.format(question=q, answer=a)

            best_output = None
            is_valid = False

            for attempt in range(1, max_retries + 1):
                output, valid = generate_single(
                    client, model, extra_params, system_prompt, user_msg,
                    temperature, validation_rules, attempt,
                    token_tracker=token_tracker,
                    max_tokens=max_tokens,
                )
                if output:
                    best_output = output
                    is_valid = valid
                if valid:
                    break
                if attempt < max_retries:
                    stats["total_retries"] += 1
                    time.sleep(1)

            if best_output is None:
                print("  SKIPPED: No output generated")
                stats["skipped"] += 1
                continue

            if is_valid:
                print("  Valid output")
                stats["valid"] += 1
            else:
                print("  Kept invalid output (will filter later)")
                stats["invalid_kept"] += 1

            jsonl_line = {"instruction": q, "output": best_output, "valid": is_valid}
            out_f.write(json.dumps(jsonl_line) + "\n")
            out_f.flush()
            stats["processed"] += 1

            if stats["processed"] % 25 == 0:
                valid_pct = 100 * stats["valid"] / stats["processed"]
                print(f"\n--- Progress: {idx}/{total} | Valid: {valid_pct:.1f}% ---\n")

            if rate_delay > 0:
                time.sleep(rate_delay)

    gpu_tracker.stop()
    _print_summary(stats, output_path)
    token_tracker.report()
    gpu_tracker.report()


def run_test(
    client: openai.OpenAI,
    model: str,
    extra_params: dict,
    task_config: dict,
    input_data: list[dict],
    num_samples: int = 5,
    max_tokens: int = 4096,
):
    """
    Test mode — generate a few samples with detailed validation output.

    Args:
        num_samples: Number of test samples to generate.
    """
    gen_config = task_config["generation"]
    system_prompt = gen_config["system_prompt"]
    user_template = gen_config["user_prompt_template"]
    temperature = gen_config.get("temperature", 0.2)
    validation_rules = task_config.get("validation", {})
    rate_delay = extra_params.get("_rate_limit_delay", 0)

    # Build domain checks from task config
    domain_checks = _build_domain_checks(task_config)

    samples = input_data[:num_samples]

    print(f"Generating {len(samples)} test samples with {model}...")
    print("=" * 80)

    token_tracker = TokenTracker()
    gpu_tracker = GPUTracker()
    gpu_tracker.start()

    results = []
    for i, entry in enumerate(samples):
        q = entry["question"]
        a = entry["answer"]

        print(f"\n[{i+1}/{len(samples)}] QUESTION:")
        print(q[:200] + ("..." if len(q) > 200 else ""))
        print()

        user_msg = user_template.format(question=q, answer=a)

        start = time.time()
        output, valid = generate_single(
            client, model, extra_params, system_prompt, user_msg,
            temperature, validation_rules,
            token_tracker=token_tracker,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - start

        if output:
            print(f"GENERATED ({elapsed:.1f}s):")
            print("-" * 40)
            print(output)
            print("-" * 40)

            validation = _detailed_validation(output, domain_checks)
            _print_detailed_validation(validation, domain_checks)

            quality = _score_quality(validation, domain_checks)
            print(f"\n  >>> {quality}")

            results.append({
                "question": q,
                "generated": output,
                "validation": validation,
                "quality": quality,
                "time": elapsed,
            })
        else:
            print("FAILED to generate")
            results.append({"question": q, "generated": None, "quality": "ERROR", "time": elapsed})

        print("=" * 80)

        if rate_delay > 0:
            time.sleep(rate_delay)

    gpu_tracker.stop()
    _print_test_summary(results, len(samples), domain_checks)
    token_tracker.report()
    gpu_tracker.report()


def _build_domain_checks(task_config: dict) -> dict:
    """Build domain check config from task YAML.

    Looks for `validation.domain_checks` in the task config. Each check is a
    dict with `name` and `pattern` (regex). If not present, returns empty checks.

    Example task config:
        validation:
          domain_checks:
            - name: "Hilbert space"
              pattern: "hilbert|dimension|\\\\mathbb\\{C\\}"
            - name: "Hamiltonian"
              pattern: "hamiltonian|\\\\hat\\{H\\}"
    """
    validation = task_config.get("validation", {})
    checks = validation.get("domain_checks", [])
    require_latex = validation.get("require_latex", False)
    min_domain_score = validation.get("min_domain_score", 0)
    return {
        "checks": checks,
        "require_latex": require_latex,
        "min_domain_score": min_domain_score,
    }


def _detailed_validation(content: str, domain_checks: dict | None = None) -> dict:
    """Comprehensive validation with configurable domain checks."""
    domain_checks = domain_checks or {"checks": [], "require_latex": False, "min_domain_score": 0}
    lower = content.lower()
    result = {
        "has_think_open": "<think>" in lower,
        "has_think_close": "</think>" in lower,
        "has_answer_open": "<answer>" in lower,
        "has_answer_close": "</answer>" in lower,
        "has_latex": bool(re.search(r"\\[a-zA-Z]+|\\\[|\$", content)),
        "has_steps": bool(re.search(r"step\s*\d|step\s*[1-4]:", lower)),
    }
    result["all_tags_valid"] = all([
        result["has_think_open"], result["has_think_close"],
        result["has_answer_open"], result["has_answer_close"],
    ])

    # Domain-specific checks from task config
    domain_results = {}
    for check in domain_checks.get("checks", []):
        name = check["name"]
        pattern = check["pattern"]
        domain_results[name] = bool(re.search(pattern, content, re.IGNORECASE))
    result["domain_results"] = domain_results
    result["domain_score"] = sum(1 for v in domain_results.values() if v)
    result["domain_total"] = len(domain_results)

    return result


def _score_quality(validation: dict, domain_checks: dict | None = None) -> str:
    """Score quality using configurable thresholds."""
    domain_checks = domain_checks or {"checks": [], "require_latex": False, "min_domain_score": 0}
    require_latex = domain_checks.get("require_latex", False)
    min_domain = domain_checks.get("min_domain_score", 0)

    if not validation["all_tags_valid"]:
        return "FAIL"

    if require_latex and not validation["has_latex"]:
        return "FAIL"

    if min_domain > 0 and validation["domain_score"] < min_domain:
        if validation["all_tags_valid"] and (not require_latex or validation["has_latex"]):
            return "MARGINAL"
        return "FAIL"

    return "PASS"


def _print_detailed_validation(v: dict, domain_checks: dict | None = None):
    ok = lambda b: "Y" if b else "N"
    print(f"\nVALIDATION:")
    print(f"  Tags:          {ok(v['all_tags_valid'])} "
          f"(think: {ok(v['has_think_open'])}/{ok(v['has_think_close'])}, "
          f"answer: {ok(v['has_answer_open'])}/{ok(v['has_answer_close'])})")
    print(f"  LaTeX:         {ok(v['has_latex'])}")
    print(f"  Step format:   {ok(v['has_steps'])}")

    domain_results = v.get("domain_results", {})
    if domain_results:
        total = v.get("domain_total", len(domain_results))
        score = v.get("domain_score", 0)
        print(f"\n  DOMAIN CHECKS ({score}/{total}):")
        for name, passed in domain_results.items():
            print(f"    {name:20s} {ok(passed)}")


def _print_test_summary(results: list, num_samples: int, domain_checks: dict | None = None):
    passed = sum(1 for r in results if r.get("quality") == "PASS")
    marginal = sum(1 for r in results if r.get("quality") == "MARGINAL")
    failed = sum(1 for r in results if r.get("quality") == "FAIL")
    errors = sum(1 for r in results if r.get("quality") == "ERROR")
    avg_time = sum(r["time"] for r in results) / len(results) if results else 0

    domain_scores = [
        r.get("validation", {}).get("domain_score", 0)
        for r in results if r.get("validation")
    ]
    domain_totals = [
        r.get("validation", {}).get("domain_total", 0)
        for r in results if r.get("validation")
    ]
    avg_domain = sum(domain_scores) / len(domain_scores) if domain_scores else 0
    max_domain = max(domain_totals) if domain_totals else 0

    print(f"\n{'=' * 80}")
    print("QUALITY SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total samples:       {num_samples}")
    print(f"PASS:                {passed} ({100*passed/num_samples:.0f}%)")
    print(f"MARGINAL:            {marginal} ({100*marginal/num_samples:.0f}%)")
    print(f"FAIL:                {failed} ({100*failed/num_samples:.0f}%)")
    print(f"ERROR:               {errors}")
    if max_domain > 0:
        print(f"\nDomain checks avg:   {avg_domain:.1f}/{max_domain}")
    print(f"Avg time/sample:     {avg_time:.1f}s")

    usable = passed + marginal
    if num_samples > 0 and usable / num_samples >= 0.8:
        print(f"\nQuality looks good! {usable}/{num_samples} usable samples.")
        print("Ready for full generation.")
    elif num_samples > 0:
        print(f"\nOnly {usable}/{num_samples} usable. Consider:")
        print("  - Adjusting the system prompt in the task config")
        print("  - Using a different model/provider")
        print("  - Post-filtering with: sdgs filter <output.jsonl>")


def _print_summary(stats: dict, output_path: str):
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total processed:   {stats['processed']}")
    if stats["processed"] > 0:
        print(f"Valid outputs:     {stats['valid']} ({100*stats['valid']/stats['processed']:.1f}%)")
    print(f"Invalid (kept):    {stats['invalid_kept']}")
    print(f"Skipped:           {stats['skipped']}")
    print(f"Total retries:     {stats['total_retries']}")
    print(f"\nOutput saved to:   {output_path}")
    print(f"\nNext step: sdgs filter {output_path}")
