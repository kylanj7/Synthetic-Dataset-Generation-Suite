"""Shared validation and healing utilities for generated outputs."""
import re
from typing import Optional

# ── Compiled patterns ──────────────────────────────────────────────────
THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
ANSWER_OPEN = re.compile(r"<answer>", re.IGNORECASE)
ANSWER_CLOSE = re.compile(r"</answer>", re.IGNORECASE)

BOXED_PATTERN = re.compile(
    r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL
)
FINAL_ANSWER_PATTERNS = [
    re.compile(
        r'(?:final\s+)?(?:answer|result)\s*(?:is|:)\s*["\']?([^"\'.\n]+)',
        re.IGNORECASE,
    ),
    re.compile(r"(?:therefore|thus|hence)[,\s]+([^.\n]+)", re.IGNORECASE),
    re.compile(r"=\s*([^=\n]+)$", re.MULTILINE),
]

LATEX_INLINE = re.compile(r"\$[^$]+\$")
LATEX_DISPLAY = re.compile(r"\\\[.+?\\\]", re.DOTALL)
LATEX_OPERATORS = re.compile(
    r"\\(sigma|psi|phi|omega|alpha|beta|gamma|delta|epsilon|hbar|nabla|"
    r"partial|frac|sqrt|sum|prod|int|ket|bra|langle|rangle)"
)

PLAIN_TEXT_PHYSICS = [
    r"\bsigma_[xyz]\b",
    r"\bpsi\b(?![_^\\])",
    r"\bH\s*=\s*[^$\\]",
    r"\bhbar\b(?!\\)",
]


# ── Core validation ────────────────────────────────────────────────────

def validate_output(content: str, rules: dict) -> tuple[bool, str]:
    """
    Validate generated output against configurable rules from a task config.

    Args:
        content: The generated text.
        rules: Validation config dict, e.g.:
            {
                "require_tags": [["<think>", "</think>"], ["<answer>", "</answer>"]],
                "require_latex": true,
                "min_reasoning_length": 100,
                "min_answer_length": 10,
            }
    Returns:
        (is_valid, issues_string)
    """
    issues = []
    lower = content.lower()

    # Tag validation
    for tag_pair in rules.get("require_tags", []):
        open_tag, close_tag = tag_pair[0].lower(), tag_pair[1].lower()
        if open_tag not in lower:
            issues.append(f"missing {tag_pair[0]}")
        if close_tag not in lower:
            issues.append(f"missing {tag_pair[1]}")

    # Tag ordering
    if not issues and rules.get("require_tags"):
        positions = []
        for tag_pair in rules["require_tags"]:
            positions.append(lower.find(tag_pair[0].lower()))
            positions.append(lower.find(tag_pair[1].lower()))
        if positions != sorted(positions):
            issues.append("tags out of order")

    # LaTeX check
    if rules.get("require_latex", False):
        has_latex = bool(re.search(r"\\[a-zA-Z]+|\\\[|\$", content))
        if not has_latex:
            issues.append("no LaTeX detected")

    # Length checks (only if tags are present)
    min_reasoning = rules.get("min_reasoning_length", 0)
    min_answer = rules.get("min_answer_length", 0)
    if min_reasoning > 0:
        think_match = re.search(
            r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE
        )
        if think_match and len(think_match.group(1).strip()) < min_reasoning:
            issues.append(f"reasoning too short (< {min_reasoning} chars)")

    if min_answer > 0:
        answer_match = re.search(
            r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE
        )
        if answer_match and len(answer_match.group(1).strip()) < min_answer:
            issues.append(f"answer too short (< {min_answer} chars)")

    return len(issues) == 0, ", ".join(issues) if issues else "valid"


# ── Detailed validators (used by filter) ───────────────────────────────

def validate_tags(output: str) -> tuple[bool, list[str]]:
    """Validate <think>/<answer> tags are properly opened, closed, and ordered."""
    issues = []

    think_opens = len(THINK_OPEN.findall(output))
    think_closes = len(THINK_CLOSE.findall(output))
    answer_opens = len(ANSWER_OPEN.findall(output))
    answer_closes = len(ANSWER_CLOSE.findall(output))

    if think_opens == 0:
        issues.append("Missing <think> tag")
    if think_closes == 0:
        issues.append("Missing </think> tag")
    if answer_opens == 0:
        issues.append("Missing <answer> tag")
    if answer_closes == 0:
        issues.append("Missing </answer> tag - CRITICAL: model learned to never conclude")

    if think_opens != think_closes:
        issues.append(f"Unbalanced think tags: {think_opens} opens, {think_closes} closes")
    if answer_opens != answer_closes:
        issues.append(f"Unbalanced answer tags: {answer_opens} opens, {answer_closes} closes")

    if not issues:
        think_start = output.lower().find("<think>")
        think_end = output.lower().find("</think>")
        answer_start = output.lower().find("<answer>")
        answer_end = output.lower().find("</answer>")
        if not (think_start < think_end < answer_start < answer_end):
            issues.append("Tags not in correct order: <think>...</think><answer>...</answer>")

    return len(issues) == 0, issues


def strip_math_environments(text: str) -> str:
    """Remove content inside LaTeX math environments to avoid false positives."""
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$[^$]+(?<!\\)\$", " ", text)
    text = re.sub(r"\\\(.*?\\\)", " ", text, flags=re.DOTALL)
    text = re.sub(
        r"\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}",
        " ",
        text,
        flags=re.DOTALL,
    )
    return text


def validate_latex(output: str) -> tuple[bool, list[str]]:
    """Check for LaTeX consistency."""
    issues = []

    text_outside_math = strip_math_environments(output)
    for pattern in PLAIN_TEXT_PHYSICS:
        if re.search(pattern, text_outside_math):
            issues.append(f"Plain text physics found (should be LaTeX): {pattern}")

    has_latex = bool(
        LATEX_INLINE.search(output)
        or LATEX_DISPLAY.search(output)
        or LATEX_OPERATORS.search(output)
    )
    if not has_latex:
        issues.append("No LaTeX math detected - physics response should contain LaTeX")

    dollar_count = output.count("$") - output.count("\\$")
    if dollar_count % 2 != 0:
        issues.append("Unclosed LaTeX $ delimiter")

    return len(issues) == 0, issues


def validate_reasoning_quality(output: str, rules: dict | None = None) -> tuple[bool, list[str]]:
    """Check for reasoning quality issues."""
    issues = []
    rules = rules or {}
    min_reasoning = rules.get("min_reasoning_length", 100)
    min_answer = rules.get("min_answer_length", 10)

    think_match = re.search(r"<think>(.*?)</think>", output, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_content = think_match.group(1)
        if len(think_content.strip()) < min_reasoning:
            issues.append(f"Reasoning too short (< {min_reasoning} chars)")
        has_steps = any(
            marker in think_content.lower()
            for marker in ["step 1", "first,", "1.", "1)", "let us", "we begin", "starting with"]
        )
        if not has_steps:
            issues.append("No clear step-by-step structure detected")

    answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL | re.IGNORECASE)
    if answer_match:
        if len(answer_match.group(1).strip()) < min_answer:
            issues.append("Answer too short - may be incomplete")

    sentences = output.split(".")
    if len(sentences) > 5:
        unique_starts = set(s.strip()[:30].lower() for s in sentences if len(s.strip()) > 30)
        if len(unique_starts) < len(sentences) * 0.5:
            issues.append("Possible reasoning loop - repetitive sentence structures")

    return len(issues) == 0, issues


# ── Healing functions ──────────────────────────────────────────────────

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content from \\boxed{} LaTeX command."""
    matches = BOXED_PATTERN.findall(text)
    return matches[-1].strip() if matches else None


def extract_final_answer(text: str) -> Optional[str]:
    """Try to extract final answer from conversational patterns."""
    for pattern in FINAL_ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r"\s*[.,:;]?\s*$", "", answer)
            if len(answer) > 3:
                return answer
    return None


def heal_answer_tag(output: str) -> tuple[str, bool, str]:
    """Attempt to heal a broken <answer> tag. Returns (output, was_healed, method)."""
    if ANSWER_OPEN.search(output) and ANSWER_CLOSE.search(output):
        answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL | re.IGNORECASE)
        if answer_match and answer_match.group(1).strip():
            return output, False, "already_valid"

    # Strategy 1: Missing </answer> with existing content
    if ANSWER_OPEN.search(output) and not ANSWER_CLOSE.search(output):
        answer_start = re.search(r"<answer>", output, re.IGNORECASE)
        if answer_start:
            content_after = output[answer_start.end():].strip()
            if len(content_after) > 20:
                return output.rstrip() + "\n</answer>", True, "close_tag"

    # Strategy 2: No answer tags — extract from \boxed{}
    if not ANSWER_OPEN.search(output):
        boxed = extract_boxed_answer(output)
        if boxed and THINK_CLOSE.search(output):
            replacement = f"</think>\n\n<answer>\n{boxed}\n</answer>"
            healed = THINK_CLOSE.sub(lambda m: replacement, output, count=1)
            return healed, True, "boxed_extraction"

    # Strategy 3: No answer tags — extract from conversational patterns
    if not ANSWER_OPEN.search(output):
        extracted = extract_final_answer(output)
        if extracted and THINK_CLOSE.search(output):
            replacement = f"</think>\n\n<answer>\n{extracted}\n</answer>"
            healed = THINK_CLOSE.sub(lambda m: replacement, output, count=1)
            return healed, True, "pattern_extraction"

    # Strategy 4: Answer tag exists but content is empty — inject boxed
    if ANSWER_OPEN.search(output):
        answer_match = re.search(r"<answer>(.*?)(?:</answer>|$)", output, re.DOTALL | re.IGNORECASE)
        if answer_match and len(answer_match.group(1).strip()) < 20:
            boxed = extract_boxed_answer(output)
            if boxed:
                replacement = f"<answer>\n{boxed}\n</answer>"
                healed = re.sub(
                    r"<answer>.*?(?:</answer>|$)",
                    lambda m: replacement,
                    output,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                return healed, True, "answer_replacement"

    # Strategy 5: Fallback — just add closing tag
    if ANSWER_OPEN.search(output) and not ANSWER_CLOSE.search(output):
        return output.rstrip() + "\n</answer>", True, "close_tag_fallback"

    return output, False, "unrecoverable"


def heal_think_tag(output: str) -> tuple[str, bool, str]:
    """Attempt to heal broken <think> tags."""
    has_open = THINK_OPEN.search(output)
    has_close = THINK_CLOSE.search(output)

    if has_open and has_close:
        return output, False, "already_valid"

    if not has_open and has_close:
        return "<think>\n" + output, True, "add_open_tag"

    if has_open and not has_close:
        if ANSWER_OPEN.search(output):
            healed = ANSWER_OPEN.sub(
                lambda m: "</think>\n\n<answer>", output, count=1
            )
            return healed, True, "add_close_before_answer"
        conclusion_match = re.search(
            r"(therefore|thus|hence|finally|in conclusion)[^.]*\.",
            output,
            re.IGNORECASE,
        )
        if conclusion_match:
            pos = conclusion_match.end()
            healed = output[:pos] + "\n</think>\n" + output[pos:]
            return healed, True, "add_close_at_conclusion"

    return output, False, "unrecoverable"


def heal_sample(output: str) -> tuple[str, bool, list[str]]:
    """Attempt to heal all issues in a sample. Returns (output, was_healed, methods)."""
    methods = []
    healed = output
    any_healed = False

    healed, think_healed, think_method = heal_think_tag(healed)
    if think_healed:
        methods.append(f"think:{think_method}")
        any_healed = True

    healed, answer_healed, answer_method = heal_answer_tag(healed)
    if answer_healed:
        methods.append(f"answer:{answer_method}")
        any_healed = True

    return healed, any_healed, methods
