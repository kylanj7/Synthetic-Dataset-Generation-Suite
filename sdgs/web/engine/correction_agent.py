"""Claude-powered self-correction agent for failing evaluation samples."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

from ..services.broadcast import BroadcastType, enqueue_broadcast


class CorrectionAgent:
    """Synthesises gold Q&A pairs for samples that scored below threshold."""

    def __init__(
        self,
        api_key: str,
        score_threshold: float = 50.0,
        model: str = "claude-opus-4-20250916",
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.score_threshold = score_threshold
        self.model = model

    def find_failing_samples(
        self, per_sample_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return samples whose overall_accuracy is below threshold."""
        failing = []
        for sample in per_sample_results:
            score = sample.get("overall_accuracy") or sample.get("score", 0)
            if score < self.score_threshold:
                failing.append(sample)
        return failing

    def synthesize_correction(
        self,
        sample: Dict[str, Any],
        paper_context: str,
    ) -> Optional[Dict[str, str]]:
        """Call Claude to produce a corrected {instruction, output} pair."""
        instruction = sample.get("instruction", "")
        original_output = sample.get("model_output") or sample.get("output", "")
        reference_output = sample.get("reference", "") or sample.get("expected", "")

        prompt = f"""You are a scientific QA gold-standard editor. A fine-tuned model produced a poor answer. Your job is to write the ideal answer.

## Original Question
{instruction}

## Model's Answer (poor quality)
{original_output}

## Reference Answer
{reference_output}

## Source Paper Context
{paper_context}

## Task
Write the best possible answer to the question, drawing on the paper context and reference answer. Return ONLY valid JSON:
{{"instruction": "<the original question, unchanged>", "output": "<your ideal answer>"}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Extract JSON from response (handle markdown code fences)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            return json.loads(text)
        except (json.JSONDecodeError, IndexError, anthropic.APIError) as exc:
            print(f"Correction synthesis failed: {exc}")
            return None

    def append_to_dataset(
        self,
        dataset_path: str,
        corrected_samples: List[Dict[str, str]],
    ) -> int:
        """Append corrected Q&A pairs to the training JSONL. Returns count appended."""
        path = Path(dataset_path)
        if not path.exists():
            return 0
        count = 0
        with open(path, "a") as f:
            for sample in corrected_samples:
                if sample.get("instruction") and sample.get("output"):
                    f.write(json.dumps(sample) + "\n")
                    count += 1
        return count

    def run_correction(
        self,
        eval_id: int,
        per_sample_results: List[Dict[str, Any]],
        dataset_path: str,
        articles_log: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Full correction pipeline: filter → synthesize → append → broadcast."""
        # Build paper context from articles_log
        paper_context = ""
        if articles_log:
            snippets = []
            for article in articles_log[:10]:
                title = article.get("title", "")
                abstract = article.get("abstract", "")
                if title:
                    snippets.append(f"### {title}\n{abstract}")
            paper_context = "\n\n".join(snippets)

        failing = self.find_failing_samples(per_sample_results)

        enqueue_broadcast(BroadcastType.CORRECTION, eval_id, {
            "type": "started",
            "total_failing": len(failing),
            "monologue": f"Found {len(failing)} failing samples to correct.",
        })

        corrected: List[Dict[str, str]] = []
        errors = 0

        for i, sample in enumerate(failing):
            enqueue_broadcast(BroadcastType.CORRECTION, eval_id, {
                "type": "progress",
                "current": i + 1,
                "total": len(failing),
                "monologue": f"Correcting sample {i + 1}/{len(failing)}...",
            })

            result = self.synthesize_correction(sample, paper_context)
            if result:
                corrected.append(result)
            else:
                errors += 1

        appended = self.append_to_dataset(dataset_path, corrected)

        summary = {
            "total_failing": len(failing),
            "corrected": len(corrected),
            "errors": errors,
            "appended_to_dataset": appended,
            "dataset_path": dataset_path,
            "corrected_samples": corrected,
        }

        enqueue_broadcast(BroadcastType.CORRECTION, eval_id, {
            "type": "complete",
            "monologue": f"Correction done: {len(corrected)} corrected, {errors} errors, {appended} appended.",
            **{k: v for k, v in summary.items() if k != "corrected_samples"},
        })

        return summary
