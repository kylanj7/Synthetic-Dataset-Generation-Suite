"""ModelEvaluator — RAG-grounded judge evaluation engine.

Refactored from Qwen-Fine-Tuning-Lab/evaluate_model.py.  All interactive
prompts, argparse, wandb, and config discovery have been stripped.  Embeds
OllamaClient, SemanticScholarRAG, and GGUFModel inline.
"""

import gc
import json
import math
import os
import re
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_JUDGE_MODEL = "gpt-oss:120b"
RAG_MAX_PAPERS = 5
MAX_RETRIES_ON_ZERO = 2


# ---------------------------------------------------------------------------
# OllamaClient  (from evaluate_model.py:58-133)
# ---------------------------------------------------------------------------

class OllamaClient:
    """Client for Ollama API (judge model)."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def list_models(self) -> List[str]:
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def generate(
        self, model: str, prompt: str,
        temperature: float = 0.0, max_tokens: int = 200, max_retries: int = 3,
    ) -> Optional[str]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_gpu": 60},
        }
        for attempt in range(max_retries):
            try:
                resp = self.session.post(f"{self.base_url}/api/generate", json=payload, timeout=300)
                if resp.status_code == 500:
                    wait = 2 ** (attempt + 1)
                    print(f"[Server error, retrying in {wait}s... ({attempt+1}/{max_retries})]", end=" ", flush=True)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json().get("response", "")
            except requests.exceptions.Timeout:
                time.sleep(2 ** (attempt + 1))
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                print(f"[Error after {max_retries} attempts: {e}]")
                return None
        return None

    def unload_model(self, model: str):
        try:
            self.session.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
                timeout=30,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SemanticScholarRAG  (from evaluate_model.py:139-327)
# ---------------------------------------------------------------------------

class SemanticScholarRAG:
    """RAG client using Semantic Scholar API for grounding."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    SEARCH_FIELDS = "paperId,title,abstract,year,citationCount,authors,openAccessPdf,isOpenAccess"

    def __init__(self, max_papers: int = 3, api_key: Optional[str] = None):
        self.max_papers = max_papers
        self.session = requests.Session()
        self.api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        self.session.headers.update(headers)
        self._cache: Dict[str, List[Dict]] = {}
        self._last_request_time = 0.0
        self._min_request_interval = 1.1 if self.api_key else 3.0

    def _extract_keywords(self, question: str, max_keywords: int = 5) -> str:
        text = re.sub(r'\$[^$]+\$', ' ', question)
        text = re.sub(r'\\\[.*?\\\]', ' ', text, flags=re.DOTALL)
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', text)
        text = re.sub(r'\\[a-zA-Z]+', ' ', text)
        text = re.sub(r'[{}_^<>\\|]', ' ', text)

        stopwords = {
            'what', 'which', 'how', 'why', 'when', 'where', 'who', 'whom',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'into', 'through', 'during',
            'this', 'that', 'these', 'those', 'it', 'its', 'can', 'may', 'might',
            'explain', 'describe', 'define', 'discuss', 'compare', 'contrast',
            'give', 'provide', 'list', 'name', 'identify', 'answer', 'question',
            'following', 'example', 'examples', 'please', 'briefly', 'detail',
            'given', 'assume', 'using', 'use', 'find', 'denotes', 'all', 'pairs',
            'sum', 'sigma', 'beta', 'left', 'right', 'frac', 'text',
        }

        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        words = text.split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        seen: set = set()
        unique = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        unique.sort(key=len, reverse=True)
        return ' '.join(unique[:max_keywords])

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def search_papers(self, query: str, limit: Optional[int] = None) -> List[Dict]:
        if limit is None:
            limit = self.max_papers
        cache_key = f"{query}:{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        for attempt in range(3):
            try:
                self._rate_limit()
                url = f"{self.BASE_URL}/paper/search?query={quote_plus(query)}&limit={limit}&fields={self.SEARCH_FIELDS}"
                resp = self.session.get(url, timeout=10)
                if resp.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                    continue
                resp.raise_for_status()
                papers = resp.json().get("data", [])
                papers = [p for p in papers if p.get("abstract")]
                papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
                result = papers[:limit]
                self._cache[cache_key] = result
                return result
            except Exception:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return []
        return []

    def retrieve_context(self, question: str) -> str:
        keywords = self._extract_keywords(question)
        if not keywords:
            return ""
        papers = self.search_papers(keywords)
        if not papers:
            return ""
        parts = []
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Unknown")
            year = paper.get("year", "N/A")
            abstract = paper.get("abstract", "")
            citations = paper.get("citationCount", 0)
            authors = paper.get("authors", [])
            author_names = ", ".join([a.get("name", "") for a in authors[:3]])
            if len(authors) > 3:
                author_names += " et al."
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            parts.append(
                f"[{i}] {title} ({year})\n"
                f"    Authors: {author_names}\n"
                f"    Citations: {citations}\n"
                f"    Abstract: {abstract}"
            )
        return "\n\n".join(parts)

    def get_grounded_context(self, question: str, reference: str = "") -> str:
        question_context = self.retrieve_context(question)
        reference_context = ""
        if reference and len(reference) > 50:
            ref_keywords = self._extract_keywords(reference[:300])
            if ref_keywords:
                ref_papers = self.search_papers(ref_keywords, limit=2)
                if ref_papers:
                    reference_context = self.retrieve_context(reference[:300])
        if question_context and reference_context:
            return f"Academic context from question:\n{question_context}\n\nAdditional context from reference topic:\n{reference_context}"
        return question_context or reference_context or ""


# ---------------------------------------------------------------------------
# GGUFModel  (from evaluate_model.py:334-397)
# ---------------------------------------------------------------------------

class GGUFModel:
    """Wrapper for GGUF model inference via llama-cpp."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        self.load()

    def load(self):
        if self.llm is not None:
            return
        from llama_cpp import Llama  # lazy import
        print("Loading test model into VRAM...", flush=True)
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        print("Model loaded!")

    def unload(self):
        if self.llm is None:
            return
        print("Unloading test model from VRAM...", flush=True)
        del self.llm
        self.llm = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(self, prompt: str, max_tokens: int = 3072, temperature: float = 0.1) -> str:
        """Generate (non-streaming) response."""
        self.load()
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        try:
            output = self.llm(
                formatted,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            print(f"[Generation error: {e}]")
            return ""


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT_RAG = """You are a rigorous fact-checker evaluating a model's response using academic sources as ground truth.

QUERY:
{question}

MODEL OUTPUT:
{answer}

REFERENCE ANSWER:
{reference}

ACADEMIC SOURCES (ground truth):
{rag_context}

Score the model output on three dimensions (0-100 each):

1. FACTUAL_ACCURACY: Do the claims match the academic sources and reference?
   - 100: All claims verified and accurate
   - 70-99: Minor inaccuracies that don't affect core understanding
   - 40-69: Some significant errors but partial correctness
   - 0-39: Major factual errors or contradictions

2. COMPLETENESS: Does the response fully address the query?
   - 100: Comprehensive, addresses all aspects
   - 70-99: Addresses main points, minor omissions
   - 40-69: Partial coverage, missing key elements
   - 0-39: Severely incomplete or off-topic

3. TECHNICAL_PRECISION: Are equations, terminology, and methods correct?
   - 100: Flawless technical presentation
   - 70-99: Minor notation or terminology issues
   - 40-69: Some technical errors but approach is sound
   - 0-39: Fundamental technical mistakes

Respond in this exact format:
FACTUAL_ACCURACY: [score]
COMPLETENESS: [score]
TECHNICAL_PRECISION: [score]
JUSTIFICATION: [brief explanation citing specific issues or strengths]"""


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """RAG-grounded judge evaluation engine."""

    def __init__(
        self,
        model_path: str,
        test_dataset_path: str,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        max_samples: int = 50,
        cancel_event: Optional[threading.Event] = None,
        s2_api_key: Optional[str] = None,
    ):
        self.model_path = model_path
        self.test_dataset_path = test_dataset_path
        self.judge_model = judge_model
        self.max_samples = max_samples
        self.cancel_event = cancel_event

        self.gguf_model: Optional[GGUFModel] = None
        self.judge = OllamaClient()
        self.rag = SemanticScholarRAG(max_papers=RAG_MAX_PAPERS, api_key=s2_api_key)

        self.results: List[Dict] = []
        self.article_logs: List[Dict] = []

    def _check_cancel(self):
        if self.cancel_event and self.cancel_event.is_set():
            raise InterruptedError("Evaluation cancelled by user")

    # ------------------------------------------------------------------
    # Purity & Entropy  (new metrics)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_purity(dataset_path: str) -> float:
        """Percentage of well-formed valid samples.

        A sample is "pure" when it:
        - Parses as valid JSON with both ``instruction`` and ``output`` fields
        - Both fields are non-empty and meet minimum length (10 chars each)
        - ``instruction`` != ``output`` (not copied)
        """
        path = Path(dataset_path)
        if not path.exists():
            return 0.0

        total = 0
        valid = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                instruction = (obj.get("instruction") or "").strip()
                output = (obj.get("output") or "").strip()
                if len(instruction) < 10 or len(output) < 10:
                    continue
                if instruction == output:
                    continue
                valid += 1

        return (valid / total * 100.0) if total > 0 else 0.0

    @staticmethod
    def compute_entropy(dataset_path: str) -> float:
        """Normalised Shannon entropy over the unigram distribution.

        Returns a value in [0, 1] where 1 means perfectly uniform and 0 means
        all tokens are identical.
        """
        path = Path(dataset_path)
        if not path.exists():
            return 0.0

        counter: Counter = Counter()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                text = f"{obj.get('instruction', '')} {obj.get('output', '')}"
                counter.update(text.lower().split())

        total = sum(counter.values())
        if total == 0 or len(counter) <= 1:
            return 0.0

        entropy = -sum((c / total) * math.log2(c / total) for c in counter.values())
        max_entropy = math.log2(len(counter))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------

    def _load_test_data(self) -> List[Dict]:
        path = Path(self.test_dataset_path)
        print(f"Loading test data: {path}")

        data: List[Dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                data.append({
                    "question": obj.get("instruction", ""),
                    "reference": obj.get("output", ""),
                })

        data = data[: self.max_samples]
        print(f"Loaded {len(data)} test samples")
        return data

    # ------------------------------------------------------------------
    # Evaluate single sample  (from evaluate_model.py:618-865)
    # ------------------------------------------------------------------

    def _evaluate_one(self, item: Dict, idx: int, total: int) -> Dict:
        self._check_cancel()

        question = item["question"]
        reference = item.get("reference", "")

        print(f"Evaluating sample {idx}/{total}", flush=True)
        answer = self.gguf_model.generate(question)

        if not answer:
            return {"index": idx, "skipped": True, "reason": "No response"}

        # RAG retrieval
        keywords = self.rag._extract_keywords(question)
        papers = self.rag.search_papers(keywords) if keywords else []
        rag_context = self.rag.get_grounded_context(question, reference)

        # Log articles
        article_entry: Dict[str, Any] = {
            "question_index": idx,
            "question": question[:500],
            "search_keywords": keywords,
            "papers_retrieved": [
                {
                    "paper_id": p.get("paperId"),
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "citation_count": p.get("citationCount", 0),
                    "abstract": (p.get("abstract") or "")[:1000],
                }
                for p in papers
            ],
        }
        self.article_logs.append(article_entry)

        if not rag_context:
            return {"index": idx, "question": question, "answer": answer, "skipped": True, "reason": "No Semantic Scholar articles found"}

        # Unload test model, run judge, unload judge
        self.gguf_model.unload()

        judge_prompt = JUDGE_PROMPT_RAG.format(
            question=question[:1500],
            answer=answer[:2000],
            reference=reference[:1500] if reference else "No reference provided",
            rag_context=rag_context[:3000],
        )
        judge_response = self.judge.generate(self.judge_model, judge_prompt, max_tokens=4096)
        self.judge.unload_model(self.judge_model)

        if judge_response is None:
            return {"index": idx, "question": question, "answer": answer, "skipped": True, "reason": "Judge model failed"}

        # Parse scores
        scores = {"factual_accuracy": 0, "completeness": 0, "technical_precision": 0}
        for key, pattern in [
            ("factual_accuracy", r"FACTUAL_ACCURACY:\s*(\d+)"),
            ("completeness", r"COMPLETENESS:\s*(\d+)"),
            ("technical_precision", r"TECHNICAL_PRECISION:\s*(\d+)"),
        ]:
            match = re.search(pattern, judge_response, re.IGNORECASE)
            if match:
                scores[key] = min(100, max(0, int(match.group(1))))

        justification = ""
        just_match = re.search(r"JUSTIFICATION:\s*(.+)", judge_response, re.IGNORECASE | re.DOTALL)
        if just_match:
            justification = just_match.group(1).strip()

        overall_score = (
            scores["factual_accuracy"] * 0.50
            + scores["completeness"] * 0.30
            + scores["technical_precision"] * 0.20
        )

        print(
            f"SCORES: factual_accuracy={scores['factual_accuracy']}, "
            f"completeness={scores['completeness']}, "
            f"technical_precision={scores['technical_precision']}",
            flush=True,
        )

        return {
            "index": idx,
            "question": question,
            "answer": answer,
            "reference": reference,
            "scores": scores,
            "overall_score": overall_score,
            "justification": justification,
            "skipped": False,
        }

    def _evaluate_with_retry(self, item: Dict, idx: int, total: int) -> Dict:
        for attempt in range(MAX_RETRIES_ON_ZERO + 1):
            result = self._evaluate_one(item, idx, total)

            should_retry = False
            if result.get("reason") == "No response":
                should_retry = True
            elif result.get("skipped") and result.get("reason") == "Judge model failed":
                should_retry = True
            elif result.get("skipped") and "articles" in result.get("reason", "").lower():
                should_retry = True
            elif not result.get("skipped"):
                scores = result.get("scores", {})
                if result.get("overall_score", -1) == 0 and all(scores.get(k, 0) == 0 for k in scores):
                    should_retry = True

            if not should_retry or attempt >= MAX_RETRIES_ON_ZERO:
                return result

            print(f"Retrying question {idx} (attempt {attempt+2}/{MAX_RETRIES_ON_ZERO+1})", flush=True)
            if self.article_logs and self.article_logs[-1].get("question_index") == idx:
                self.article_logs.pop()
            time.sleep(2.0)

        return result

    # ------------------------------------------------------------------
    # run_evaluation  — main entry point for job_runner
    # ------------------------------------------------------------------

    def run_evaluation(self) -> Dict[str, Any]:
        """Run the full evaluation pipeline.

        Returns structured results dict with all 5 metrics + per-sample data.
        """
        # Compute purity & entropy first (cheap, no GPU)
        purity = self.compute_purity(self.test_dataset_path)
        entropy = self.compute_entropy(self.test_dataset_path)
        print(f"Dataset purity: {purity:.1f}%")
        print(f"Dataset entropy: {entropy:.4f}")

        # Load GGUF model
        self.gguf_model = GGUFModel(self.model_path)

        # Load test data
        test_data = self._load_test_data()
        if not test_data:
            return {
                "purity": purity,
                "entropy": entropy,
                "error": "No test data loaded",
            }

        # Evaluate
        for i, item in enumerate(test_data, 1):
            self._check_cancel()
            result = self._evaluate_with_retry(item, i, len(test_data))
            self.results.append(result)
            time.sleep(0.2)

        # Aggregate
        scored = [r for r in self.results if not r.get("skipped") and r.get("overall_score", 0) > 0]
        skipped = [r for r in self.results if r.get("skipped")]
        failed = [r for r in self.results if not r.get("skipped") and r.get("overall_score", 0) == 0]

        n_scored = len(scored)
        avg = lambda key: sum(r["scores"][key] for r in scored) / n_scored if n_scored else 0.0

        return {
            "factual_accuracy": avg("factual_accuracy"),
            "completeness": avg("completeness"),
            "technical_precision": avg("technical_precision"),
            "overall_accuracy": sum(r["overall_score"] for r in scored) / n_scored if n_scored else 0.0,
            "purity": purity,
            "entropy": entropy,
            "samples_scored": n_scored,
            "samples_skipped": len(skipped),
            "samples_failed": len(failed),
            "results": self.results,
            "articles": self.article_logs,
        }
