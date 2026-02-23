"""ModelEvaluator — RAG-grounded judge evaluation engine.

Ported from Qwen-Fine-Tuning-Lab/evaluate_model.py.  All interactive
prompts, argparse, wandb, and config discovery have been stripped.  Embeds
OllamaClient, SemanticScholarRAG, and GGUFModel inline.

Supports:
- Streaming model output (generate_stream)
- Full article metadata (authors, pdf_url, semantic_scholar_url, is_open_access)
- RAG sources per scored result
- Evaluation log file (.txt)
- Custom field mapping from dataset_config
- HuggingFace dataset loading with config-driven splits
- Structured JSON output with config metadata
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
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "evaluation_results"


# ---------------------------------------------------------------------------
# OllamaClient
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
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
        max_retries: int = 3,
    ) -> Optional[str]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_gpu": 60,
            },
        }
        for attempt in range(max_retries):
            try:
                resp = self.session.post(
                    f"{self.base_url}/api/generate", json=payload, timeout=300
                )
                if resp.status_code == 500:
                    wait = 2 ** (attempt + 1)
                    print(
                        f"[Server error, retrying in {wait}s... ({attempt+1}/{max_retries})]",
                        end=" ",
                        flush=True,
                    )
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
# SemanticScholarRAG
# ---------------------------------------------------------------------------


class SemanticScholarRAG:
    """RAG client using Semantic Scholar API for grounding."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    SEARCH_FIELDS = (
        "paperId,title,abstract,year,citationCount,authors,openAccessPdf,isOpenAccess"
    )

    def __init__(
        self, max_papers: int = 3, api_key: Optional[str] = None
    ):
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
        text = re.sub(r"\$[^$]+\$", " ", question)
        text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
        text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
        text = re.sub(r"\\[a-zA-Z]+", " ", text)
        text = re.sub(r"[{}_^<>\\|]", " ", text)

        stopwords = {
            "what", "which", "how", "why", "when", "where", "who", "whom",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "the", "a", "an", "and", "or", "but", "in",
            "on", "at", "to", "for", "of", "with", "by", "from", "as",
            "into", "through", "during", "this", "that", "these", "those",
            "it", "its", "can", "may", "might", "explain", "describe",
            "define", "discuss", "compare", "contrast", "give", "provide",
            "list", "name", "identify", "answer", "question", "following",
            "example", "examples", "please", "briefly", "detail", "given",
            "assume", "using", "use", "find", "denotes", "all", "pairs",
            "sum", "sigma", "beta", "left", "right", "frac", "text",
        }

        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        words = text.split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        seen: set = set()
        unique = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        unique.sort(key=len, reverse=True)
        return " ".join(unique[:max_keywords])

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def search_papers(
        self, query: str, limit: Optional[int] = None
    ) -> List[Dict]:
        if limit is None:
            limit = self.max_papers
        cache_key = f"{query}:{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        for attempt in range(3):
            try:
                self._rate_limit()
                url = (
                    f"{self.BASE_URL}/paper/search"
                    f"?query={quote_plus(query)}&limit={limit}"
                    f"&fields={self.SEARCH_FIELDS}"
                )
                resp = self.session.get(url, timeout=10)
                if resp.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                    continue
                resp.raise_for_status()
                papers = resp.json().get("data", [])
                papers = [p for p in papers if p.get("abstract")]
                papers.sort(
                    key=lambda p: p.get("citationCount", 0), reverse=True
                )
                result = papers[:limit]
                self._cache[cache_key] = result
                return result
            except Exception:
                if attempt < 2:
                    time.sleep(2**attempt)
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
            return (
                f"Academic context from question:\n{question_context}\n\n"
                f"Additional context from reference topic:\n{reference_context}"
            )
        return question_context or reference_context or ""


# ---------------------------------------------------------------------------
# GGUFModel
# ---------------------------------------------------------------------------


class GGUFModel:
    """Wrapper for GGUF model inference via llama-cpp with streaming support."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        self.load()

    def load(self):
        if self.llm is not None:
            return
        from llama_cpp import Llama

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

    def generate(
        self, prompt: str, max_tokens: int = 3072, temperature: float = 0.1
    ) -> str:
        """Generate (non-streaming) response."""
        self.load()
        formatted = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
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

    def generate_stream(
        self, prompt: str, max_tokens: int = 3072, temperature: float = 0.1
    ) -> str:
        """Generate with streaming output. Reloads model if it was unloaded."""
        self.load()
        formatted = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        try:
            stream = self.llm(
                formatted,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|endoftext|>"],
                stream=True,
            )
            full_response = ""
            for output in stream:
                chunk = output["choices"][0]["text"]
                full_response += chunk
                print(chunk, end="", flush=True)
            print()
            return full_response.strip()
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
    """RAG-grounded judge evaluation engine with streaming, logging, and
    full article metadata."""

    def __init__(
        self,
        model_path: str,
        test_dataset_path: str,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        max_samples: int = 50,
        cancel_event: Optional[threading.Event] = None,
        s2_api_key: Optional[str] = None,
        dataset_config: Optional[Dict[str, Any]] = None,
        use_streaming: bool = True,
    ):
        self.model_path = model_path
        self.test_dataset_path = test_dataset_path
        self.judge_model = judge_model
        self.max_samples = max_samples
        self.cancel_event = cancel_event
        self.dataset_config = dataset_config
        self.use_streaming = use_streaming

        # Field mapping from dataset_config
        if dataset_config:
            fields = dataset_config.get("fields", {})
            self.instruction_field = fields.get("instruction", "instruction")
            self.response_field = fields.get("response", "response")
            self.domain = dataset_config.get(
                "domain", dataset_config.get("name", "General")
            )
        else:
            self.instruction_field = "instruction"
            self.response_field = "output"
            self.domain = "General"

        self.gguf_model: Optional[GGUFModel] = None
        self.judge = OllamaClient()
        self.rag = SemanticScholarRAG(
            max_papers=RAG_MAX_PAPERS, api_key=s2_api_key
        )

        self.results: List[Dict] = []
        self.article_logs: List[Dict] = []

        # Log file setup
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = (
            OUTPUT_DIR / f"eval_{self.domain.lower().replace(' ', '_')}_{timestamp}.txt"
        )
        self.json_path = (
            OUTPUT_DIR / f"eval_{self.domain.lower().replace(' ', '_')}_{timestamp}.json"
        )
        self.articles_log_path = (
            OUTPUT_DIR
            / f"articles_{self.domain.lower().replace(' ', '_')}_{timestamp}.json"
        )

    def _check_cancel(self):
        if self.cancel_event and self.cancel_event.is_set():
            raise InterruptedError("Evaluation cancelled by user")

    def log(self, text: str, also_print: bool = False):
        """Write to log file."""
        with open(self.log_path, "a") as f:
            f.write(text + "\n")
        if also_print:
            print(text)

    # ------------------------------------------------------------------
    # Purity & Entropy
    # ------------------------------------------------------------------

    @staticmethod
    def compute_purity(dataset_path: str) -> float:
        """Percentage of well-formed valid samples."""
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
        """Normalised Shannon entropy over the unigram distribution."""
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

        entropy = -sum(
            (c / total) * math.log2(c / total) for c in counter.values()
        )
        max_entropy = math.log2(len(counter))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    # ------------------------------------------------------------------
    # Load test data — HF dataset or local JSONL
    # ------------------------------------------------------------------

    def _load_test_data(self) -> List[Dict]:
        """Load test data from HuggingFace dataset (via config) or local JSONL."""
        if self.dataset_config and not self.dataset_config.get("is_local", False):
            return self._load_test_data_from_hf()
        return self._load_test_data_from_jsonl()

    def _load_test_data_from_hf(self) -> List[Dict]:
        """Load test data from HuggingFace, applying config-driven splits."""
        from datasets import load_dataset as hf_load_dataset

        dcfg = self.dataset_config
        dataset_name = dcfg["dataset_name"]
        split = dcfg.get("split", "train")
        split_config = dcfg.get("train_val_test_split")

        print(f"Loading HF dataset: {dataset_name}")
        self.log(f"Loading dataset: {dataset_name}")

        dataset = hf_load_dataset(dataset_name, split=split)

        # Apply split to extract test set
        if split_config:
            test_ratio = split_config.get("test", 0.2)
            val_ratio = split_config.get("val", 0.2)
            seed = split_config.get("seed", 3407)

            temp_size = val_ratio + test_ratio
            split1 = dataset.train_test_split(test_size=temp_size, seed=seed)

            if test_ratio > 0:
                ratio = test_ratio / temp_size
                split2 = split1["test"].train_test_split(
                    test_size=ratio, seed=seed
                )
                dataset = split2["test"]
            else:
                dataset = split1["test"]

        data = []
        for item in dataset:
            data.append(
                {
                    "question": item.get(self.instruction_field, ""),
                    "reference": item.get(self.response_field, ""),
                    "topic": item.get(
                        "topic", item.get("sub_topic", self.domain)
                    ),
                }
            )

        data = data[: self.max_samples]
        print(f"Loaded {len(data)} test samples")
        self.log(f"Test samples: {len(data)}\n")
        return data

    def _load_test_data_from_jsonl(self) -> List[Dict]:
        """Load test data from a local JSONL file."""
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
                data.append(
                    {
                        "question": obj.get(self.instruction_field, ""),
                        "reference": obj.get(self.response_field, ""),
                    }
                )

        data = data[: self.max_samples]
        print(f"Loaded {len(data)} test samples")
        return data

    # ------------------------------------------------------------------
    # Evaluate single sample
    # ------------------------------------------------------------------

    def _evaluate_one(self, item: Dict, idx: int, total: int) -> Dict:
        self._check_cancel()

        question = item["question"]
        reference = item.get("reference", "")

        print(f"Evaluating sample {idx}/{total}", flush=True)
        self.log(f"\n{'='*70}\n[{idx}/{total}] QUESTION\n{'='*70}")
        self.log(question)

        # Generate model response (streaming or non-streaming)
        if self.use_streaming:
            answer = self.gguf_model.generate_stream(question)
        else:
            answer = self.gguf_model.generate(question)
        self.log(answer)

        if not answer:
            return {"index": idx, "skipped": True, "reason": "No response"}

        # RAG retrieval
        keywords = self.rag._extract_keywords(question)
        papers = self.rag.search_papers(keywords) if keywords else []
        rag_context = self.rag.get_grounded_context(question, reference)

        # Log articles with full metadata
        article_entry: Dict[str, Any] = {
            "question_index": idx,
            "question": question[:500],
            "search_keywords": keywords,
            "papers_retrieved": [],
        }
        for paper in papers:
            open_access_pdf = paper.get("openAccessPdf")
            pdf_url = open_access_pdf.get("url") if open_access_pdf else None

            article_entry["papers_retrieved"].append(
                {
                    "paper_id": paper.get("paperId"),
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "authors": [
                        a.get("name") for a in paper.get("authors", [])[:5]
                    ],
                    "citation_count": paper.get("citationCount", 0),
                    "abstract": (paper.get("abstract") or "")[:1000],
                    "semantic_scholar_url": (
                        f"https://www.semanticscholar.org/paper/{paper.get('paperId')}"
                    ),
                    "is_open_access": paper.get("isOpenAccess", False),
                    "pdf_url": pdf_url,
                }
            )
        self.article_logs.append(article_entry)

        if not rag_context:
            return {
                "index": idx,
                "question": question,
                "answer": answer,
                "skipped": True,
                "reason": "No Semantic Scholar articles found",
            }

        # Unload test model, run judge, unload judge
        self.gguf_model.unload()

        judge_prompt = JUDGE_PROMPT_RAG.format(
            question=question[:1500],
            answer=answer[:2000],
            reference=reference[:1500] if reference else "No reference provided",
            rag_context=rag_context[:3000],
        )
        judge_response = self.judge.generate(
            self.judge_model, judge_prompt, max_tokens=4096
        )
        self.judge.unload_model(self.judge_model)

        if judge_response is None:
            return {
                "index": idx,
                "question": question,
                "answer": answer,
                "skipped": True,
                "reason": "Judge model failed",
            }

        # Parse scores
        scores = {
            "factual_accuracy": 0,
            "completeness": 0,
            "technical_precision": 0,
        }
        for key, pattern in [
            ("factual_accuracy", r"FACTUAL_ACCURACY:\s*(\d+)"),
            ("completeness", r"COMPLETENESS:\s*(\d+)"),
            ("technical_precision", r"TECHNICAL_PRECISION:\s*(\d+)"),
        ]:
            match = re.search(pattern, judge_response, re.IGNORECASE)
            if match:
                scores[key] = min(100, max(0, int(match.group(1))))

        justification = ""
        just_match = re.search(
            r"JUSTIFICATION:\s*(.+)", judge_response, re.IGNORECASE | re.DOTALL
        )
        if just_match:
            justification = just_match.group(1).strip()

        overall_score = (
            scores["factual_accuracy"] * 0.50
            + scores["completeness"] * 0.30
            + scores["technical_precision"] * 0.20
        )

        # Log scores
        score_text = (
            f"SCORES: factual_accuracy={scores['factual_accuracy']}, "
            f"completeness={scores['completeness']}, "
            f"technical_precision={scores['technical_precision']}, "
            f"overall={overall_score:.1f}"
        )
        print(score_text, flush=True)
        self.log(score_text)
        self.log(f"JUSTIFICATION: {justification}")

        # Build RAG sources for this result
        rag_sources = []
        if self.article_logs:
            last_article = self.article_logs[-1]
            if last_article.get("question_index") == idx:
                for paper in last_article.get("papers_retrieved", []):
                    rag_sources.append(
                        {
                            "title": paper.get("title", ""),
                            "year": paper.get("year"),
                            "authors": paper.get("authors", []),
                            "url": paper.get("semantic_scholar_url", ""),
                            "pdf_url": paper.get("pdf_url"),
                            "is_open_access": paper.get("is_open_access", False),
                        }
                    )

        return {
            "index": idx,
            "question": question,
            "answer": answer,
            "reference": reference,
            "scores": scores,
            "overall_score": overall_score,
            "justification": justification,
            "topic": item.get("topic", ""),
            "skipped": False,
            "rag_sources": rag_sources,
        }

    def _evaluate_with_retry(self, item: Dict, idx: int, total: int) -> Dict:
        for attempt in range(MAX_RETRIES_ON_ZERO + 1):
            result = self._evaluate_one(item, idx, total)

            should_retry = False
            if result.get("reason") == "No response":
                should_retry = True
            elif (
                result.get("skipped")
                and result.get("reason") == "Judge model failed"
            ):
                should_retry = True
            elif result.get("skipped") and "articles" in result.get(
                "reason", ""
            ).lower():
                should_retry = True
            elif not result.get("skipped"):
                scores = result.get("scores", {})
                if result.get("overall_score", -1) == 0 and all(
                    scores.get(k, 0) == 0 for k in scores
                ):
                    should_retry = True

            if not should_retry or attempt >= MAX_RETRIES_ON_ZERO:
                return result

            print(
                f"Retrying question {idx} (attempt {attempt+2}/{MAX_RETRIES_ON_ZERO+1})",
                flush=True,
            )
            self.log(f"[RETRY {attempt + 1}] Question {idx}")
            if (
                self.article_logs
                and self.article_logs[-1].get("question_index") == idx
            ):
                self.article_logs.pop()
            time.sleep(2.0)

        return result

    # ------------------------------------------------------------------
    # run_evaluation — main entry point
    # ------------------------------------------------------------------

    def run_evaluation(self) -> Dict[str, Any]:
        """Run the full evaluation pipeline.

        Returns structured results dict with all 5 metrics + per-sample data.
        """
        # Write log header
        self.log("=" * 70)
        self.log("MODEL EVALUATION LOG")
        self.log("=" * 70)
        self.log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Model: {Path(self.model_path).name}")
        self.log(f"Domain: {self.domain}")
        self.log(f"Judge: {self.judge_model}")
        self.log(f"RAG: Semantic Scholar ({RAG_MAX_PAPERS} papers)")
        self.log(f"Max Samples: {self.max_samples}")
        self.log("=" * 70)

        # Compute purity & entropy (cheap, no GPU)
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
        scored = [
            r
            for r in self.results
            if not r.get("skipped") and r.get("overall_score", 0) > 0
        ]
        skipped = [r for r in self.results if r.get("skipped")]
        failed = [
            r
            for r in self.results
            if not r.get("skipped") and r.get("overall_score", 0) == 0
        ]

        n_scored = len(scored)
        avg = (
            lambda key: sum(r["scores"][key] for r in scored) / n_scored
            if n_scored
            else 0.0
        )

        result_dict = {
            "factual_accuracy": avg("factual_accuracy"),
            "completeness": avg("completeness"),
            "technical_precision": avg("technical_precision"),
            "overall_accuracy": (
                sum(r["overall_score"] for r in scored) / n_scored
                if n_scored
                else 0.0
            ),
            "purity": purity,
            "entropy": entropy,
            "samples_scored": n_scored,
            "samples_skipped": len(skipped),
            "samples_failed": len(failed),
            "results": self.results,
            "articles": self.article_logs,
        }

        # Save structured JSON with config metadata
        self._save_results(result_dict)

        return result_dict

    def _save_results(self, result_dict: Dict[str, Any]) -> None:
        """Save evaluation results and article logs to disk."""
        # JSON results with config metadata
        json_output = {
            "config": {
                "test_model": Path(self.model_path).name,
                "judge_model": self.judge_model,
                "domain": self.domain,
                "samples_attempted": len(self.results),
                "samples_scored": result_dict["samples_scored"],
                "samples_skipped": result_dict["samples_skipped"],
                "rag_source": "Semantic Scholar API",
                "rag_papers": RAG_MAX_PAPERS,
            },
            "metrics": {
                "overall_accuracy": result_dict["overall_accuracy"],
                "factual_accuracy": result_dict["factual_accuracy"],
                "completeness": result_dict["completeness"],
                "technical_precision": result_dict["technical_precision"],
                "purity": result_dict["purity"],
                "entropy": result_dict["entropy"],
            },
            "results": self.results,
        }

        if self.dataset_config:
            json_output["config"]["dataset"] = self.dataset_config.get(
                "dataset_name"
            )

        with open(self.json_path, "w") as f:
            json.dump(json_output, f, indent=2)

        # Article logs
        with open(self.articles_log_path, "w") as f:
            json.dump(
                {
                    "evaluation_info": {
                        "model": Path(self.model_path).name,
                        "domain": self.domain,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "total_questions": len(self.results),
                        "rag_source": "Semantic Scholar API",
                    },
                    "article_logs": self.article_logs,
                },
                f,
                indent=2,
            )

        print(f"Log saved:     {self.log_path}")
        print(f"JSON saved:    {self.json_path}")
        print(f"Articles log:  {self.articles_log_path}")
