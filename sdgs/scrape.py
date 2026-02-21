"""Scholarly paper search, full-text extraction, and Q&A generation pipeline."""
import io
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .tracker import GPUTracker, TokenTracker
from .validate import validate_output


# ── Browser-like headers for PDF fetching ─────────────────────────────

_PDF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*;q=0.8",
}


# ── Retry session ─────────────────────────────────────────────────────

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """Return a shared requests.Session with automatic retry on transient errors."""
    global _session
    if _session is None:
        _session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


# ── Rate limiting ──────────────────────────────────────────────────────

_last_request = {}


def _rate_limit(source: str, delay: float):
    """Simple per-source rate limiter."""
    now = time.time()
    last = _last_request.get(source, 0)
    wait = delay - (now - last)
    if wait > 0:
        time.sleep(wait)
    _last_request[source] = time.time()


# ── Paper search ───────────────────────────────────────────────────────

def _search_arxiv(topic: str, max_results: int) -> list[dict]:
    """Search arXiv via its API and return paper metadata dicts."""
    papers = []
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in client.results(search):
            _rate_limit("arxiv", 0.34)  # ~3 req/sec
            arxiv_id = result.entry_id.split("/abs/")[-1]
            papers.append({
                "paper_id": f"arxiv:{arxiv_id}",
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "year": result.published.year if result.published else None,
                "doi": result.doi,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "source": "arxiv",
                "citation_count": 0,
                "full_text": None,
            })
    except Exception as e:
        print(f"  arXiv search error: {e}")
    return papers


def _search_semantic_scholar(topic: str, max_results: int) -> list[dict]:
    """Search Semantic Scholar API and return paper metadata dicts."""
    papers = []
    try:
        _rate_limit("semantic_scholar", 1.0)
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": topic,
            "limit": min(max_results, 100),
            "fields": "paperId,title,authors,abstract,year,externalIds,url,citationCount,isOpenAccess,openAccessPdf",
        }
        headers = {}
        s2_key = os.environ.get("S2_API_KEY")
        if s2_key:
            headers["x-api-key"] = s2_key
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("data", []):
            ext_ids = item.get("externalIds") or {}
            arxiv_id = ext_ids.get("ArXiv")
            doi = ext_ids.get("DOI")
            oa_pdf = item.get("openAccessPdf") or {}
            pdf_url = oa_pdf.get("url")

            # Build paper_id
            if arxiv_id:
                paper_id = f"arxiv:{arxiv_id}"
            elif doi:
                paper_id = f"doi:{doi}"
            else:
                paper_id = f"s2:{item.get('paperId', 'unknown')}"

            papers.append({
                "paper_id": paper_id,
                "title": item.get("title", ""),
                "authors": [a.get("name", "") for a in (item.get("authors") or [])],
                "abstract": item.get("abstract") or "",
                "year": item.get("year"),
                "doi": doi,
                "url": item.get("url", ""),
                "pdf_url": pdf_url,
                "source": "semantic_scholar",
                "citation_count": item.get("citationCount", 0) or 0,
                "full_text": None,
            })
    except Exception as e:
        print(f"  Semantic Scholar search error: {e}")
    return papers


def search_papers(topic: str, max_results: int = 20) -> list[dict]:
    """Search arXiv + Semantic Scholar, deduplicate, return merged results.

    Args:
        topic: Research topic query string.
        max_results: Max papers to return total.

    Returns:
        List of paper metadata dicts, deduplicated by paper_id.
    """
    print(f"Searching for papers on: {topic}")

    # Search both sources
    arxiv_papers = _search_arxiv(topic, max_results)
    print(f"  arXiv: {len(arxiv_papers)} results")

    s2_papers = _search_semantic_scholar(topic, max_results)
    print(f"  Semantic Scholar: {len(s2_papers)} results")

    # Deduplicate — prefer arXiv entries (they have direct PDF URLs)
    seen_ids = set()
    seen_titles = set()
    merged = []

    for paper in arxiv_papers + s2_papers:
        pid = paper["paper_id"]
        title_key = paper["title"].lower().strip()

        if pid in seen_ids or title_key in seen_titles:
            continue
        seen_ids.add(pid)
        seen_titles.add(title_key)
        merged.append(paper)

    # Sort by citation count descending
    merged.sort(key=lambda p: p["citation_count"], reverse=True)

    # Trim to max_results
    merged = merged[:max_results]
    print(f"  Merged (deduplicated): {len(merged)} papers")
    return merged


# ── Full-text extraction ───────────────────────────────────────────────

def _fetch_pdf_text(pdf_url: str, paper_id: str = "") -> tuple[str | None, str | None]:
    """Download PDF, save to disk, extract text with PyMuPDF.

    Returns:
        Tuple of (extracted_text, pdf_path) where pdf_path is the saved file
        path relative to the project root, or None if download/extraction failed.
    """
    try:
        import pymupdf

        resp = _get_session().get(pdf_url, headers=_PDF_HEADERS, timeout=60, stream=True)
        resp.raise_for_status()

        raw_bytes = resp.content

        # Save PDF to data/pdfs/
        pdf_dir = Path(__file__).resolve().parent.parent / "data" / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', paper_id)[:100] if paper_id else "unknown"
        pdf_file = pdf_dir / f"{safe_name}.pdf"
        pdf_file.write_bytes(raw_bytes)
        pdf_path = str(pdf_file)

        pdf_bytes = io.BytesIO(raw_bytes)
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()

        full_text = "\n".join(text_parts).strip()
        if len(full_text) > 100:
            return full_text, pdf_path
        return None, pdf_path
    except Exception as e:
        print(f"    PDF extraction failed: {e}")
        return None, None


def _unpaywall_pdf_url(doi: str) -> str | None:
    """Try to find an open-access PDF via Unpaywall."""
    if not doi:
        return None
    try:
        url = f"https://api.unpaywall.org/v2/{doi}"
        params = {"email": "sdgs-tool@example.com"}
        resp = _get_session().get(url, params=params, headers=_PDF_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        best_oa = data.get("best_oa_location") or {}
        return best_oa.get("url_for_pdf")
    except Exception:
        return None


def _pmc_pdf_url(doi: str) -> str | None:
    """Try to find an open-access PDF via PubMed Central (for biomedical papers)."""
    if not doi:
        return None
    try:
        # Step 1: DOI → PMCID via NCBI ID Converter
        conv_url = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
        conv_params = {"ids": doi, "format": "json"}
        resp = _get_session().get(conv_url, params=conv_params, headers=_PDF_HEADERS, timeout=15)
        resp.raise_for_status()
        records = resp.json().get("records", [])
        if not records:
            return None
        pmcid = records[0].get("pmcid")
        if not pmcid:
            return None

        # Step 2: PMCID → PDF link via OA service
        oa_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
        oa_params = {"id": pmcid}
        resp = _get_session().get(oa_url, params=oa_params, headers=_PDF_HEADERS, timeout=15)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        for link in root.iter("link"):
            href = link.get("href", "")
            fmt = link.get("format", "")
            if fmt == "pdf" or href.endswith(".pdf"):
                # Convert FTP URLs to HTTPS
                if href.startswith("ftp://"):
                    href = href.replace("ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/", "https://pmc.ncbi.nlm.nih.gov/articles/", 1)
                return href
        return None
    except Exception:
        return None


def fetch_full_text(paper: dict) -> str | None:
    """Fetch full text for a paper via multi-level fallback chain.

    Also sets paper["pdf_path"] if a PDF is successfully saved to disk.

    Fallback chain:
        1. Direct pdf_url (from search metadata)
        2. arXiv canonical PDF (if paper_id starts with "arxiv:")
        3. PubMed Central (if paper has DOI)
        4. Unpaywall (if paper has DOI)

    Args:
        paper: Paper metadata dict (modified in-place with pdf_path).

    Returns:
        Extracted text string, or None if unavailable.
    """
    title_short = paper["title"][:60]
    paper_id = paper.get("paper_id", "")
    tried_urls: set[str] = set()

    def _try_url(url: str) -> str | None:
        if not url or url in tried_urls:
            return None
        tried_urls.add(url)
        text, pdf_path = _fetch_pdf_text(url, paper_id)
        if pdf_path:
            paper["pdf_path"] = pdf_path
        return text

    # 1. Direct PDF URL from search metadata
    if paper.get("pdf_url"):
        print(f"    Fetching PDF: {title_short}...")
        text = _try_url(paper["pdf_url"])
        if text:
            return text

    # 2. arXiv canonical PDF fallback
    if paper_id.startswith("arxiv:"):
        arxiv_id = paper_id.removeprefix("arxiv:")
        arxiv_pdf = f"https://arxiv.org/pdf/{arxiv_id}"
        if arxiv_pdf not in tried_urls:
            print(f"    Trying arXiv PDF fallback: {title_short}...")
            _rate_limit("arxiv", 0.34)
            text = _try_url(arxiv_pdf)
            if text:
                return text

    # 3. PubMed Central fallback (biomedical papers)
    if paper.get("doi"):
        print(f"    Trying PMC for: {title_short}...")
        pmc_url = _pmc_pdf_url(paper["doi"])
        if pmc_url:
            text = _try_url(pmc_url)
            if text:
                return text

    # 4. Unpaywall fallback
    if paper.get("doi"):
        print(f"    Trying Unpaywall for: {title_short}...")
        oa_url = _unpaywall_pdf_url(paper["doi"])
        if oa_url:
            text = _try_url(oa_url)
            if text:
                return text

    return None


# ── Q&A generation from papers ─────────────────────────────────────────

def _parse_qa_pairs(content: str, paper: dict) -> list[dict]:
    """Parse <qa> blocks from LLM output into JSONL-compatible dicts."""
    pairs = []
    qa_blocks = re.findall(
        r"<qa>(.*?)</qa>", content, re.DOTALL | re.IGNORECASE
    )

    for block in qa_blocks:
        q_match = re.search(
            r"<question>(.*?)</question>", block, re.DOTALL | re.IGNORECASE
        )
        think_match = re.search(
            r"<think>(.*?)</think>", block, re.DOTALL | re.IGNORECASE
        )
        a_match = re.search(
            r"<answer>(.*?)</answer>", block, re.DOTALL | re.IGNORECASE
        )

        if q_match and a_match:
            think_text = think_match.group(1).strip() if think_match else ""
            answer_text = a_match.group(1).strip()

            output_text = ""
            if think_text:
                output_text += f"<think>{think_text}</think>\n"
            output_text += f"<answer>{answer_text}</answer>"

            entry = {
                "instruction": q_match.group(1).strip(),
                "output": output_text,
                "valid": True,
                "source_paper": paper["paper_id"],
                "source_title": paper["title"],
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "abstract": paper.get("abstract", ""),
                "url": paper.get("url", ""),
                "doi": paper.get("doi"),
                "source": paper.get("source", ""),
                "citation_count": paper.get("citation_count", 0),
            }
            if paper.get("pdf_path"):
                entry["pdf_path"] = paper["pdf_path"]
            pairs.append(entry)

    return pairs


def generate_qa_for_paper(
    client,
    model: str,
    extra_params: dict,
    task_config: dict,
    paper: dict,
    token_tracker: TokenTracker | None = None,
    max_tokens: int = 4096,
) -> list[dict]:
    """Generate Q&A pairs for a single paper using the LLM.

    Args:
        client: OpenAI-compatible client.
        model: Model name.
        extra_params: Provider-specific params.
        task_config: Task YAML config.
        paper: Paper metadata dict with full_text or abstract.
        token_tracker: Optional tracker to accumulate usage.

    Returns:
        List of Q&A dicts.
    """
    gen_config = task_config["generation"]
    system_prompt = gen_config["system_prompt"]
    user_template = gen_config["user_prompt_template"]
    temperature = gen_config.get("temperature", 0.3)
    max_retries = gen_config.get("max_retries", 2)

    content = paper.get("full_text") or paper.get("abstract") or ""
    if not content:
        return []

    # Truncate very long texts to avoid token limits
    if len(content) > 30000:
        content = content[:30000] + "\n\n[Content truncated for length]"

    authors_str = ", ".join(paper.get("authors", [])[:5])
    user_msg = user_template.format(
        title=paper["title"],
        authors=authors_str,
        content=content,
    )

    api_params = {k: v for k, v in extra_params.items() if not k.startswith("_")}

    for attempt in range(1, max_retries + 1):
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

            if token_tracker and hasattr(response, "usage"):
                token_tracker.update(response.usage)

            result = response.choices[0].message.content
            pairs = _parse_qa_pairs(result, paper)

            if pairs:
                return pairs

            if attempt < max_retries:
                print(f"    Retry {attempt}: no valid Q&A pairs parsed")
                time.sleep(1)

        except Exception as e:
            print(f"    Attempt {attempt} error: {e}")
            if attempt < max_retries:
                time.sleep(2)

    return []


# ── Pipeline orchestrator ──────────────────────────────────────────────

def run_scrape(
    topic: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    task_name: str,
    max_papers: int,
    top_n: int,
    output_path: str,
    collect_only: bool = False,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """Orchestrate the full scrape pipeline.

    1. Search for papers
    2. Rank by citation count
    3. Fetch full text for top N
    4. Generate Q&A pairs via LLM
    5. Write JSONL + citations.json

    Args:
        topic: Research topic to search.
        provider: LLM provider name (required unless collect_only).
        model: Model override.
        api_key: API key override.
        task_name: Task config name for generation prompts.
        max_papers: Max papers to search for.
        top_n: Fetch full text for top N papers.
        output_path: Output file path.
        collect_only: If True, just save paper metadata.
    """
    # ── Step 1: Search ──
    papers = search_papers(topic, max_results=max_papers)

    if not papers:
        raise RuntimeError(f"No papers found for topic '{topic}'. APIs may be rate-limited — check arXiv and Semantic Scholar access.")

    # ── Collect-only mode ──
    if collect_only:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Strip full_text from output (it's None anyway at this point)
        save_data = []
        for p in papers:
            entry = {k: v for k, v in p.items() if k != "full_text"}
            save_data.append(entry)
        with open(output, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nCollected {len(papers)} papers → {output_path}")
        return

    # ── Step 2: Fetch full text for top N ──
    print(f"\nFetching full text for top {top_n} papers...")
    full_text_count = 0
    for i, paper in enumerate(papers[:top_n]):
        text = fetch_full_text(paper)
        if text:
            paper["full_text"] = text
            full_text_count += 1
            print(f"  [{i+1}/{top_n}] Full text: {paper['title'][:50]}... ({len(text):,} chars)")
        else:
            print(f"  [{i+1}/{top_n}] Abstract only: {paper['title'][:50]}...")

    print(f"\nFull text obtained for {full_text_count}/{top_n} papers")
    abstract_count = sum(1 for p in papers if p.get("abstract") and not p.get("full_text"))
    print(f"Abstract-only papers: {abstract_count}")

    # Filter out papers with no content at all
    papers_with_content = [
        p for p in papers if p.get("full_text") or p.get("abstract")
    ]
    print(f"Papers with usable content: {len(papers_with_content)}")

    if not papers_with_content:
        raise RuntimeError("No papers with extractable content (no abstracts or full text available).")

    # ── Step 3: Setup LLM client ──
    from .providers import get_client

    # Load task config
    task_config_path = Path(__file__).parent.parent / "configs" / "tasks" / f"{task_name}.yaml"
    if not task_config_path.exists():
        from pathlib import Path as P
        available = ", ".join(
            sorted(p.stem for p in (P(__file__).parent.parent / "configs" / "tasks").glob("*.yaml"))
        )
        print(f"Unknown task: '{task_name}'. Available: {available}")
        return

    import yaml
    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)

    # Apply user overrides from web UI
    if system_prompt:
        task_config["generation"]["system_prompt"] = system_prompt
    if temperature is not None:
        task_config["generation"]["temperature"] = temperature

    effective_max_tokens = max_tokens if max_tokens is not None else 4096

    client, model_name, extra_params = get_client(provider, model=model, api_key=api_key)
    rate_delay = extra_params.get("_rate_limit_delay", 0)

    print(f"\nProvider: {provider} | Model: {model_name}")
    print(f"Generating Q&A from {len(papers_with_content)} papers...")
    print("=" * 60)

    # ── Step 4: Generate Q&A ──
    token_tracker = TokenTracker()
    gpu_tracker = GPUTracker()
    gpu_tracker.start()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    citations = []

    for i, paper in enumerate(papers_with_content):
        title_short = paper["title"][:60]
        content_type = "full text" if paper.get("full_text") else "abstract"
        print(f"\n[{i+1}/{len(papers_with_content)}] {title_short}... ({content_type})")

        pairs = generate_qa_for_paper(
            client, model_name, extra_params, task_config, paper, token_tracker,
            max_tokens=effective_max_tokens,
        )

        if pairs:
            all_pairs.extend(pairs)
            print(f"  → {len(pairs)} Q&A pairs generated")
        else:
            print(f"  → No Q&A pairs generated")

        citations.append({
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "authors": paper.get("authors", []),
            "year": paper.get("year"),
            "doi": paper.get("doi"),
            "url": paper.get("url", ""),
            "source": paper.get("source", ""),
            "qa_pairs_generated": len(pairs),
        })

        if rate_delay > 0:
            time.sleep(rate_delay)

    gpu_tracker.stop()

    # ── Step 5: Write output ──
    with open(output_file, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    citations_path = output_file.parent / "citations.json"
    with open(citations_path, "w") as f:
        json.dump(citations, f, indent=2)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SCRAPE COMPLETE")
    print("=" * 60)
    print(f"Papers searched:     {len(papers)}")
    print(f"Papers with content: {len(papers_with_content)}")
    print(f"Full text fetched:   {full_text_count}")
    print(f"Q&A pairs generated: {len(all_pairs)}")
    print(f"Output:              {output_path}")
    print(f"Citations:           {citations_path}")

    token_tracker.report()
    gpu_tracker.report()
