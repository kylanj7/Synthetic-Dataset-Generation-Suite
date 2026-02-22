"""Galaxy graph construction service — builds nodes/links from datasets and papers."""
import math
import re
from collections import defaultdict

from sqlalchemy.orm import Session

from ..db.models import Dataset, Paper, QAPair

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
    "were", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "this", "that", "these", "those", "not", "no", "its", "we", "our",
    "their", "using", "based", "via", "into", "over", "about", "between",
    "through", "during", "before", "after", "above", "below", "up", "down",
    "than", "other", "such", "only", "also", "more", "most", "very",
    "each", "every", "all", "both", "few", "many", "some", "any",
    "new", "first", "two", "one", "which", "when", "how", "what",
}

CLUSTER_COLORS = [
    "#7eb8ff", "#6ee7d8", "#ffd666", "#c084fc", "#ff7eb3",
    "#a3e635", "#fb923c", "#60a5fa", "#f472b6", "#34d399",
]


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """Extract keywords from text via simple stop-word removal."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    filtered = [w for w in words if w not in STOP_WORDS]
    # Count and return top keywords
    freq = defaultdict(int)
    for w in filtered:
        freq[w] += 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:max_keywords]]


def build_galaxy_data(db: Session, user_id: int) -> dict:
    """Build the galaxy graph from datasets and papers (QA pairs excluded for performance)."""
    datasets = (
        db.query(Dataset)
        .filter(Dataset.user_id == user_id, Dataset.status == "completed")
        .all()
    )
    papers = db.query(Paper).filter(Paper.user_id == user_id).all()
    qa_pairs = db.query(QAPair).filter(QAPair.user_id == user_id).all()

    if not datasets and not papers:
        return {"nodes": [], "links": [], "clusters": []}

    # ── Pre-build lookup dicts ──
    papers_by_dataset: dict[int, list] = defaultdict(list)
    for p in papers:
        if p.dataset_id:
            papers_by_dataset[p.dataset_id].append(p)

    # Count QA pairs per paper and per dataset (for sizing, not for nodes)
    qa_count_by_paper: dict[int, int] = defaultdict(int)
    qa_count_by_dataset: dict[int, int] = defaultdict(int)
    for qa in qa_pairs:
        if qa.paper_id:
            qa_count_by_paper[qa.paper_id] += 1
        if qa.dataset_id:
            qa_count_by_dataset[qa.dataset_id] += 1

    # Extract keywords and build paper keyword map
    paper_keywords: dict[int, list[str]] = {}
    for p in papers:
        text = f"{p.title or ''} {p.abstract or ''}"
        paper_keywords[p.id] = extract_keywords(text)

    # Use datasets as clusters (one cluster per dataset)
    cluster_infos = []
    paper_cluster: dict[int, int] = {}
    dataset_cluster: dict[int, int] = {}

    for i, ds in enumerate(datasets):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ds_papers = papers_by_dataset[ds.id]
        cluster_infos.append({
            "id": i,
            "label": ds.topic[:30] if ds.topic else f"Dataset {ds.id}",
            "color": color,
            "paper_count": len(ds_papers),
        })
        dataset_cluster[ds.id] = i
        for p in ds_papers:
            paper_cluster[p.id] = i

    # Assign orphan papers (no dataset) to a misc cluster
    orphan_papers = [p for p in papers if p.id not in paper_cluster]
    if orphan_papers:
        misc_idx = len(cluster_infos)
        cluster_infos.append({
            "id": misc_idx,
            "label": "uncategorized",
            "color": "#888888",
            "paper_count": len(orphan_papers),
        })
        for p in orphan_papers:
            paper_cluster[p.id] = misc_idx

    # Build nodes
    nodes = []

    # Dataset nodes — large central hubs
    for ds in datasets:
        cid = dataset_cluster.get(ds.id, 0)
        color = cluster_infos[cid]["color"] if cid < len(cluster_infos) else "#7eb8ff"
        ds_qa_count = qa_count_by_dataset[ds.id]
        ds_paper_count = len(papers_by_dataset[ds.id])
        size = 12 + math.log2(ds_qa_count + 1) * 3
        nodes.append({
            "id": f"dataset-{ds.id}",
            "type": "dataset",
            "label": ds.topic or ds.name,
            "size": round(size, 1),
            "color": color,
            "cluster": cid,
            "year": None,
            "citation_count": None,
            "abstract": f"{ds_paper_count} papers, {ds_qa_count} Q&A pairs | {ds.provider or 'unknown'}/{ds.model or 'default'}",
            "authors": None,
            "url": None,
        })

    # Paper nodes — sized by QA count + citation count
    for p in papers:
        cid = paper_cluster.get(p.id, 0)
        color = cluster_infos[cid]["color"] if cid < len(cluster_infos) else "#7eb8ff"
        qa_count = qa_count_by_paper[p.id]
        size = 4 + math.log2(qa_count + 1) * 2 + math.log2((p.citation_count or 0) + 1) * 2
        nodes.append({
            "id": f"paper-{p.id}",
            "type": "paper",
            "label": p.title or "Untitled",
            "size": round(size, 1),
            "color": color,
            "cluster": cid,
            "year": p.year,
            "citation_count": p.citation_count or 0,
            "abstract": (p.abstract or "")[:300],
            "authors": p.authors or [],
            "url": p.url,
            "qa_pair_count": qa_count,
        })

    # Build links
    links = []

    # Dataset -> Paper links
    for p in papers:
        if p.dataset_id:
            links.append({
                "source": f"dataset-{p.dataset_id}",
                "target": f"paper-{p.id}",
                "weight": 0.5,
                "type": "dataset_paper",
            })

    # Shared keyword links between papers (cap at 200 papers to avoid O(n²) blowup)
    paper_list = list(papers[:200])
    for i in range(len(paper_list)):
        for j in range(i + 1, len(paper_list)):
            p1 = paper_list[i]
            p2 = paper_list[j]
            kw1 = set(paper_keywords.get(p1.id, []))
            kw2 = set(paper_keywords.get(p2.id, []))
            overlap = len(kw1 & kw2)
            if overlap >= 2:
                links.append({
                    "source": f"paper-{p1.id}",
                    "target": f"paper-{p2.id}",
                    "weight": overlap / 10,
                    "type": "keyword",
                })

    return {
        "nodes": nodes,
        "links": links,
        "clusters": cluster_infos,
    }


def get_paper_detail(db: Session, paper_id: int, user_id: int) -> dict | None:
    """Get paper detail with its Q&A pairs, scoped to user."""
    paper = db.query(Paper).filter(Paper.id == paper_id, Paper.user_id == user_id).first()
    if not paper:
        return None

    qa_pairs = db.query(QAPair).filter(QAPair.paper_id == paper.id).all()

    return {
        "paper_id": paper.paper_id or str(paper.id),
        "title": paper.title,
        "authors": paper.authors or [],
        "abstract": paper.abstract,
        "year": paper.year,
        "citation_count": paper.citation_count or 0,
        "url": paper.url,
        "qa_pairs": [
            {
                "instruction": qa.instruction,
                "output": qa.output,
                "is_valid": qa.is_valid,
                "was_healed": qa.was_healed,
                "think_text": qa.think_text,
                "answer_text": qa.answer_text,
            }
            for qa in qa_pairs
        ],
    }
