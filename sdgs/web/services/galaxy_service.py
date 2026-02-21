"""Galaxy graph construction service — builds nodes/links from papers and Q&A pairs."""
import math
import re
from collections import defaultdict

from sqlalchemy.orm import Session

from ..db.models import Paper, QAPair

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
    """Build the full galaxy graph from database contents for a specific user."""
    papers = db.query(Paper).filter(Paper.user_id == user_id).all()
    qa_pairs = db.query(QAPair).filter(QAPair.user_id == user_id).all()

    if not papers:
        return {"nodes": [], "links": [], "clusters": []}

    # Extract keywords and build paper keyword map
    paper_keywords: dict[int, list[str]] = {}
    for p in papers:
        text = f"{p.title or ''} {p.abstract or ''}"
        paper_keywords[p.id] = extract_keywords(text)
        p._keywords = paper_keywords[p.id]

    # Cluster papers by keyword overlap (greedy grouping)
    clusters: list[list[Paper]] = []
    assigned = set()

    for p in sorted(papers, key=lambda x: x.citation_count or 0, reverse=True):
        if p.id in assigned:
            continue
        cluster = [p]
        assigned.add(p.id)
        p_kw = set(paper_keywords[p.id])

        for other in papers:
            if other.id in assigned:
                continue
            other_kw = set(paper_keywords[other.id])
            overlap = len(p_kw & other_kw)
            if overlap >= 2:
                cluster.append(other)
                assigned.add(other.id)

        clusters.append(cluster)

    # Assign remaining unassigned papers to cluster 0
    for p in papers:
        if p.id not in assigned:
            if clusters:
                clusters[0].append(p)
            else:
                clusters.append([p])
            assigned.add(p.id)

    # Build cluster info and paper->cluster mapping
    paper_cluster: dict[int, int] = {}
    cluster_infos = []
    for i, cluster_papers in enumerate(clusters):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        # Label from most common keyword in cluster
        all_kw = []
        for p in cluster_papers:
            all_kw.extend(paper_keywords[p.id])
        freq = defaultdict(int)
        for w in all_kw:
            freq[w] += 1
        label = max(freq, key=freq.get) if freq else f"cluster-{i}"

        cluster_infos.append({
            "id": i,
            "label": label,
            "color": color,
            "paper_count": len(cluster_papers),
        })
        for p in cluster_papers:
            paper_cluster[p.id] = i

    # Build paper->qa lookup
    paper_qa_map: dict[int, list[QAPair]] = defaultdict(list)
    for qa in qa_pairs:
        if qa.paper_id:
            paper_qa_map[qa.paper_id].append(qa)

    # Build nodes
    nodes = []
    for p in papers:
        cid = paper_cluster.get(p.id, 0)
        color = cluster_infos[cid]["color"] if cid < len(cluster_infos) else "#7eb8ff"
        size = 4 + math.log2((p.citation_count or 0) + 1) * 3
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
        })

    for qa in qa_pairs:
        cid = paper_cluster.get(qa.paper_id, 0) if qa.paper_id else 0
        color = cluster_infos[cid]["color"] if cid < len(cluster_infos) else "#6ee7d8"
        nodes.append({
            "id": f"qa-{qa.id}",
            "type": "qa",
            "label": (qa.instruction or "")[:50],
            "size": 2,
            "color": color,
            "cluster": cid,
            "instruction": qa.instruction,
            "output_preview": (qa.output or "")[:200],
        })

    # Build links
    links = []

    # Paper -> QA links
    for qa in qa_pairs:
        if qa.paper_id:
            links.append({
                "source": f"paper-{qa.paper_id}",
                "target": f"qa-{qa.id}",
                "weight": 0.3,
                "type": "paper_qa",
            })

    # Shared keyword links between papers
    paper_list = list(papers)
    for i in range(len(paper_list)):
        for j in range(i + 1, len(paper_list)):
            p1 = paper_list[i]
            p2 = paper_list[j]
            kw1 = set(paper_keywords[p1.id])
            kw2 = set(paper_keywords[p2.id])
            overlap = len(kw1 & kw2)
            if overlap >= 2:
                links.append({
                    "source": f"paper-{p1.id}",
                    "target": f"paper-{p2.id}",
                    "weight": overlap / 10,
                    "type": "keyword",
                })

    # Co-topic links (same dataset)
    dataset_papers: dict[int, list[Paper]] = defaultdict(list)
    for p in papers:
        if p.dataset_id:
            dataset_papers[p.dataset_id].append(p)

    for ds_id, dpapers in dataset_papers.items():
        for i in range(len(dpapers)):
            for j in range(i + 1, len(dpapers)):
                # Only add if not already linked by keywords
                key = (f"paper-{dpapers[i].id}", f"paper-{dpapers[j].id}")
                already = any(
                    (l["source"], l["target"]) == key or (l["target"], l["source"]) == key
                    for l in links if l["type"] == "keyword"
                )
                if not already:
                    links.append({
                        "source": f"paper-{dpapers[i].id}",
                        "target": f"paper-{dpapers[j].id}",
                        "weight": 0.1,
                        "type": "co_topic",
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
