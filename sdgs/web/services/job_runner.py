"""Dataset execution engine using ThreadPoolExecutor with stdout capture."""
import io
import sys
import queue
import threading
import datetime
import re
import traceback
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.orm import Session

from ..db.database import SessionLocal
from ..db.models import Dataset

_executor = ThreadPoolExecutor(max_workers=2)
_job_queues: dict[int, queue.Queue] = {}
_job_futures: dict[int, object] = {}
_lock = threading.Lock()


class StdoutCapture(io.TextIOBase):
    """Captures stdout writes and pushes lines into a queue."""

    def __init__(self, dataset_id: int, q: queue.Queue):
        self.dataset_id = dataset_id
        self.q = q
        self._buffer = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.q.put({"type": "log", "data": line})
        return len(s)

    def flush(self):
        if self._buffer:
            self.q.put({"type": "log", "data": self._buffer})
            self._buffer = ""


def get_job_queue(dataset_id: int) -> queue.Queue | None:
    with _lock:
        return _job_queues.get(dataset_id)


def submit_job(ds_id: int, run_fn, **kwargs):
    """Submit a dataset pipeline for background execution."""
    q = queue.Queue()
    with _lock:
        _job_queues[ds_id] = q

    future = _executor.submit(_run_job, ds_id, q, run_fn, kwargs)
    with _lock:
        _job_futures[ds_id] = future
    return future


def cancel_job(dataset_id: int) -> bool:
    """Attempt to cancel a running dataset pipeline."""
    with _lock:
        future = _job_futures.get(dataset_id)
    if future and future.cancel():
        db = SessionLocal()
        try:
            ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if ds:
                ds.status = "cancelled"
                ds.completed_at = datetime.datetime.utcnow()
                db.commit()
        finally:
            db.close()
        q = get_job_queue(dataset_id)
        if q:
            q.put({"type": "status", "data": "cancelled"})
            q.put(None)  # sentinel
        return True
    return False


def _run_job(dataset_id: int, q: queue.Queue, run_fn, kwargs: dict):
    """Execute the pipeline function with stdout capture."""
    db = SessionLocal()
    old_stdout = sys.stdout
    capture = StdoutCapture(dataset_id, q)

    try:
        # Mark as running
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not ds:
            return
        ds.status = "running"
        ds.started_at = datetime.datetime.utcnow()
        db.commit()
        q.put({"type": "status", "data": "running"})

        # Redirect stdout
        sys.stdout = capture
        result = run_fn(**kwargs)
        capture.flush()
        sys.stdout = old_stdout

        # Collect all captured output for stats parsing
        all_lines = []
        while not q.empty():
            item = q.get_nowait()
            if item and item.get("type") == "log":
                all_lines.append(item["data"])

        # Re-queue the lines so SSE still gets them
        for line in all_lines:
            q.put({"type": "log", "data": line})

        # Parse stats from stdout
        stdout_text = "\n".join(all_lines)
        stats = _parse_stats(stdout_text)

        # Now parse the output files and store papers + QA pairs
        from .dataset_service import parse_dataset_results
        from ..db.models import Paper, QAPair

        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if result and isinstance(result, dict):
            output_path = result.get("output_path", "")
            filtered_path = result.get("filtered_path", output_path)

            parsed = parse_dataset_results(output_path, filtered_path)

            ds.output_path = filtered_path or output_path
            ds.actual_size = parsed["actual_size"]
            ds.valid_count = parsed["valid_count"]
            ds.invalid_count = parsed["invalid_count"]
            ds.healed_count = parsed["healed_count"]

            # Store papers
            for p_data in parsed["papers"]:
                paper = Paper(
                    paper_id=p_data.get("paper_id"),
                    title=p_data.get("title", "Unknown"),
                    authors=p_data.get("authors", []),
                    abstract=p_data.get("abstract", ""),
                    year=p_data.get("year"),
                    url=p_data.get("url", ""),
                    citation_count=p_data.get("citation_count", 0),
                    user_id=ds.user_id,
                    dataset_id=ds.id,
                )
                db.add(paper)
            db.flush()

            # Build paper lookup for FK
            paper_lookup = {}
            for paper in db.query(Paper).filter(Paper.dataset_id == ds.id).all():
                if paper.paper_id:
                    paper_lookup[paper.paper_id] = paper.id

            # Store QA pairs
            for qa_data in parsed["qa_pairs"]:
                source_pid = qa_data.get("source_paper_id", "")
                qa = QAPair(
                    instruction=qa_data["instruction"],
                    output=qa_data["output"],
                    is_valid=qa_data.get("is_valid", True),
                    was_healed=qa_data.get("was_healed", False),
                    source_paper_id=source_pid,
                    source_title=qa_data.get("source_title", ""),
                    think_text=qa_data.get("think_text", ""),
                    answer_text=qa_data.get("answer_text", ""),
                    user_id=ds.user_id,
                    paper_id=paper_lookup.get(source_pid),
                    dataset_id=ds.id,
                )
                db.add(qa)

            # Update paper QA counts
            for paper in db.query(Paper).filter(Paper.dataset_id == ds.id).all():
                paper.qa_pair_count = db.query(QAPair).filter(
                    QAPair.paper_id == paper.id
                ).count()

        # Mark completed
        ds.status = "completed"
        ds.completed_at = datetime.datetime.utcnow()
        ds.prompt_tokens = stats.get("prompt_tokens", 0)
        ds.completion_tokens = stats.get("completion_tokens", 0)
        ds.total_tokens = stats.get("total_tokens", 0)
        ds.gpu_kwh = stats.get("gpu_kwh", 0.0)
        ds.duration_seconds = (
            (ds.completed_at - ds.started_at).total_seconds()
            if ds.started_at else 0.0
        )
        db.commit()

        q.put({"type": "status", "data": "completed"})

    except Exception as e:
        sys.stdout = old_stdout
        capture.flush()
        tb = traceback.format_exc()

        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if ds:
            ds.status = "failed"
            ds.completed_at = datetime.datetime.utcnow()
            ds.error_message = str(e)
            if ds.started_at:
                ds.duration_seconds = (ds.completed_at - ds.started_at).total_seconds()
            db.commit()

        q.put({"type": "error", "data": str(e)})
        q.put({"type": "log", "data": tb})
        q.put({"type": "status", "data": "failed"})

    finally:
        sys.stdout = old_stdout
        q.put(None)  # sentinel to end SSE stream
        db.close()

        # Cleanup after a delay
        def _cleanup():
            import time
            time.sleep(60)
            with _lock:
                _job_queues.pop(dataset_id, None)
                _job_futures.pop(dataset_id, None)

        threading.Thread(target=_cleanup, daemon=True).start()


def _parse_stats(text: str) -> dict:
    """Parse token and GPU stats from captured stdout."""
    stats: dict = {}

    m = re.search(r"Prompt tokens:\s+([\d,]+)", text)
    if m:
        stats["prompt_tokens"] = int(m.group(1).replace(",", ""))

    m = re.search(r"Completion tokens:\s+([\d,]+)", text)
    if m:
        stats["completion_tokens"] = int(m.group(1).replace(",", ""))

    m = re.search(r"Total tokens:\s+([\d,]+)", text)
    if m:
        stats["total_tokens"] = int(m.group(1).replace(",", ""))

    m = re.search(r"Total energy:\s+([\d.]+)\s*kWh", text)
    if m:
        stats["gpu_kwh"] = float(m.group(1))

    return stats


def shutdown_runner():
    """Shut down the thread pool executor."""
    _executor.shutdown(wait=False)
