"""Server-Sent Events endpoint for real-time dataset pipeline progress."""
import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..services.job_runner import get_job_queue

router = APIRouter()


@router.get("/datasets/{dataset_id}")
async def dataset_events(dataset_id: int):
    """SSE stream for a running dataset pipeline — sends log lines, status updates, completion."""

    async def event_generator():
        q = get_job_queue(dataset_id)
        if q is None:
            yield f"data: {json.dumps({'type': 'error', 'data': 'Dataset not found or already completed'})}\n\n"
            return

        while True:
            try:
                # Non-blocking check with asyncio sleep
                item = None
                try:
                    item = q.get_nowait()
                except Exception:
                    await asyncio.sleep(0.1)
                    continue

                if item is None:
                    # Sentinel — stream is done
                    yield f"data: {json.dumps({'type': 'done', 'data': 'stream_end'})}\n\n"
                    return

                yield f"data: {json.dumps(item)}\n\n"

            except asyncio.CancelledError:
                return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
