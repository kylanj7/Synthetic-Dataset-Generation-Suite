"""Server-Sent Events endpoint for real-time dataset pipeline progress."""
import asyncio
import json

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from ..services.job_runner import get_job_queue, get_job_logs

router = APIRouter()


@router.get("/datasets/{dataset_id}")
async def dataset_events(dataset_id: int, last_id: int = Query(0, ge=0)):
    """SSE stream for a running dataset pipeline — sends log lines, status updates, completion.

    Supports reconnection via `last_id` query param: the client sends the last
    event ID it received, and the server replays any stored logs after that point
    before switching to live queue streaming.
    """

    async def event_generator():
        # Replay stored logs for reconnection (skip already-received events)
        stored = get_job_logs(dataset_id)
        event_id = len(stored)  # next live event starts after stored logs

        for i, item in enumerate(stored):
            if i < last_id:
                continue
            yield f"id: {i}\ndata: {json.dumps(item)}\n\n"

        # Stream live items from queue
        q = get_job_queue(dataset_id)
        if q is None:
            # Job already finished — just send done after replay
            yield f"id: {event_id}\ndata: {json.dumps({'type': 'done', 'data': 'stream_end'})}\n\n"
            return

        while True:
            try:
                item = None
                try:
                    item = q.get_nowait()
                except Exception:
                    await asyncio.sleep(0.1)
                    continue

                if item is None:
                    # Sentinel — stream is done
                    yield f"id: {event_id}\ndata: {json.dumps({'type': 'done', 'data': 'stream_end'})}\n\n"
                    return

                yield f"id: {event_id}\ndata: {json.dumps(item)}\n\n"
                event_id += 1

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
