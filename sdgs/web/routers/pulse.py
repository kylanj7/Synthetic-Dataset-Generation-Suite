"""WebSocket endpoint for live training/evaluation/correction metrics."""
import asyncio
import json
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from ..services.broadcast import BroadcastMessage, get_broadcast_queue

router = APIRouter()


class PulseManager:
    """Registry of active WebSocket connections keyed by 'type:id'."""

    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, key: str, ws: WebSocket) -> None:
        await ws.accept()
        if key not in self._connections:
            self._connections[key] = set()
        self._connections[key].add(ws)

    def disconnect(self, key: str, ws: WebSocket) -> None:
        conns = self._connections.get(key)
        if conns:
            conns.discard(ws)
            if not conns:
                del self._connections[key]

    async def broadcast(self, key: str, data: dict) -> None:
        conns = self._connections.get(key)
        if not conns:
            return
        payload = json.dumps(data)
        dead: list[WebSocket] = []
        for ws in conns:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            conns.discard(ws)


pulse_manager = PulseManager()


@router.websocket("/ws/pulse/{run_type}/{run_id}")
async def pulse_websocket(ws: WebSocket, run_type: str, run_id: int):
    key = f"{run_type}:{run_id}"
    await pulse_manager.connect(key, ws)
    try:
        while True:
            # Keep connection alive; client can send pings or we just wait
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive ping
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        pulse_manager.disconnect(key, ws)


async def broadcast_consumer() -> None:
    """Async task that polls the broadcast queue and forwards to WebSocket clients."""
    bq = get_broadcast_queue()
    while True:
        try:
            msg: BroadcastMessage = bq.get_nowait()
            key = f"{msg.broadcast_type.value}:{msg.entity_id}"
            await pulse_manager.broadcast(key, msg.message)
        except Exception:
            # Queue empty or other transient error — throttle
            await asyncio.sleep(0.05)
