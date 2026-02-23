"""Thread-safe broadcast queue bridging sync training threads to async WebSocket consumers."""
import queue
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class BroadcastType(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"
    CORRECTION = "correction"


@dataclass
class BroadcastMessage:
    broadcast_type: BroadcastType
    entity_id: int
    message: Dict[str, Any] = field(default_factory=dict)


_broadcast_queue: queue.Queue[BroadcastMessage] = queue.Queue()


def enqueue_broadcast(
    broadcast_type: BroadcastType,
    entity_id: int,
    message: Dict[str, Any],
) -> None:
    """Push a message from any thread into the broadcast queue."""
    _broadcast_queue.put(BroadcastMessage(
        broadcast_type=broadcast_type,
        entity_id=entity_id,
        message=message,
    ))


def get_broadcast_queue() -> queue.Queue[BroadcastMessage]:
    """Return the module-level broadcast queue for async consumption."""
    return _broadcast_queue
