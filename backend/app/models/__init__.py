"""backend.app.models — SQLAlchemy ORM(全部模型统一在此注册)。"""

from __future__ import annotations
from .base import Base
from .camera import Camera
from .identity import Identity, IdentityEmbedding
from .event import Event, EventType, RecognitionLog

__all__ = [
    "Base",
    "Camera",
    "Identity",
    "IdentityEmbedding",
    "Event",
    "EventType",
    "RecognitionLog",
]
