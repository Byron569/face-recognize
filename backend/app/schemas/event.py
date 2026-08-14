from __future__ import annotations
from datetime import datetime

from pydantic import BaseModel


class EventOut(BaseModel):
    id: int
    event_type: str
    camera_id: str
    track_id: int | None = None
    identity_id: str | None = None
    identity_name: str | None = None
    confidence: float = 0
    payload: dict = {}
    snapshot_path: str | None = None
    acknowledged: bool = False
    acknowledged_at: datetime | None = None
    created_at: datetime


class EventListOut(BaseModel):
    items: list[EventOut]
    total: int


class RecognitionLogOut(BaseModel):
    id: int
    camera_id: str
    identity_id: str | None = None
    track_id: int
    similarity: float
    latency_ms: float | None = None
    created_at: datetime


class RecognitionLogListOut(BaseModel):
    items: list[RecognitionLogOut]
    total: int
