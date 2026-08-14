from __future__ import annotations
import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class IdentityOut(BaseModel):
    id: uuid.UUID
    name: str
    avatar_path: str | None = None
    notes: str = ""
    embedding_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


class IdentityListOut(BaseModel):
    items: list[IdentityOut]
    total: int


class IdentityUpdateIn(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=128)
    notes: str | None = None


class FaceSearchOut(BaseModel):
    identity_id: str | None = None
    name: str | None = None
    similarity: float
