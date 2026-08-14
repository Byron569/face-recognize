from __future__ import annotations
from datetime import datetime

from pydantic import BaseModel, Field


class CameraCreate(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    name: str = ""
    source: str = Field(..., min_length=1, max_length=512)
    width: int = 640
    height: int = 480
    profile: str = "desktop"
    enabled: bool = False
    config: dict = Field(default_factory=dict)


class CameraUpdate(BaseModel):
    name: str | None = None
    source: str | None = None
    width: int | None = None
    height: int | None = None
    profile: str | None = None
    enabled: bool | None = None
    config: dict | None = None


class CameraResolutionUpdate(BaseModel):
    """分辨率设置(0 = 原生/不缩放)。"""

    capture_width: int = Field(0, ge=0, le=7680)
    capture_height: int = Field(0, ge=0, le=7680)
    stream_max_height: int = Field(0, ge=0, le=7680)


class CameraOut(BaseModel):
    id: str
    name: str
    source: str
    width: int
    height: int
    profile: str
    enabled: bool
    config: dict = {}
    status: str = "stopped"
    metrics: dict | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}
