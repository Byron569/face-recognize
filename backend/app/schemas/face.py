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

class RegistrationFrameAnalysisOut(BaseModel):
    """单帧质量分析结果(摄像头/视频注册)。"""

    frame_id: str
    timestamp_ms: int
    accepted: bool
    reason: str | None = None
    pose: str | None = None
    bbox: list[float] | None = None
    det_score: float | None = None
    yaw_ratio: float | None = None
    pitch_ratio: float | None = None
    blur_score: float | None = None
    quality_score: float | None = None


class RegistrationAnalyzeOut(BaseModel):
    """批量分析结果(不写库)。"""

    sampled_count: int
    accepted_count: int
    recommended_frame_ids: list[str]
    frames: list[RegistrationFrameAnalysisOut]


class RegistrationCommitOut(BaseModel):
    """提交注册结果(create / append)。"""

    mode: str
    identity_id: uuid.UUID
    embedding_count_added: int
