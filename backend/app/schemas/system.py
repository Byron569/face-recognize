from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel


class SystemStatusOut(BaseModel):
    cpu_percent: float
    memory_percent: float
    gpu_name: str | None = None
    gpu_utilization: float | None = None
    gpu_memory_percent: float | None = None
    camera_count: int
    active_camera_count: int
    engine_count: int = 0
    gallery_size: int = 0


# ── 阶段10 M2:跌倒检测只读健康 ─────────────────────────────

class FallRuntimeHealthOut(BaseModel):
    """GET /system/fall-runtime 响应(DISABLED 时 worker/gpu/model/delivery 为 None)。"""

    schema_version: int = 1
    enabled: bool
    mode: Optional[str] = None
    runtime_key: Optional[str] = None
    state: str
    error: Optional[str] = None
    worker: Optional[dict] = None
    gpu: Optional[dict] = None
    model: Optional[dict] = None
    delivery: Optional[dict] = None
    cameras: list[dict] = []
    _ignored: Any = None
