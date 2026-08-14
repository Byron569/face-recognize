from __future__ import annotations
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
