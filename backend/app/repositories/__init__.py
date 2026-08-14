"""backend.app.repositories — 数据访问层(全部 SQL 集中于此,services 不含裸 SQL)。"""

from __future__ import annotations
from .camera_repo import CameraRepository
from .event_repo import EventRepository
from .identity_repo import IdentityRepository

__all__ = ["CameraRepository", "IdentityRepository", "EventRepository"]
