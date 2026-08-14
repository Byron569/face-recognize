"""backend.app.schemas — Pydantic 出入参模型。"""

from __future__ import annotations
from .camera import CameraCreate, CameraOut, CameraUpdate
from .event import EventOut, EventListOut, RecognitionLogOut, RecognitionLogListOut
from .face import FaceSearchOut, IdentityListOut, IdentityOut, IdentityUpdateIn
from .system import SystemStatusOut
from .task import TaskInfoOut, TaskListOut

__all__ = [
    "CameraCreate",
    "CameraOut",
    "CameraUpdate",
    "EventOut",
    "EventListOut",
    "RecognitionLogOut",
    "RecognitionLogListOut",
    "FaceSearchOut",
    "IdentityListOut",
    "IdentityOut",
    "IdentityUpdateIn",
    "SystemStatusOut",
    "TaskInfoOut",
    "TaskListOut",
]
