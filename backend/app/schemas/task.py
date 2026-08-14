from __future__ import annotations
from pydantic import BaseModel


class TaskInfoOut(BaseModel):
    name: str
    enabled: bool
    class_path: str | None = None
    loaded: bool = False


class TaskListOut(BaseModel):
    items: list[TaskInfoOut]
