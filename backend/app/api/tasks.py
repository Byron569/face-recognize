"""任务注册表查询(展示可用/启用的可插拔任务)。"""

from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import load_profile_config
from ..deps import get_db
from ..schemas.task import TaskInfoOut, TaskListOut

router = APIRouter()


@router.get("/tasks", response_model=TaskListOut)
async def list_tasks(profile: str = "desktop", db: AsyncSession = Depends(get_db)):
    """返回配置中登记的任务(含预留扩展任务)。"""
    tasks_cfg = load_profile_config(profile).get("tasks", {})
    items = [
        TaskInfoOut(
            name=name,
            enabled=bool(cfg.get("enabled", False)) if isinstance(cfg, dict) else False,
            class_path=cfg.get("class_path") if isinstance(cfg, dict) else None,
            loaded=False,
        )
        for name, cfg in tasks_cfg.items()
    ]
    return TaskListOut(items=items)
