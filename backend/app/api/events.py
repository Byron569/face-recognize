"""事件:列表 / 详情 / 确认 / 类型枚举。"""

from __future__ import annotations
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_db
from ..models.event import EventType
from ..schemas.event import EventListOut, EventOut
from ..services.event_service import EventService

router = APIRouter()


@router.get("/events", response_model=EventListOut)
async def list_events(
    page: int = 1,
    page_size: int = 20,
    event_type: str | None = Query(None),
    camera_id: str | None = None,
    acknowledged: bool | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    db: AsyncSession = Depends(get_db),
):
    items, total = await EventService(db).list_events(
        page, page_size, event_type, camera_id, acknowledged, start, end
    )
    return EventListOut(items=[EventOut(**it) for it in items], total=total)


@router.get("/events/types")
async def event_types():
    """全部事件类型(含为扩展任务预留的类型)。"""
    return {"types": [e.value for e in EventType]}


@router.get("/events/{event_id}", response_model=EventOut)
async def get_event(event_id: int, db: AsyncSession = Depends(get_db)):
    item = await EventService(db).get_event(event_id)
    if not item:
        raise HTTPException(404, "Event not found")
    return EventOut(**item)


@router.post("/events/{event_id}/acknowledge")
async def acknowledge_event(event_id: int, db: AsyncSession = Depends(get_db)):
    ok = await EventService(db).acknowledge(event_id)
    if not ok:
        raise HTTPException(404, "Event not found")
    return {"acknowledged": True}


@router.delete("/events")
async def delete_events(
    ids: str | None = Query(None, description="逗号分隔的事件 id,如 ids=1,2,3"),
    all: bool = Query(False, description="删除当前筛选条件下的全部记录(与 ids 二选一)"),
    event_type: str | None = Query(None),
    camera_id: str | None = Query(None),
    acknowledged: bool | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """批量删除事件记录。

    - 指定 ids:删除指定 id(逗号分隔)
    - all=true:删除 event_type/camera_id/acknowledged 筛选条件下的全部
    """
    svc = EventService(db)
    if all:
        deleted = await svc.delete_events_filtered(event_type, camera_id, acknowledged)
    elif ids:
        id_list = [int(x) for x in ids.split(",") if x.strip()]
        deleted = await svc.delete_events(id_list)
    else:
        raise HTTPException(422, "ids 或 all=true 至少提供一个")
    return {"deleted": deleted}
