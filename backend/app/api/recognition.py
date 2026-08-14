"""识别记录查询。"""

from __future__ import annotations
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_db
from ..schemas.event import RecognitionLogListOut, RecognitionLogOut
from ..services.event_service import EventService

router = APIRouter()


@router.get("/recognition-logs", response_model=RecognitionLogListOut)
async def list_recognition_logs(
    page: int = 1,
    page_size: int = 20,
    camera_id: str | None = None,
    identity_id: str | None = Query(None),
    start: datetime | None = None,
    end: datetime | None = None,
    db: AsyncSession = Depends(get_db),
):
    items, total = await EventService(db).list_recognition_logs(
        page, page_size, camera_id, identity_id, start, end
    )
    return RecognitionLogListOut(items=[RecognitionLogOut(**it) for it in items], total=total)


@router.get("/recognition-logs/{log_id}", response_model=RecognitionLogOut)
async def get_recognition_log(log_id: int, db: AsyncSession = Depends(get_db)):
    item = await EventService(db).get_recognition_log(log_id)
    if not item:
        raise HTTPException(404, "Recognition log not found")
    return RecognitionLogOut(**item)
