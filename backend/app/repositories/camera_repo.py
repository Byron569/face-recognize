"""摄像头表数据访问。"""

from __future__ import annotations
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.camera import Camera


class CameraRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_all(self) -> list[Camera]:
        q = select(Camera).order_by(Camera.id)
        return list((await self.db.execute(q)).scalars().all())

    async def get(self, camera_id: str) -> Optional[Camera]:
        return await self.db.get(Camera, camera_id)

    async def create(self, camera: Camera) -> Camera:
        self.db.add(camera)
        await self.db.commit()
        await self.db.refresh(camera)
        return camera

    async def update(self, camera: Camera, data: dict) -> Camera:
        for key, value in data.items():
            if value is not None and hasattr(camera, key):
                setattr(camera, key, value)
        await self.db.commit()
        await self.db.refresh(camera)
        return camera

    async def delete(self, camera_id: str) -> bool:
        camera = await self.db.get(Camera, camera_id)
        if camera is None:
            return False
        await self.db.delete(camera)
        await self.db.commit()
        return True
