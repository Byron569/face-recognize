"""
services.camera_service — 摄像头配置业务。
"""


from __future__ import annotations
from typing import Optional

from ..models.camera import Camera
from ..repositories.camera_repo import CameraRepository


class CameraService:
    def __init__(self, db):
        self._repo = CameraRepository(db)

    async def list(self) -> list[Camera]:
        return await self._repo.list_all()

    async def get(self, camera_id: str) -> Optional[Camera]:
        return await self._repo.get(camera_id)

    async def create(self, data: dict) -> Camera:
        return await self._repo.create(Camera(**data))

    async def update(self, camera_id: str, data: dict) -> Optional[Camera]:
        camera = await self._repo.get(camera_id)
        if camera is None:
            return None
        return await self._repo.update(camera, data)

    async def delete(self, camera_id: str) -> bool:
        return await self._repo.delete(camera_id)
