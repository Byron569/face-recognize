"""摄像头管理:CRUD + 启停 + 抓拍 + profile 切换。"""

from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import build_camera_config
from ..deps import get_db
from ..schemas.camera import CameraCreate, CameraOut, CameraResolutionUpdate, CameraUpdate
from ..services.camera_service import CameraService
from ..services.pipeline_manager import get_pipeline_manager

router = APIRouter()

VALID_PROFILES = ("desktop", "balanced", "edge_minimal")


def _to_out(camera, mgr) -> CameraOut:
    status = mgr.get_status(camera.id)
    return CameraOut(
        id=camera.id,
        name=camera.name,
        source=camera.source,
        width=camera.width,
        height=camera.height,
        profile=camera.profile,
        enabled=camera.enabled,
        config=camera.config or {},
        status=("running" if status and status.get("alive") else "stopped"),
        metrics=status,
        created_at=camera.created_at,
        updated_at=camera.updated_at,
    )


@router.get("/cameras", response_model=list[CameraOut])
async def list_cameras(db: AsyncSession = Depends(get_db)):
    mgr = get_pipeline_manager()
    cameras = await CameraService(db).list()
    return [_to_out(c, mgr) for c in cameras]


@router.get("/cameras/{camera_id}", response_model=CameraOut)
async def get_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    camera = await CameraService(db).get(camera_id)
    if not camera:
        raise HTTPException(404, "Camera not found")
    return _to_out(camera, get_pipeline_manager())


@router.post("/cameras", response_model=CameraOut, status_code=201)
async def create_camera(body: CameraCreate, db: AsyncSession = Depends(get_db)):
    svc = CameraService(db)
    if await svc.get(body.id):
        raise HTTPException(409, "Camera already exists")
    if body.profile not in VALID_PROFILES:
        raise HTTPException(400, f"Invalid profile: {body.profile}")
    camera = await svc.create(body.model_dump())
    return _to_out(camera, get_pipeline_manager())


@router.put("/cameras/{camera_id}", response_model=CameraOut)
async def update_camera(camera_id: str, body: CameraUpdate, db: AsyncSession = Depends(get_db)):
    svc = CameraService(db)
    camera = await svc.update(camera_id, body.model_dump(exclude_none=True))
    if not camera:
        raise HTTPException(404, "Camera not found")
    return _to_out(camera, get_pipeline_manager())


@router.delete("/cameras/{camera_id}")
async def delete_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    mgr = get_pipeline_manager()
    if camera_id in mgr.list_cameras():
        await mgr.stop_camera(camera_id)
    ok = await CameraService(db).delete(camera_id)
    if not ok:
        raise HTTPException(404, "Camera not found")
    return {"deleted": True}


@router.post("/cameras/{camera_id}/start")
async def start_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    """启动摄像头推理流水线(配置 = profile 级联 + 摄像头个性化 config)。"""
    camera = await CameraService(db).get(camera_id)
    if not camera:
        raise HTTPException(404, "Camera not found")

    # 最终配置:default → profile → 摄像头个性化(不写死任何参数)
    config = build_camera_config(camera.profile, camera.config)

    mgr = get_pipeline_manager()
    if camera_id in mgr.list_cameras():
        return {"started": True, "status": "already_running"}

    try:
        ok = await mgr.start_camera(camera_id, camera.source, config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Pipeline start failed: {exc}") from exc

    await CameraService(db).update(camera_id, {"enabled": True})
    return {"started": ok, "status": "running" if ok else "failed"}


@router.post("/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    mgr = get_pipeline_manager()
    stopped = await mgr.stop_camera(camera_id)
    await CameraService(db).update(camera_id, {"enabled": False})
    return {"stopped": stopped, "status": "stopped"}


@router.post("/cameras/{camera_id}/snapshot")
async def snapshot(camera_id: str):
    """抓拍当前帧,返回 JPEG。"""
    jpeg = get_pipeline_manager().snapshot_jpeg(camera_id)
    if jpeg is None:
        raise HTTPException(404, "No frame available — camera not running")
    return Response(content=jpeg, media_type="image/jpeg")


@router.put("/cameras/{camera_id}/profile")
async def switch_profile(camera_id: str, body: dict, db: AsyncSession = Depends(get_db)):
    """运行时切换部署档位:停旧流水线 → 更新配置 → 按新档位重启。"""
    profile: Optional[str] = body.get("profile")
    if profile not in VALID_PROFILES:
        raise HTTPException(400, f"Invalid profile: {profile}")

    svc = CameraService(db)
    camera = await svc.get(camera_id)
    if not camera:
        raise HTTPException(404, "Camera not found")

    mgr = get_pipeline_manager()
    was_running = camera_id in mgr.list_cameras()
    if was_running:
        await mgr.stop_camera(camera_id)

    camera = await svc.update(camera_id, {"profile": profile})

    if was_running:
        config = build_camera_config(camera.profile, camera.config)
        await mgr.start_camera(camera_id, camera.source, config)
        await svc.update(camera_id, {"enabled": True})
    return {"profile": profile, "restarted": was_running, "updated": True}


@router.put("/cameras/{camera_id}/resolution")
async def update_resolution(
    camera_id: str, body: CameraResolutionUpdate, db: AsyncSession = Depends(get_db)
):
    """设置采集/推理分辨率与推流分辨率(0 = 原生/不缩放)。

    写入 cameras.config JSONB;若流水线正在运行,自动重启使其立即生效。
    """
    svc = CameraService(db)
    camera = await svc.get(camera_id)
    if not camera:
        raise HTTPException(404, "Camera not found")

    cfg = dict(camera.config or {})
    cam_defaults = dict(cfg.get("camera_defaults") or {})
    cam_defaults["width"] = body.capture_width
    cam_defaults["height"] = body.capture_height
    cam_defaults.setdefault("max_width", 0)  # 保持已有值,缺省 0=不缩放
    cfg["camera_defaults"] = cam_defaults
    stream = dict(cfg.get("stream") or {})
    stream["max_height"] = body.stream_max_height
    cfg["stream"] = stream

    camera = await svc.update(camera_id, {"config": cfg})

    mgr = get_pipeline_manager()
    was_running = camera_id in mgr.list_cameras()
    if was_running:
        await mgr.stop_camera(camera_id)
        config = build_camera_config(camera.profile, camera.config)
        try:
            ok = await mgr.start_camera(camera_id, camera.source, config)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Pipeline restart failed: {exc}") from exc
        if not ok:
            raise HTTPException(500, "Pipeline restart failed — check camera source")
        await svc.update(camera_id, {"enabled": True})

    capture = (
        f"{body.capture_width}x{body.capture_height}"
        if body.capture_width and body.capture_height
        else "native"
    )
    return {
        "updated": True,
        "restarted": was_running,
        "capture": capture,
        "stream_max_height": body.stream_max_height,
    }
