"""
AI Monitor 后端入口(FastAPI)。

启动流程(lifespan):
    1. 初始化 PipelineManager(绑定事件循环);
    2. 启动数据清理定时任务(APScheduler);
    3. 停机时优雅停止全部摄像头流水线并释放推理引擎。

API 文档: 运行时访问 /docs(OpenAPI)与 /redoc。
"""

from __future__ import annotations
import logging
import os
import sys
from contextlib import asynccontextmanager

# 项目根路径加入 sys.path,保证 vision/ 与 configs/ 可导入
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from .api.router import api_router, ws_router  # noqa: E402
from .config import Settings, get_settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

settings: Settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    from .services.pipeline_manager import get_pipeline_manager

    mgr = get_pipeline_manager()
    mgr.set_event_loop(asyncio.get_running_loop())

    # 预加载人脸底库快照(注册/搜索接口与识别热路径都依赖它)
    try:
        await mgr.load_gallery()
    except Exception as exc:  # noqa: BLE001
        logger.warning("gallery preload skipped (database unavailable?): %s", exc)

    # 启动时恢复"enabled"摄像头
    try:
        from .deps import AsyncSessionLocal
        from .services.camera_service import CameraService

        async with AsyncSessionLocal() as db:
            cameras = await CameraService(db).list()
        restored = 0
        for cam in cameras:
            if cam.enabled:
                from .config import build_camera_config

                config = build_camera_config(cam.profile, cam.config)
                try:
                    await mgr.start_camera(cam.id, cam.source, config)
                    restored += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error("restore camera %s failed: %s", cam.id, exc)
        if restored:
            logger.info("restored %s camera pipeline(s)", restored)
    except Exception as exc:  # noqa: BLE001
        logger.warning("camera restore skipped (database unavailable?): %s", exc)

    # 数据清理定时任务
    scheduler = None
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            _cleanup_old_events,
            CronTrigger(hour=settings.cleanup_cron_hour, minute=0),
            kwargs={"retention_days": settings.event_retention_days},
        )
        scheduler.start()
        app.state.scheduler = scheduler
    except Exception as exc:  # noqa: BLE001
        logger.warning("cleanup scheduler unavailable: %s", exc)

    yield

    # 停机清理
    if scheduler:
        try:
            scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass
    await mgr.shutdown()


async def _cleanup_old_events(retention_days: int = 30):
    from .deps import AsyncSessionLocal
    from .services.event_service import EventService

    async with AsyncSessionLocal() as db:
        deleted = await EventService(db).cleanup(retention_days)
        if deleted > 0:
            logger.info("cleaned up %s old events/logs", deleted)


app = FastAPI(
    title="AI Monitor API",
    description=(
        "基于 InsightFace 的实时人脸识别监控系统。\n\n"
        "- 推理内核:SCRFD 检测 + ArcFace 识别(默认 GPU)\n"
        "- 可插拔任务架构:跌倒检测等扩展任务通过配置接入,无需改动主流程\n"
        "- 接口文档:本页(/docs)为 OpenAPI 自动文档,详见 docs/API.md"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.model_dump().get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)


@app.get("/")
async def root():
    return {"name": "AI Monitor", "docs": "/docs", "health": "/api/health"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port)
