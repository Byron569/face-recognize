"""系统状态 / 性能指标 / 部署档位 / 运行配置。"""

from __future__ import annotations
import psutil
from fastapi import APIRouter

from ..config import build_camera_config, load_profile_config, resolve_project_path
from ..schemas.system import SystemStatusOut
from ..services.model_manager import get_engine_pool
from ..services.pipeline_manager import get_pipeline_manager

router = APIRouter()

# 部署档位摘要(与 configs/profiles/*.yaml 一致,供前端展示)
PROFILES = {
    "desktop": {"device": "cuda", "det_size": "640px", "desc": "NVIDIA GPU 工作站(推荐,推理走 CUDA)"},
    "balanced": {"device": "cuda", "det_size": "480px", "desc": "中等 GPU,降低检测频率"},
    "edge_minimal": {"device": "cpu", "det_size": "320px", "desc": "低算力边缘设备"},
}


@router.get("/system/status", response_model=SystemStatusOut)
async def system_status():
    mgr = get_pipeline_manager()
    all_ids = mgr.list_cameras()
    active = [cid for cid in all_ids if (mgr.get_status(cid) or {}).get("alive")]

    gpu_name = gpu_util = gpu_mem = None
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode()
        parts = [x.strip() for x in out.split(",")]
        if len(parts) >= 4:
            gpu_name = parts[0]
            gpu_util = float(parts[1])
            mem_used, mem_total = float(parts[2]), float(parts[3])
            gpu_mem = (mem_used / mem_total) * 100 if mem_total > 0 else 0
    except Exception:  # noqa: BLE001
        pass

    return SystemStatusOut(
        cpu_percent=psutil.cpu_percent(interval=0.5),
        memory_percent=psutil.virtual_memory().percent,
        gpu_name=gpu_name,
        gpu_utilization=gpu_util,
        gpu_memory_percent=gpu_mem,
        camera_count=len(all_ids),
        active_camera_count=len(active),
        engine_count=get_engine_pool().count,
        gallery_size=mgr.gallery.size,
    )


@router.get("/system/metrics")
async def system_metrics():
    """各摄像头实时性能指标(FPS 由帧率折算,阶段耗时由 pipeline 上报)。"""
    mgr = get_pipeline_manager()
    cameras = {}
    for cid in mgr.list_cameras():
        m = mgr.get_status(cid) or {}
        cameras[cid] = {
            "fps": round(m.get("fps", 0), 1),
            "frames": m.get("frames", 0),
            "tracks": m.get("tracks", 0),
            "uptime_seconds": m.get("uptime_seconds", 0),
            "stage_ms": m.get("stage_ms", {}),
            "stream": mgr.get_stream_metrics(cid),
        }
    return {
        "cameras": cameras,
        "global": {
            "camera_count": len(mgr.list_cameras()),
            "engine_count": get_engine_pool().count,
            "gallery_size": mgr.gallery.size,
        },
    }


@router.get("/system/profiles")
async def system_profiles():
    """部署档位清单(来自 configs/profiles/,前端可据此渲染选择器)。"""
    return {"profiles": [{"name": name, **params} for name, params in PROFILES.items()]}


@router.get("/system/config")
async def get_runtime_config(profile: str = "desktop"):
    """查看指定档位的运行时合并配置(不含敏感信息)。"""
    return load_profile_config(profile)


@router.put("/system/config")
async def update_runtime_config(body: dict):
    """预留:运行时热更新全局配置(当前返回提示,实施时写回 YAML 或配置中心)。"""
    return {"updated": False, "message": "runtime config update is reserved"}
