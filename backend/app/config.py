"""
backend.app.config — 配置中枢。

职责:
    1. pydantic-settings 读取 .env / 环境变量(服务级:数据库、端口等);
    2. 加载 configs/default.yaml + configs/profiles/{profile}.yaml 深合并(推理级);
    3. 为每个摄像头产出最终配置(级联:default → profile → 摄像头个性化 JSONB)。

级联优先级(高到低):
    摄像头 cameras.config(JSONB) > profile 文件 > default.yaml
"""


from __future__ import annotations
import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """服务级配置(环境变量 / .env)。"""

    database_url: str = "postgresql+asyncpg://postgres:123456@localhost:5432/ai_monitor"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False

    project_root: str = "."
    configs_dir: str = "configs"
    models_dir: str = "models"
    avatars_dir: str = "face_db/avatars"

    stream_max_height: int = 480
    stream_jpeg_quality: int = 70
    stream_push_fps: int = 10

    event_retention_days: int = 30
    cleanup_cron_hour: int = 3

    model_config = {"env_file": ".env", "env_prefix": "AIM_", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ──────────────────────────────────────────────────────────────
# YAML 级联合并
# ──────────────────────────────────────────────────────────────

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归深合并:override 覆盖 base(不存在键直接写入)。"""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _configs_path(settings: Settings) -> Path:
    root = Path(settings.project_root).resolve()
    return root / settings.configs_dir


def load_profile_config(profile: Optional[str] = None) -> Dict[str, Any]:
    """加载 default + 指定 profile 的合并配置。profile 为空时只用 default。"""
    settings = get_settings()
    base_dir = _configs_path(settings)
    merged = _read_yaml(base_dir / "default.yaml")
    if profile:
        merged = _deep_merge(merged, _read_yaml(base_dir / "profiles" / f"{profile}.yaml"))
    return merged


def resolve_project_path(relative: str) -> str:
    """把相对路径解析到项目根(模型目录、头像目录等)。"""
    settings = get_settings()
    p = Path(relative)
    if p.is_absolute():
        return str(p)
    return str((Path(settings.project_root).resolve() / p))


def build_camera_config(
    profile: str,
    camera_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """为单个摄像头产出最终配置(default → profile → camera_extra)。

    顺带把相对路径(vision.models_root 等)解析为绝对路径,
    保证任何工作目录下运行都正确。
    """
    merged = load_profile_config(profile)
    # 摄像头个性化配置直接合并到根(extra 里可含 vision/tasks/... 任意节)
    merged = _deep_merge(merged, camera_extra or {})
    vision_cfg = merged.get("vision") or {}
    if isinstance(vision_cfg, dict) and vision_cfg.get("models_root"):
        vision_cfg["models_root"] = resolve_project_path(str(vision_cfg["models_root"]))
    return merged
