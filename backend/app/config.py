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
    stream_push_fps: int = 20

    event_retention_days: int = 30
    cleanup_cron_hour: int = 3

    # CORS 白名单(环境变量 AIM_CORS_ORIGINS,JSON 数组格式;
    # 生产同源部署(nginx 反代)无需跨域,保持默认即可)
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    model_config = {"env_file": ".env", "env_prefix": "AIM_", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ──────────────────────────────────────────────────────────────
# 集成测试模式(默认关闭,仅显式环境注入才启用)
# ──────────────────────────────────────────────────────────────
# 与 AI_MONITOR_TEST_DATABASE_URL 一致,读取无前缀的环境变量(不走 Settings 的
# env_prefix),确保测试编排能按既定变量名控制。关闭时 /health 绝不暴露任何
# marker,也绝不注入 run_id,避免扩大生产 API 面或泄漏随机标识。

def test_mode_enabled() -> bool:
    """AI_MONITOR_TEST_MODE 显式为 1/true/yes/on 才启用(默认关闭)。"""
    val = os.environ.get("AI_MONITOR_TEST_MODE", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def test_run_id() -> str:
    """本次集成测试注入的随机 run id;未启用/未注入则返回空。"""
    return os.environ.get("AI_MONITOR_TEST_RUN_ID", "").strip()


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
        profile_path = base_dir / "profiles" / f"{profile}.yaml"
        if not profile_path.exists():
            raise ValueError(f"未知 profile: {profile}(可用: desktop/balanced/edge_minimal)")
        merged = _deep_merge(merged, _read_yaml(profile_path))
    return merged


def resolve_project_path(relative: str) -> str:
    """把相对路径解析到项目根(模型目录、头像目录等)。"""
    settings = get_settings()
    p = Path(relative)
    if p.is_absolute():
        return str(p)
    return str((Path(settings.project_root).resolve() / p))


_FALL_PATH_KEYS = (
    ("worker", "python"),
    ("model", "path"),
    ("model", "sha256_file"),
    ("runtime", "capacity_manifest_path"),
    ("runtime", "worker_journal_path"),
    ("runtime", "event_spool_path"),
)


def _resolve_fall_paths(merged: Dict[str, Any]) -> Dict[str, Any]:
    """把 tasks.fall_detection 的相对路径解析为基于项目根(仓库根)的绝对路径。

    ai_monitor_pose.config.FallTaskConfig 构造期硬约束这些字段必须为绝对路径
    (design contract),故在产出最终摄像头配置时统一解析,做到 clone 即跑的便携部署。
    """
    fd = ((merged.get("tasks") or {}).get("fall_detection") or {})
    if not isinstance(fd, dict):
        return merged
    for section, key in _FALL_PATH_KEYS:
        node = fd.get(section)
        if isinstance(node, dict) and node.get(key):
            node[key] = resolve_project_path(str(node[key]))
    return merged


def build_camera_config(
    profile: str,
    camera_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """为单个摄像头产出最终配置(default → profile → camera_extra)。

    顺带把相对路径(vision.models_root、tasks.fall_detection 各持久化/worker 路径等)
    解析为绝对路径,保证任何工作目录下运行都正确。
    """
    merged = load_profile_config(profile)
    # 摄像头个性化配置直接合并到根(extra 里可含 vision/tasks/... 任意节)
    merged = _deep_merge(merged, camera_extra or {})
    vision_cfg = merged.get("vision") or {}
    if isinstance(vision_cfg, dict) and vision_cfg.get("models_root"):
        vision_cfg["models_root"] = resolve_project_path(str(vision_cfg["models_root"]))
    _resolve_fall_paths(merged)
    return merged
