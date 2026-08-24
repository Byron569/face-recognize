"""Capacity manifest 校验与注册 FPS 门禁（阶段9）。

manifest(如 models/capacity-cuda0.json) 由本机真实 GPU smoke 从空 venv 重建的环境生成,
只读不做修改。校验严格按"部署证据"对待:
    - schema_version 固定为 1;
    - 硬门禁: cuda_available=true 且 cpu_fallback_count=0;
    - 环境指纹(设备名 regex + model/package/torch/cuda/driver)精确匹配;
    - raw sustained FPS 取 300 秒持续阶段的逐秒窗口最小值(而非均值), 保证安全;
    - generated_utc 必须在 max_manifest_age_s 内, 过期证据拒绝;
    - safe_total_fps = floor(basis * headroom_ratio), basis = min(窗口最小值, max_fps_meeting_threshold);
    - effective_max_total_fps = min(safe_total_fps, 配置请求的 requested_max_total_fps)。

任何失败都不留半成品: 抛出 CapacityError, 调用方不启动 Worker。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CAPACITY_SCHEMA_VERSION = 1


class CapacityError(Exception):
    """capacity manifest 校验失败(损坏/过期/指纹不匹配/门禁不达)。"""


@dataclass(frozen=True, slots=True)
class CapacityQuery:
    """配置侧期望值, 用于与 manifest 证据比对。"""

    expected_device_regex: str
    model_sha256: str
    package_sha256: str
    torch_version: str
    cuda_version: str
    driver_version: str
    requested_max_total_fps: int
    headroom_ratio: float
    max_manifest_age_s: float = 86400.0


@dataclass(frozen=True, slots=True)
class CapacityManifest:
    """通过校验后的可消费证据。"""

    schema_version: int
    device_name: str
    raw_sustained_fps: float
    max_fps_meeting_threshold: float
    headroom_ratio: float
    safe_total_fps: int
    effective_max_total_fps: int
    capacity_basis_revision: str
    generated_utc: str
    data: dict  # 原始 payload(含每路结果与完整 audit), 仅审计用


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise CapacityError(msg)


def load_capacity_manifest(path: str) -> dict:
    """读取原始 JSON; 文件缺失或无法解析视为损坏。"""
    p = Path(path)
    if not p.is_file():
        raise CapacityError(f"capacity manifest 不存在: {path}")
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise CapacityError(f"capacity manifest 损坏(无法解析 JSON): {exc}") from None
    if not isinstance(data, dict):
        raise CapacityError("capacity manifest 必须是 JSON 对象")
    return data


def validate_capacity_manifest(
    data: dict,
    query: CapacityQuery,
    now: datetime | None = None,
) -> CapacityManifest:
    """校验损坏/过期/指纹, 通过则返回可消费证据; 失败抛 CapacityError。"""
    now = now or datetime.now(timezone.utc)

    _require(data.get("schema_version") == CAPACITY_SCHEMA_VERSION, "schema_version 必须为 1")

    # 硬门禁: 无 CPU 兜底
    _require(data.get("cuda_available") is True, "capacity 必须 cuda_available=true")
    _require(int(data.get("cpu_fallback_count", -1)) == 0, "capacity 不允许 CPU fallback")

    device_name = data.get("device_name")
    _require(
        isinstance(device_name, str) and re.search(query.expected_device_regex, device_name),
        f"capacity device_name 不匹配 {query.expected_device_regex!r}: {device_name!r}",
    )

    # 环境指纹精确匹配
    for key in (
        "model_sha256",
        "package_sha256",
        "torch_version",
        "cuda_version",
        "driver_version",
    ):
        expected = getattr(query, key)
        actual = data.get(key)
        _require(actual == expected, f"capacity {key} 指纹不匹配: 期望 {expected!r} 实际 {actual!r}")

    # 时长窗口: raw sustained 取逐秒窗口最小值
    windows = data.get("sustained_windows_total") or []
    _require(isinstance(windows, list) and len(windows) > 0, "缺少 sustained_windows_total 窗口序列")
    raw_sustained = min(float(w) for w in windows)
    _require(math.isfinite(raw_sustained) and raw_sustained > 0, "raw sustained FPS 非法")
    max_fps = float(data.get("max_fps_meeting_threshold", 0.0))
    _require(math.isfinite(max_fps) and max_fps > 0, "max_fps_meeting_threshold 非法")

    # 过期证据拒绝
    generated_utc = data.get("generated_utc")
    _require(isinstance(generated_utc, str) and generated_utc, "缺少 generated_utc")
    try:
        generated = datetime.fromisoformat(str(generated_utc).replace("Z", "+00:00"))
    except ValueError:
        raise CapacityError("generated_utc 格式无效") from None
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    age = abs((now - generated).total_seconds())
    _require(age <= query.max_manifest_age_s, f"capacity manifest 过期(age={age:.0f}s)")

    basis = min(raw_sustained, max_fps)
    safe_total = math.floor(basis * query.headroom_ratio)
    _require(safe_total > 0, "safe_total_fps 必须为正")

    effective = min(safe_total, query.requested_max_total_fps)

    return CapacityManifest(
        schema_version=CAPACITY_SCHEMA_VERSION,
        device_name=device_name,
        raw_sustained_fps=raw_sustained,
        max_fps_meeting_threshold=max_fps,
        headroom_ratio=query.headroom_ratio,
        safe_total_fps=safe_total,
        effective_max_total_fps=effective,
        capacity_basis_revision=str(data.get("capacity_basis_revision", "")),
        generated_utc=str(generated_utc),
        data=data,
    )


def capacity_explains_registration(
    cap: CapacityManifest,
    per_camera_target_fps: int,
    num_cameras: int,
) -> bool:
    """注册门禁: 目标总需求必须落在 effective 能力内, 否则拒绝新注册。"""
    return per_camera_target_fps * num_cameras <= cap.effective_max_total_fps