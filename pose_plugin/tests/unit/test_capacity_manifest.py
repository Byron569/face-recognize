"""阶段9:capacity manifest 单元测试(固定样本, 无 GPU)。

覆盖(与 融合实施.md 阶段9 一致):
    - 窗口最小值: raw sustained FPS 取逐秒窗口最小值而非均值;
    - headroom 向下取整: safe_total_fps = floor(basis * headroom);
    - 环境指纹匹配: device_name regex 与 model/package/torch/cuda/driver 指纹;
    - 损坏/过期证据拒绝: JSON 损坏、缺字段、指纹错、过期均抛出 CapacityError;
    - 配置 requested 上限: effective = min(safe, requested);
    - 注册总 FPS 门禁: 需求总和 <= effective_max_total_fps。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from pathlib import Path

import pytest

from ai_monitor_pose.capacity import (
    CapacityError,
    CapacityQuery,
    capacity_explains_registration,
    load_capacity_manifest,
    validate_capacity_manifest,
)

_DEVICE_RE = r"(?i)RTX 4060"
_SHA = "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212"


def _now():
    return datetime.now(timezone.utc)


def _manifest(**overrides):
    data = {
        "schema_version": 1,
        "cuda_available": True,
        "device_name": "NVIDIA GeForce RTX 4060",
        "model_sha256": _SHA,
        "package_sha256": "pkgsha",
        "torch_version": "2.7.0+cu126",
        "cuda_version": "12.6",
        "driver_version": "31.0.15",
        "cpu_fallback_count": 0,
        "sustained_windows_total": [34.0, 33.0, 32.0],
        "max_fps_meeting_threshold": 46.0,
        "capacity_basis_revision": "abcd1234",
        "generated_utc": _now().isoformat(),
    }
    data.update(overrides)
    return data


def _query(**overrides):
    q = CapacityQuery(
        expected_device_regex=_DEVICE_RE,
        model_sha256=_SHA,
        package_sha256="pkgsha",
        torch_version="2.7.0+cu126",
        cuda_version="12.6",
        driver_version="31.0.15",
        requested_max_total_fps=32,
        headroom_ratio=0.75,
    )
    return replace(q, **overrides)


# ── 窗口最小值 ────────────────────────────────────────────

def test_raw_sustained_takes_window_minimum_not_average() -> None:
    m = _manifest(sustained_windows_total=[40.0, 38.0, 30.0], max_fps_meeting_threshold=100.0)
    cap = validate_capacity_manifest(m, _query(requested_max_total_fps=64))
    assert cap.raw_sustained_fps == 30.0  # 不是 (40+38+30)/3=36, 也不是 40


# ── headroom 向下取整 ─────────────────────────────────────

def test_safe_total_floors_headroom() -> None:
    cap = validate_capacity_manifest(
        _manifest(sustained_windows_total=[46.4], max_fps_meeting_threshold=46.4),
        _query(requested_max_total_fps=64),  # 提高 requested 以观察纯 floor
    )
    # basis=46.4, 46.4*0.75=34.8 -> floor 34 (既非 34.8 也非 35)
    assert cap.safe_total_fps == 34


# ── 配置 requested 上限 ───────────────────────────────────

def test_effective_caps_at_requested_max_total_fps() -> None:
    cap = validate_capacity_manifest(
        _manifest(sustained_windows_total=[46.4], max_fps_meeting_threshold=46.4),
        _query(requested_max_total_fps=32),
    )
    # safe=34, requested=32 -> effective=32
    assert cap.safe_total_fps == 34
    assert cap.effective_max_total_fps == 32


def test_effective_obeys_lower_of_safe_and_requested() -> None:
    cap = validate_capacity_manifest(
        _manifest(sustained_windows_total=[30.0], max_fps_meeting_threshold=30.0),
        _query(requested_max_total_fps=64),
    )
    # safe=floor(30*0.75)=22 < requested -> effective=22
    assert cap.effective_max_total_fps == 22


# ── 环境指纹 ─────────────────────────────────────────────

def test_device_name_regex_mismatch_rejected() -> None:
    with pytest.raises(CapacityError):
        validate_capacity_manifest(_manifest(device_name="Intel UHD Graphics"), _query())


def test_device_name_regex_case_insensitive_match() -> None:
    cap = validate_capacity_manifest(_manifest(device_name="nvidia geforce rtx 4060 LAPTOP"), _query())
    assert cap.device_name.lower() == "nvidia geforce rtx 4060 laptop"


@pytest.mark.parametrize(
    "override_key,bad_val",
    [
        ("model_sha256", "0" * 64),
        ("package_sha256", "wrongpkg"),
        ("torch_version", "2.6.0"),
        ("cuda_version", "12.4"),
        ("driver_version", "999.99"),
    ],
)
def test_environment_fingerprint_mismatch_rejected(override_key: str, bad_val: str) -> None:
    with pytest.raises(CapacityError):
        validate_capacity_manifest(_manifest(**{override_key: bad_val}), _query())


# ── 损坏 / 过期证据 ──────────────────────────────────────

def test_corrupt_json_file_rejected(tmp_path: Path) -> None:
    f = tmp_path / "capacity.json"
    f.write_text("{not json", encoding="utf-8")
    with pytest.raises(CapacityError):
        load_capacity_manifest(str(f))


def test_missing_manifest_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(CapacityError):
        load_capacity_manifest(str(tmp_path / "nope.json"))


def test_missing_schema_version_rejected() -> None:
    with pytest.raises(CapacityError):
        validate_capacity_manifest(_manifest(schema_version=2), _query())


def test_cpu_fallback_evidence_rejected() -> None:
    with pytest.raises(CapacityError):
        validate_capacity_manifest(_manifest(cpu_fallback_count=1), _query())


def test_expired_evidence_rejected() -> None:
    stale = (_now() - timedelta(seconds=120)).isoformat()
    with pytest.raises(CapacityError):
        validate_capacity_manifest(
            _manifest(generated_utc=stale),
            _query(max_manifest_age_s=30),
        )


def test_fresh_evidence_accepted() -> None:
    cap = validate_capacity_manifest(_manifest(), _query())
    assert cap.schema_version == 1
    assert cap.effective_max_total_fps >= 1


def test_load_and_validate_roundtrip(tmp_path: Path) -> None:
    f = tmp_path / "capacity.json"
    f.write_text(json.dumps(_manifest()), encoding="utf-8")
    data = load_capacity_manifest(str(f))
    assert isinstance(data, dict)
    cap = validate_capacity_manifest(data, _query())
    assert cap.generated_utc == data["generated_utc"]


# ── 注册总 FPS 门禁 ──────────────────────────────────────

def test_registration_total_fps_gate() -> None:
    # effective=32: 4 路 x 8 FPS(需求=32) 恰好通过
    cap = validate_capacity_manifest(
        _manifest(sustained_windows_total=[46.4], max_fps_meeting_threshold=46.4),
        _query(requested_max_total_fps=32),
    )
    assert capacity_explains_registration(cap, per_camera_target_fps=8, num_cameras=4) is True


def test_registration_fails_when_requirement_exceeds_gate() -> None:
    # effective=22: 3x8=24 超出 -> 关闭新摄或者降低每路 target
    cap = validate_capacity_manifest(
        _manifest(sustained_windows_total=[30.0], max_fps_meeting_threshold=30.0),
        _query(requested_max_total_fps=64),
    )
    assert cap.effective_max_total_fps == 22
    assert capacity_explains_registration(cap, per_camera_target_fps=8, num_cameras=3) is False
    # 降低每路 target 后满足
    assert capacity_explains_registration(cap, per_camera_target_fps=5, num_cameras=4) is True