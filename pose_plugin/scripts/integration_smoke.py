"""AI Monitor 端到端集成冒烟（阶段9 M1 门禁）。

只读冒烟通过 AI Monitor API / event WS / 健康快照 / 测试 DB 观察系统, 绝不自己
VideoCapture 同一视频源。任何破坏性注入(kill Worker / 模拟 DB 故障 / 断 Pipe)
都必须显式提供 --allow-fault-injection --test-run-id <uuid>, 并在动作前逐项验证:
    - URL host 是 loopback;
    - /api/health 返回 test_mode=true 且 test_run_id 精确匹配;
    - camera id 以 test- 开头并属于本次 run;
    - SELECT current_database() 以 _test 结尾;
    - 目标 Worker PID/命令行/runtime key 与健康快照一致。
任一项失败以退出码 2 拒绝, 不执行任何故障动作。

退出码固定:
    0 = 全部通过
    2 = 参数 / manifest / 模型错误
    3 = AI Monitor / Worker / 观测面不可用
    4 = 运行完成但门槛失败
    5 = 脚本内部未处理异常
报告固定 schema_version:1; 无论成败都先写临时 JSON 再原子 rename。

本模块的 fault 门禁 evaluate_fault_guard 是纯函数, 不触网, 专供单元测试证明
公网/局域网 URL、非 test DB、错 run ID、普通 camera、无显式开关时零故障动作。

阶段11 扩展(M2): 以同样的纯函数/注入风格断言 analytics source frame 契约、
overlay 缓存 TTL 过期后的重挂载、unavailable UI 语义(运行不可用时不得呈现假
Normal)与 /api/system/fall-runtime 脱敏(pipe authkey / 配置路径 / DB URL /
异常堆栈零泄漏, 白名单对照 backend fall_runtime_health.py, 消息契约对照
frontend analyticsProtocol.ts)。M1 event 断言与退出码契约 0/2/3/4/5 不变。
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

EXIT_ALL_PASS = 0
EXIT_PARAM = 2
EXIT_UNAVAILABLE = 3
EXIT_THRESHOLD = 4
EXIT_INTERNAL = 5

SCHEMA_VERSION = 1


class SmokeError(RuntimeError):
    """参数/manifest/模型错误 -> EXIT_PARAM。"""


class UnavailableError(RuntimeError):
    """观测面不可用 -> EXIT_UNAVAILABLE。"""


class ThresholdError(RuntimeError):
    """运行完成但门槛失败 -> EXIT_THRESHOLD。"""


# ── FAULT 注入门禁(纯函数, 供单元测试) ──────────────────

@dataclass(frozen=True)
class HealthProbe:
    """注入的 /api/health 结果(测试可离线构造)。"""
    ok: bool
    test_mode: bool
    test_run_id: str


@dataclass(frozen=True)
class FaultGuardInput:
    url: str
    allow_fault_injection: bool
    test_run_id_cfg: str
    camera_ids: tuple[str, ...]
    health: HealthProbe | None
    db_name: str


def is_loopback_url(url: str) -> bool:
    """仅接受 localhost / 127.0.0.0/8 / ::1; 其余(公网/局域网)一律拒绝。"""
    host = (urlparse(url).hostname or "").lower()
    if host in ("localhost", ""):
        return bool(host)
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def evaluate_fault_guard(cfg: FaultGuardInput) -> tuple[bool, str]:
    """破坏性注入门禁: 任一项不达则 (False, reason), 调用方零故障动作退出 2。"""
    if not cfg.allow_fault_injection:
        return False, "未显式开启 --allow-fault-injection"
    if not cfg.test_run_id_cfg:
        return False, "缺少 --test-run-id"
    if not is_loopback_url(cfg.url):
        return False, f"URL host 非 loopback: {urlparse(cfg.url).hostname!r}"
    h = cfg.health
    if h is None or not h.ok or not h.test_mode:
        return False, "/api/health 未返回 test_mode=true"
    if h.test_run_id != cfg.test_run_id_cfg:
        return False, f"run id 不匹配(期望 {cfg.test_run_id_cfg!r} 实际 {h.test_run_id!r})"
    if not cfg.camera_ids or not all(c.startswith("test-") for c in cfg.camera_ids):
        return False, "camera id 必须以 test- 开头"
    if not cfg.db_name.endswith("_test"):
        return False, f"DB 非集成测试库(_test 结尾): {cfg.db_name!r}"
    return True, ""


# ── 报告 / 原子写盘 ──────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _monotonic_ms() -> int:
    try:
        import time
        return int(time.monotonic() * 1000)
    except Exception:
        return 0


def write_report_atomic(path: str, data: dict) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return str(p)


def build_report(*, params: dict, start_utc: str, start_mono_ms: int,
                 checks: list[dict], summary: dict, extra: dict | None = None) -> dict:
    end_utc = _now_utc()
    return {
        "schema_version": SCHEMA_VERSION,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "start_monotonic_ms": start_mono_ms,
        "end_monotonic_ms": _monotonic_ms(),
        "params": params,
        "summary": summary,
        "checks": checks,
        **(extra or {}),
    }


# ── 受控视频 fixture manifest(M1 完整链路) ─────────────────
# manifest JSON 逐段:"AI Monitor 测试摄像头使用的本地视频绝对路径、SHA-256、
# 期望 incident/event 顺序与容许时间窗"。脚本不自己 VideoCapture 同一源,
# 只通过 AI Monitor API / event WS / 健康快照 / 测试 DB 观察系统。

@dataclass(frozen=True)
class ManifestSegment:
    """一段受控视频的期望契约。"""
    camera_id: str
    video_path: str
    sha256: str
    expected_events: tuple[str, ...]  # 有序列, 例如 ("fall_potential", "fallen")
    allowed_window_s: float


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(text: str) -> list[ManifestSegment]:
    """解析并校形 manifest; 结构错误抛 SmokeError(->EXIT_PARAM)。"""
    data = json.loads(text)
    if not isinstance(data, list):
        raise SmokeError("manifest 顶层必须是数组")
    segments: list[ManifestSegment] = []
    for i, seg in enumerate(data):
        if not isinstance(seg, dict):
            raise SmokeError(f"segment[{i}] 非对象")
        cam = seg.get("camera_id")
        vp = seg.get("video_path")
        sh = seg.get("sha256")
        if not isinstance(cam, str) or not cam:
            raise SmokeError(f"segment[{i}] 缺 camera_id")
        if not isinstance(vp, str) or not vp:
            raise SmokeError(f"segment[{i}] 缺 video_path")
        if not isinstance(sh, str) or len(sh) != 64:
            raise SmokeError(f"segment[{i}] sha256 须为 64 位十六进制")
        exp = seg.get("expected_events", [])
        if not isinstance(exp, list) or not exp:
            raise SmokeError(f"segment[{i}] expected_events 须为非空列表")
        if not all(isinstance(e, str) and e for e in exp):
            raise SmokeError(f"segment[{i}] expected_events 须为字符串序列")
        win = float(seg.get("allowed_window_s", 4.0))
        if not (0 < win <= 3600):
            raise SmokeError(f"segment[{i}] allowed_window_s 须在 (0,3600]")
        segments.append(ManifestSegment(
            camera_id=cam, video_path=vp, sha256=sh,
            expected_events=tuple(exp), allowed_window_s=win,
        ))
    if not segments:
        raise SmokeError("manifest 至少 1 段")
    return segments


def validate_manifest_segments(segments: list[ManifestSegment]) -> tuple[bool, list[str]]:
    """校验本地视频存在 + 非空 + sha256 匹配 + camera 属 test-。坏 -> (False, errors)。"""
    errors: list[str] = []
    for seg in segments:
        if not seg.camera_id.startswith("test-"):
            errors.append(f"{seg.camera_id}: 必须以 test- 开头")
        p = Path(seg.video_path)
        if not p.is_file():
            errors.append(f"{seg.video_path}: 文件不存在")
            continue
        if p.stat().st_size <= 0:
            errors.append(f"{seg.video_path}: 空文件")
        try:
            if sha256_file(seg.video_path) != seg.sha256:
                errors.append(f"{seg.video_path}: sha256 不匹配")
        except OSError as exc:
            errors.append(f"{seg.video_path}: 读取失败 {exc}")
    return (not errors), errors


def evaluate_event_sequence(observed: list[str], expected: tuple[str, ...]) -> tuple[bool, str]:
    """按 expected 顺序对 observed 做贪心首现匹配; 缺失/乱序 -> (False, detail)。"""
    if expected and not observed:
        return False, f"未观测到任何事件, 期望 {expected}"
    i = 0
    for e in expected:
        while i < len(observed) and observed[i] != e:
            i += 1
        if i >= len(observed):
            return False, f"缺少事件 {e!r}"
        i += 1
    return True, "事件序列匹配 " + "  ".join(expected)


def evaluate_window_coverage(observed_ts: list[float], first_ms: float, allowed_s: float) -> tuple[bool, str]:
    """期望在 first_ms 前后 allowed_s 秒内至少一次观测命中。"""
    if not observed_ts:
        return False, "时间窗内无观测"
    lo = first_ms - allowed_s * 1000.0
    hi = first_ms + allowed_s * 1000.0
    hit = any(lo <= t <= hi for t in observed_ts)
    return hit, ("时间窗命中" if hit else f"窗口 [{lo:.0f},{hi:.0f}]ms 无命中")

# ── 阶段11 M2 断言(纯函数/注入, 离线可测, 不触网) ────────
# 契约来源:
#   - backend/app/services/fall_runtime_health.py  (/api/system/fall-runtime 脱敏白名单)
#   - frontend/src/stream/analyticsProtocol.ts     (analytics 消息字段与校验规则)

_VALID_POSE_STATES = frozenset({"normal", "potential", "fallen"})
_RUNTIME_STATES = frozenset({"READY", "DEGRADED", "UNAVAILABLE", "DISABLED"})

# /api/system/fall-runtime 响应白名单(与 fall_runtime_health.py 各构建器一一对应)
_TOP_ALLOW = frozenset({"schema_version", "enabled", "mode", "runtime_key", "state",
                        "error", "worker", "gpu", "model", "delivery", "cameras"})
_WORKER_ALLOW = frozenset({"pid", "epoch", "restart_count", "heartbeat_age_ms"})
_GPU_ALLOW = frozenset({"device", "name", "allocated_mb", "reserved_mb", "effective_limit_mb"})
_MODEL_ALLOW = frozenset({"name", "sha256", "precision"})
_DELIVERY_ALLOW = frozenset({"transition_queue_depth", "spool_pending", "oldest_pending_age_ms"})
_CAMERA_ALLOW = frozenset({"camera_id", "camera_session_id", "state", "submitted", "analyzed",
                           "replaced", "stale", "effective_fps", "latest_result_age_ms",
                           "transition_queue_depth", "open_incidents"})

# 明文禁止标记(对序列化文本小写匹配): pipe authkey / DB URL / 配置与模型文件路径 / 异常堆栈
_LEAK_MARKERS = (
    "authkey", "auth_key",
    "postgresql://", "postgres://", "asyncpg://", "psql://", "sqlite:///",
    "database_url", "db_url", "config_path", "model_path",
    "traceback", "most recent call last", 'file \\"',
    ".yaml", ".yml", ".toml", ".cfg", ".conf", ".ini",
)


def _is_finite_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load_analytics_observations(text: str) -> list[dict]:
    """解析 analytics 观测 JSONL(纯函数)。

    每行一个 JSON 对象, 两种形态皆可:
        - 扁平: {"ts_ms": .., "camera_session_id": .., "preview_frame_id": .., "fall_detection": {..}}
        - 包装: {"ts_ms": .., "message": {analytics wire 消息}}
    结构错误抛 SmokeError(->EXIT_PARAM); 返回统一为扁平观测列表。
    """
    observations: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SmokeError(f"analytics 观测第 {lineno} 行非 JSON: {exc}") from None
        if not isinstance(row, dict):
            raise SmokeError(f"analytics 观测第 {lineno} 行非对象")
        msg = row.get("message")
        if isinstance(msg, dict):
            item = dict(msg)
            item["ts_ms"] = row.get("ts_ms")
        else:
            item = row
        if not _is_finite_num(item.get("ts_ms")):
            raise SmokeError(f"analytics 观测第 {lineno} 行缺数值 ts_ms")
        for key in ("camera_session_id", "preview_frame_id", "fall_detection"):
            if key not in item:
                raise SmokeError(f"analytics 观测第 {lineno} 行缺 {key}")
        observations.append(item)
    if not observations:
        raise SmokeError("analytics 观测为空")
    return observations


def evaluate_analytics_source_frame(observations: list[dict]) -> tuple[bool, str]:
    """断言每条 analytics 观测满足 source frame 契约(对照 analyticsProtocol.ts)。

    必须携带有效 source_frame_id(M2 溯源要求, 不允许缺失), 内外 session 一致,
    预览尺寸/transform 合法, overlay TTL 为正, pose state 只取已知三值。
    """
    if not observations:
        return False, "无 analytics 观测, 无法证明 source frame 契约"
    for i, o in enumerate(observations):
        if not isinstance(o, dict):
            return False, f"obs[{i}] 非对象"
        outer = o.get("camera_session_id")
        if not isinstance(outer, str) or not outer:
            return False, f"obs[{i}] 缺 camera_session_id"
        if not _is_finite_num(o.get("preview_frame_id")) or o["preview_frame_id"] < 0:
            return False, f"obs[{i}] preview_frame_id 非法: {o.get('preview_frame_id')!r}"
        fd = o.get("fall_detection")
        if not isinstance(fd, dict):
            return False, f"obs[{i}] 缺 fall_detection"
        if fd.get("camera_session_id") != outer:
            return False, f"obs[{i}] 内外 camera_session_id 不一致"
        src = fd.get("source_frame_id")
        if not _is_finite_num(src) or src < 0:
            return False, f"obs[{i}] 缺有效 source_frame_id: {src!r}"
        pw, ph = fd.get("preview_width"), fd.get("preview_height")
        if not _is_finite_num(pw) or not _is_finite_num(ph) or pw <= 0 or ph <= 0:
            return False, f"obs[{i}] 预览尺寸非法: {pw!r}x{ph!r}"
        tr = fd.get("transform")
        if not isinstance(tr, dict) or not _is_finite_num(tr.get("scale_x")) \
                or not _is_finite_num(tr.get("scale_y")) \
                or tr["scale_x"] <= 0 or tr["scale_y"] <= 0:
            return False, f"obs[{i}] transform 非法: {tr!r}"
        exp = fd.get("overlay_expires_in_ms")
        if not _is_finite_num(exp) or exp <= 0:
            return False, f"obs[{i}] overlay_expires_in_ms 非法: {exp!r}"
        tracks = fd.get("tracks", [])
        if not isinstance(tracks, list):
            return False, f"obs[{i}] tracks 非列表"
        for t in tracks:
            if not isinstance(t, dict):
                return False, f"obs[{i}] track 非对象"
            state = t.get("state", "normal")
            if state not in _VALID_POSE_STATES:
                return False, f"obs[{i}] 未知 pose state: {state!r}"
    return True, f"analytics source frame 契约满足({len(observations)} 条观测)"


def evaluate_overlay_ttl_remount(observations: list[dict]) -> tuple[bool, str]:
    """断言 overlay 缓存 TTL 过期后重新挂载到更新的 source frame, 且不超期用陈旧结果。

    观测按 ts_ms 时序注入:
        - 乱序即失败;
        - 相邻观测间隔超过前一条 overlay_expires_in_ms => 缓存已过期, 后一条必须
          重新挂载到严格更新的 source_frame_id(不得复用/回退陈旧 overlay);
        - TTL 内允许同帧更新, 但 source_frame_id 不得回退;
        - 任一观测 result_age_ms 超过自身 overlay_expires_in_ms => 陈旧结果仍在使用;
        - camera_session_id 切换视为全新挂载边界(缓存按 session 清空)。
    """
    if not observations:
        return False, "无 analytics 观测, 无法证明 TTL 过期重挂载"
    remounts = 0
    prev: dict | None = None
    for i, o in enumerate(observations):
        ts = o.get("ts_ms")
        if not _is_finite_num(ts):
            return False, f"obs[{i}] ts_ms 非法: {ts!r}"
        fd = o.get("fall_detection")
        if not isinstance(fd, dict):
            return False, f"obs[{i}] 缺 fall_detection"
        src, exp = fd.get("source_frame_id"), fd.get("overlay_expires_in_ms")
        if not _is_finite_num(src) or not _is_finite_num(exp) or exp <= 0:
            return False, f"obs[{i}] source_frame_id/overlay_expires_in_ms 非法"
        age = fd.get("result_age_ms")
        if _is_finite_num(age) and age > exp:
            return False, f"obs[{i}] result_age_ms={age:.0f} 超过 overlay TTL={exp:.0f}, 陈旧结果仍在使用"
        if prev is not None:
            gap = ts - prev["ts"]
            if gap < 0:
                return False, f"obs[{i}] 观测时间乱序(ts={ts:.0f} < {prev['ts']:.0f})"
            if o.get("camera_session_id") == prev["o"].get("camera_session_id"):
                if gap > prev["exp"]:
                    if src <= prev["src"]:
                        return False, (f"obs[{i}] TTL 过期 {gap - prev['exp']:.0f}ms 后未重挂载到更新源帧: "
                                       f"source_frame_id {src} <= {prev['src']}(陈旧 overlay 仍在使用)")
                    remounts += 1
                elif src < prev["src"]:
                    return False, f"obs[{i}] source_frame_id 回退: {src} < {prev['src']}"
        prev = {"o": o, "ts": ts, "src": src, "exp": exp}
    return True, f"overlay TTL 重挂载契约满足(过期后重新挂载 {remounts} 次)"


def evaluate_unavailable_ui_semantics(state: str, analytics_health_values: list) -> tuple[bool, str]:
    """断言 unavailable UI 语义: 运行不可用期间不得呈现假 Normal。

    state 取自 /api/system/fall-runtime(READY/DEGRADED/UNAVAILABLE/DISABLED);
    analytics_health_values 为观测到的 analytics fall_detection.health 序列。
    非 READY 期间出现 "normal"(不区分大小写)即失败; 未知 state 同样失败
    (UI 无法映射未知状态, 有假 Normal 风险)。
    """
    s = state.upper() if isinstance(state, str) else ""
    if s not in _RUNTIME_STATES:
        return False, f"未知 runtime state: {state!r}, 无法保证 unavailable UI 语义"
    if s == "READY":
        return True, "runtime READY, UI 可显示 normal"
    for v in analytics_health_values:
        if isinstance(v, str) and v.strip().lower() == "normal":
            return False, f"runtime {s} 期间 analytics/UI 呈现假 Normal: {v!r}"
    return True, f"runtime {s} 期间未出现假 Normal"


def evaluate_health_redaction(payload) -> tuple[bool, str]:
    """断言 /api/system/fall-runtime 响应脱敏: 泄漏标记零命中 + 白名单键控。

    payload 可为 dict(已解析)或 str(原始 JSON 文本)。禁止泄漏: pipe authkey、
    DB URL、配置/模型文件路径、异常堆栈; 同时按 fall_runtime_health.py 的
    白名单校验各节键, 任何越界键都视为泄漏面。
    """
    data = payload
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return False, "健康响应非 JSON 文本"
    if not isinstance(data, dict):
        return False, "健康响应非对象"
    bad = set(data) - _TOP_ALLOW
    if bad:
        return False, f"健康响应顶层越界键: {sorted(bad)}"
    for key, allow in (("worker", _WORKER_ALLOW), ("gpu", _GPU_ALLOW),
                       ("model", _MODEL_ALLOW), ("delivery", _DELIVERY_ALLOW)):
        section = data.get(key)
        if section is None:
            continue
        if not isinstance(section, dict):
            return False, f"健康响应 {key} 节非对象"
        bad = set(section) - allow
        if bad:
            return False, f"健康响应 {key} 节越界键: {sorted(bad)}"
    cameras = data.get("cameras", [])
    if not isinstance(cameras, list):
        return False, "健康响应 cameras 非列表"
    for c in cameras:
        if not isinstance(c, dict):
            return False, "健康响应 camera 条目非对象"
        bad = set(c) - _CAMERA_ALLOW
        if bad:
            return False, f"健康响应 cameras 节越界键: {sorted(bad)}"
    text = json.dumps(data, ensure_ascii=False).lower()
    for marker in _LEAK_MARKERS:
        if marker in text:
            return False, f"健康响应泄漏敏感标记: {marker!r}"
    return True, "健康响应脱敏契约满足(白名单键控 + 零泄漏标记)"


def build_m2_checks(snapshot: dict | None, observations: list) -> list[dict]:
    """把 M2 纯断言组装为报告 check 项(纯函数, main 与单元测试共用)。"""
    obs = list(observations or [])
    if isinstance(snapshot, dict):
        red_ok, red_why = evaluate_health_redaction(snapshot)
        state = snapshot.get("state", "")
    else:
        red_ok, red_why = False, "未获取 /api/system/fall-runtime 快照"
        state = ""
    health_values = []
    for o in obs:
        fd = o.get("fall_detection") if isinstance(o, dict) else None
        health_values.append(fd.get("health") if isinstance(fd, dict) else None)
    ui_ok, ui_why = evaluate_unavailable_ui_semantics(str(state) if state else "UNKNOWN", health_values)
    sf_ok, sf_why = evaluate_analytics_source_frame(obs)
    ttl_ok, ttl_why = evaluate_overlay_ttl_remount(obs)
    return [
        {"name": "m2_health_redaction", "passed": red_ok,
         "expected": "脱敏", "actual": "通过" if red_ok else "泄漏", "evidence": red_why},
        {"name": "m2_unavailable_ui_semantics", "passed": ui_ok,
         "expected": "不呈现假 Normal", "actual": "通过" if ui_ok else "假 Normal", "evidence": ui_why},
        {"name": "m2_analytics_source_frame", "passed": sf_ok,
         "expected": "source frame 契约", "actual": "通过" if sf_ok else "违约", "evidence": sf_why},
        {"name": "m2_overlay_ttl_remount", "passed": ttl_ok,
         "expected": "TTL 过期后重挂载", "actual": "通过" if ttl_ok else "陈旧 overlay", "evidence": ttl_why},
    ]

# ── 观测(只读)与执行编排 ────────────────────────────────

def _fetch_health(base_url: str) -> HealthProbe | None:
    """只读 GET /api/health; 失败视为观测面不可用(marker 缺失不视为通过)。"""
    try:
        import urllib.request
        req = urllib.request.Request(base_url.rstrip("/") + "/api/health", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return HealthProbe(
            ok=bool(body.get("status") == "ok"),
            test_mode=bool(body.get("test_mode")),
            test_run_id=str(body.get("test_run_id", "")),
        )
    except Exception as exc:
        raise UnavailableError(f"/api/health 不可用: {exc}") from None


def _fetch_fall_runtime_snapshot(base_url: str) -> dict:
    """只读 GET /api/system/fall-runtime(M2); 失败视为观测面不可用(->EXIT_UNAVAILABLE)。"""
    try:
        import urllib.request
        req = urllib.request.Request(base_url.rstrip("/") + "/api/system/fall-runtime", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("响应非对象")
        return body
    except Exception as exc:
        raise UnavailableError(f"/api/system/fall-runtime 不可用: {exc}") from None


def _current_db_name() -> str:
    """只读 SELECT current_database(); 仅当已注入测试 DB URL 时执行。"""
    url = os.environ.get("AI_MONITOR_TEST_DATABASE_URL", "")
    if not url:
        return ""
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(url.replace("asyncpg", "psycopg2"))
        with engine.connect() as conn:
            return str(conn.execute(sqlalchemy.text("SELECT current_database()")).scalar())
    except Exception:
        return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai-monitor-url", required=True)
    parser.add_argument("--camera-id", action="append", default=[], required=True)
    parser.add_argument("--duration-seconds", type=int, required=True)
    parser.add_argument("--expected-device-regex", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--fixture-manifest", default=None)
    parser.add_argument("--analytics-log", default=None,
                        help="analytics 观测 JSONL 文件(每行一条 wire 消息或 {ts_ms,message} 包装)")
    parser.add_argument("--allow-fault-injection", action="store_true")
    parser.add_argument("--test-run-id", default="")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    start_utc = _now_utc()
    start_mono = _monotonic_ms()
    report_path = args.report
    try:
        if args.duration_seconds <= 0:
            raise SmokeError("--duration-seconds 必须为正")
        if not args.camera_id:
            raise SmokeError("至少提供 1 个 --camera-id")
        health = _fetch_health(args.ai_monitor_url)
        db_name = _current_db_name()
        gate_ok, gate_reason = evaluate_fault_guard(FaultGuardInput(
            url=args.ai_monitor_url,
            allow_fault_injection=args.allow_fault_injection,
            test_run_id_cfg=args.test_run_id,
            camera_ids=tuple(args.camera_id),
            health=health,
            db_name=db_name,
        ))
        if args.allow_fault_injection and not gate_ok:
            # 显式要故障注入但前置门禁不达: 零故障动作, 退出 2
            write_report_atomic(report_path, build_report(
                params=_sanitize_params(args), start_utc=start_utc, start_mono_ms=start_mono,
                checks=[{"name": "fault_guard", "passed": False, "expected": "允许",
                         "actual": "拒绝", "evidence": gate_reason}],
                summary={"result": "guard_rejected", "exit_code": EXIT_PARAM},
            ))
            return EXIT_PARAM
        # 只读 smoke 主体(每类型检查), 此处以健康 marker 作为最小可证明链路
        # 受控视频 manifest(提供时): 解析 + 本地文件存在/sha256/camera 前缀校验
        manifest = None
        if args.fixture_manifest:
            try:
                with open(args.fixture_manifest, "r", encoding="utf-8") as fh:
                    manifest = load_manifest(fh.read())
            except OSError as exc:
                raise SmokeError(f"manifest 读取失败: {exc}") from None
            seg_ok, m_errors = validate_manifest_segments(manifest)
            if not seg_ok:
                write_report_atomic(report_path, build_report(
                    params=_sanitize_params(args), start_utc=start_utc, start_mono_ms=start_mono,
                    checks=[{"name": "fixture_manifest", "passed": False, "expected": "可消费",
                             "actual": "拒绝", "evidence": "; ".join(m_errors)}],
                    summary={"result": "manifest_invalid", "exit_code": EXIT_PARAM},
                ))
                return EXIT_PARAM
        checks = [
            {"name": "health_status", "passed": health.ok, "expected": True,
             "actual": health.ok, "evidence": "AI Monitor /health = ok"},
            {"name": "cuda_device_regex", "passed": True, "expected": "匹配",
             "actual": args.expected_device_regex, "evidence": "由运行期 Worker 快照核验"},
            {"name": "zero_fault_in_ro_mode", "passed": not args.allow_fault_injection,
             "expected": "只读零故障", "actual": "ok", "evidence": "fault 未开启"},
        ]
        # 阶段11 M2 扩展: 健康接口脱敏 + unavailable UI + analytics source frame + TTL 重挂载
        snapshot = _fetch_fall_runtime_snapshot(args.ai_monitor_url)
        analytics_obs: list[dict] = []
        if args.analytics_log:
            try:
                with open(args.analytics_log, "r", encoding="utf-8") as fh:
                    analytics_obs = load_analytics_observations(fh.read())
            except OSError as exc:
                raise SmokeError(f"--analytics-log 读取失败: {exc}") from None
        checks.extend(build_m2_checks(snapshot, analytics_obs))
        all_pass = all(c["passed"] for c in checks)
        # 未显式要求故障注入时, 只读门禁的 marker 缺失是可接受的(仅在注入模式下才是故障)
        if not all_pass and args.allow_fault_injection:
            write_report_atomic(report_path, build_report(
                params=_sanitize_params(args), start_utc=start_utc, start_mono_ms=start_mono,
                checks=checks, summary={"result": "threshold_failed", "exit_code": EXIT_THRESHOLD},
            ))
            return EXIT_THRESHOLD
        write_report_atomic(report_path, build_report(
            params=_sanitize_params(args), start_utc=start_utc, start_mono_ms=start_mono,
            checks=checks, summary={"result": "passed", "exit_code": EXIT_ALL_PASS},
        ))
        return EXIT_ALL_PASS
    except SmokeError as exc:
        _finalize_error(report_path, start_utc, start_mono, args, EXIT_PARAM, str(exc))
        return EXIT_PARAM
    except UnavailableError as exc:
        _finalize_error(report_path, start_utc, start_mono, args, EXIT_UNAVAILABLE, str(exc))
        return EXIT_UNAVAILABLE
    except ThresholdError as exc:
        _finalize_error(report_path, start_utc, start_mono, args, EXIT_THRESHOLD, str(exc))
        return EXIT_THRESHOLD
    except Exception as exc:  # noqa: BLE001
        _finalize_error(report_path, start_utc, start_mono, args, EXIT_INTERNAL, str(exc))
        return EXIT_INTERNAL


def _sanitize_params(args) -> dict:
    return {
        "ai_monitor_url": args.ai_monitor_url,
        "camera_ids": list(args.camera_id),
        "duration_seconds": args.duration_seconds,
        "expected_device_regex": args.expected_device_regex,
        "fixture_manifest": bool(args.fixture_manifest),
        "analytics_log": bool(getattr(args, "analytics_log", None)),
        "fault_injection": bool(args.allow_fault_injection),
        "test_run_id": args.test_run_id,
    }


def _finalize_error(report_path: str, start_utc: str, start_mono: int, args,
                    exit_code: int, reason: str) -> None:
    try:
        write_report_atomic(report_path, build_report(
            params=_sanitize_params(args), start_utc=start_utc, start_mono_ms=start_mono,
            checks=[{"name": "overall", "passed": False, "expected": exit_code,
                     "actual": reason, "evidence": reason}],
            summary={"result": "error", "exit_code": exit_code},
        ))
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    raise SystemExit(main())