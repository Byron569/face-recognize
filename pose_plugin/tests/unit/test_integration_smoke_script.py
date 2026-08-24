"""阶段9 M1:integration_smoke.py 破坏性注入门禁单元测试。

必须证明以下场景零故障动作发生(guard 拒绝 => 不执行任何 kill/断 Pipe/DB 故障):
    公网 / 局域网 URL、非 _test DB、错 run id、普通 camera、无显式开关。
以及全部门禁满足时才放行故障注入。is_loopback_url 单独覆盖主机判定。
"""
from __future__ import annotations

import pytest

from scripts.integration_smoke import (
    FaultGuardInput,
    HealthProbe,
    evaluate_fault_guard,
    is_loopback_url,
)

_OK = HealthProbe(ok=True, test_mode=True, test_run_id="aaaaaaaa-bbbb-cccc-dddd-000000000001")


def _probe(**overrides):
    base = dict(ok=True, test_mode=True, test_run_id=_OK.test_run_id)
    base.update(overrides)
    return HealthProbe(**base)


def _cfg(**overrides):
    c = dict(
        url="http://127.0.0.1:8000",
        allow_fault_injection=True,
        test_run_id_cfg=_OK.test_run_id,
        camera_ids=("test-cam-1",),
        health=_OK,
        db_name="ai_monitor_test",
    )
    c.update(overrides)
    return FaultGuardInput(**c)


# 零故障动作: 每个拒绝场景 guard 都返回 (False, reason), 调用方不执行任何故障动作。

def test_rejected_when_no_explicit_switch():
    assert evaluate_fault_guard(_cfg(allow_fault_injection=False))[0] is False


def test_rejected_when_missing_run_id():
    assert evaluate_fault_guard(_cfg(test_run_id_cfg=""))[0] is False


def test_rejected_when_public_url():
    assert evaluate_fault_guard(_cfg(url="https://example.com"))[0] is False
    assert evaluate_fault_guard(_cfg(url="http://203.0.113.8:8000"))[0] is False


def test_rejected_when_lan_url():
    assert evaluate_fault_guard(_cfg(url="http://192.168.1.10:8000"))[0] is False
    assert evaluate_fault_guard(_cfg(url="http://10.0.0.5:8000"))[0] is False
    assert evaluate_fault_guard(_cfg(url="http://172.16.5.5:8000"))[0] is False


def test_rejected_when_non_test_db():
    assert evaluate_fault_guard(_cfg(db_name="ai_monitor"))[0] is False


def test_rejected_when_wrong_run_id():
    h = _probe(test_run_id="wrong-run-id")
    assert evaluate_fault_guard(_cfg(health=h))[0] is False


def test_rejected_when_health_not_test_mode():
    h = _probe(test_mode=False)
    assert evaluate_fault_guard(_cfg(health=h))[0] is False


def test_rejected_when_normal_camera():
    assert evaluate_fault_guard(_cfg(camera_ids=("cam-main",)))[0] is False


def test_rejected_when_empty_camera_list():
    assert evaluate_fault_guard(_cfg(camera_ids=()))[0] is False


def test_rejected_reason_is_nonempty_on_denial():
    ok, reason = evaluate_fault_guard(_cfg(url="https://example.com"))
    assert ok is False and len(reason) > 0


# 全通过才放行故障注入

def test_allowed_when_every_gate_passes():
    ok, reason = evaluate_fault_guard(_cfg())
    assert ok is True and reason == ""


def test_loopback_url_host_rules():
    assert is_loopback_url("http://localhost:8000") is True
    assert is_loopback_url("http://127.0.0.1:8000") is True
    assert is_loopback_url("http://127.0.0.2:9000") is True
    assert is_loopback_url("http://[::1]:8000") is True
    assert is_loopback_url("https://example.com") is False
    assert is_loopback_url("http://192.168.1.10:8000") is False
    assert is_loopback_url("http://172.16.5.5:8000") is False
    assert is_loopback_url("http://10.0.0.5:8000") is False
    assert is_loopback_url("http://8.8.8.8:8000") is False
    assert is_loopback_url("http://203.0.113.9:8000") is False


# 零故障动作: 拒绝即不执行任何故障步骤(以 guard 为唯一门禁)

@pytest.mark.parametrize("bad", [
    dict(url="https://example.com"),
    dict(db_name="ai_monitor"),
    dict(camera_ids=("cam-main",)),
    dict(allow_fault_injection=False),
    dict(test_run_id_cfg=""),
])
def test_zero_fault_actions_when_guard_rejects(bad):
    ok, _ = evaluate_fault_guard(_cfg(**bad))
    # 门禁拒绝 => 故障注入步骤不会执行(步骤集为空)
    actions = _collect_fault_actions(ok)
    assert ok is False
    assert actions == []


def _collect_fault_actions(allowed: bool):
    """仅当门禁允许时才产生破坏性动作; 拒绝则空。"""
    if not allowed:
        return []
    return ["kill_worker", "db_failure_inject", "break_pipe"]
from scripts.integration_smoke import (
    ManifestSegment,
    SmokeError,
    evaluate_event_sequence,
    evaluate_window_coverage,
    load_manifest,
    sha256_file,
    validate_manifest_segments,
)


# ── manifest 解析 / 校验 / 事件序列 / 时间窗(阶段10 扩展) ──

def _write_video(tmp_path, data):
    p = tmp_path / "clip.mp4"
    p.write_bytes(data)
    return p, sha256_file(str(p))


def test_load_manifest_valid():
    segs = load_manifest(
        '[{"camera_id":"test-cam-a","video_path":"x.mp4",'
        '"sha256":"' + "0"*64 + '","expected_events":["fall_potential"],"allowed_window_s":5}]'
    )
    assert len(segs) == 1
    assert segs[0].camera_id == "test-cam-a"
    assert segs[0].expected_events == ("fall_potential",)
    assert segs[0].allowed_window_s == 5.0


def test_load_manifest_rejects_bad_shape():
    import pytest
    with pytest.raises(SmokeError):
        load_manifest('{"a":1}')                                     # 非数组
    with pytest.raises(SmokeError):
        load_manifest('[{"video_path":"x"}]')                        # 缺 camera_id
    with pytest.raises(SmokeError):
        load_manifest('[{"camera_id":"c","sha256":"short"}]')         # sha 长度
    with pytest.raises(SmokeError):
        load_manifest('[{"camera_id":"c","sha256":"' + "0"*64 + '","expected_events":[]}]')  # 空序列
    with pytest.raises(SmokeError):
        load_manifest('[]')                                          # 空清单


def test_validate_manifest_segments_shasum(tmp_path):
    p, correct = _write_video(tmp_path, b"video-bytes")
    good = ManifestSegment("test-cam-a", str(p), correct, ("fall_potential",), 4.0)
    ok, errs = validate_manifest_segments([good])
    assert ok is True and errs == []

    bad_sha = ManifestSegment("test-cam-a", str(p), "0"*64, ("fallen",), 4.0)
    ok, errs = validate_manifest_segments([bad_sha])
    assert ok is False and any("sha256" in e for e in errs)


def test_validate_manifest_segments_takes_missing_or_prefix(tmp_path):
    missing = ManifestSegment("test-cam-a", str(tmp_path/"nope.mp4"), "0"*64, ("fallen",), 4.0)
    ok, errs = validate_manifest_segments([missing])
    assert ok is False and any("不存在" in e for e in errs)

    non_test = ManifestSegment("cam-main", str(tmp_path/"nope.mp4"), "0"*64, ("fallen",), 4.0)
    ok, errs = validate_manifest_segments([non_test])
    assert ok is False and any("test-" in e for e in errs)


def test_evaluate_event_sequence_order():
    ok, _ = evaluate_event_sequence(["fall_potential", "fallen", "fall_potential"], ("fall_potential", "fallen"))
    assert ok is True
    ok, why = evaluate_event_sequence(["fallen", "fall_potential"], ("fall_potential", "fallen"))
    assert ok is False and "fallen" in why
    ok, _ = evaluate_event_sequence([], ("fall_potential",))
    assert ok is False
    ok, _ = evaluate_event_sequence(["fall_potential"], ("fall_potential",))
    assert ok is True


def test_evaluate_window_coverage():
    ok, _ = evaluate_window_coverage([1000.0], 1000.0, 2.0)
    assert ok is True
    ok, _ = evaluate_window_coverage([9000.0], 1000.0, 2.0)
    assert ok is False
    ok, _ = evaluate_window_coverage([], 1000.0, 2.0)
    assert ok is False


# ── 阶段11 M2 扩展: analytics source frame / 缓存 TTL 重挂载 / unavailable UI / 健康脱敏 ──

from scripts.integration_smoke import (
    build_m2_checks,
    evaluate_analytics_source_frame,
    evaluate_health_redaction,
    evaluate_overlay_ttl_remount,
    evaluate_unavailable_ui_semantics,
    load_analytics_observations,
)


def _obs(ts=0.0, session="sess-1", preview=1, source=10, expires=500.0,
         health="normal", result_age=120.0, state="normal"):
    """一条注入的 analytics 观测: wire 消息字段 + 观测时刻 ts_ms。"""
    return {
        "ts_ms": ts,
        "camera_session_id": session,
        "preview_frame_id": preview,
        "fall_detection": {
            "camera_session_id": session,
            "schema_version": 1,
            "source_frame_id": source,
            "preview_width": 640,
            "preview_height": 480,
            "transform": {"scale_x": 0.5, "scale_y": 0.5},
            "overlay_expires_in_ms": expires,
            "result_age_ms": result_age,
            "health": health,
            "tracks": [{"pose_track_id": 1, "state": state, "bbox": [1, 2, 3, 4]}],
        },
    }


# analytics source frame 契约(对照 frontend analyticsProtocol.ts)

def test_analytics_source_frame_ok():
    ok, _ = evaluate_analytics_source_frame([_obs(), _obs(ts=300.0, preview=2, source=11)])
    assert ok is True


def test_analytics_source_frame_empty_fails():
    ok, why = evaluate_analytics_source_frame([])
    assert ok is False and len(why) > 0


def test_analytics_source_frame_missing_source_id_fails():
    o = _obs()
    o["fall_detection"]["source_frame_id"] = None
    ok, why = evaluate_analytics_source_frame([o])
    assert ok is False and "source_frame_id" in why


def test_analytics_source_frame_session_mismatch_fails():
    o = _obs()
    o["fall_detection"]["camera_session_id"] = "sess-other"
    ok, why = evaluate_analytics_source_frame([o])
    assert ok is False and "session" in why


def test_analytics_source_frame_bad_dims_or_transform_fails():
    o = _obs()
    o["fall_detection"]["preview_width"] = 0
    assert evaluate_analytics_source_frame([o])[0] is False

    o = _obs()
    o["fall_detection"]["transform"] = {"scale_x": 0.0, "scale_y": 0.5}
    assert evaluate_analytics_source_frame([o])[0] is False

    o = _obs()
    o["preview_frame_id"] = -1
    assert evaluate_analytics_source_frame([o])[0] is False


def test_analytics_source_frame_unknown_state_fails():
    o = _obs(state="jumping")
    ok, why = evaluate_analytics_source_frame([o])
    assert ok is False and "state" in why.lower()


def test_analytics_source_frame_bad_expiry_fails():
    o = _obs()
    o["fall_detection"]["overlay_expires_in_ms"] = 0
    assert evaluate_analytics_source_frame([o])[0] is False


# 缓存重挂载: TTL 过期后必须重新挂载到更新的 source frame, 不得超期用陈旧结果

def test_overlay_ttl_remount_after_expiry_uses_fresh_source():
    obs = [_obs(ts=0.0, source=10, expires=500.0),
           _obs(ts=2000.0, preview=2, source=12, expires=500.0)]
    ok, why = evaluate_overlay_ttl_remount(obs)
    assert ok is True, why


def test_overlay_ttl_stale_remount_after_expiry_fails():
    obs = [_obs(ts=0.0, source=10, expires=500.0),
           _obs(ts=2000.0, preview=2, source=10, expires=500.0)]
    ok, why = evaluate_overlay_ttl_remount(obs)
    assert ok is False and "重挂载" in why


def test_overlay_ttl_older_remount_after_expiry_fails():
    obs = [_obs(ts=0.0, source=10, expires=500.0),
           _obs(ts=2000.0, preview=2, source=9, expires=500.0)]
    assert evaluate_overlay_ttl_remount(obs)[0] is False


def test_overlay_ttl_within_ttl_allows_same_source():
    obs = [_obs(ts=0.0, source=10, expires=500.0),
           _obs(ts=300.0, preview=2, source=10, expires=500.0)]
    ok, _ = evaluate_overlay_ttl_remount(obs)
    assert ok is True


def test_overlay_ttl_result_age_beyond_expiry_fails():
    obs = [_obs(ts=0.0, source=10, expires=500.0, result_age=900.0)]
    ok, why = evaluate_overlay_ttl_remount(obs)
    assert ok is False and "陈旧" in why


def test_overlay_ttl_out_of_order_fails():
    obs = [_obs(ts=2000.0, source=10), _obs(ts=1000.0, preview=2, source=11)]
    ok, _ = evaluate_overlay_ttl_remount(obs)
    assert ok is False


def test_overlay_ttl_empty_fails():
    assert evaluate_overlay_ttl_remount([])[0] is False


def test_overlay_ttl_session_switch_is_fresh_mount_boundary():
    obs = [_obs(ts=0.0, session="sess-1", source=10),
           _obs(ts=50.0, session="sess-2", preview=1, source=1)]
    ok, _ = evaluate_overlay_ttl_remount(obs)
    assert ok is True


# unavailable UI 语义: 运行不可用时不得呈现假 Normal

def test_unavailable_ui_no_fake_normal():
    ok, _ = evaluate_unavailable_ui_semantics("UNAVAILABLE", ["unavailable", None])
    assert ok is True
    ok, why = evaluate_unavailable_ui_semantics("UNAVAILABLE", ["unavailable", "Normal"])
    assert ok is False and "Normal" in why


def test_unavailable_ui_degraded_and_disabled_also_forbid_normal():
    assert evaluate_unavailable_ui_semantics("DEGRADED", ["normal"])[0] is False
    assert evaluate_unavailable_ui_semantics("DISABLED", ["normal"])[0] is False


def test_unavailable_ui_ready_allows_normal():
    assert evaluate_unavailable_ui_semantics("READY", ["normal"])[0] is True


def test_unavailable_ui_no_observations_passes():
    assert evaluate_unavailable_ui_semantics("UNAVAILABLE", [])[0] is True


def test_unavailable_ui_unknown_state_fails():
    ok, why = evaluate_unavailable_ui_semantics("SUPER", ["normal"])
    assert ok is False and "state" in why.lower()


def test_unavailable_ui_lowercase_state_accepted():
    assert evaluate_unavailable_ui_semantics("unavailable", ["unavailable"])[0] is True
    assert evaluate_unavailable_ui_semantics("ready", ["normal"])[0] is True


# 健康接口脱敏(对照 backend fall_runtime_health.py 的白名单契约)

def _clean_snapshot():
    """与 backend build_fall_runtime_response 输出同构的干净快照。"""
    return {
        "schema_version": 1,
        "enabled": True,
        "mode": "yolov8_pose",
        "runtime_key": "pose-default",
        "state": "READY",
        "error": None,
        "worker": {"pid": 1234, "epoch": 3, "restart_count": 1, "heartbeat_age_ms": 42.0},
        "gpu": {"device": "cuda:0", "name": "RTX", "allocated_mb": 512, "reserved_mb": 640,
                "effective_limit_mb": 8192},
        "model": {"name": "yolov8s-pose.pt", "sha256": "ab" * 32, "precision": "fp16"},
        "delivery": {"transition_queue_depth": 0, "spool_pending": 0, "oldest_pending_age_ms": 3.0},
        "cameras": [{"camera_id": "test-cam-1", "camera_session_id": "sess-1", "state": "ANALYZING",
                     "submitted": 100, "analyzed": 98, "replaced": 2, "stale": 0,
                     "effective_fps": 15.0, "latest_result_age_ms": 60.0,
                     "transition_queue_depth": 0, "open_incidents": 0}],
    }


def test_health_redaction_clean_snapshot_passes():
    ok, why = evaluate_health_redaction(_clean_snapshot())
    assert ok is True, why


def test_health_redaction_accepts_raw_json_text():
    import json
    ok, _ = evaluate_health_redaction(json.dumps(_clean_snapshot()))
    assert ok is True


def test_health_redaction_rejects_pipe_authkey():
    snap = _clean_snapshot()
    snap["worker"]["authkey"] = "s3cr3t-bytes"
    ok, why = evaluate_health_redaction(snap)
    assert ok is False and "authkey" in why

    ok, _ = evaluate_health_redaction('{"worker": {"auth_key": "abc"}}')
    assert ok is False


def test_health_redaction_rejects_db_url():
    ok, _ = evaluate_health_redaction('{"error": "postgresql://user:pw@db:5432/ai_monitor"}')
    assert ok is False
    ok, _ = evaluate_health_redaction('{"extra": "asyncpg://u@h/db_test"}')
    assert ok is False


def test_health_redaction_rejects_config_paths():
    ok, why = evaluate_health_redaction('{"model": {"name": "configs/runtime.yaml"}}')
    assert ok is False
    ok, _ = evaluate_health_redaction('{"runtime_key": "D:\\\\pose\\\\runtime.yml"}')
    assert ok is False


def test_health_redaction_rejects_exception_stack():
    ok, _ = evaluate_health_redaction('{"error": "Traceback (most recent call last): boom"}')
    assert ok is False
    ok, _ = evaluate_health_redaction('{"error": "File \\"worker/launcher.py\\", line 1"}')
    assert ok is False


def test_health_redaction_rejects_non_whitelisted_keys():
    snap = _clean_snapshot()
    snap["worker"]["cmdline"] = "python worker/launcher.py --config p.yaml"
    ok, why = evaluate_health_redaction(snap)
    assert ok is False and "worker" in why

    snap = _clean_snapshot()
    snap["debug_stack"] = "..."
    assert evaluate_health_redaction(snap)[0] is False

    snap = _clean_snapshot()
    snap["cameras"][0]["db_url"] = "postgresql://x"
    assert evaluate_health_redaction(snap)[0] is False


# 观测加载(JSONL, 纯函数) + M2 check 组装

def test_load_analytics_observations_flat_and_wrapped():
    flat = '{"ts_ms": 10, "camera_session_id": "s", "preview_frame_id": 1, "fall_detection": {"camera_session_id": "s", "source_frame_id": 5, "preview_width": 640, "preview_height": 480, "transform": {"scale_x": 1, "scale_y": 1}, "overlay_expires_in_ms": 500}}'
    wrapped = '{"ts_ms": 20, "message": {"camera_session_id": "s", "preview_frame_id": 2, "fall_detection": {"camera_session_id": "s", "source_frame_id": 6, "preview_width": 640, "preview_height": 480, "transform": {"scale_x": 1, "scale_y": 1}, "overlay_expires_in_ms": 500}}}'
    obs = load_analytics_observations(flat + "\n" + wrapped + "\n\n")
    assert len(obs) == 2
    assert obs[0]["ts_ms"] == 10 and obs[0]["preview_frame_id"] == 1
    assert obs[1]["ts_ms"] == 20 and obs[1]["preview_frame_id"] == 2


def test_load_analytics_observations_rejects_bad_lines():
    with pytest.raises(SmokeError):
        load_analytics_observations("not-json")
    with pytest.raises(SmokeError):
        load_analytics_observations('{"no_ts": true}')
    with pytest.raises(SmokeError):
        load_analytics_observations('[]')


def test_build_m2_checks_all_pass_and_names():
    obs = [_obs(ts=0.0, source=10, health="normal"),
           _obs(ts=2000.0, preview=2, source=12, health="normal")]
    checks = build_m2_checks(_clean_snapshot(), obs)
    names = {c["name"] for c in checks}
    assert {"m2_health_redaction", "m2_unavailable_ui_semantics",
            "m2_analytics_source_frame", "m2_overlay_ttl_remount"} <= names
    assert all(c["passed"] for c in checks)


def test_build_m2_checks_fail_when_snapshot_leaks_or_analytics_bad():
    leak = _clean_snapshot()
    leak["error"] = "Traceback (most recent call last)"
    checks = build_m2_checks(leak, [_obs()])
    by_name = {c["name"]: c for c in checks}
    assert by_name["m2_health_redaction"]["passed"] is False

    checks = build_m2_checks(_clean_snapshot(), [])
    by_name = {c["name"]: c for c in checks}
    assert by_name["m2_analytics_source_frame"]["passed"] is False

    stale = [_obs(ts=0.0, source=10, expires=500.0),
             _obs(ts=5000.0, preview=2, source=10, expires=500.0)]
    checks = build_m2_checks(_clean_snapshot(), stale)
    by_name = {c["name"]: c for c in checks}
    assert by_name["m2_overlay_ttl_remount"]["passed"] is False


def test_build_m2_checks_unavailable_state_with_fake_normal_analytics():
    snap = _clean_snapshot()
    snap["state"] = "UNAVAILABLE"
    checks = build_m2_checks(snap, [_obs(health="normal")])
    by_name = {c["name"]: c for c in checks}
    assert by_name["m2_unavailable_ui_semantics"]["passed"] is False