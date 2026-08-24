"""阶段3：latest-only 公平调度器测试。"""
from __future__ import annotations

from ai_monitor_pose.scheduler import FallScheduler, OfferOutcome


def _sched():
    return FallScheduler(target_fps=8, batch_size=1)


def test_latest_frame_replaces_older_pending_same_camera() -> None:
    s = _sched()
    s.register_camera("cam-a")
    s.register_camera("cam-b")
    now = int(1e12)
    assert s.offer("cam-a", frame_id=1, generation=1, now_ns=now) is OfferOutcome.ACCEPTED
    assert s.offer("cam-a", frame_id=2, generation=2, now_ns=now) is OfferOutcome.REPLACED_OLDER_FRAME
    assert s.offer("cam-a", frame_id=3, generation=3, now_ns=now) is OfferOutcome.REPLACED_OLDER_FRAME
    assert s.offer("cam-b", frame_id=1, generation=1, now_ns=now) is OfferOutcome.ACCEPTED

    seen = set()
    while len(seen) < 2:
        d = s.pick(now)
        assert d is not None
        cam, gen = d
        assert (cam, gen) not in seen
        seen.add((cam, gen))
        s.complete(cam, gen, ok=True)
    assert ("cam-a", 3) in seen   # A 最新帧被保留
    assert ("cam-b", 1) in seen
    assert all((c, g) not in seen for c, g in [("cam-a", 1), ("cam-a", 2)])


def test_duplicate_and_out_of_order_frames_are_rejected() -> None:
    s = _sched()
    s.register_camera("cam-c")
    now = int(2e12)
    assert s.offer("cam-c", 5, 5, now) is OfferOutcome.ACCEPTED
    # 消费掉第一个 pending，避免后续被 latest-only 判定为覆盖
    d = s.pick(now)
    assert d == ("cam-c", 5)
    s.complete("cam-c", 5, ok=True)
    assert s.offer("cam-c", 5, 6, now) is OfferOutcome.DUPLICATE_FRAME
    assert s.offer("cam-c", 4, 7, now) is OfferOutcome.STALE_FRAME
    assert s.offer("cam-c", 6, 8, now) is OfferOutcome.ACCEPTED


def test_round_robin_prevents_busy_camera_starvation() -> None:
    s = _sched()
    s.register_camera("busy")
    s.register_camera("quiet")
    now0 = int(3e12)
    dispatched = {"busy": 0, "quiet": 0}
    for i in range(40):
        # 每轮向两个摄像头各供给一帧（latest-only 会逐帧覆盖）
        s.offer("busy", i, i, now0)
        s.offer("quiet", i, i, now0)
        # 每轮最多服务两帧（每个摄像头一轮一次）
        for _ in range(2):
            d = s.pick(now0)
            if d is None:
                break
            cam, gen = d
            dispatched[cam] += 1
            s.complete(cam, gen, ok=True)
    # 高频源不能垄断：安静摄像头也应获得合理份额
    assert dispatched["busy"] >= 15
    assert dispatched["quiet"] >= 15
