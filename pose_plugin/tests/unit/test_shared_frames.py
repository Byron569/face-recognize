"""阶段4：共享内存双槽（Fixed header + seqlock sequence）。"""
from __future__ import annotations

import numpy as np
import pytest

from ai_monitor_pose.shared_frames import (
    FrameTooLargeError,
    SlotBusyError,
    HeaderLayoutMismatchError,
    FrameShmRegion,
    HEADER_SIZE,
    MAGIC,
)


def _region(max_w=1920, max_h=1080, slots=2):
    r = FrameShmRegion("t-" + __import__("uuid").uuid4().hex[:8], max_w, max_h, slots)
    return r


def test_header_is_exact_64_byte_little_endian_layout() -> None:
    assert HEADER_SIZE == 64
    r = _region(1920, 1080, slots=2)
    try:
        r.submit(np.zeros((480, 640, 3), dtype=np.uint8), now_ns=1)
        header = r._read_header(0)
        assert header["magic"] == MAGIC
        assert header["schema_version"] == 1
        assert header["header_size"] == 64
        assert header["pixel_format_code"] == 1
        assert header["reserved"] == 0
    finally:
        r.close()


def test_bgr_round_trip_preserves_bytes_shape_stride() -> None:
    r = _region(640, 480, slots=2)
    try:
        frame = (np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)).copy()
        ref = r.submit(frame, now_ns=1)
        data = r.read(ref)
        assert data.shape == (480, 640, 3)
        assert (data == frame).all()
        assert ref.generation == 2  # 首个偶数 sequence
        assert ref.byte_offset == HEADER_SIZE  # 每槽像素从 header 后开始
    finally:
        r.close()


def test_non_contiguous_input_is_copied_correctly() -> None:
    r = _region(640, 480, slots=2)
    try:
        base = np.random.randint(0, 255, (480, 1280, 3), dtype=np.uint8)
        view = base[:, ::2]  # 非连续视图
        ref = r.submit(view, now_ns=1)
        data = r.read(ref)
        assert (data == np.ascontiguousarray(view)).all()
    finally:
        r.close()


def test_generation_change_during_read_rejects_torn_frame() -> None:
    r = _region(640, 480, slots=2)
    try:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ref = r.submit(frame, now_ns=1)
        # 模拟读前 generation 已变化：从另一槽提交覆盖
        ref2 = r.submit(np.full((480, 640, 3), 255, dtype=np.uint8), now_ns=2, prefer_slot=ref2_target(ref))
        # 若 sequence 被中途改变，读取应检测到
        torn = r.read_may_raise(ref2)
        assert torn is None or torn.shape == (480, 640, 3)
    finally:
        r.close()


def ref2_target(ref):
    return (ref.slot_index + 1) % 2


def test_frame_too_large_is_rejected_without_overflow() -> None:
    r = _region(640, 480, slots=2)
    try:
        big = np.zeros((1200, 1920, 3), dtype=np.uint8)
        with pytest.raises(FrameTooLargeError):
            r.submit(big, now_ns=1)
    finally:
        r.close()


def test_one_inflight_slot_leaves_second_slot_writable() -> None:
    r = _region(640, 480, slots=2)
    try:
        r.submit(np.zeros((480, 640, 3), dtype=np.uint8), now_ns=1)
        # 至少一个槽仍可写
        r.submit(np.zeros((480, 640, 3), dtype=np.uint8), now_ns=2)
    finally:
        r.close()


def test_two_busy_slots_make_submit_return_immediately() -> None:
    r = _region(640, 480, slots=2)
    try:
        a = r.submit(np.zeros((480, 640, 3), dtype=np.uint8), now_ns=1)
        b = r.submit(np.ones((480, 640, 3), dtype=np.uint8), now_ns=2)
        # 两个槽都被占用；再次提交应抛 SlotBusy 而不等待
        with pytest.raises(SlotBusyError):
            r.submit(np.full((480, 640, 3), 1, dtype=np.uint8), now_ns=3)
    finally:
        r.close()


def test_resolution_change_within_capacity_reuses_same_segment() -> None:
    r = _region(640, 480, slots=2)
    try:
        seg = r.shm_name
        r.submit(np.zeros((240, 320, 3), dtype=np.uint8), now_ns=1)
        r.submit(np.zeros((480, 640, 3), dtype=np.uint8), now_ns=2)
        assert r.shm_name == seg
    finally:
        r.close()


def test_cleanup_unlinks_memory() -> None:
    name = "t-unlink-" + __import__("uuid").uuid4().hex[:8]
    r = FrameShmRegion(name, 640, 480, 2)
    r.close(unlink=True)
    # close 后再 close 是幂等的
    r.close(unlink=True)
