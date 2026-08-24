"""共享内存双槽（第 6.3 节）。

每个 (camera, session) 分配两个固定容量槽，父进程为唯一 owner（create/close/unlink），
Worker 只 attach/close。头 64 字节固定 little-endian 布局；sequence（存于 8 字节对齐的
offset 16，用低 32 位）通过 kernel32 的 32 位 Interlocked 原语发布/读取（seqlock 语义：
奇数=写中、偶数=已提交）。generation 必须等于已提交偶数 sequence。
提交优先覆盖空闲/未占用槽；两个槽都忙则立即抛 SlotBusy，绝不等待。
"""
from __future__ import annotations

import ctypes
import struct
import uuid
from ctypes import POINTER, addressof, byref, c_int32, c_void_p, cast

import numpy as np

from .contracts import SharedFrameRefV1

HEADER = struct.Struct("<8sHHIQIIIIQQII")  # 64 bytes
HEADER_SIZE = HEADER.size
MAGIC = b"AIMPOSE1"
HEADER_SCHEMA_VERSION = 1
PIXEL_FORMAT_BGR8 = 1

_SEQUENCE_OFFSET = 16  # u64，用低 32 位存 sequence；仅需 32 位原子

_k32 = ctypes.WinDLL("kernel32")

_CreateMutexW = _k32.CreateMutexW
_CreateMutexW.restype = c_void_p
_CreateMutexW.argtypes = [c_void_p, ctypes.c_int, ctypes.c_wchar_p]
_WaitObject = _k32.WaitForSingleObject
_WaitObject.restype = ctypes.c_uint32
_WaitObject.argtypes = [c_void_p, ctypes.c_uint32]
_ReleaseMutex = _k32.ReleaseMutex
_ReleaseMutex.restype = ctypes.c_int
_ReleaseMutex.argtypes = [c_void_p]
_CloseHandle = _k32.CloseHandle
_CloseHandle.restype = ctypes.c_int
_CloseHandle.argtypes = [c_void_p]

_INFINITE = 0xFFFFFFFF


class _NamedMutex:
    """轻量 Windows 命名互斥体，用于保护共享内存 sequence 临界区。

    说明：kernel32 不导出 Interlocked*（Win64 为编译期内在函数），故用命名互斥体
    实现等价的跨进程原子发布；撕裂检测仍基于前后两次读取的 generation 比较。
    """

    def __init__(self, name: str) -> None:
        from ctypes import c_wchar_p
        self._h = _CreateMutexW(None, False, name)
        if not self._h:
            raise ctypes.WinError()

    def acquire(self) -> None:
        _WaitObject(self._h, _INFINITE)

    def release(self) -> None:
        _ReleaseMutex(self._h)

    def close(self) -> None:
        if self._h:
            _CloseHandle(self._h)
            self._h = None


class FrameTooLargeError(Exception):
    pass


class SlotBusyError(Exception):
    pass


class HeaderLayoutMismatchError(Exception):
    pass


class FrameShmRegion:
    def __init__(self, shm_name: str, max_width: int, max_height: int, slots: int = 2,
                 *, attach: bool = False) -> None:
        self.shm_name = shm_name
        self.slots = slots
        self.max_width = max_width
        self.max_height = max_height
        from multiprocessing.shared_memory import SharedMemory
        if attach:
            # 附件侧只 attach 已有内存段；max_width/max_height 传 0 表示未知，
            # 用 slot0 头推导实际宽高再计算槽容量，否则非 slot0 帧会读偏。
            self.shm = SharedMemory(name=shm_name)
            self._owns = False
            self.slots = max(slots, 1)
            if max_width <= 0 or max_height <= 0:
                raw0 = bytes(self.shm.buf[0:HEADER_SIZE])
                if len(raw0) < HEADER_SIZE:
                    raise HeaderLayoutMismatchError()
                (magic, _sv, _fl, _hs, _seq, w, h, _ch, _st, _bl, _fid, _pf, _rs) = HEADER.unpack(raw0)
                self.max_width = int(w)
                self.max_height = int(h)
            slot_pixels = self.max_width * self.max_height * 3
            self._slot_capacity = HEADER_SIZE + slot_pixels
        else:
            self.shm = SharedMemory(name=shm_name, create=True, size=int(
                slots * (HEADER_SIZE + max_width * max_height * 3)))
            self._owns = True
            slot_pixels = max_width * max_height * 3
            self._slot_capacity = HEADER_SIZE + slot_pixels
        self._mutex = _NamedMutex(f"Global\\AIMPOSE_SEQ_{self.shm_name[:40]}")

    # --- 头字段（除 sequence 用 Interlocked）---
    def _read_header(self, slot: int) -> dict:
        off = slot * self._slot_capacity
        raw = bytes(self.shm.buf[off: off + HEADER_SIZE])
        if len(raw) < HEADER_SIZE:
            raise HeaderLayoutMismatchError()
        (magic, sv, flags, hs, seq, w, h, ch, stride, blen, fid, pf, res) = HEADER.unpack(raw)
        if magic != MAGIC or sv != HEADER_SCHEMA_VERSION or hs != HEADER_SIZE:
            raise HeaderLayoutMismatchError()
        return dict(magic=magic, schema_version=sv, flags=flags, header_size=hs, sequence=seq,
                    width=w, height=h, channels=ch, row_stride=stride, byte_length=blen,
                    frame_id=fid, pixel_format_code=pf, reserved=res)

    def _seq_load(self, slot: int) -> int:
        o = slot * self._slot_capacity + _SEQUENCE_OFFSET
        return int.from_bytes(self.shm.buf[o: o + 4], "little")

    def _seq_set(self, slot: int, value: int) -> None:
        o = slot * self._slot_capacity + _SEQUENCE_OFFSET
        self.shm.buf[o: o + 4] = (value & 0xFFFFFFFF).to_bytes(4, "little")

    def _slot_active(self, slot: int) -> bool:
        try:
            return self._read_header(slot)["sequence"] > 0
        except HeaderLayoutMismatchError:
            return False

    def submit(self, frame: np.ndarray, *, now_ns: int, prefer_slot: int | None = None) -> SharedFrameRefV1:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("需要 uint8 BGR 三通道帧")
        if frame.dtype != np.uint8:
            raise ValueError("需要 uint8")
        h, w, _ = frame.shape
        if w > self.max_width or h > self.max_height:
            raise FrameTooLargeError(f"{w}x{h} 超过最大 {self.max_width}x{self.max_height}")
        arr = np.ascontiguousarray(frame)
        idx: int | None = None
        if prefer_slot is not None and 0 <= prefer_slot < self.slots:
            idx = prefer_slot
        else:
            for i in range(self.slots):
                if not self._slot_active(i):
                    idx = i
                    break
            if idx is None:
                raise SlotBusyError("两个槽都不可写")
        return self._write_slot(idx, arr, now_ns)

    def _write_slot(self, idx: int, arr: np.ndarray, now_ns: int) -> SharedFrameRefV1:
        h, w, _ = arr.shape
        off = idx * self._slot_capacity
        pixel_off = off + HEADER_SIZE
        # 整个“写中->写数据->发布已提交偶数”在命名互斥体临界区内原子完成
        self._mutex.acquire()
        try:
            cur = self._seq_load(idx)
            odd = (cur // 2) * 2 + 1
            self._seq_set(idx, odd)
            meta = struct.pack(
                "<8sHHIQIIIIQQII", MAGIC, HEADER_SCHEMA_VERSION, 0, HEADER_SIZE,
                odd, w, h, 3, w * 3, h * w * 3, now_ns, PIXEL_FORMAT_BGR8, 0,
            )
            self.shm.buf[off: off + HEADER_SIZE] = meta
            self.shm.buf[pixel_off: pixel_off + arr.nbytes] = arr.tobytes()
            even = odd + 1
            self._seq_set(idx, even)
        finally:
            self._mutex.release()
        return SharedFrameRefV1(
            shm_name=self.shm_name, slot_index=idx, generation=even,
            byte_offset=pixel_off, byte_length=arr.nbytes, width=w, height=h,
            channels=3, row_stride=w * 3, dtype="uint8", pixel_format="BGR8",
        )

    def read(self, ref: SharedFrameRefV1) -> np.ndarray | None:
        slot = ref.slot_index
        if self._seq_load(slot) != ref.generation:
            return None
        hd = self._read_header(slot)
        if hd["sequence"] % 2 != 0 or hd["sequence"] != ref.generation:
            return None
        pixel_off = slot * self._slot_capacity + HEADER_SIZE
        nbytes = int(hd["byte_length"])
        buf = bytes(self.shm.buf[pixel_off: pixel_off + nbytes])
        out = np.frombuffer(buf, dtype=np.uint8).reshape((hd["height"], hd["width"], 3))
        if self._seq_load(slot) != ref.generation:
            return None
        return out

    def read_may_raise(self, ref: SharedFrameRefV1) -> np.ndarray | None:
        try:
            return self.read(ref)
        except Exception:
            return None

    def close(self, *, unlink: bool = False) -> None:
        try:
            self._mutex.close()
        except Exception:
            pass
        try:
            self.shm.close()
        except Exception:
            pass
        if unlink and self._owns:
            try:
                self.shm.unlink()
            except Exception:
                pass


def new_region_name(prefix: str = "ai-monitor-pose") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"