
"""机器级命名互斥体（第 6.2 / worker_lock）。

用 Win32 CreateMutexW 获取机器级 `Global\\AI_MONITOR_POSE_CUDA_<index>`；
禁止退化为仅当前 Session 的 Local 命名空间。mutex 由 Worker 持有，进程退出/崩溃由
操作系统自动释放。
"""
from __future__ import annotations

import ctypes

_k32 = ctypes.WinDLL("kernel32")
_CreateMutexW = _k32.CreateMutexW
_CreateMutexW.restype = ctypes.c_void_p
_CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
_ERROR_ALREADY_EXISTS = 183
_CloseHandle = _k32.CloseHandle


class WorkerLockHeldError(Exception):
    pass


class WorkerLockPermissionError(Exception):
    pass


def make_lock_name(device_index: int) -> str:
    return f"Global\\AI_MONITOR_POSE_CUDA_{device_index}"


class GpuWorkerLock:
    def __init__(self, device_index: int) -> None:
        self.name = make_lock_name(device_index)
        self._h = _CreateMutexW(None, False, self.name)
        if not self._h:
            # ACCESS_DENIED 等 -> 权限
            raise WorkerLockPermissionError(self.name)
        if ctypes.windll.kernel32.GetLastError() == _ERROR_ALREADY_EXISTS:
            _CloseHandle(self._h)
            self._h = None
            raise WorkerLockHeldError(self.name)

    def release(self) -> None:
        if self._h:
            _CloseHandle(self._h)
            self._h = None
