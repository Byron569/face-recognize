"""父进程 Win32 启动器（第 6.x / win_job）。

用 CreateProcessW(CREATE_SUSPENDED|CREATE_NO_WINDOW) 创建 Worker，先加入带
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE 的 Job Object，再 ResumeThread。父进程异常退出时
Job Object 会把 Worker 一起杀死，避免孤儿竞态；assign 或 resume 失败则终止子进程并关闭句柄。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as w
import os

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kd32 = ctypes.WinDLL("kernel32", use_last_error=True)

CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
EXTENDED_STARTUPINFO_PRESENT = 0x00080000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9

IdlePriorityClass = 0x40  # 无关，占位避免魔法数误读

_INFINITE = 0xFFFFFFFF


class LaunchError(Exception):
    pass


class ProcessHandle:
    """持有进程/线程/Job 句柄；wait/kill/close 封装。"""

    def __init__(self, pid: int, process_handle: int, thread_handle: int, job_handle: int) -> None:
        self.pid = pid
        self._ph = process_handle
        self._th = thread_handle
        self._jh = job_handle
        self._exited = False

    @property
    def is_alive(self) -> bool:
        if self._exited:
            return False
        code = w.DWORD()
        ok = _kd32.GetExitCodeProcess(self._ph, ctypes.byref(code))
        if not ok:
            return False
        return code.value == 259  # STILL_ACTIVE

    def wait(self, timeout_ms: int = _INFINITE) -> int:
        r = _kd32.WaitForSingleObject(self._ph, timeout_ms)
        code = w.DWORD()
        _kd32.GetExitCodeProcess(self._ph, ctypes.byref(code))
        return int(code.value)

    def terminate(self, code: int = 1) -> None:
        _kd32.TerminateProcess(self._ph, code)

    def close(self) -> None:
        for h in (self._th, self._ph, self._jh):
            if h:
                _kd32.CloseHandle(h)
        self._ph = self._th = self._jh = 0


class WorkerLauncher:
    """CREATE_SUSPENDED + kill-on-close Job Object 的 Worker 启动器。"""

    def __init__(self) -> None:
        self._crash_priority = 0

    def create_job_object(self) -> int:
        job = _k32.CreateJobObjectW(None, None)
        if not job:
            raise LaunchError(f"CreateJobObjectW 失败: {ctypes.get_last_error()}")
        info = (ctypes.c_uint64 * 8)(0) * 1  # placeholder
        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", ctypes.c_byte * 64),
                ("IoInfo", ctypes.c_byte * 48),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        ctypes.memset(ctypes.byref(info), 0, ctypes.sizeof(info))
        # BasicLimitInformation 前 4 字节是 32 位 LimitFlags
        ctypes.cast(ctypes.addressof(info) + 0, ctypes.POINTER(ctypes.c_uint32))[0] = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not _k32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        ):
            _k32.CloseHandle(job)
            raise LaunchError(f"SetInformationJobObject 失败: {ctypes.get_last_error()}")
        return job

    def launch(self, python: str, argv: list[str],
               env: dict[str, str] | None = None,
               cwd: str | None = None) -> ProcessHandle:
        job = self.create_job_object()
        cmdline = _quote_cmdline([python, *argv])
        env_block = _build_environment_block(env or dict(os.environ))

        class STARTUPINFOEXW(ctypes.Structure):
            _fields_ = [
                ("cb", w.DWORD),
                ("lpReserved", w.LPWSTR),
                ("lpDesktop", w.LPWSTR),
                ("lpTitle", w.LPWSTR),
                ("dwX", w.DWORD), ("dwY", w.DWORD),
                ("dwXSize", w.DWORD), ("dwYSize", w.DWORD),
                ("dwXCountChars", w.DWORD), ("dwYCountChars", w.DWORD),
                ("dwFillAttribute", w.DWORD),
                ("dwFlags", w.DWORD),
                ("wShowWindow", w.WORD),
                ("cbReserved2", w.WORD),
                ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
                ("hStdInput", w.HANDLE),
                ("hStdOutput", w.HANDLE),
                ("hStdError", w.HANDLE),
                ("lpAttributeList", ctypes.POINTER(ctypes.c_void_p)),
            ]
        si = STARTUPINFOEXW()
        si.cb = ctypes.sizeof(si)
        pi = w.PROCESS_INFORMATION()
        ok = _k32.CreateProcessW(
            python,
            cmdline,
            None, None, False,
            CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
            env_block,
            cwd,
            ctypes.byref(si),
            ctypes.byref(pi),
        )
        if not ok:
            _k32.CloseHandle(job)
            raise LaunchError(f"CreateProcessW 失败: {ctypes.get_last_error()}")

        def _fail(err: str):
            _kd32.TerminateProcess(pi.hProcess, 1)
            for h in (pi.hThread, pi.hProcess, job):
                _kd32.CloseHandle(h)
            raise LaunchError(err)

        if not _k32.AssignProcessToJobObject(job, pi.hProcess):
            _fail(f"AssignProcessToJobObject 失败: {ctypes.get_last_error()}")
        if _kd32.ResumeThread(pi.hThread) == 0xFFFFFFFF:
            _fail(f"ResumeThread 失败: {ctypes.get_last_error()}")
        _kd32.CloseHandle(pi.hThread)
        return ProcessHandle(int(pi.dwProcessId), int(pi.hProcess), 0, int(job))


def _quote_arg(arg: str) -> str:
    # 简单 Windows 命令行引号转义：含空格则用双引号包裹，内部双引号转义为 \\\"。
    if arg and not any(ch in arg for ch in ' \t"'):
        return arg
    return '"' + arg.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _quote_cmdline(argv: list[str]) -> str:
    return " ".join(_quote_arg(a) for a in argv)


def _build_environment_block(env: dict[str, str]) -> bytes:
    block = "".join(f"{k}={v}\0" for k, v in env.items())
    return (block + "\0").encode("utf-16le")