"""生产 GPU Worker 拉起器（w2 打通生产 spawn 链路）。

PipedChild 满足 PoseRuntime 的子进程契约：write / stdout / is_alive / terminate /
kill / wait / close / pid。用匿名管道 + STARTF_USESTDHANDLES + bInheritHandles 在
CREATE_SUSPENDED 的 kill-on-close Job Object 上拉起 ``python -m ai_monitor_pose.worker``；
父端句柄用 msvcrt.open_osfhandle 包成 file-like 供 runtime 对称读写（协议见 ipc.py：
4 字节大端长度前缀 + UTF-8 JSON）。

本模块只依赖 win_job 的句柄/Job 工具，不含任何 Torch/Ultralytics import，
保证被后端（仅 sys.path）安全导入。
"""
from __future__ import annotations

import json
import os
import threading

import ctypes
import ctypes.wintypes as w
import msvcrt

from ..win_job import (
    CREATE_NO_WINDOW,
    CREATE_SUSPENDED,
    CREATE_UNICODE_ENVIRONMENT,
    LaunchError,
    ProcessHandle,
    _build_environment_block,
    _quote_cmdline,
)

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kd32 = ctypes.WinDLL("kernel32", use_last_error=True)

_INFINITE = 0xFFFFFFFF

# 本模块自带的 Win32 常量/结构（不强制改动 win_job.py，保持其稳定面向已测契约）
_HANDLE_FLAG_INHERIT = 0x00000001
_STARTF_USESTDHANDLES = 0x00000100
_GENERIC_WRITE = 0x40000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_OPEN_EXISTING = 3
_OPEN_ALWAYS = 4
_FILE_APPEND_DATA = 0x00000004
_INVALID_HANDLE_VALUE = 0xFFFFFFFFFFFFFFFF
# 父进程通过该环境变量指定 Worker stderr 日志文件路径（追加模式）；未设置时保持 NUL
_STDERR_ENV = "AI_MONITOR_POSE_WORKER_STDERR"

_k32.CreateFileW.restype = w.HANDLE


class _SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", w.DWORD),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", w.BOOL),
    ]


class _STARTUPINFOEXW(ctypes.Structure):
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


def _create_pipe_pair():
    sa = _SECURITY_ATTRIBUTES()
    sa.nLength = ctypes.sizeof(sa)
    sa.bInheritHandle = True
    r = w.HANDLE()
    wr = w.HANDLE()
    if not _k32.CreatePipe(ctypes.byref(r), ctypes.byref(wr), ctypes.byref(sa), 0):
        raise LaunchError(f"CreatePipe 失败: {ctypes.get_last_error()}")
    return int(r.value), int(wr.value)


def _set_inherit(handle: int, on: bool) -> None:
    if not _k32.SetHandleInformation(w.HANDLE(handle), _HANDLE_FLAG_INHERIT, 1 if on else 0):
        raise LaunchError(f"SetHandleInformation 失败: {ctypes.get_last_error()}")


def _open_nul():
    sa = _SECURITY_ATTRIBUTES()
    sa.nLength = ctypes.sizeof(sa)
    sa.bInheritHandle = True
    h = _k32.CreateFileW(
        "NUL", _GENERIC_WRITE,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        ctypes.byref(sa), _OPEN_EXISTING, 0, None,
    )
    if not h:
        raise LaunchError(f"无法打开 NUL: {ctypes.get_last_error()}")
    return int(h)


def _open_stderr_log(path: str) -> int:
    """以追加模式打开 Worker stderr 日志文件，返回可被子进程继承的句柄。

    FILE_APPEND_DATA 使每次写都原子追加到文件末尾（多次拉起共写同一日志也不互相
    覆盖）；OPEN_ALWAYS 在文件不存在时创建。句柄用 SetHandleInformation 显式标记
    继承，供 STARTF_USESTDHANDLES 的 hStdError 使用。打开失败按 LaunchError 上抛，
    不静默回退 NUL 丢失 Worker 排障信息。
    """
    sa = _SECURITY_ATTRIBUTES()
    sa.nLength = ctypes.sizeof(sa)
    sa.bInheritHandle = True
    h = _k32.CreateFileW(
        path, _FILE_APPEND_DATA,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        ctypes.byref(sa), _OPEN_ALWAYS, 0, None,
    )
    if not h or int(h) in (-1, _INVALID_HANDLE_VALUE):
        raise LaunchError(
            f"无法以追加模式打开 stderr 日志文件 {path!r}: {ctypes.get_last_error()}")
    handle = int(h)
    _set_inherit(handle, True)
    return handle


class PipedChild:
    """一个带 stdin/stdout 匿名管道的 Worker 子进程，满足 PoseRuntime 合同。"""

    def __init__(self, handle: ProcessHandle, stdin_file, stdout_file) -> None:
        self._handle = handle
        self.pid = handle.pid
        self._stdin = stdin_file  # 父端 -> 子端的写 file-like
        self._stdout = stdout_file  # 子端 -> 父端的读 file-like
        # 写互斥:offer_frame 线程与心跳看门狗线程会并发 write 同一管道,
        # 4 字节长度前缀帧绝不能交错;同时保证部分写时循环写满
        self._write_lock = threading.Lock()

    @property
    def stdout(self):
        """返回 file-like(父端读句柄)；runtime._read_loop 用 .read(n) 读取。"""
        return self._stdout

    def write(self, msg: dict) -> None:
        from ..ipc import encode_message  # 惰性，避免模块导入开销

        data = encode_message(msg)
        with self._write_lock:
            view = memoryview(data)
            while view:
                n = self._stdin.write(view)
                if not n:  # raw FileIO 部分写语义:0 字节=管道断开
                    raise BrokenPipeError("worker pipe write returned 0 bytes")
                view = view[n:]
            self._stdin.flush()

    def is_alive(self) -> bool:
        return bool(self._handle.is_alive)

    def terminate(self) -> None:
        self._handle.terminate(1)

    def kill(self) -> None:
        try:
            self._handle.terminate(9)
        except Exception:  # noqa: BLE001
            pass

    def wait(self, timeout=None) -> int:
        if timeout is None:
            return self._handle.wait(_INFINITE)
        return self._handle.wait(int(max(0, timeout * 1000)))

    def close(self) -> None:
        for f in (self._stdin, self._stdout):
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass
        self._handle.close()


JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", w.HANDLE),
        ("hThread", w.HANDLE),
        ("dwProcessId", w.DWORD),
        ("dwThreadId", w.DWORD),
    ]


def _create_job_object() -> int:
    """创建 kill-on-close 的 Job Object 句柄。"""
    job = _k32.CreateJobObjectW(None, None)
    if not job:
        raise LaunchError(f'CreateJobObjectW 失败: {ctypes.get_last_error()}')

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('BasicLimitInformation', ctypes.c_byte * 64),
            ('IoInfo', ctypes.c_byte * 48),
            ('ProcessMemoryLimit', ctypes.c_size_t),
            ('JobMemoryLimit', ctypes.c_size_t),
            ('PeakProcessMemoryUsed', ctypes.c_size_t),
            ('PeakJobMemoryUsed', ctypes.c_size_t),
        ]

    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    ctypes.memset(ctypes.byref(info), 0, ctypes.sizeof(info))
    ctypes.cast(ctypes.addressof(info), ctypes.POINTER(ctypes.c_uint32))[0] = (
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    if not _k32.SetInformationJobObject(
        job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(info), ctypes.sizeof(info),
    ):
        _k32.CloseHandle(job)
        raise LaunchError(f'SetInformationJobObject 失败: {ctypes.get_last_error()}')
    return int(job)


def launch_piped(python: str, argv: list[str], *, env: dict | None = None,
                 cwd: str | None = None) -> PipedChild:
    """创建匿名管道并以继承句柄方式拉起子进程，返回 PipedChild。

    成功即返回（工作已 CreateProcessW 恢复线程）；进程可能随后才真正加载模型，
    是否 READY 由父端通过协议感知。

    通信不走标准 stdin/stdout，而走两条专用管道：
      rx: 父写 rx_w -> 子读 rx_r（HELLO / INFER_FRAME / SHUTDOWN ...)
      tx: 子写 tx_w -> 父读 tx_r（WORKER_READY / INFERENCE_RESULT ...）
    句柄值经 env 传给子进程（AI_MONITOR_POSE_RX_HANDLE / TX_HANDLE）。原因：
    torch/ultralytics predict 在 Windows 上会破坏标准 stdin/stdout 的 OS 句柄，
    使 os.read(0)/os.write(1) 及 GetStdHandle 写返回 EINVAL/ERROR_NO_DATA(232)，
    消息到不了父端。专用句柄不受影响。
    """
    throw_in_r, throw_in_w = _create_pipe_pair()   # hStdInput 占位（子进程不使用）
    throw_out_r, throw_out_w = _create_pipe_pair()  # hStdOutput 占位（子进程不使用）
    rx_r, rx_w = _create_pipe_pair()   # 父写, 子读
    tx_r, tx_w = _create_pipe_pair()   # 子写, 父读
    # stderr：环境变量指定日志文件则追加写入，否则保持原 NUL 行为
    stderr_path = (env or {}).get(_STDERR_ENV) or os.environ.get(_STDERR_ENV)
    stderr_h = _open_stderr_log(str(stderr_path)) if stderr_path else None
    nul = None if stderr_h is not None else _open_nul()
    child_stderr = stderr_h if stderr_h is not None else nul

    for h in (throw_in_w, throw_out_r, rx_w, tx_r, nul):
        if h is None:
            continue
        _set_inherit(h, False)
    for h in (throw_in_r, throw_out_w, rx_r, tx_w, child_stderr):
        _set_inherit(h, True)

    job = _create_job_object()
    cmdline = _quote_cmdline([python, *argv])
    use_env = {**os.environ, **(env or {})}
    use_env["AI_MONITOR_POSE_RX_HANDLE"] = str(int(rx_r))
    use_env["AI_MONITOR_POSE_TX_HANDLE"] = str(int(tx_w))
    env_block = _build_environment_block(use_env)

    class _SI(_STARTUPINFOEXW):
        pass

    si = _SI()
    si.cb = ctypes.sizeof(si)
    si.dwFlags = _STARTF_USESTDHANDLES
    si.hStdInput = w.HANDLE(throw_in_r)
    si.hStdOutput = w.HANDLE(throw_out_w)
    si.hStdError = w.HANDLE(child_stderr)
    pi = _PROCESS_INFORMATION()

    def _cleanup_handles() -> None:
        for h in (throw_in_r, throw_in_w, throw_out_r, throw_out_w,
                  rx_r, rx_w, tx_r, tx_w, child_stderr):
            if h is None:
                continue
            _k32.CloseHandle(w.HANDLE(h))

    ok = _k32.CreateProcessW(
        python,
        cmdline,
        None, None,
        True,  # bInheritHandles
        CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
        env_block,
        cwd,
        ctypes.byref(si),
        ctypes.byref(pi),
    )
    if not ok:
        _k32.CloseHandle(job)
        _cleanup_handles()
        raise LaunchError(f"CreateProcessW 失败: {ctypes.get_last_error()}")

    # 子进程已复制句柄，父端关掉子端副本（含 stderr 日志句柄）
    for h in (throw_in_r, throw_out_w, rx_r, tx_w, child_stderr):
        if h is None:
            continue
        _k32.CloseHandle(w.HANDLE(h))

    def _fail(err: str):
        _kd32.TerminateProcess(pi.hProcess, 1)
        _kd32.CloseHandle(pi.hThread)
        _kd32.CloseHandle(pi.hProcess)
        _k32.CloseHandle(job)
        _k32.CloseHandle(w.HANDLE(rx_w))
        _k32.CloseHandle(w.HANDLE(tx_r))
        _k32.CloseHandle(w.HANDLE(throw_in_w))
        _k32.CloseHandle(w.HANDLE(throw_out_r))
        raise LaunchError(err)

    if not _k32.AssignProcessToJobObject(job, pi.hProcess):
        _fail(f"AssignProcessToJobObject 失败: {ctypes.get_last_error()}")
    if _kd32.ResumeThread(pi.hThread) == 0xFFFFFFFF:
        _fail(f"ResumeThread 失败: {ctypes.get_last_error()}")
    _kd32.CloseHandle(pi.hThread)

    # 父端保留 rx_w(写子) 与 tx_r(读子)；丢弃占位端
    _k32.CloseHandle(w.HANDLE(throw_in_w))
    _k32.CloseHandle(w.HANDLE(throw_out_r))
    # 必须显式 O_BINARY: CRT 默认文本模式会把长度前缀中的 0x0A 翻译成 CRLF,破坏二进制协议
    stdin_fd = msvcrt.open_osfhandle(rx_w, os.O_WRONLY | os.O_BINARY)
    out_file = os.fdopen(stdin_fd, "wb", buffering=0)
    stdout_fd = msvcrt.open_osfhandle(tx_r, os.O_RDONLY | os.O_BINARY)
    in_file = os.fdopen(stdout_fd, "rb", buffering=0)

    handle = ProcessHandle(int(pi.dwProcessId), int(pi.hProcess), 0, int(job))
    return PipedChild(handle, out_file, in_file)


def build_worker_process_factory(python_exe: str, config_dict: dict, *,
                                 module: str = "ai_monitor_pose.worker",
                                 cwd: str | None = None):
    """构造生产 process_factory 闭包：每次调用拉一个真实 Worker 子进程。

    config_dict 序列化进 WORKER_CONF 环境变量，worker._load_config 据此构建
    FallTaskConfig（与父端 Task 的 config 同一来源）。
    """
    python_exe = str(python_exe)
    argv = ["-m", module]
    env_stub = {"WORKER_CONF": json.dumps(config_dict, ensure_ascii=False, separators=(",", ":"))}
    root = cwd or str(pathlib_path(__file__))

    def factory():
        return launch_piped(python_exe, argv, env=env_stub, cwd=root)

    factory.__doc__ = f"拉起 {python_exe} -m {module} 的真实 GPU Worker 子进程"
    return factory


def pathlib_path(p):
    from pathlib import Path

    return str(Path(p).resolve().parent.parent.parent)
