"""fake_worker fixture：一个真实子进程，用控制协议（4字节大端长度前缀+UTF-8 JSON）与父端对话。

不需要 GPU；可用环境变量/INFER payload 指示在指定帧后 os._exit(17) 模拟崩溃。
对 HELLO 先回 WORKER_STARTING 再回 WORKER_READY；支持 PING/PONG、INFER_FRAME(RESULT/REJECT)、
SHUTDOWN/STOPPED、REGISTER_CAMERA/CAMERA_REGISTERED。EOF 前会发 EOF 标记。
"""
from __future__ import annotations

FAKE_WORKER_CODE = r'''
import json, os, struct, sys, time, uuid

def write_msg(obj):
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(struct.pack(">I", len(body)) + body)
    sys.stdout.buffer.flush()

def read_msg():
    hdr = sys.stdin.buffer.read(4)
    if not hdr or len(hdr) < 4:
        return None
    (n,) = struct.unpack(">I", hdr)
    n = int(n)
    data = b""
    while len(data) < n:
        chunk = sys.stdin.buffer.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return json.loads(data.decode("utf-8"))

epoch = uuid.uuid4().hex
iid = uuid.uuid4().hex
cfg = json.loads(os.environ.get("FAKE_WORKER_CFG", "{}"))
crash_after = int(os.environ.get("FAKE_WORKER_CRASH_AFTER", "0"))
infer_count = [0]

while True:
    m = read_msg()
    if m is None:
        break
    mt = m.get("message_type")
    mid = m.get("message_id")
    if mt == "HELLO":
        write_msg({"message_type": "WORKER_STARTING", "worker_epoch": epoch,
                   "correlation_id": mid, "payload": {
                       "selected_schema_version": 1, "worker_epoch": epoch,
                       "worker_instance_id": iid, "worker_pid": os.getpid(),
                       "parent_pid": m.get("payload", {}).get("parent_pid"),
                       "challenge_sha256": "fake"}})
        write_msg({"message_type": "WORKER_READY", "worker_epoch": epoch,
                   "correlation_id": mid, "payload": {
                       "worker_epoch": epoch, "worker_instance_id": iid,
                       "worker_pid": os.getpid(),
                       "device": cfg.get("device", "cuda:0"),
                       "device_name": cfg.get("device_name", "Fake GPU"),
                       "model_sha256": "fake", "precision": "fp16"}})
    elif mt == "PING":
        write_msg({"message_type": "PONG", "worker_epoch": epoch,
                   "correlation_id": mid, "payload": {
                       "ping_id": m.get("payload", {}).get("ping_id"),
                       "parent_sent_ns": m.get("payload", {}).get("parent_sent_ns"),
                       "worker_received_ns": time.time_ns(),
                       "worker_sent_ns": time.time_ns()}})
    elif mt == "INFER_FRAME":
        infer_count[0] += 1
        if crash_after and infer_count[0] >= crash_after:
            os._exit(17)
        p = m.get("payload", {})
        if p.get("fake_action") == "crash":
            os._exit(17)
        if p.get("fake_action") == "reject":
            write_msg({"message_type": "FRAME_REJECTED", "worker_epoch": epoch,
                       "correlation_id": mid, "payload": {
                           "request_id": p.get("request_id", ""),
                           "camera_id": p.get("camera_id", ""),
                           "camera_session_id": p.get("camera_session_id", ""),
                           "frame_id": p.get("frame_id", 0),
                           "error_code": "FAKE_REJECT", "error_message": "faked",
                           "retryable": False}})
        else:
            write_msg({"message_type": "INFERENCE_RESULT", "worker_epoch": epoch,
                       "correlation_id": mid, "payload": {
                           "request_id": p.get("request_id", ""),
                           "worker_epoch": epoch, "worker_instance_id": iid,
                           "camera_id": p.get("camera_id", ""),
                           "camera_session_id": p.get("camera_session_id", ""),
                           "source_frame_id": p.get("frame_id", 0),
                           "status": "ok", "tracks": [], "config_revision": "r"}})
    elif mt == "REGISTER_CAMERA":
        write_msg({"message_type": "CAMERA_REGISTERED", "worker_epoch": epoch,
                   "correlation_id": mid, "payload": {
                       "camera_id": m.get("payload", {}).get("camera_id"),
                       "camera_session_id": m.get("payload", {}).get("camera_session_id"),
                       "camera_config_revision": "r"}})
    elif mt == "UNREGISTER_CAMERA":
        write_msg({"message_type": "CAMERA_UNREGISTERED", "worker_epoch": epoch,
                   "correlation_id": mid, "payload": {
                       "camera_id": m.get("payload", {}).get("camera_id"),
                       "camera_session_id": m.get("payload", {}).get("camera_session_id"),
                       "pending_transition_count": 0}})
    elif mt == "SHUTDOWN":
        write_msg({"message_type": "STOPPED", "worker_epoch": epoch,
                   "correlation_id": mid, "payload": {
                       "reason": "down", "pending_transition_count": 0,
                       "exit_code_intent": 0}})
        sys.exit(0)
'''

import os
import struct
import subprocess
import sys
import threading

from ai_monitor_pose.ipc import decode_message, encode_message


class FakeWorkerProcess:
    """把 fake_worker 代码作为真实子进程运行，并暴露父端 I/O 接口。"""

    def __init__(self, *, crash_after: int = 0, cfg: dict | None = None) -> None:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        if crash_after:
            env["FAKE_WORKER_CRASH_AFTER"] = str(int(crash_after))
        if cfg:
            import json as _json
            env["FAKE_WORKER_CFG"] = _json.dumps(cfg, ensure_ascii=False)
        self._proc = subprocess.Popen(
            [sys.executable, "-c", FAKE_WORKER_CODE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, bufsize=0,
        )
        self.stdin = self._proc.stdin
        self.stdout = self._proc.stdout
        self._lock = threading.Lock()

    @property
    def pid(self) -> int:
        return self._proc.pid

    def write(self, envelope: dict) -> None:
        if self._proc.poll() is not None:
            raise BrokenPipeError("child already exited")
        with self._lock:
            self.stdin.write(encode_message(envelope))
            self.stdin.flush()

    def read_frame(self, timeout: float = 3.0):
        """阻塞读一帧；超时抛 TimeoutError。便于测试直接读写。"""
        self._proc._child_created = True  # noqa
        recv = self._read_exact(timeout)
        if recv is None:
            raise TimeoutError("no frame within timeout")
        return decode_message(recv)

    def _read_exact(self, timeout: float) -> bytes | None:
        import socket
        old = self.stdout if False else None
        # 用 selectors 在 Windows 上对管道不友好，退化为阻塞读（子进程稳定即时回复）
        hdr = self.stdout.read(4)
        if not hdr or len(hdr) < 4:
            return None
        (n,) = struct.unpack(">I", hdr)
        body = self.stdout.read(int(n))
        return hdr + body

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def poll(self):
        return self._proc.poll()

    def terminate(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def kill(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:
                pass

    def wait(self, timeout: float | None = None) -> int | None:
        try:
            return self._proc.wait(timeout)
        except subprocess.TimeoutExpired:
            return None

    def close(self) -> None:
        try:
            self.stdin.close()
        except Exception:
            pass


def make_process_factory(*, crash_after: int = 0, cfg: dict | None = None):
    """返回一个无参 callable，实例化一个新的 FakeWorkerProcess，测试注入 runtime。"""
    return lambda: FakeWorkerProcess(crash_after=crash_after, cfg=cfg)