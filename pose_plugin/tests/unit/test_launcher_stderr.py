"""修复1回归：launch_piped 的 stderr 重定向。

环境变量 AI_MONITOR_POSE_WORKER_STDERR 指定日志文件时，子进程 stderr 以追加模式
写入该文件（句柄需被子进程继承）；未设置时保持原 NUL 行为不变。
"""
from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Win32 launcher only")

from ai_monitor_pose.worker.launcher import launch_piped  # noqa: E402

_PY = sys.executable


def _run_child(args, env=None, timeout: float = 20.0) -> int:
    child = launch_piped(_PY, args, env=env)
    try:
        return child.wait(timeout=timeout)
    finally:
        child.close()


def test_stderr_env_redirects_child_stderr_to_log_file(tmp_path):
    log = tmp_path / "worker-stderr.log"
    rc = _run_child(
        ["-c", "import sys; sys.stderr.write('pose-worker-stderr-mark')"],
        env={"AI_MONITOR_POSE_WORKER_STDERR": str(log)},
    )
    assert rc == 0
    assert "pose-worker-stderr-mark" in log.read_text(encoding="utf-8", errors="replace")


def test_stderr_env_is_read_from_process_environ(tmp_path, monkeypatch):
    log = tmp_path / "worker-stderr-osenv.log"
    monkeypatch.setenv("AI_MONITOR_POSE_WORKER_STDERR", str(log))
    rc = _run_child(["-c", "import sys; sys.stderr.write('from-os-environ')"])
    assert rc == 0
    assert "from-os-environ" in log.read_text(encoding="utf-8", errors="replace")


def test_stderr_env_opens_in_append_mode(tmp_path):
    log = tmp_path / "append.log"
    log.write_text("first-line\n", encoding="utf-8")
    rc = _run_child(
        ["-c", "import sys; sys.stderr.write('second-line')"],
        env={"AI_MONITOR_POSE_WORKER_STDERR": str(log)},
    )
    assert rc == 0
    text = log.read_text(encoding="utf-8", errors="replace")
    assert "first-line" in text
    assert "second-line" in text
    assert text.index("first-line") < text.index("second-line")


def test_stderr_env_with_unopenable_path_fails_loudly(tmp_path):
    from ai_monitor_pose.win_job import LaunchError

    bad = tmp_path / "no-such-dir" / "x.log"
    with pytest.raises(LaunchError):
        launch_piped(_PY, ["-c", "pass"], env={"AI_MONITOR_POSE_WORKER_STDERR": str(bad)})


def test_without_stderr_env_keeps_nul_behavior(tmp_path, monkeypatch):
    monkeypatch.delenv("AI_MONITOR_POSE_WORKER_STDERR", raising=False)
    # 未设置环境变量：stderr 指向 NUL，子进程写 stderr 不报错、正常退出
    rc = _run_child(["-c", "import sys; sys.stderr.write('to-nul')"], env={})
    assert rc == 0
    # 管道通路不受影响：父端仍可向子进程写消息（rx 通道）
    child = launch_piped(_PY, ["-c", "import sys; sys.stdin.buffer.read(1)"])
    try:
        child.write({"message_type": "HELLO", "message_id": "m1", "payload": {}})
        rc = child.wait(timeout=20.0)
    finally:
        child.close()
    assert rc == 0
