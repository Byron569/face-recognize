"""阶段1：client 进程导入不得拉入 Torch/Ultralytics；不得触达旧 main。

说明：Task 继承真实 vision.tasks.VisionTask 会经 vision 包载入后端既有依赖(如 cv2)，
故禁用项只含重依赖 torch/ultralytics 与旧算法入口。
"""
from __future__ import annotations

import subprocess
import sys

CODE = (
    "import sys; "
    "import ai_monitor_pose.task; "
    "assert 'torch' not in sys.modules, 'torch imported'; "
    "assert 'ultralytics' not in sys.modules, 'ultralytics imported'; "
    "assert 'main' not in sys.modules and 'camera_process' not in sys.modules; "
    "assert FallDetectionTask.__module__ == 'ai_monitor_pose.task' "
    "if 'FallDetectionTask' not in dir(sys.modules['ai_monitor_pose.task']) else True; "
    "print('LIGHT_IMPORT_OK')"
)


def test_importing_task_does_not_import_torch_ultralytics_or_old_main() -> None:
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "LIGHT_IMPORT_OK" in r.stdout
