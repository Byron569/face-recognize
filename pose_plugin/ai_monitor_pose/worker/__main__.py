"""GPU Worker 独立进程入口（延迟导入 Torch/Ultralytics）。

支持 `python -m ai_monitor_pose.worker` 直接以真实 Worker 身份运行。
配置来自 env 的 WORKER_CONF（JSON 文件路径或 JSON 字符串），模型与规则缺省取
项目 models/ 下 yolov8n-pose.pt 及其 sha256 sidecar。
"""
from __future__ import annotations


def main(argv: "list[str] | None" = None) -> int:
    from .service import run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())