# 模型目录
真实权重与 capacity manifest 见阶段9。

## TensorRT engine（可选 GPU 加速产物）

- worker 启动时若存在与 .pt 同 stem 的 `*.engine`（TensorRT FP16, imgsz=640）且
  修改时间不早于 .pt，自动优先加载 engine（跳过 `.to`/`.half`，精度已烘焙进
  引擎）；engine 缺失、陈旧（早于 .pt）或加载失败时自动回退 .pt，绝不因 engine
  阻断 worker 启动。GPU-only 约束不变（engine 本身即 GPU 后端）。
- 生成命令（在 `.venv-worker` 内、于本目录执行，需先 `pip install tensorrt`）：

      python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt').export(format='engine', half=True, imgsz=640, device=0)"

- engine 与 GPU 型号 / 驱动 / TensorRT 版本绑定，跨机器不可移植；损坏时 worker
  自动回退 .pt。
- 同目录的 `*.onnx` 为 engine 导出过程的中间产物，worker 不直接使用。
