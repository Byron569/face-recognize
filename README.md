# AI Monitor — 基于 InsightFace 的人脸识别 + YOLOv8-Pose 摔倒检测监控系统

实时视频监控 Web 平台。后端 FastAPI 直连 [InsightFace](https://github.com/deepinsight/insightface)
官方推理内核（SCRFD 检测 + ArcFace 识别，默认 GPU），前端 React + Ant Design 5，
数据落 PostgreSQL，支持多摄像头与**可插拔视觉任务**（人脸识别、摔倒检测等）。

摔倒检测采用**独立 GPU Worker 进程**增强：AI Monitor 摄像头流水线把 raw 帧写入共享内存，
由虚拟环境中的唯一 PyTorch CUDA 进程运行 YOLOv8n-pose 模型，逐摄像头维护跟踪器与
**时间制摔倒状态机**，状态转换通过持久事件泵映射为原生 `VisionEvent` 落库并经宿主 ACK 幂等投递。

## 特性

### 人脸识别（InsightFace 官方内核）
- **InsightFace 官方内核**：`FaceAnalysis` 一次推理完成检测 + 关键点 + 512-d embedding；切换 `buffalo_s/buffalo_l` 只需改配置
- **推理默认 GPU**：CUDAExecutionProvider 优先，无 GPU 自动降级 CPU（日志告警）
- **ByteTrack 多目标跟踪**：卡尔曼滤波 + 两阶段关联（高/低置信度框分层匹配），稳定 track_id；识别按冷却策略触发（新 track 优先 → 冷却 → 失败退避 → 重验证）
- **向量化识别**：人脸底库载入内存 numpy 矩阵，一次点积完成检索；注册/删除即时生效
- **摄像头实时注册**（动作引导五步）：`getUserMedia` 实时取景 + 检测框叠加，正脸→左转→右转→抬头→低头，达标自动采帧，多角度特征原子入库，即时生效；不含活体防伪，摄像头流不落盘

### 摔倒检测（YOLOv8-Pose 独立 GPU Worker）
- **GPU-hard 无 CPU 回退**：模型前向、预处理/后处理、NMS、关键点计算全部在 `cuda:0` + FP16；`torch.cuda.is_available()` 与真实模型加载双重校验，禁止任何隐式 `try/except → cpu` 降级
- **唯一引擎共享**：多摄像头共享单一模型 + CUDA 上下文，带公平帧调度与容量 manifest（`CapacityManifest`）门禁，按环境指纹（GPU 型号/驱动/CUDA 版本/模型哈希）校验
- **IPC：共享内存 + Windows 命名管道**：raw BGR 帧走共享内存双槽，控制消息走 `AF_PIPE`；管道以 `O_BINARY` 打开避免 CRT 文本模式破坏二进制长度前缀；崩溃重连、健康心跳、`Worker Journal` 持久化状态转换
- **时间制摔倒状态机**：长期状态 `NORMAL / POTENTIAL / FALLEN`（`RECOVERED` 为转换事件），只用单调时间判持续时间，证据空洞中断连续性；`fall_potential / fall_detected / fall_recovered` 三态事件
- **原子事件投递**：状态转换经 Worker Journal → 父端 `runtime._drain_journal`（成功才 ACK、退避重试、毒丸落父端 `EventSpool`）→ `EventIngress` → PostgreSQL + WebSocket 推送；事件去重（event_id / dedupe_key）
- **实时骨架叠加**：Canvas 按预览像素绘制 17 关键点 + bbox，`FallOverlayCache` TTL 会话无效化；每路 >=300ms 超阈值拒绝过期 overlay
- **健康接口**：`GET /system/fall-runtime` 返回脱敏健康快照（worker/gpu/model/delivery/cameras），不泄漏 pipe authkey / 配置路径 / DB URL / 异常堆栈

### 平台共性
- **可插拔任务架构**：`class_path` / `module_path` 动态加载，扩展任务零改动接入（见 [docs/PLUGIN_GUIDE.md](docs/PLUGIN_GUIDE.md)）
- **全参数配置化**：`default.yaml` → `profiles/{desktop,balanced,edge_minimal}.yaml` → 摄像头个性化 JSONB，代码零硬编码
- **工业化工程**：Alembic 版本化迁移、api→services→repositories 分层、引擎池共享显存、断线自动重连、事件自动落库与 WS 推送、迁移保护脚本
- **流畅预览**：`binary_jpeg_v1` 二进制 JPEG、单路编码一次、多订阅者单槽位丢旧帧，慢客户端不拖其他客户端

## 快速开始

> **GPU 环境版本对应（重要）**：`onnxruntime-gpu` 与 CUDA 运行库必须匹配，否则人脸推理会**静默降级为 CPU**。
> 排查：`python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`，列表含 `CUDAExecutionProvider` 即 GPU 就绪。
> 推荐 `pip install onnxruntime-gpu==1.21.0`（匹配 CUDA 12.6）。项目自带 GPU 就绪虚拟环境 `.venv`。

```bash
# 1. 数据库(PostgreSQL 16)
docker run -d --name ai-monitor-db -p 5432:5432 \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=123456 -e POSTGRES_DB=ai_monitor \
  postgres:16-alpine

# 2. 后端(项目根目录)
cd backend
pip install -r requirements.txt
alembic upgrade head          # 建表(版本化迁移)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. 前端(另一个终端)
cd frontend
npm install
npm run dev                   # http://localhost:3000(已代理 /api 与 /ws)
```

> 模型：默认高精度档 `models/buffalo_l/`（SCRFD-10G 检测 + ArcFace ResNet50 识别）；
> `buffalo_l` 超 GitHub 单文件限制不随仓库提供，首次使用请从官方 Release 下载解压到 `models/buffalo_l/`；
> 轻量档 `models/buffalo_s/` 已随仓库提供，切换 `vision.model_pack` 即可。切换模型包后 embedding 空间不兼容，**底库需重新注册**。

### 启用摔倒检测

摔倒检测默认关闭（`tasks.fall_detection.enabled: false`）。开启需要：

1. **准备姿态 Worker 环境**（独立虚拟环境）：安装 `torch==2.7.0+cu126`（CUDA wheel）、`torchvision==0.22.0+cu126`、`ultralytics==8.4.126`、`scipy`、`opencv-python`；YOLO 权重 `models/yolov8n-pose.pt` 及其 `.sha256` 摘要文件（禁止自动联网下载）。
2. **配置 `configs/default.yaml`**：
   ```yaml
   tasks:
     fall_detection:
       enabled: true                  # 开启
       class_path: app.tasks.builtin.fall_detection_task.FallDetectionTask
   fall_detection:
     worker:
       python: "D:/YOLOv8-Pose/-YOLOv8-Pose--main/.venv-worker/Scripts/python.exe"
       module: ai_monitor_pose.worker
     model:
       path: "models/yolov8n-pose.pt"
       sha256_file: "models/yolov8n-pose.pt.sha256"
     runtime:
       capacity_manifest_path: "models/capacity-cuda0.json"
       worker_journal_path: "var/worker-transition-journal.sqlite3"
       event_spool_path: "var/event-spool.sqlite3"
       overlay_ttl_ms: 1200
   ```
3. **注册摄像头**：前端「添加摄像头」选择视频源（RTSP / 本地摄像头 / 本地文件），对每路摄像头可覆盖阈值（如 `min_fall_pose_duration_s`、`max_trigger_gap_s`）。
   > 摄像头必须由**后端所能访问**的地址提供（RTSP 网络流 / 后端本地摄像头 / 后端可读的视频文件，支持循环播放）。前端「本地摄像头」用的是访问者浏览器（需 localhost/HTTPS）。
4. 重启服务，事件自动进入 `/ws/events`、`events` 表与前端事件页。

> **诊断**：Worker 侧异常默认被 NUL 吞掉；设环境变量 `AI_MONITOR_POSE_WORKER_STDERR=<日志文件>` 即可把 GPU Worker 的 stderr 重定向到文件排障。
> 容量基准：`scripts/gpu_smoke.py` / `scripts/integration_smoke.py --fixture-manifest fixtures/fall_manifest.json` 可离线验证 GPU 吞吐与 M1/M2 链路。

## 文档

| 文档 | 内容 |
|------|------|
| [docs/API.md](docs/API.md) | 全部 REST/WebSocket 接口说明（运行时另见 `/docs` OpenAPI） |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 分层架构、关键设计决策、数据模型 |
| [docs/PLUGIN_GUIDE.md](docs/PLUGIN_GUIDE.md) | 如何开发扩展任务（跌倒检测等） |

## 配置速览

```yaml
# configs/default.yaml(节选)
vision:
  model_pack: buffalo_l
  device: cuda            # 推理默认 GPU;cpu/auto 可选
  det_size: [640, 640]
  det_interval: 2
  track: { iou_threshold: 0.3, max_lost: 15, min_hits: 2 }
  recognition:
    threshold: 0.40
    quality: { min_det_score: 0.60, min_face_size: 80 }
    temporal: { min_valid_samples: 3, max_samples_per_track: 8, top_k: 3 }
stream:
  max_height: 480       # 默认预览高度;局域网高画质可设 720
  jpeg_quality: 70
  push_fps: 20
tasks:
  face_recognition:
    enabled: true
    class_path: app.tasks.builtin.face_recognition_task.FaceRecognitionTask
  fall_detection:
    enabled: false      # 摔倒检测(独立 GPU Worker),见上文“启用摔倒检测”
    class_path: app.tasks.builtin.fall_detection_task.FallDetectionTask

fall_detection:         # 任务专属配置节(完整节见上)
  algorithm:
    min_fall_pose_duration_s: 3.5   # 判定 FALLEN 的最短躺平持续时长
    max_trigger_gap_s: 0.25         # 证据空洞容忍,超过则中断连续性
```

级联优先级：摄像头 `cameras.config`（JSONB）> profile 文件 > default.yaml。

预览分辨率只影响 WebSocket JPEG，不改变推理分辨率。默认 480p 适合局域网；已有摄像头保存的 `stream.max_height: 0`（原生）不会被自动覆盖，需在系统设置手动调整。

## 项目结构

```
backend/       FastAPI 服务(api → services → repositories → models → migrations)
vision/        纯推理内核(camera / engine / tracker / pipeline / tasks,依赖 InsightFace)
configs/       全参数 YAML(default + profile 级联 + 摔倒检测/容量)
frontend/      React 18 + AntD 5(样式/菜单/配色 src/config;WS 去重、骨架叠加、事件页)
models/        本地模型(buffalo_s 人脸;yolov8n-pose.pt + capacity-cuda0.json 见上)
tests/         pytest(单元/集成/迁移/任务契约/摔倒检测协议)
docs/          接口 / 架构 / 插件开发文档
face_db/       头像与旧 pickle 数据(可选迁移)
fixtures/      M1/M2 受控验证用视频与 manifest(可选)
```

摔倒检测增强的前端关键文件：`src/stream/analyticsProtocol.ts`（解析/强校验）、`src/stream/fallOverlay.ts`（Canvas 骨架+bbox 叠加+TTL）、`src/components/CameraCell.tsx`（帧匹配叠加）、`src/stream/eventDedupe.ts`（LRU 去重）。后端：`backend/app/services/fall_runtime_health.py`、`event_ingress.py`；迁移：`backend/migrations/versions/0002_event_outbox.py`。

## 测试

```bash
# AI Monitor 侧(AI_MONITOR_TEST_DATABASE_URL 指向以 _test 结尾的库时有更多迁移/集成用例)
python -m pytest tests/ -v

# 摔倒检测姿态侧(独立 Worker 环境,含真实 CUDA 推理门禁)
# $PosePython = D:\YOLOv8-Pose\-YOLOv8-Pose--main\.venv-worker\Scripts\python.exe
& $PosePython -m pytest tests/ -v

# 前端
cd frontend
npm test
npm run build
```