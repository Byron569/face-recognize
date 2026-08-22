# AI Monitor — 基于 InsightFace 的人脸识别监控系统

实时视频监控 + 人脸识别 Web 平台:后端 FastAPI 直连 [InsightFace](https://github.com/deepinsight/insightface)
官方推理内核(SCRFD 检测 + ArcFace 识别,默认 GPU),前端 React + Ant Design 5,
数据落 PostgreSQL,支持多摄像头与可插拔扩展任务(跌倒检测等)。

## 特性

- **InsightFace 官方内核**:`FaceAnalysis` 一次推理完成检测 + 关键点 + 512-d embedding;切换 `buffalo_s/buffalo_l` 只需改配置
- **推理默认 GPU**:CUDAExecutionProvider 优先,无 GPU 自动降级 CPU(日志告警)
- **ByteTrack 多目标跟踪**:卡尔曼滤波 + 两阶段关联(高/低置信度框分层匹配),稳定 track_id;识别按冷却策略触发(新 track 优先 → 冷却 → 失败退避 → 重验证)
- **向量化识别**:人脸底库载入内存 numpy 矩阵,一次点积完成检索;注册/删除即时生效
- **可插拔任务架构**:`class_path` 动态加载,跌倒检测等扩展任务零改动接入(见 docs/PLUGIN_GUIDE.md)
- **全参数配置化**:`default.yaml` → `profiles/{desktop,balanced,edge_minimal}.yaml` → 摄像头个性化 JSONB,代码零硬编码
- **工业化工程**:Alembic 版本化迁移、api→services→repositories 分层、引擎池共享显存、断线自动重连、事件自动落库与 WS 推送
- **流畅预览**:摄像头预览使用 `binary_jpeg_v1` 二进制 JPEG、单路编码一次、多订阅者单槽位丢旧帧,慢客户端不会拖住其他客户端
- **摄像头实时注册**(动作引导五步):`getUserMedia` 实时取景 + 检测框叠加,系统引导 正脸→左转→右转→抬头→低头,达标自动采帧,多角度特征原子入库(`source='camera'`),即时生效;不含活体防伪,摄像头流不落盘

## 快速开始

> **GPU 环境版本对应（重要）**:`onnxruntime-gpu` 与 CUDA 运行库必须匹配，否则推理会**静默降级为 CPU**
>（启动日志出现 `请求 CUDA 但实际使用 CPUExecutionProvider` / `cublasLtXX_XX.dll 缺失` 即为该问题）：
>
> | onnxruntime-gpu | CUDA 运行库 | 适用场景 |
> |---|---|------|
> | 1.20.x ~ 1.22.x | CUDA 12.x（NVIDIA 546+ 驱动自带） | RTX 30/40 系推荐 |
> | 1.29.x | CUDA 13.x（需另装运行库） | 高端卡 + 手动装 CUDA 13 |
>
> 排查：`python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`，
> 若列表含 `CUDAExecutionProvider` 则 GPU 就绪；仅 `CPUExecutionProvider` 则需按上表重装匹配版本
>（推荐 `pip install onnxruntime-gpu==1.21.0`，与 CUDA 12.6 实测匹配，detect ~285ms→~10ms）。更省事的做法是直接用项目自带 GPU 就绪虚拟环境：`D:\ai-monitor-1.1.0\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000`。`requirements.txt` 已锁定 CUDA 12.x 推荐档。
> 真判据：用 `InferenceSession` 实证(`get_available_providers()` 是假阳性)——`python -c "import onnxruntime as ort; s=ort.InferenceSession(r'D:\ai-monitor-1.1.0\models\buffalo_s\det_500m.onnx', providers=['CUDAExecutionProvider','CPUExecutionProvider']); print(s.get_providers())"`，打印 `['CUDAExecutionProvider','CPUExecutionProvider']` 即 GPU 真正可用。


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

> 模型:默认高精度档 `models/buffalo_l/`(SCRFD-10G 检测 + ArcFace ResNet50 识别);
> buffalo_l 约 275MB、识别器 166MB 超 GitHub 单文件限制,不随仓库提供 —— 首次使用请下载:
> `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip`
> 解压到 `models/buffalo_l/`(zip 内文件直接放在该目录)。
> 轻量档 `models/buffalo_s/`(det_500m + w600k_mbf)已随仓库提供,切换 `vision.model_pack` 即可;
> 若本地无模型,insightface 也会按官方通道自动下载。
> ⚠️ 切换模型包后 embedding 空间不兼容,**底库需重新注册**。

## 文档

| 文档 | 内容 |
|------|------|
| [docs/API.md](docs/API.md) | 全部 REST/WebSocket 接口说明(运行时另见 `/docs` OpenAPI) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 分层架构、关键设计决策、数据模型 |
| [docs/PLUGIN_GUIDE.md](docs/PLUGIN_GUIDE.md) | 如何开发扩展任务(跌倒检测等) |

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
  recognition:
    threshold: 0.40
    quality: { min_det_score: 0.60, min_face_size: 80 }
    temporal: { min_valid_samples: 3, max_samples_per_track: 8, top_k: 3 }
stream:
  max_height: 480       # 默认预览高度;局域网高画质可设 720
  jpeg_quality: 70
  push_fps: 20
tasks:
  face_recognition: { enabled: true, class_path: app.tasks.builtin.face_recognition_task.FaceRecognitionTask }
  fall_detection:  { enabled: false, class_path: null }   # 扩展任务预留
```

级联优先级:摄像头 `cameras.config`(JSONB)> profile 文件 > default.yaml。

实时识别只使用通过质量筛选的检测帧，并按同一轨迹/候选身份聚合 Top-K 相似度后确认。未来如需切换 embedding 模型，建议为每条 embedding 保存 `model_pack`、模型指纹和维度，避免混用不同特征空间；本次不做数据库迁移或底库重建。
实时识别只使用通过质量筛选的检测帧，并按同一轨迹/候选身份聚合 Top-K 相似度后确认。未来如需切换 embedding 模型，建议为每条 embedding 保存 `model_pack`、模型指纹和维度，避免混用不同特征空间；本次不做数据库迁移或底库重建。

预览分辨率只影响 WebSocket JPEG，不改变 InsightFace 的采集/推理分辨率。默认 480p 适合普通局域网；需要更清晰的局域网预览时，将摄像头的 `stream.max_height` 显式改为 720。已有摄像头保存的 `stream.max_height: 0`(原生)不会被自动覆盖，需在系统设置中手动调整。

## 项目结构

```
backend/       FastAPI 服务(api → services → repositories → models → migrations)
vision/        纯推理内核(camera / engine / tracker / pipeline / tasks,唯一依赖 InsightFace)
configs/       全参数 YAML(default + 3 档 profile 级联)
frontend/      React 18 + AntD 5(样式/菜单/配色集中 src/config)
models/        本地 InsightFace 模型(buffalo_s)
docs/          接口 / 架构 / 插件开发文档
face_db/       头像与旧 pickle 数据(可选迁移)
```

## 测试

```bash
python -m pytest tests/ -v
```
