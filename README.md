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

## 快速开始

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
tasks:
  face_recognition: { enabled: true, class_path: app.tasks.builtin.face_recognition_task.FaceRecognitionTask }
  fall_detection:  { enabled: false, class_path: null }   # 扩展任务预留
```

级联优先级:摄像头 `cameras.config`(JSONB)> profile 文件 > default.yaml。

实时识别只使用通过质量筛选的检测帧，并按同一轨迹/候选身份聚合 Top-K 相似度后确认。未来如需切换 embedding 模型，建议为每条 embedding 保存 `model_pack`、模型指纹和维度，避免混用不同特征空间；本次不做数据库迁移或底库重建。

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
