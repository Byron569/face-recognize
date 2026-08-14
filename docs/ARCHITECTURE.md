# 系统架构(Architecture)

## 1. 总体分层

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser(React 18 + Ant Design 5)                                │
│  Monitor / FaceLibrary / EventLog / Settings                     │
│  · 主题/菜单/配色集中配置(src/config)                              │
└──────────────┬───────────────────────────────┬───────────────────┘
               │ REST /api/*                    │ WebSocket /ws/*
┌──────────────▼───────────────────────────────▼───────────────────┐
│  backend/(FastAPI)                                               │
│  api/          路由层 — 只做参数校验与响应组装                      │
│  schemas/      Pydantic 出入参模型                                 │
│  services/     业务编排(pipeline_manager / face / event / camera) │
│  repositories/ 数据访问(全部 SQL 集中于此)                          │
│  models/       SQLAlchemy ORM                                     │
│  tasks/        可插拔任务(内置 face_recognition;扩展任务注册点)     │
└──────────────┬───────────────────────────────┬───────────────────┘
               │ 依赖注入(配置/底库/引擎)         │ SQLAlchemy async
┌──────────────▼───────────────────────────────▼───────────────────┐
│  vision/(纯推理内核,唯一依赖 InsightFace)                          │
│  camera.py    帧采集(本地/RTSP/文件,自动重连)                      │
│  engine.py    InsightFace 官方 FaceAnalysis 封装(GPU 优先)         │
│  tracker.py   ByteTrack 多目标跟踪(卡尔曼 + 两阶段关联,移植自 IFzhang/ByteTrack) │
│  tasks.py     可插拔任务接口(VisionTask ABC)                       │
│  pipeline.py  工业化主循环:采集→检测→跟踪→任务→输出回调             │
│  events.py    内核数据模型(FaceResult/TrackResult/VisionEvent)     │
│  config.py    全部参数 dataclass(由 YAML 注入)                     │
└──────────────┬───────────────────────────────┬───────────────────┘
               │                               │ Alembic 迁移
┌──────────────▼──────────────┐   ┌────────────▼───────────────────┐
│  models/buffalo_s/          │   │  PostgreSQL                     │
│  det_500m.onnx(SCRFD)       │   │  identities / identity_embeddings │
│  w600k_mbf.onnx(ArcFace)    │   │  recognition_logs / events / cameras │
└─────────────────────────────┘   └────────────────────────────────┘
```

## 2. 关键设计决策

| 决策 | 理由 |
|------|------|
| 内核直用 InsightFace 官方 `FaceAnalysis` | 官方 `get()` 一次完成 SCRFD 检测 + 关键点 + ArcFace embedding,无自研算法,跟上官方模型迭代(切换模型包只需改配置 `model_pack`) |
| 识别比对走内存底库快照(FaceGallery) | embedding 全量载入 numpy 矩阵,比对 = 一次点积(O(N×512),微秒级);底库变更时刷新,避免识别热路径碰数据库 |
| 识别调度在任务插件内 | 新 track 优先 → 冷却 → 失败 backoff → 已识别重验证,参数全部可配;与检测/跟踪完全解耦 |
| EnginePool 共享模型 | 多摄像头共享同一份模型/显存;注册/搜索接口不再重复加载模型 |
| 任务插件 class_path 动态加载 | 新增任务(如跌倒检测)零改动主循环/路由/前端,见 PLUGIN_GUIDE.md |
| 配置级联 default → profile → camera.config | 全参数可配;单摄像头个性化覆盖任意键;代码零硬编码 |
| Alembic 版本化 schema | 数据库演进可追踪、可回滚;启动自动恢复 enabled 摄像头 |

## 3. 单摄像头流水线(线程内)

```
OpenCVFrameSource.read() ──失败──▶ 指数退避重连
        │
        ▼
[det_interval 降频] InsightFaceEngine.detect(frame)
        │  FaceResult: bbox / det_score / kps / embedding(512-d 归一化)
        ▼
ByteTracker.update(detections)    卡尔曼预测 + 两阶段 IoU 关联,输出稳定 track_id
        │  TrackResult: track_id / bbox / identity / embedding
        ▼
for task in tasks:                VisionTask 接口(可插拔)
   should_run(frame_id, ctx) → run(frame, ctx) → [VisionEvent]
        │
        ▼
on_frame 回调 ──▶ PipelineManager ──▶ JPEG 缩放编码 ──▶ /ws/cameras/{id}
on_event 回调 ──▶ EventBridge ──▶ /ws/events 广播 + PostgreSQL 落库
```

线程模型:主线程 = asyncio loop(FastAPI);每个摄像头一个 `VisionPipeline` daemon 线程;
帧/事件通过 `asyncio.run_coroutine_threadsafe` 桥接回 loop,保证 WS/DB 操作在单 loop 内。

## 4. 配置体系

```
configs/default.yaml           全部键的基线(server/database/stream/vision/tasks/camera_defaults/cleanup)
configs/profiles/desktop.yaml  覆盖项(640px, det_interval=2, cuda)
configs/profiles/balanced.yaml 覆盖项(480px, det_interval=3, cuda)
configs/profiles/edge_minimal.yaml 覆盖项(320px, cpu, 低功耗)
cameras.config(JSONB)          单摄像头个性化,覆盖上述任意键
backend/.env                   服务级:AIM_DATABASE_URL / AIM_PROJECT_ROOT / 流参数 / 保留天数
```

推理默认 `device: cuda`(CUDA 优先;无 GPU 环境自动降级 CPU 并告警日志)。

## 5. 数据模型

```
identities 1──N identity_embeddings(ARRAY(Float)[512],L2 归一化)
identities 1──N recognition_logs(ON DELETE SET NULL)
identities 1──N events(ON DELETE SET NULL)
cameras    1──N recognition_logs / events(camera_id 字符串关联)
events.event_type 为 PostgreSQL enum(含 fall_*/intrusion/loitering 预留值)
```

## 6. 目录速查

| 位置 | 内容 | 修改频度 |
|------|------|---------|
| `vision/` | 推理内核 | 算法级改动 |
| `backend/app/tasks/builtin/` | 内置识别任务 | 识别策略调整 |
| `backend/app/api/` | REST/WS 路由 | 接口变更 |
| `backend/app/services/` | 业务编排 | 业务规则 |
| `backend/app/repositories/` | SQL | 查询优化 |
| `backend/migrations/versions/` | 迁移脚本 | schema 变更 |
| `configs/` | 全部可调参数 | 日常调优 |
| `frontend/src/config/` | 主题/菜单/元数据 | 外观调整 |
