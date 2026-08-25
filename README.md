# AI Smart Monitoring System

<p align="center">
  <img src="https://img.shields.io/badge/Version-v2.1-brightgreen" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.6-green?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/FastAPI-async-red?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-18-61dafb?logo=react" alt="React">
  <img src="https://img.shields.io/badge/InsightFace-0.7.3-orange" alt="InsightFace">
  <img src="https://img.shields.io/badge/YOLOv8--Pose-TensorRT_FP16-76b900?logo=nvidia" alt="YOLOv8-Pose">
  <img src="https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/License-LYUN-lightgrey" alt="License">
</p>

<p align="center">
  <b>A production-grade real-time edge AI monitoring platform — SCRFD detection + ByteTrack tracking + ArcFace recognition<br>+ YOLOv8-Pose fall detection in an isolated GPU Worker (TensorRT FP16) + React web UI + PostgreSQL persistence + zero-loss event pipeline.</b>
</p>

---

## Introduction

**AI Smart Monitoring System** is a production-grade real-time AI monitoring platform that fuses **face recognition** and **fall detection** into a single unified web system for edge deployment.

It combines state-of-the-art components — SCRFD for lightweight face detection, ByteTrack for multi-object tracking, InsightFace ArcFace for face recognition, YOLOv8-Pose (TensorRT FP16) for human pose estimation, a time-based fall state machine, an isolated GPU Worker process with crash-safe event journaling, and a pluggable VisionTask layer — into an optimized pipeline delivered as a full web application (FastAPI backend + React frontend + PostgreSQL).

Designed for scenarios where cloud dependency, latency, or subscription cost is unacceptable:

- **Smart Security** — face-based access control, visitor logging, stranger alerts
- **Elderly Care** — real-time fall detection with camera isolation, nursing home monitoring
- **Smart Hospital** — patient movement monitoring, bed-exit alerts, identity confirmation
- **Retail Analytics** — customer counting, recognition-based analytics
- **Smart Home** — family member recognition, automation triggers, in-home fall alerts
- **Edge AI Research** — multi-camera pipeline benchmarking, ONNX vs TensorRT comparison

---

## Demo

```
┌───────────────────────── AI Monitor Web UI (React) ─────────────────────────┐
│  Monitor Grid                                    Events   Faces   Settings │
│ ┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────────┐ │
│ │ cam-0  ● LIVE        │  │ test-fall-cam  ● LIVE│  │ cam-2  ● LIVE      │ │
│ │  ┌────────┐          │  │     ┌────────┐       │  │   (no person)      │ │
│ │  │ Byron  │ skeleton │  │     │ ⚠ #1   │       │  │                    │ │
│ │  │ (0.87) │ overlay  │  │     │ fallen │       │  │                    │ │
│ │  └────────┘          │  │     └────────┘       │  │                    │ │
│ │  face bbox (green)   │  │  17-kpt skeleton(red)│  │                    │ │
│ └──────────────────────┘  └──────────────────────┘  └────────────────────┘ │
│  [EVENT] fall_detected  cam=test-fall-cam  track=1  score=0.90             │
│  [EVENT] recognition    cam=cam-0  name=Byron  similarity=0.87             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> *Face bounding-box (green=identified / gray=unknown) + 17-keypoint skeleton + fall status (teal Normal / orange Potential / red Fallen), all rendered on Canvas over a WebSocket binary JPEG preview stream.*

---

## Features

### Face Recognition Pipeline

- [x] **SCRFD Face Detection** — InsightFace official kernel (buffalo_l: det_10g / buffalo_s: det_500m), CUDA GPU inference, 5-point landmarks
- [x] **ByteTrack Multi-Object Tracking** — Kalman filter + two-tier association (high/low confidence), stable `track_id`
- [x] **Detection Throttling** — `det_interval` skips frames between detections; tracker prediction fills the gaps
- [x] **ArcFace Recognition** — 512-dim embeddings via InsightFace 0.7.3; cooldown scheduler (new track → cooldown → exponential backoff → re-verify)
- [x] **Vectorized Gallery Search** — in-memory numpy matrix, single `np.dot` cosine similarity; register/delete takes effect instantly
- [x] **Temporal Stability Aggregation** — top-K sample averaging before identity confirmation, rejects single-frame mis-hits
- [x] **Face Quality Gate** — min detection score + min face size pre-screening before embedding consumption
- [x] **Guided Camera Registration** — 5-step action guidance (front → left → right → up → down), auto-capture on threshold, multi-angle atomic commit with server-side re-validation
- [x] **Batch Import / Photo Registration / Video Registration** — up to 32 photos per batch, registration-quality screening (blur / yaw / pitch / duplicate similarity)

### Fall Detection (v2.1 — TensorRT)

- [x] **YOLOv8-Pose** — 17-keypoint human pose estimation (COCO format)
- [x] **TensorRT FP16 Engine** — auto-preferred when `yolov8n-pose.engine` exists (mtime-guarded); transparent fallback to `.pt`; **4.65ms vs 9.81ms per frame (2.11×)**, 100% detection consistency vs PyTorch
- [x] **GPU-hard, no CPU fallback** — model forward, pre/post-processing, NMS and keypoint math all on `cuda:0` + FP16; dual verification (`torch.cuda.is_available()` + real model load)
- [x] **Isolated GPU Worker Process** — own venv (`pose_plugin/.venv-worker`), never blocks the web/event loop; single shared engine + CUDA context across cameras
- [x] **Dedicated rx/tx Pipes** — bypasses the Windows stdio handle corruption caused by torch/ultralytics; binary mode (`O_BINARY`) protects the 4-byte length prefix; per-write lock prevents frame interleaving
- [x] **Shared-Memory Frame Transport** — double-slot regions with seqlock + named mutex; attach cache in the Worker (zero per-frame kernel-object churn)
- [x] **Heartbeat Watchdog** — 1s PING / 3s timeout, automatic worker restart with new epoch (limit 3) then circuit-break
- [x] **Time-based Fall State Machine** — long-term states `NORMAL / POTENTIAL / FALLEN` (`RECOVERED` is a transition event); monotonic-time durations, evidence-gap continuity break
- [x] **Four-path Evidence** — geometry (AR + torso inclination) + physics (rotation energy / gravity factor) + head descent + fast-dynamic; upright-gated EMA body-height baseline, standing head baseline
- [x] **Camera Isolation** — each `(camera_id, session)` has an independent tracker + state machines
- [x] **Health API** — `GET /api/system/fall-runtime` redacted snapshot: worker state, heartbeat age, GPU name, spool backlog, per-camera submitted/analyzed/replaced counters

### Real-time Streaming & UI

- [x] **WebSocket Binary Preview** — `binary_jpeg_v1` packets (type + u64 frame_id + JPEG), single encode per frame, multi-subscriber fan-out
- [x] **Bounded Sender per Subscriber** — single-slot latest-frame semantics; a slow client drops its own frames, never stalls others
- [x] **Canvas Overlay** — face bbox + identity labels, pose skeleton + fall status projected to preview pixels (source→preview coordinate transform, clamped)
- [x] **Overlay TTL + Frame Matching** — stale analytics (>1200ms) expire; frame-id mismatch refuses drawing (no ghost skeletons on wrong frames)
- [x] **Frontend Event Dedup** — LRU cache (4096 entries, 24h TTL) + sessionStorage, keyed by `event_id` first, `dedupe_key` second; protocol-violating events rejected, not guessed
- [x] **Reconnect Guard** — exponential backoff WebSocket reconnection with generation check (no zombie sockets)

### Reliability & Event Pipeline

- [x] **Transactional Outbox** — events + outbox rows committed in a single PostgreSQL transaction; at-least-once WebSocket delivery with exponential backoff
- [x] **Exactly-once Ingest** — `dedupe_key` unique constraint + atomic `ON CONFLICT`; `incident_id` in the key keeps re-detected falls distinct after worker restarts
- [x] **Crash-safe Worker Journal** — SQLite WAL + `synchronous=FULL`; parent drain ACK-on-success, backoff retry, poison-pill spill to parent `EventSpool` after 5 failed attempts
- [x] **Bounded Queues Everywhere** — ingress queue (1024), outbox capacity (10k) with advisory-lock protection, journal/spool capacity gates
- [x] **Graceful Degradation** — GPU/Worker failure sets fall health to `unavailable` without affecting face recognition or core web functions

### Platform

- [x] **Pluggable VisionTask** — tasks loaded dynamically by `class_path` from YAML, zero changes to main loop / routes / frontend (see `docs/PLUGIN_GUIDE.md`)
- [x] **Full Config Cascade** — `configs/default.yaml` → `configs/profiles/{desktop,balanced,edge_minimal}.yaml` → per-camera JSONB; zero hard-coded params
- [x] **Industrialized Backend** — Alembic versioned migrations, api → services → repositories layering, engine-pool shared VRAM, RTSP open-timeout + exponential backoff reconnect
- [x] **Async Broadcast** — `asyncio.gather` fan-out; one dead socket never blocks the event loop
- [x] **Measured Telemetry** — per-stage latency (capture/detect/tasks/emit), GPU inference ms, queue-wait ms, end-to-end ms on every fall result
- [x] **Full Test Suites** — 132 backend + 248 pose-plugin + 49 frontend tests

### Coming Soon

- [ ] Multi-camera RTSP management in the web UI
- [ ] Labeled-dataset fall evaluation harness (`evaluate_fall_dataset.py`)
- [ ] Mobile app push notifications
- [ ] Authentication layer for the API
- [ ] OpenVINO / Jetson custom backends

---

## System Architecture

### End-to-End Pipeline

```
┌─────────────────────────────── Backend (FastAPI, .venv) ────────────────────────┐
│  Camera Pipeline (vision/, one thread per camera)                                │
│    OpenCV source ─▶ SCRFD(降频) ─▶ ByteTrack ─▶ VisionTask plugins               │
│         │                              │                                          │
│         │                    FallDetectionTask (host adapter)                     │
│         │                              │ shared-memory frame + INFER_FRAME       │
│         ▼                              ▼                                          │
│    preview encode ──▶ WS binary JPEG   PoseRuntime (lifecycle, heartbeat,        │
│         │                             epoch filtering, journal drain)             │
│         ▼                              │                                          │
│    EventIngress ◀── parent drain ◀── Worker Journal (ack/retry/poison-pill)      │
│         │                                                                        │
│         ├─▶ PostgreSQL events + event_outbox (single tx) ─▶ /api/events           │
│         └─▶ WebSocket /ws/events (outbox dispatcher, at-least-once)               │
└──────────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────── GPU Worker (pose_plugin/.venv-worker) ───────────────┐
│  dedicated rx/tx pipes (bypasses torch stdio corruption on Windows)              │
│    INFER_FRAME ─▶ read shared-memory slot ─▶ YOLOv8n-Pose TensorRT FP16          │
│      ─▶ per-camera PoseTracker (Hungarian) ─▶ evidence features                  │
│      ─▶ time-based state machine ─▶ transitions ─▶ Worker Journal (SQLite WAL)   │
│      ─▶ INFERENCE_RESULT (tracks + timing metrics) ─▶ parent overlay mount       │
└──────────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────── Frontend (React 18) ─────────────────────────────┐
│  Monitor grid (Canvas) ─▶ face bbox + pose skeleton + fall status overlay        │
│  WS consumers ─▶ eventDedupe ─▶ AlertBanner / EventLog / FaceLibrary / Settings  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Tech per Stage

| Stage | Technology | Runs In |
|-------|-----------|---------|
| Capture | OpenCV VideoCapture (camera / RTSP / file) | backend thread |
| Face Detect + Embed | SCRFD + ArcFace (InsightFace, onnxruntime-gpu CUDA) | backend thread |
| Face Track | ByteTrack (Kalman + two-tier association) | backend thread |
| Preview Encode | OpenCV JPEG (resize + quality, per-camera thread) | backend encode thread |
| WS Transport | FastAPI WebSocket, binary packets + JSON messages | event loop |
| Pose Inference | YOLOv8n-Pose, TensorRT FP16 engine (fallback .pt) | **GPU Worker process** |
| Pose Track | PoseTracker (Hungarian, per camera+session) | GPU Worker |
| Fall Judgment | 4-path evidence + time-based state machine | GPU Worker |
| Event Handoff | Worker Journal → parent drain → EventIngress | Worker + backend |
| Persistence | PostgreSQL events + event_outbox (single tx) | backend event loop |
| Overlay | Canvas 2D, source→preview projection | browser |

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Fall inference in an isolated GPU Worker process | Heavy torch stack never imported by the web process; single shared engine + VRAM; crash isolation with epoch-based restart |
| Dedicated rx/tx pipes instead of stdin/stdout | torch/ultralytics corrupt Windows CRT stdio handles; dedicated inheritable handles are immune |
| `O_BINARY` pipes + per-write lock | CRT text mode would CRLF-corrupt the 4-byte length prefix; lock prevents heartbeat/infer frame interleaving |
| TensorRT engine with mtime guard + `.pt` fallback | 2.11× speedup when the engine exists; engine is GPU/driver-bound, so any staleness or corruption silently falls back to `.pt` |
| Shared memory (frames) + pipe (control) | Zero-copy-ish frame path; control messages stay small and structured |
| Heartbeat watchdog (1s/3s) | A hung (not dead) worker previously stayed `READY` forever; now detected and restarted, then circuit-broken |
| Journal ACK-on-success + backoff + poison pill | No event loss on crash or DB outage; unrecoverable events spill to a durable spool instead of being dropped |
| `incident_id` in `dedupe_key` | Worker restart resets track IDs; without it, a real second fall is rejected as a duplicate (event loss) |
| Transactional outbox | DB write and broadcast request committed atomically; WebSocket delivery retried with backoff |
| Time-based (not frame-based) fall thresholds | Robust against pose-track ID switches and variable FPS |
| Latest-only analytics + TTL overlay | Overlay always reflects the newest pose; stale results expire instead of ghosting |
| Upright-gated EMA baselines | Body-height / head-position baselines track perspective changes but freeze during falls (no self-confirming evidence) |
| Bounded sender per WS subscriber | Slow clients drop their own frames; the encode thread and other viewers are never stalled |

---

## Tech Stack

### Face Recognition

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| Face Detection | SCRFD det_10g (InsightFace) | 0.7.3 | Face detection + 5-point landmarks |
| Recognition | ArcFace w600k_r50 (buffalo_l) | 0.7.3 | 512-dim embedding extraction |
| Inference Runtime | onnxruntime-gpu | **1.21.0** (CUDA 12.6) | GPU execution provider |
| Tracking | ByteTrack (custom impl) | — | Kalman + two-tier association |
| Gallery Search | numpy `np.dot` | — | Vectorized cosine similarity |

### Fall Detection (isolated GPU Worker)

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| Pose Estimation | YOLOv8n-Pose | ultralytics 8.4.x | 17-keypoint COCO pose |
| Inference Engine | **TensorRT FP16** (auto) / PyTorch fallback | TRT 10/11 · torch 2.7.0+cu126 | 4.65ms/frame measured (RTX 4060 Laptop) |
| IPC | Dedicated anonymous pipes (`O_BINARY`) | — | Length-prefixed JSON control protocol |
| Frame Transport | Shared memory double-slot + seqlock + named mutex | — | Torn-frame detection, zero-copy reads |
| State Machine | Time-based NORMAL/POTENTIAL/FALLEN | — | Monotonic-clock durations |
| Persistence | SQLite WAL Worker Journal + EventSpool | — | Crash-safe event handoff |
| Watchdog | Heartbeat thread (PING/PONG) | — | 1s interval, 3s timeout, 3 restarts → circuit break |

### Platform

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| Backend | FastAPI + SQLAlchemy async + asyncpg | — | REST + WebSocket + services/repos layering |
| Database | PostgreSQL | 16 | events, outbox, cameras, identities |
| Migrations | Alembic | — | Versioned schema (`backend/migrations`) |
| Frontend | React 18 + Ant Design 5 + Vite | — | Monitor grid, overlays, registration, event log |
| Realtime | WebSocket (binary JPEG + JSON) | — | Preview stream, detections, analytics, events |
| Tests | pytest + Vitest/JSDOM | — | 132 + 248 + 49 tests |
| Deployment Profiles | desktop / balanced / edge_minimal | — | YAML cascade per hardware tier |

---

## Project Structure

```
ai-monitor/
│
├── install_python.ps1                  # One-click setup (backend venv + worker venv + models)
├── publish_release.ps1                 # Release packaging/distribution script
├── docker-compose.yml                  # db + backend + frontend (web-only; see Deployment)
│
├── backend/                            # FastAPI service
│   ├── app/
│   │   ├── api/                        # REST + WebSocket routes (cameras/events/faces/system/health)
│   │   ├── services/                   # pipeline_manager, event_ingress, model_manager, ...
│   │   ├── repositories/               # DB access (event outbox, camera, identity)
│   │   ├── models/ / schemas/          # SQLAlchemy models + pydantic schemas
│   │   └── tasks/builtin/              # FaceRecognitionTask (VisionTask plugin)
│   ├── migrations/                     # Alembic versions
│   └── requirements.txt
│
├── vision/                             # Pure inference core (no web deps)
│   ├── camera.py                       # Frame sources: camera / RTSP / file + reconnect backoff
│   ├── engine.py                       # InsightFace wrapper (GPU-first, fail-fast on missing models)
│   ├── pipeline.py                     # Per-camera thread: capture → detect → track → tasks
│   ├── tracker.py                      # ByteTrack
│   └── tasks.py                        # VisionTask ABC
│
├── pose_plugin/                        # Fall-detection plugin (self-contained)
│   ├── ai_monitor_pose/
│   │   ├── task.py                     # FallDetectionTask (host-side VisionTask adapter)
│   │   ├── runtime.py / runtime_registry.py   # Worker lifecycle, heartbeat, journal drain
│   │   ├── shared_frames.py            # Shared-memory double-slot regions
│   │   ├── event_spool.py              # Durable poison-pill spool
│   │   └── worker/                     # GPU Worker process (service, launcher, tracker, state machine, features)
│   ├── scripts/                        # build_trt_engine.py, engine_benchmark.py, gpu_smoke.py, soak.py
│   ├── tests/                          # 248 tests (unit/integration/gpu)
│   └── models/                         # yolov8n-pose.pt (+ optional .engine, capacity manifest)
│
├── configs/
│   ├── default.yaml                    # All tunable parameters (baseline)
│   └── profiles/                       # desktop / balanced / edge_minimal overrides
│
├── frontend/                           # React 18 + AntD 5 + Vite
│   └── src/                            # pages, components, stream protocol, hooks
│
├── tests/                              # 132 backend/integration tests
├── docs/                               # API.md / ARCHITECTURE.md / PLUGIN_GUIDE.md
├── models/                             # buffalo_s shipped; buffalo_l downloaded (see below)
└── face_db/                            # Face avatars (runtime data)
```

---

## Installation

### Prerequisites

| Software | Version | Check | Required For |
|----------|---------|-------|-------------|
| Python | 3.12 | `python --version` | All components |
| Node.js | 16+ | `node -v` | Frontend |
| PostgreSQL | 16 | `pg_isready` | Persistence |
| NVIDIA GPU + driver | GTX 1060 6GB+ | `nvidia-smi` | Face GPU + fall Worker |
| CUDA runtime | 12.6 (matches wheels) | driver report | onnxruntime-gpu 1.21.0 / torch cu126 |

> Fall detection (pose Worker) is **Windows-native** (named pipes + shared memory). On Linux, face recognition + the web platform work; the fall Worker is not yet supported.

### One-click Setup (Windows)

```powershell
Set-ExecutionPolicy -Scope Process Bypass   # one-time
.\install_python.ps1
```

The script creates **both** virtual environments (`.venv` backend + `pose_plugin/.venv-worker` GPU worker), installs PyTorch 2.7.0+cu126, verifies GPU availability, downloads & SHA-256-checks `yolov8n-pose.pt`, and optionally builds the frontend.

| Flag | Effect |
|---|---|
| `-SkipFrontend` | skip `npm install` + build |
| `-SkipModels` | skip model download |
| `-OnlyBackend` / `-OnlyWorker` | install only one environment |
| `-Mirror` | use a CN PyPI mirror for backend deps |

### Manual Setup

```bash
# 1. Database
docker run -d --name ai-monitor-db -p 5432:5432 \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=123456 -e POSTGRES_DB=ai_monitor \
  postgres:16-alpine

# 2. Backend (project root .venv, Python 3.12)
pip install -r backend/requirements.txt
cd backend && alembic upgrade head
# configure backend/.env (AIM_DATABASE_URL, AIM_PROJECT_ROOT, ...)
uvicorn app.main:app --host 0.0.0.0 --port 8000   # or: python -m app.main

# 3. Frontend (another terminal)
cd frontend && npm install && npm run dev   # http://localhost:3000 (proxies /api, /ws)
```

### Models

| Model | Location | Notes |
|---|---|---|
| SCRFD + ArcFace **buffalo_s** | `models/buffalo_s/` | Shipped with the repo (lightweight) |
| SCRFD + ArcFace **buffalo_l** | `models/buffalo_l/` | **Default (high accuracy)**; 166MB file exceeds GitHub limits — download the [official zip](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) and extract here |
| YOLOv8n-Pose | `pose_plugin/models/yolov8n-pose.pt` | Downloaded + SHA-256-verified by the install script; never auto-downloaded at runtime |

> ⚠️ **Embeddings are model-pack-bound.** The face gallery must be re-registered if you switch `vision.model_pack` (buffalo_l ↔ buffalo_s spaces are not interoperable). Missing model files fail fast at engine construction — the system never silently downloads 280MB at runtime.

### Optional: TensorRT Engine (recommended, ~2× pose inference)

With the worker venv active and `tensorrt` installed:

```bash
cd pose_plugin
python -c "from ultralytics import YOLO; YOLO('models/yolov8n-pose.pt').export(format='engine', half=True, imgsz=640, device=0)"
# alternative (no modelopt dependency):
python scripts/build_trt_engine.py
```

The Worker auto-prefers `yolov8n-pose.engine` when it exists and is not older than the `.pt`; any load failure falls back to `.pt` with a stderr notice. Engines are GPU/driver/TensorRT-bound — rebuild per machine (`pose_plugin/models/README.md`).

Verify:

```bash
python scripts/engine_benchmark.py    # latency + accuracy + VRAM vs .pt
```

---

## Usage

### Start

```bash
# Backend  (from backend/, using the project .venv)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (from frontend/)
npm run dev        # http://localhost:3000
```

Enabled cameras auto-restore on backend startup. OpenAPI docs at `http://localhost:8000/docs`.

### Web UI

| Page | Function |
|------|----------|
| **Monitor** | Live camera grid: face boxes + identity, pose skeleton + fall status, snapshot download |
| **Event Log** | Fall/recognition events with filters, acknowledgement, batch delete |
| **Face Library** | Register (guided 5-pose capture / photo / batch import / video), edit, delete |
| **Settings** | Camera management (source / profile / resolution), start/stop, runtime info |

### Toggle Fall Detection

```yaml
# configs/default.yaml
tasks:
  fall_detection:
    enabled: true     # ON (default)
    mode: "shadow"    # shadow = persist only; alert = persist + real-time WS push
```

### Runtime Health

```bash
curl http://localhost:8000/api/health                 # liveness
curl http://localhost:8000/api/system/fall-runtime    # worker state, heartbeat age, spool backlog
```

### Worker Diagnostics

Worker stderr is discarded by default. Redirect it to a file before starting the backend:

```powershell
$env:AI_MONITOR_POSE_WORKER_STDERR = "D:\ai-monitor\pose_plugin\var\worker-stderr.log"
```

GPU / capacity checks: `pose_plugin/scripts/gpu_smoke.py`, `integration_smoke.py`, long-run `soak.py`.

### Snapshot & Events API

See [docs/API.md](docs/API.md) for the full reference — cameras CRUD, start/stop, snapshot JPEG, events list/ack/delete, face registration analyze/commit, search, batch import, WebSocket channels (`/ws/cameras/{id}`, `/ws/events`).

---

## Configuration

### Profiles

| Profile | Vision input | Face det interval | Fall target FPS | Intended hardware |
|---------|-------------|-------------------|-----------------|-------------------|
| `desktop` | 640px | 2f | 8 | GPU workstation (default baseline) |
| `balanced` | 480px | 3f | 6 | Mid-tier CPU/GPU |
| `edge_minimal` | 320px | 5f | 4 | Low-power edge |

**Priority: per-camera JSONB > profile > `default.yaml`** — everything is a parameter; no hard-coded values. Only `enabled` / `mode` / `target_fps` differ per profile; the full baseline lives in `default.yaml`.

### Key Parameters (`configs/default.yaml`)

```yaml
vision:
  model_pack: "buffalo_l"     # gallery is pack-bound — re-register faces if changed
  device: "cuda"
  det_size: [640, 640]
  det_interval: 2
  recognition:
    threshold: 0.40
    cooldown_frames: 300

stream:
  max_height: 480             # preview downscale
  jpeg_quality: 70
  push_fps: 20

tasks:
  fall_detection:
    enabled: true
    mode: "shadow"            # shadow | alert
    worker:                   # heartbeat watchdog (implemented)
      heartbeat_interval_s: 1
      heartbeat_timeout_s: 3
      restart_max_attempts: 3
    model:
      imgsz: 640
      confidence: 0.35
    runtime:
      overlay_ttl_ms: 1200
      worker_journal_path: "pose_plugin/var/worker-transition-journal.sqlite3"
      event_spool_path: "pose_plugin/var/fall-event-spool.sqlite3"
    algorithm:
      min_fall_pose_duration_s: 3.50
      recovery_duration_s: 1.00
```

### Quick Tuning

```
Goal                     │ Setting
─────────────────────────┼────────────────────────────────────────────────────
Faster pose inference    │ Build the TensorRT engine (2.11×, see Installation)
Lower GPU load           │ scheduler.target_fps: 4, vision.det_interval: 3
Real-time fall alerts    │ mode: "alert" (persist + WS push)
Fewer fall false alarms  │ algorithm.min_fall_pose_duration_s: 5.0
Lighter preview          │ stream.max_height: 360, jpeg_quality: 60
CPU-only box (no fall)   │ vision.device: cpu; tasks.fall_detection.enabled: false
```

---

## Performance

Measured on **RTX 4060 Laptop GPU / CUDA 12.6 / torch 2.7.0+cu126 / TensorRT FP16**.

### Pose Inference (yolov8n-pose, imgsz 640, conf 0.35)

| Engine | avg | P50 | P95 | VRAM peak |
|--------|-----|-----|-----|-----------|
| PyTorch FP16 (`.pt`) | 9.81 ms | 9.16 ms | 13.30 ms | 16.5 MB |
| **TensorRT FP16 (`.engine`)** | **4.65 ms** | **4.43 ms** | **6.24 ms** | **12.3 MB** |

Accuracy sanity (20 sampled real frames): **100% frame-level detection consistency**, keypoint-visibility delta +0.05 targets/frame (no degradation).

### Per-Stage Latency (face pipeline, single camera)

| Stage | Latency (typical) |
|-------|-------------------|
| Capture (file source) | 4–9 ms |
| SCRFD detect + track (det_interval=2) | ~11 ms amortized |
| Recognition search (gallery hit) | < 0.5 ms (vectorized) |
| Tasks + emit | ~2 ms |
| Preview encode (480px JPEG) | per-camera thread, off the event loop |

### Reliability Numbers (production observation)

- Worker journal: **27/27 transitions ACKed**, outbox backlog **0** under continuous load
- Heartbeat watchdog: PONG round-trip observed 250–1000 ms
- Event pipeline: end-to-end (offer → persisted) ~35 ms typical, measured per event (`end_to_end_ms`, `queue_wait_ms`, `gpu_inference_ms` on every result)

### Multi-Camera Scaling

| Cameras | Face detect | Fall inference (shared worker) | Notes |
|---------|-------------|-------------------------------|-------|
| 1–2 | Per-camera threads | 8 fps each, latest-only | Native |
| 3–4 | Engine pool shares VRAM | Fair latest-frame scheduling | Capacity-manifest gated |
| 4+ | det_interval ↑ | target_fps ↓ | Watch `system/fall-runtime` counters |

---

## Deployment Guide

### Windows Native (recommended — full features)

```powershell
.\install_python.ps1          # envs + models
# start PostgreSQL, run alembic, start backend + frontend (see Usage)
```

### Docker (web-only fallback)

```bash
docker compose up -d    # db + backend + frontend
```

> ⚠️ The fall-detection Worker requires Windows named pipes + shared memory and is **not containerized** — in Docker the web platform and face recognition run, fall detection stays disabled. Use Windows-native for the full pipeline.

### Capacity Planning

`pose_plugin/models/capacity-cuda0.json` records the measured GPU throughput fingerprint (GPU / driver / CUDA / model hash). The runtime gates camera admission by this manifest (`capacity_headroom_ratio: 0.75`). After a GPU/driver change, re-run the capacity benchmark and update the manifest.

---

## Test

```powershell
# Backend + integration (132 tests)
.venv\Scripts\python.exe -m pytest tests/ -q

# Pose plugin (248 tests; PYTHONPATH must include the repo root)
$env:PYTHONPATH = "D:\ai-monitor-1.1.0"
.venv\Scripts\python.exe -m pytest pose_plugin/tests/ -q

# Frontend (49 tests)
cd frontend; npm test
```

---

## Changelog

### v2.1 — TensorRT + Reliability & Algorithm Batch (2026-08)

- **新增** TensorRT FP16 engine auto-preference with mtime guard + `.pt` fallback (2.11× pose speedup, zero accuracy loss)
- **新增** `scripts/build_trt_engine.py` (modelopt-free engine builder) + `scripts/engine_benchmark.py` (+ tests)
- **新增** Heartbeat watchdog (1s/3s) — hung workers now restart instead of staying `READY` forever
- **修复** `dedupe_key` now includes `incident_id` — second real falls no longer rejected as duplicates after worker restarts (event loss)
- **修复** Dedicated-pipe write lock + partial-write loop (heartbeat/infer frame interleaving)
- **修复** WS endpoints hold DB sessions only briefly (pool exhaustion with 16+ sockets); EnginePool leak; concurrent `start_camera` race; sequential broadcast blocking
- **修复** RTSP open timeout — FFMPEG env set *before* `VideoCapture` construction (was ineffective after)
- **修复** Fall algorithm semantics: real vertical-velocity backfill (was rule-score, ~8× gravity distortion), upright-gated EMA body-height baseline, standing-head baseline, real timing metrics
- **改进** Shared-memory attach cache, drain throttle (250 ms), lock-scope fix in `offer_frame`, analytics serialization hoisted, no-viewer frame-copy skip
- **改进** health snapshot: real `mode` / heartbeat age / spool pending; missing-model fail-fast (no silent 280 MB downloads)
- **测试** 132 backend + 248 pose + 49 frontend, all green

### v2.0 — Web Platform Rewrite

- FastAPI + React + PostgreSQL architecture; isolated GPU Worker plugin (`pose_plugin`); transactional outbox; WS binary streaming; guided registration; Alembic migrations; pluggable VisionTask; config cascade.

### v1.x — Legacy Standalone Pipeline

- Single-process desktop pipeline (SCRFD / ByteTrack / NCNN edge backends / behavior layer), see the v10 line for history.

---

## Roadmap

```
v2.1 ✅  TensorRT FP16 engine + reliability/algorithm batch (current)
v2.0 ✅  Web platform rewrite (FastAPI + React + PostgreSQL + GPU Worker plugin)
v1.x ✅  Legacy standalone pipeline (SCRFD / ByteTrack / NCNN / behavior layer)
v2.2 🔜  RTSP multi-camera management UI
v2.3 🔜  Labeled-dataset fall evaluation harness
v2.4 🔜  API authentication layer
v2.5 🔜  Mobile push notifications
v2.6 🔜  OpenVINO / Jetson backends
```

---

## FAQ

<details>
<summary><b>Faces all show Unknown after changing model_pack</b></summary>

`buffalo_l` and `buffalo_s` embedding spaces are **not interoperable**. Switching packs invalidates the whole gallery — re-register all faces. The default is `buffalo_l`; `models/buffalo_l/` must be downloaded manually (166 MB exceeds GitHub limits). Missing files now fail fast at startup instead of silently downloading.
</details>

<details>
<summary><b>GPU utilization looks lower than before — is inference on CPU?</b></summary>

Check `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"` — `CUDAExecutionProvider` must be present. With TensorRT, pose inference is 2.11× faster, so *lower* GPU utilization at the same FPS is the expected outcome. The worker stderr log confirms: `[worker] TensorRT engine: yolov8n-pose.engine`.
</details>

<details>
<summary><b>Fall detection not triggering when I fall</b></summary>

Confirmed `fall_detected` requires **3.5 s** of sustained fall posture (`min_fall_pose_duration_s`); shorter drops only raise `fall_potential`. `mode: "shadow"` persists events without real-time alerts — set `mode: "alert"` for push. Check `GET /api/system/fall-runtime` for worker state and per-camera submitted/analyzed counters.
</details>

<details>
<summary><b>Worker stderr shows "engine load failed; fallback to pt"</b></summary>

The TensorRT engine is bound to GPU model / driver / TensorRT version. Rebuild it on this machine (`scripts/build_trt_engine.py`). The system continues correctly on PyTorch in the meantime.
</details>

<details>
<summary><b>High CPU usage with video-file cameras</b></summary>

File sources are read at maximum speed (no real-time throttling), so the pipeline processes e.g. 60+ fps per file. Real cameras/RTSP are rate-limited by the source itself. This is by design for benchmarking.
</details>

<details>
<summary><b>Two backend instances fighting over port 8000</b></summary>

Start the backend exactly once (check `netstat -ano | findstr :8000`). If you run from an IDE, make sure its run configuration points at the project `.venv` interpreter, not the system Python — dependency versions are only guaranteed in `.venv`.
</details>

<details>
<summary><b>How does the event pipeline guarantee no loss?</b></summary>

Worker writes transitions to a WAL journal before responding → parent drains with ACK-on-success + backoff retry (5 attempts) → poison pill spills to a durable SQLite spool → backend ingests events + outbox in one PostgreSQL transaction → WS delivery retried by the outbox dispatcher. Dedup is enforced by the `dedupe_key` unique constraint.
</details>

<details>
<summary><b>How do I see worker logs?</b></summary>

Set `AI_MONITOR_POSE_WORKER_STDERR=<file>` in the backend environment before startup. Worker lifecycle notices (engine choice, fallbacks) appear there.
</details>

<details>
<summary><b>InsightFace compile error on install</b></summary>

Windows needs VS C++ Build Tools ("Desktop development with C++") before `pip install insightface`. The one-click script handles the wheel; manual installs may compile from source.
</details>

---

## License

LYUN License

> Note: Ultralytics YOLOv8 is AGPL-3.0 licensed — commercial use requires either open-sourcing your application or purchasing an Ultralytics enterprise license. InsightFace models are for non-commercial research use per their license.

---

<p align="center">
  <sub>Built for edge AI and real-time web-based vision monitoring.</sub>
</p>
