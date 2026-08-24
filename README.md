# AI Monitor — Smart Video Monitoring Platform

**A production-grade real-time edge video monitoring platform — InsightFace face recognition (SCRFD + ArcFace) + YOLOv8-Pose fall detection via an isolated GPU Worker + pluggable VisionTask architecture + React web UI + SQL persistence.**

## Introduction

**AI Monitor** is a production-grade real-time video monitoring platform that fuses **face recognition** and **fall detection** into a single unified web system for edge deployment.

It combines state-of-the-art components — SCRFD for lightweight face detection, ByteTrack for multi-object tracking, InsightFace ArcFace for face recognition, YOLOv8-Pose for human pose estimation, a time-based fall state machine, a dedicated PyTorch CUDA GPU Worker, and a pluggable VisionTask layer — into an optimized pipeline delivered as a full web application (FastAPI backend + React frontend + PostgreSQL).

Designed for scenarios where cloud dependency, latency, or subscription cost is unacceptable:

- **Smart Security** — face-based access control, visitor logging, zone alerts
- **Elderly Care** — real-time fall detection with camera isolation, nursing home monitoring
- **Smart Hospital** — patient movement monitoring, bed-exit alerts, identity confirmation
- **Retail Analytics** — customer counting, demographic analysis
- **Smart Home** — family member recognition, automation triggers, in-home fall alerts
- **Edge AI Deployment** — multi-camera pipeline benchmarking, GPU capacity planning

## Demo

```
┌───────────────────────────────────────────────────────────────────────┐
│ Live camera grid (React)                          FPS: 28  Cameras: 4 │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│ │ Byron       │  │ Unknown     │  │ (motion)    │  │ (idle)      │   │
│ │ Track #12   │  │ Track #5    │  │             │  │             │   │
│ │ 🦴 Fall: OK │  │ 🦴 Warning  │  │             │  │             │   │
│ └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│  Overlay: face bbox + 17-keypoint skeleton (Canvas)                   │
│ [EVENT] fall_potential cam=test-fall-cam track=1                      │
│ [EVENT] fall_detected  cam=test-fall-cam track=1                      │
└───────────────────────────────────────────────────────────────────────┘
```
*Live grid with face bounding-box (green=identified / gray=unknown) + 17-keypoint pose skeleton + fall status overlay, WebSocket-driven events.*

## Features

### Face Recognition (InsightFace core)
- **InsightFace Official Kernel** — `FaceAnalysis` runs detection + landmarks + 512-d embedding in one pass; switch `buffalo_s`/`buffalo_l` via config only
- **GPU-first Inference** — CUDAExecutionProvider preferred, CPU fallback with explicit warning
- **ByteTrack Multi-Object Tracking** — Kalman filter + two-tier association (high/low confidence), stable `track_id`; recognition by cooldown strategy (new track → cooldown → backoff → re-verify)
- **Vectorized Recognition** — face gallery in-memory numpy matrix, single `np.dot` cosine search; register/delete takes effect instantly
- **Guided Camera Registration** — 5-step action guidance (`getUserMedia`): front → left → right → up → down, auto-capture on threshold, multi-angle atomic registration

### Fall Detection (YOLOv8-Pose isolated GPU Worker)
- **GPU-hard, no CPU fallback** — model forward, pre/post-processing, NMS and keypoint math all on `cuda:0` + FP16; dual verification (`torch.cuda.is_available()` + real model load); no implicit `try/except → cpu` downgrade
- **Single Shared Engine** — multiple cameras share one model + CUDA context, fair frame scheduler, `CapacityManifest` gating by environment fingerprint (GPU / driver / CUDA version / model hash)
- **IPC: shared memory + Windows named pipe** — raw BGR frames via shared-memory slots, control messages via `AF_PIPE`; binary-mode pipes avoid CRT line-ending corruption; crash-reconnect, heartbeat, persistent Worker Journal
- **Time-based Fall State Machine** — long-term states `NORMAL / POTENTIAL / FALLEN` (`RECOVERED` is a transition event); monotonic-time durations, evidence-gap continuity break; events `fall_potential / fall_detected / fall_recovered`
- **Atomic Event Delivery** — Worker Journal → parent drain (ack-on-success, backoff retry, poison-pill to parent `EventSpool`) → `EventIngress` → PostgreSQL + WebSocket; dedup by `event_id`/`dedupe_key`
- **Real-time Skeleton Overlay** — Canvas draws 17 keypoints + bbox at preview resolution, `FallOverlayCache` TTL invalidation; stale overlay (>300ms threshold) rejected
- **Health API** — `GET /system/fall-runtime` returns a redacted snapshot (worker/gpu/model/delivery/cameras); leaks no pipe authkey / config path / DB URL / stack trace

### Platform
- **Pluggable VisionTask** — tasks loaded dynamically by `class_path`/`module_path`, zero changes to main loop / routes / frontend (see `docs/PLUGIN_GUIDE.md`)
- **Full Config Cascade** — `default.yaml` → `profiles/{desktop,balanced,edge_minimal}.yaml` → per-camera JSONB; zero hard-coded params
- **Industrialized** — Alembic versioned migrations, api → services → repositories layering, engine-pool shared VRAM, auto-reconnect, event persistence + WS push, migration-protection scripts
- **Smooth Preview** — `binary_jpeg_v1` binary JPEG, single-encode once, multi-subscriber single-slot drop-old-frame; slow clients never stall others
- **Frontend Event Dedup** — LRU cache (4096) + 24h TTL + sessionStorage, keyed by `event_id` first, `dedupe_key` second

## System Architecture

### End-to-End Pipeline (Fall Detection)

```
┌─────────────────────────────── Backend (FastAPI) ───────────────────────────────┐
│ Camera Pipeline (vision)                                                        │
│   OpenCV source ─▶ frame ─▶ VisionTask.FallDetectionTask ─▶ shared memory frame  │
│                                     │                                           │
│   EventIngress ◀── runtime._drain_journal ◀── Worker Journal ◀── (parent ack)   │
│        │                                                                        │
│        ├─▶ PostgreSQL (events/event_outbox)  ─▶  /api/events                     │
│        └─▶ WebSocket (raw jpeg + analytics + events)                            │
└───────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────── GPU Worker (isolated venv) ──────────────────────┐
│ Windows named pipe (binary) + shared memory                                     │
│   receive frame descriptor ─▶ YOLOv8n-Pose (cuda:0, fp16)                       │
│   ─▶ per-camera PoseTracker ─▶ time-based state machine                         │
│   ─▶ transitions ─▶ Worker Journal (SQLite, WAL)                                │
└───────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────── Frontend (React) ────────────────────────────────┐
│ Camera grid (Canvas) ─▶ face bbox + pose skeleton overlay (FallOverlay)          │
│ WS consumer ─▶ eventDedupe ─▶ AlertBanner / EventLog / Catalog                     │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Fall inference in isolated GPU Worker process | YOLO-Pose (≈11ms @640) never blocks the web/event loop; single shared engine + shared VRAM |
| GPU-hard, no CPU fallback | Strict edge accuracy guarantee; verified by real model load, not just `is_available()` |
| Shared memory (frames) + named pipe (control) | Low-latency frame transfer; handles Windows pipe handle semantics correctly |
| Binary-mode pipes (`O_BINARY`) | Prevents CRT text-mode CRLF corruption of the 4-byte length prefix |
| Worker Journal (WAL SQLite) | Crash-safe persistent transitions; poison-pill fallback if parent sink fails |
| ACK-on-success + backoff retry | No event loss; failed event stays pending and retries, then spills to EventSpool |
| Camera-scoped state isolation | Each `(camera_id, session)` has independent tracker + state machine |
| Time-based (not frame-based) fall thresholds | Robust against pose-track ID switches and variable FPS |
| Latest-only analytics delivery | Overlay always reflects newest pose; stale results expired by TTL |
| Event dedup (LRU + sessionStorage) | Replayed/looped streams produce no event flood |

## Tech Stack

### Face Recognition

| Layer | Technology | Version | Role |
|---|---|---|---|
| Face Detection | SCRFD (InsightFace) | 0.7.3+ | Lightweight face detection + 5-point landmarks |
| Tracking | ByteTrack (Kalman) | custom | Multi-object tracking, stable IDs |
| Recognition | ArcFace (buffalo_s/l) | 0.7.3+ | 512-d embedding extraction |
| Recognition Search | numpy `np.dot` matrix | — | Single pass cosine similarity vs in-memory gallery |

### Fall Detection (isolated GPU Worker)

| Layer | Technology | Version | Role |
|---|---|---|---|
| Pose Detection | YOLOv8n-pose | 8.4.x | 17-keypoint COCO pose estimation |
| Inference Engine | PyTorch CUDA | 2.7.0+cu126 | GPU model forward (fp16) on `cuda:0` |
| IPC | Win32 named pipe + shared memory | — | Frame/control/result transport |
| State Machine | time-based (NORMAL/POTENTIAL/FALLEN) | — | Fall state transitions + events |
| Persistence | SQLite WAL Worker Journal + EventSpool | — | Crash-safe event handoff |
| Capacity | CapacityManifest (environment fingerprint) | — | Per-GPU throughput gate (safe=59/eff=32 on RTX 4060) |

### Platform

| Layer | Technology | Version | Role |
|---|---|---|---|
| Backend | FastAPI + SQLAlchemy async | — | REST + WebSocket + services/repos |
| Database | PostgreSQL 16 | — | events, cameras, catalog persistence |
| Migrations | Alembic | — | versioned schema (see `backend/migrations`) |
| Frontend | React 18 + Ant Design 5 | — | Camera grid, overlays, registration, alerts |
| Tests | pytest + Vitest/JSDOM | — | backend + frontend regression |
| Deployment Profiles | desktop / balanced / edge_minimal | — | config cascade per hardware tier |

## Quick Start

### One-click setup (Windows)

> Requires **Python 3.12** and **Node.js 16+** on `PATH`, plus an NVIDIA GPU with CUDA driver for fall detection.

```powershell
Set-ExecutionPolicy -Scope Process Bypass   # pick one-time
.\install_python.ps1                        # backend venv + pose GPU worker venv + models
```

The script creates both virtual environments (`.venv` + `pose_plugin/.venv-worker`), installs PyTorch 2.7.0+cu126 from the official CUDA wheel index, verifies GPU availability, downloads & SHA-256-checks `yolov8n-pose.pt`, and (optionally) builds the frontend. Common flags:

| Flag | Effect |
|---|---|
| `-SkipFrontend` | skip `npm install` + build |
| `-SkipModels` | skip model download |
| `-OnlyBackend` / `-OnlyWorker` | install only one environment |
| `-Mirror` | use Tsinghua PyPI mirror for backend deps (CN networks) |

> **GPU version match (important)**: `onnxruntime-gpu` must match your CUDA runtime, otherwise face inference silently falls back to CPU.
> Check with `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"` — `CUDAExecutionProvider` present = GPU ready.
> Recommended `pip install onnxruntime-gpu==1.21.0` (matches CUDA 12.6). The repo ships a GPU-ready `.venv`.

### Manual setup

```bash
# 1. Database (PostgreSQL 16)
docker run -d --name ai-monitor-db -p 5432:5432 \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=123456 -e POSTGRES_DB=ai_monitor \
  postgres:16-alpine

# 2. Backend (project root)
cd backend
pip install -r requirements.txt
alembic upgrade head          # versioned schema
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. Frontend (another terminal)
cd frontend
npm install
npm run dev                   # http://localhost:3000 (proxies /api and /ws)
```

> Models: default high-accuracy `models/buffalo_l/` (SCRFD-10G + ArcFace ResNet50); `buffalo_l` exceeds the GitHub single-file limit and is not shipped — download the official zip and extract to `models/buffalo_l/`. Lightweight `models/buffalo_s/` is shipped; switch via `vision.model_pack`. Embedding space is not interoperable across packs — re-register the gallery when switching.

### Enabling Fall Detection

Fall detection is off by default (`tasks.fall_detection.enabled: false`). Enable it:

1. **Prepare the pose Worker environment** (isolated venv): install `torch==2.7.0+cu126` (CUDA wheel), `torchvision`, `ultralytics`, `scipy`, `opencv-python`; place `models/yolov8n-pose.pt` and its `.sha256` (no auto-download).
2. **Configure `configs/default.yaml`** (paths already point to `pose_plugin/` in this repo):
   ```yaml
   tasks:
     fall_detection:
       enabled: true
       class_path: app.tasks.builtin.fall_detection_task.FallDetectionTask
   fall_detection:
     worker:
       python: "D:/ai-monitor-1.1.0/pose_plugin/.venv-worker/Scripts/python.exe"
       module: ai_monitor_pose.worker
     model:
       path: "D:/ai-monitor-1.1.0/pose_plugin/models/yolov8n-pose.pt"
     runtime:
       capacity_manifest_path: "D:/ai-monitor-1.1.0/pose_plugin/models/capacity-cuda0.json"
       overlay_ttl_ms: 1200
   ```
3. **Register cameras** in the web UI (RTSP / local camera / local video file behind the backend; per-camera threshold override like `min_fall_pose_duration_s`).
4. Restart; events flow to `/ws/events`, the `events` table, and the event page.

> **Diagnosis**: worker stderr is NUL by default; set env `AI_MONITOR_POSE_WORKER_STDERR=<logfile>` to redirect GPU Worker stderr to a file.
> Capacity/link verification: `scripts/gpu_smoke.py` and `scripts/integration_smoke.py --fixture-manifest fixtures/fall_manifest.json`.

## Documentation

| Doc | Content |
|---|---|
| [docs/API.md](docs/API.md) | Full REST/WebSocket API reference (also `/docs` OpenAPI at runtime) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layered architecture, key decisions, data model |
| [docs/PLUGIN_GUIDE.md](docs/PLUGIN_GUIDE.md) | How to develop a pluggable VisionTask (fall detection, etc.) |
| [pose_plugin/](pose_plugin/) | Self-contained fall-detection plugin source (ai_monitor_pose pkg + tests + scripts) |

## Project Structure

```
backend/       FastAPI service (api → services → repositories → models → migrations)
vision/        pure inference core (camera / engine / tracker / pipeline / tasks)
configs/       all-parameter YAML (default + desktop/balanced/edge_minimal cascade)
frontend/      React 18 + AntD 5 (grid, overlay, registration, event dedup)
pose_plugin/   fall-detection plugin (ai_monitor_pose pkg, GPU Worker, tests, scripts)
models/        local models (buffalo_s faces; yolov8n-pose under pose_plugin/)
tests/         pytest (unit / integration / migration / plugin contract / fall protocol)
docs/          API / architecture / plugin guide
face_db/       face images and optional legacy pickle data
fixtures/      M1/M2 verification video + manifest (sample)
```

## Test

```bash
# Backend (migration/integration cases when AI_MONITOR_TEST_DATABASE_URL points to a *_test DB)
python -m pytest tests/ -v

# Fall-detection plugin (isolated worker env, incl. real CUDA inference gate)
# $PosePython = d:\ai-monitor-1.1.0\pose_plugin\.venv-worker\Scripts\python.exe
& $PosePython -m pytest pose_plugin/tests/ -v

# Frontend
cd frontend
npm test
npm run build
```

## Coming Soon

- TensorRT acceleration (YOLOv8-Pose 2-3x faster)
- Multi-camera RTSP streaming management in UI
- Mobile app push notifications
- 8h/24h soak testing and labeled-dataset fall evaluation
- OpenVINO / Jetson custom backends