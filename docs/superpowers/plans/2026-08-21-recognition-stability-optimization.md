# 视频流人脸识别稳定性优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** 为同一条人脸轨迹增加质量筛选和真实多帧相似度聚合，让识别事件只在稳定分数达到现有阈值后产生。

**Architecture:** 在现有 \`FaceRecognitionTask\` 内维护按 track_id 隔离的有界候选分数状态；在 \`TrackResult\`/\`STrack\` 中传递 embedding 的检测帧号，避免检测降频时重复使用旧向量。配置继续沿用现有 default → profile → camera 深合并，模型、数据库和前端接口不变。

**Tech Stack:** Python 3.12, dataclasses, collections.deque/Counter, NumPy, PyYAML, pytest, InsightFace/ONNX Runtime。

---

## 文件结构与职责

- Modify: \`vision/config.py\` — 解析 recognition.quality 与 recognition.temporal 配置。
- Modify: \`vision/events.py\` — 在 \`TrackResult\` 中传递 embedding 最近检测帧号，不改变序列化 API。
- Modify: \`vision/tracker.py\` — 在 embedding 由检测结果更新时保存帧号。
- Modify: \`backend/app/tasks/builtin/face_recognition_task.py\` — 质量筛选、轨迹候选队列、Top-K 稳定确认、状态清理和结构化日志。
- Modify: \`vision/engine.py\` — 启动日志中的实际模型文件路径和缺失 warning。
- Modify: \`configs/default.yaml\` — 增加质量与时间聚合默认配置。
- Modify: \`README.md\` — 记录模型特征空间兼容注意事项和新增配置。
- Modify: \`tests/test_vision_core.py\` — 配置默认值、嵌套覆盖和 embedding 帧标记测试。
- Modify: \`tests/test_backend_logic.py\` — 后端配置级联测试与既有识别调度回归测试。
- Create: \`tests/test_recognition_stability.py\` — 识别质量、多帧聚合、事件 confidence、候选隔离、冷却去重测试。

## Task 1: Write failing configuration and track metadata tests

**Files:**
- Modify: \`tests/test_vision_core.py\`
- Modify: \`tests/test_backend_logic.py\`

- [ ] **Step 1: Add assertions for recognition defaults and nested input.**

Add tests beside the existing \`VisionConfig\` tests:

~~~
def test_recognition_quality_and_temporal_defaults():
    cfg = VisionConfig.from_dict({})
    assert cfg.recognition.quality.min_det_score == 0.60
    assert cfg.recognition.quality.min_face_size == 80
    assert cfg.recognition.temporal.min_valid_samples == 3
    assert cfg.recognition.temporal.max_samples_per_track == 8
    assert cfg.recognition.temporal.top_k == 3


def test_recognition_quality_and_temporal_from_nested_dict():
    cfg = VisionConfig.from_dict(
        {
            "recognition": {
                "quality": {"min_det_score": 0.72, "min_face_size": 96},
                "temporal": {"min_valid_samples": 4, "max_samples_per_track": 10, "top_k": 2},
            }
        }
    )
    assert cfg.recognition.quality.min_det_score == 0.72
    assert cfg.recognition.quality.min_face_size == 96
    assert cfg.recognition.temporal.min_valid_samples == 4
    assert cfg.recognition.temporal.max_samples_per_track == 10
    assert cfg.recognition.temporal.top_k == 2
~~~

- [ ] **Step 2: Add a failing test for the embedding frame marker.**

Add this test in \`tests/test_vision_core.py\`:

~~~
def test_tracker_carries_embedding_frame_id_only_from_detection():
    tracker = ByteTracker(TrackConfig(min_hits=1))
    face = FaceResult(
        bbox=(0, 0, 100, 100),
        det_score=0.9,
        embedding=[0.1] * 512,
    )
    first = tracker.update([face], 1)[0]
    predicted = tracker.skip(2)[0]
    assert first.embedding_frame_id == 1
    assert predicted.embedding_frame_id == 1
~~~

- [ ] **Step 3: Add a failing test for camera-level nested cascade.**

Add this test in \`tests/test_backend_logic.py\`:

~~~
def test_camera_extra_overrides_nested_recognition_stability_config():
    merged = build_camera_config(
        "desktop",
        {"vision": {"recognition": {"quality": {"min_face_size": 112}, "temporal": {"top_k": 2}}}},
    )
    rec = merged["vision"]["recognition"]
    assert rec["quality"]["min_det_score"] == 0.60
    assert rec["quality"]["min_face_size"] == 112
    assert rec["temporal"]["min_valid_samples"] == 3
    assert rec["temporal"]["top_k"] == 2
~~~

- [ ] **Step 4: Run the focused tests and verify they fail for missing fields.**

Run:

~~~
.venv\Scripts\python.exe -m pytest tests/test_vision_core.py::test_recognition_quality_and_temporal_defaults tests/test_vision_core.py::test_tracker_carries_embedding_frame_id_only_from_detection tests/test_backend_logic.py::test_camera_extra_overrides_nested_recognition_stability_config -q
~~~

Expected: FAIL with missing \`RecognitionConfig\` fields and \`TrackResult.embedding_frame_id\`, while the existing configuration merge test remains importable.

## Task 2: Implement config and embedding-frame metadata

**Files:**
- Modify: \`vision/config.py\`
- Modify: \`vision/events.py\`
- Modify: \`vision/tracker.py\`
- Modify: \`configs/default.yaml\`

- [ ] **Step 1: Add minimal dataclasses and parsing.**

Define \`RecognitionQualityConfig\` and \`RecognitionTemporalConfig\` with the requested defaults. Add them as fields on \`RecognitionConfig\`, parse \`cfg.get("quality")\` and \`cfg.get("temporal")\`, and leave all existing recognition fields and defaults intact.

- [ ] **Step 2: Add non-serialized track metadata.**

Add \`embedding_frame_id: Optional[int] = None\` to \`TrackResult\`. Do not add it to \`TrackResult.to_dict()\`, so the WebSocket/frontend structure is unchanged.

- [ ] **Step 3: Mark embeddings at the actual detection frame.**

Add \`embedding_frame_id: Optional[int]\` to \`STrack\`. In every \`ByteTracker.update()\` branch that assigns a non-\`None\` detection embedding, assign the current tracker frame id as well. \`skip()\` must not change it. \`STrack.to_result()\` must copy it into \`TrackResult\`.

- [ ] **Step 4: Add YAML defaults under \`vision.recognition\`.**

Add exactly these default values without changing \`threshold\`, \`model_pack\`, or current cooldown values:

~~~
quality:
  min_det_score: 0.60
  min_face_size: 80
temporal:
  min_valid_samples: 3
  max_samples_per_track: 8
  top_k: 3
~~~

- [ ] **Step 5: Run the focused tests and full existing core/config tests.**

Run:

~~~
.venv\Scripts\python.exe -m pytest tests/test_vision_core.py tests/test_backend_logic.py -q
~~~

Expected: all existing tests and Task 1 tests PASS before changing recognition behavior.

## Task 3: Write failing recognition stability tests

**Files:**
- Create: \`tests/test_recognition_stability.py\`

- [ ] **Step 1: Add deterministic test fixtures.**

Create a \`FakeGallery\` that returns one configured tuple per embedding marker, a tracker spy that records \`set_identity\`, and a helper that builds \`PipelineContext\`/\`TrackResult\` with explicit \`embedding_frame_id\`, \`score\`, and bbox. Do not use InsightFace, a database, or real embeddings in these unit tests.

The fixture API used by the following tests should be concrete and reusable:

~~~
import pytest

from backend.app.tasks.builtin.face_recognition_task import FaceRecognitionTask
from vision.events import PipelineContext, TrackResult


class FakeGallery:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = 0

    def search(self, query, threshold):
        self.calls += 1
        return self.results.pop(0) if self.results else None


class TrackerSpy:
    def __init__(self):
        self.identities = []

    def set_identity(self, track_id, identity, similarity):
        self.identities.append((track_id, identity, similarity))
        return True


def track(score=0.90, bbox=(0, 0, 100, 100), frame_id=1, embedding_frame_id=None):
    return TrackResult(
        track_id=1,
        bbox=bbox,
        score=score,
        embedding=[0.1] * 512,
        embedding_frame_id=frame_id if embedding_frame_id is None else embedding_frame_id,
    )


def context_for(one_track):
    return PipelineContext(camera_id="cam-0", frame_id=one_track.embedding_frame_id, frame=None, tracks=[one_track])


def make_task(overrides=None):
    gallery = FakeGallery()
    tracker = TrackerSpy()
    task = FaceRecognitionTask(
        config=overrides or {},
        full_config={"vision": {"recognition": {"temporal": {"min_valid_samples": 3}}}},
        gallery=gallery,
        tracker=tracker,
    )
    return task, gallery, tracker
~~~

- [ ] **Step 2: Add quality-filter tests.**

Implement tests with these exact assertions:

~~~
def test_low_detection_score_is_skipped_without_gallery_search():
    task, gallery, _ = make_task({"quality": {"min_det_score": 0.60}})
    events = task.run(None, context_for(track(score=0.59, bbox=(0, 0, 100, 100), frame_id=1)))
    assert events == []
    assert gallery.calls == 0
    assert task._states[1].skip_reasons["low_det_score"] == 1


def test_small_face_is_skipped_without_gallery_search():
    task, gallery, _ = make_task({"quality": {"min_face_size": 80}})
    events = task.run(None, context_for(track(score=0.90, bbox=(0, 0, 60, 100), frame_id=1)))
    assert events == []
    assert gallery.calls == 0
    assert task._states[1].skip_reasons["face_too_small"] == 1
~~~

Each test runs one track through \`FaceRecognitionTask.run\`, asserts \`gallery.calls == 0\`, returns no events, and checks the task state records the corresponding skip reason.

- [ ] **Step 3: Add temporal aggregation tests.**

Implement tests for:

~~~
def test_fewer_than_min_valid_samples_does_not_confirm_identity():
    task, gallery, tracker = make_task({"temporal": {"min_valid_samples": 3}})
    gallery.results = [("id-a", "Alice", 0.70), ("id-a", "Alice", 0.71)]
    assert task.run(None, context_for(track(frame_id=1))) == []
    assert task.run(None, context_for(track(frame_id=2))) == []
    assert tracker.identities == []


def test_top_k_average_is_used_for_event_confidence():
    task, gallery, tracker = make_task({"temporal": {"min_valid_samples": 3, "top_k": 3}})
    gallery.results = [("id-a", "Alice", 0.51), ("id-a", "Alice", 0.73), ("id-a", "Alice", 0.61), ("id-a", "Alice", 0.55)]
    events = [task.run(None, context_for(track(frame_id=i))) for i in range(1, 5)]
    event = next(item for batch in events for item in batch)
    expected = (0.73 + 0.61 + 0.55) / 3
    assert event.confidence == pytest.approx(expected)
    assert tracker.identities[-1] == (1, "Alice", expected)


def test_candidate_identity_samples_are_not_mixed():
    task, gallery, tracker = make_task({"temporal": {"min_valid_samples": 3}})
    gallery.results = [("id-a", "Alice", 0.90), ("id-b", "Bob", 0.90), ("id-a", "Alice", 0.90), ("id-a", "Alice", 0.90)]
    assert task.run(None, context_for(track(frame_id=1))) == []
    assert task.run(None, context_for(track(frame_id=2))) == []
    assert task.run(None, context_for(track(frame_id=3))) == []
    assert tracker.identities == []
    events = task.run(None, context_for(track(frame_id=4)))
    assert events[0].payload["name"] == "Alice"


def test_same_embedding_frame_is_not_sampled_twice():
    task, gallery, _ = make_task({"temporal": {"min_valid_samples": 2}})
    gallery.results = [("id-a", "Alice", 0.80), ("id-a", "Alice", 0.81)]
    task.run(None, context_for(track(frame_id=1, embedding_frame_id=7)))
    task.run(None, context_for(track(frame_id=2, embedding_frame_id=7)))
    assert gallery.calls == 1
~~~

Use scores \`[0.51, 0.73, 0.61, 0.55]\` for a \`top_k=3\` case and assert the event confidence is \`(0.73 + 0.61 + 0.55) / 3\`, not a scaled score or the latest score. Use alternating identities in the isolation test and assert no identity is confirmed until one identity alone reaches the sample requirement.

- [ ] **Step 4: Add threshold, confidence, and event-suppression regression tests.**

Implement:

~~~
def test_stable_average_below_threshold_does_not_confirm():
    task, gallery, tracker = make_task({"threshold": 0.70, "temporal": {"min_valid_samples": 3}})
    gallery.results = [("id-a", "Alice", 0.60), ("id-a", "Alice", 0.61), ("id-a", "Alice", 0.62)]
    assert all(task.run(None, context_for(track(frame_id=i))) == [] for i in range(1, 4))
    assert tracker.identities == []


def test_event_confidence_equals_stable_score_and_changed_deduplicates():
    task, gallery, _ = make_task({"temporal": {"min_valid_samples": 2}})
    gallery.results = [("id-a", "Alice", 0.80), ("id-a", "Alice", 0.82)]
    first = task.run(None, context_for(track(frame_id=1)))
    second = task.run(None, context_for(track(frame_id=2)))
    stable = (0.80 + 0.82) / 2
    assert first == []
    assert second[0].confidence == pytest.approx(stable)
    assert second[0].payload["similarity"] == pytest.approx(stable)
    assert second[0].payload["changed"] is True


def test_unknown_track_keeps_existing_failure_cooldown():
    task, gallery, _ = make_task({"cooldown_frames": 10})
    gallery.results = [None, None]
    task.run(None, context_for(track(frame_id=1)))
    task.run(None, context_for(track(frame_id=5)))
    assert gallery.calls == 1


def test_confirmed_track_keeps_existing_recognized_cooldown():
    task, gallery, _ = make_task({"recognized_cooldown_frames": 10, "temporal": {"min_valid_samples": 1}})
    gallery.results = [("id-a", "Alice", 0.90), ("id-a", "Alice", 0.91)]
    first = task.run(None, context_for(track(frame_id=1)))
    task.run(None, context_for(track(frame_id=5)))
    assert first[0].payload["name"] == "Alice"
    assert gallery.calls == 1
~~~

The first test supplies three valid same-identity scores whose average is below \`threshold\` and asserts no event. The second confirms once, asserts \`event.confidence == payload["similarity"] == stable_score\`, then runs an unchanged revalidation and asserts its \`changed\` flag is false. The cooldown tests use explicit frame ids and assert the gallery is not called during the configured cooldown interval.

- [ ] **Step 5: Run the new test file and verify it fails for missing stability behavior.**

Run:

~~~
.venv\Scripts\python.exe -m pytest tests/test_recognition_stability.py -q
~~~

Expected: FAIL because the production task does not yet filter quality, track embedding frame ids, or aggregate multi-frame candidates.

## Task 4: Implement quality filtering and bounded temporal recognition

**Files:**
- Modify: \`backend/app/tasks/builtin/face_recognition_task.py\`

- [ ] **Step 1: Extend \`_TrackRecState\` with bounded sample and observability state.**

Use \`dict[str, deque[float]]\` for per-identity samples, an integer total sample count, \`last_embedding_frame_id\`, \`valid_sample_count\`, \`skipped_frame_count\`, and \`Counter[str]\` skip reasons. Set deque capacity from \`RecognitionConfig.max_samples_per_track\` and enforce the total cap when appending so the aggregate never stores more than the configured maximum.

- [ ] **Step 2: Add state cleanup and quality helpers.**

Before processing a frame, remove states whose track ids are absent from \`context.tracks\`, log one summary with valid sample count, skipped count and reason counts, and keep existing states for tracker-lost tracks still present in the snapshot. Add a helper that returns \`None\` for no new embedding, \`low_det_score\`, or \`face_too_small\`; it must use \`track.score\` and the shorter bbox side, and must not call the gallery.

- [ ] **Step 3: Gate gallery search on a new qualified embedding.**

Record a consumed \`embedding_frame_id\` before/after search so the same detection embedding cannot be searched twice. Quality skips must not update \`last_attempt_frame\`, \`fail_count\`, candidate samples, tracker identity, or event output. The existing \`max_per_frame\` counter continues to count actual gallery searches only.

- [ ] **Step 4: Preserve cooldowns while allowing the initial stability window.**

Keep the existing Unknown failure backoff when a track has no pending candidate samples. While a track has pending candidate samples and is not yet confirmed, allow the next new qualified embedding to complete its initial stability window. Once confirmed, retain the current recognized cooldown before starting a fresh revalidation window; after an unsuccessful revalidation, keep the existing identity and cooldown semantics rather than emitting an event.

- [ ] **Step 5: Aggregate and confirm only one candidate identity at a time.**

Append a gallery hit only to its identity queue. When that queue reaches \`min_valid_samples\`, sort its scores descending, take up to \`top_k\`, average them, and compare that real average to \`threshold\`. On confirmation, write the stable score to the tracker and event, include \`candidate_scores\`, \`top_k_scores\`, and \`stable_score\` in the payload, compute \`changed\` against the previous identity, and clear the temporary window.

- [ ] **Step 6: Run the new stability tests and fix only production failures.**

Run:

~~~
.venv\Scripts\python.exe -m pytest tests/test_recognition_stability.py -q
~~~

Expected: all stability tests PASS with no GPU, database, or network access.

## Task 5: Add startup model-path logging and documentation

**Files:**
- Modify: \`vision/engine.py\`
- Modify: \`README.md\`

- [ ] **Step 1: Add model path discovery for logs only.**

Resolve the effective \`{models_root}/models/{model_pack}\` directory already selected by \`_resolve_model_root\`, list detection files matching \`det*.onnx\` and recognition files from the remaining ONNX files, and log their absolute paths. If the directory, detection file, or recognition file is absent, issue a warning that InsightFace may auto-download; do not alter the selected \`model_pack\` or provider.

- [ ] **Step 2: Extend the engine-ready log.**

Include \`pack\`, model file paths, requested/resolved device, actual providers, and detection size in readable structured log fields. Never log embedding contents.

- [ ] **Step 3: Document the new configuration and future compatibility work.**

Update the README configuration example with quality/temporal defaults and add a short note that future embedding records should carry model pack, model fingerprint, and dimension; do not perform migration or re-registration now.

- [ ] **Step 4: Run engine/config import tests without constructing InsightFace.**

Run:

~~~
.venv\Scripts\python.exe -m pytest tests/test_vision_core.py::test_vision_config_defaults_gpu tests/test_vision_core.py::test_vision_config_from_yaml_shape tests/test_backend_logic.py::test_camera_extra_overrides_nested_recognition_stability_config -q
~~~

Expected: PASS; no model download or engine construction occurs.

## Task 6: Full verification and local video attempt

**Files:**
- No new production files; inspect the complete diff and test artifacts only.

- [ ] **Step 1: Run formatting/diff checks and the full test suite.**

Run:

~~~
git diff --check
.venv\Scripts\python.exe -m pytest tests/ -v
~~~

Expected: \`git diff --check\` has no output and pytest exits with zero failures. If a pre-existing environment failure occurs, capture its exact output and distinguish it from failures introduced by this change.

- [ ] **Step 2: Verify model and video prerequisites read-only.**

Check that \`D:\test6.mp4\` exists, that the configured \`models\buffalo_l\` contains detection and recognition ONNX files, and that PostgreSQL/backend dependencies are available. Do not delete, rewrite, or migrate any data.

- [ ] **Step 3: Start \`cam-0\` only if the existing application path is available.**

Use the existing API/configuration path and source \`D:\test6.mp4\`; observe at least one recognition result, capture its event confidence and the recognition/track logs, then immediately call the existing stop path for \`cam-0\`. Do not leave a video task running.

- [ ] **Step 4: Verify the final diff against the requirements.**

Check that no model pack, threshold, frontend API, embedding, history event, or face-library data was changed; confirm new logs contain no vectors/images; confirm \`confidence\` is the stable Top-K average; and list any unavailable integration prerequisite explicitly.

- [ ] **Step 5: Report changed files and evidence.**

Provide the changed-file list, configuration cascade behavior, focused and full test results, video test start/stop result, and before/after confidence comparison only when the video run produced both values. Do not claim the integration passed without fresh command output.
