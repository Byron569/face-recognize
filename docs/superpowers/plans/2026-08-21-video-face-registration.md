# Video Face Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Let a user upload a local frontal-face video, review automatically selected high-quality frames, then create a new identity or append video-derived embeddings to an existing identity without storing the original video.

**Architecture:** The browser owns the local video file and uses UploadedVideoSource to seek and extract bounded JPEG samples. The backend receives samples for analysis only, applies one-face, existing registration-quality, frontalness, blur, and diversity rules, then returns candidates. The browser sends only user-approved frames to one atomic commit endpoint. VideoRegistrationSource is the seam for a later CameraVideoSource; this release does not request browser camera permission.

**Tech Stack:** React 18, TypeScript, Ant Design, HTMLVideoElement, Canvas, Blob, FastAPI multipart, OpenCV, InsightFace 5-point landmarks, ArcFace embeddings, SQLAlchemy async sessions, PostgreSQL.

---

## File structure

- Modify: configs/default.yaml — video registration limits and thresholds.
- Create: backend/app/services/video_registration.py — pure candidate types, quality analysis, and diversity selection.
- Modify: backend/app/services/face_service.py — shared analysis and commit methods.
- Modify: backend/app/repositories/identity_repo.py — atomic multi-embedding create and append.
- Modify: backend/app/api/faces.py — analyze and commit multipart endpoints.
- Modify: backend/app/schemas/face.py — typed request responses.
- Create: tests/test_video_registration.py — quality, selection, validation, and fake persistence tests.
- Create: frontend/src/video-registration/types.ts — source, extracted-frame, and candidate types.
- Create: frontend/src/video-registration/UploadedVideoSource.ts — serial local-file seeking and JPEG extraction.
- Create: frontend/src/api/videoRegistration.ts — analyze and commit API clients.
- Create: frontend/src/components/VideoRegisterModal.tsx — setup, extraction, review, and commit workflow.
- Modify: frontend/src/pages/FaceLibraryPage.tsx — entry point and face-list refresh.
- Modify: README.md and docs/API.md — privacy, limits, and endpoint documentation.

### Task 1: Add bounded configuration and pure diversity selection

**Files:**

- Modify: configs/default.yaml
- Create: backend/app/services/video_registration.py
- Create: tests/test_video_registration.py

- [ ] **Step 1: Write failing candidate-selection tests**

~~~python
from backend.app.services.video_registration import CandidateFrame, select_diverse_candidates


def candidate(frame_id, timestamp_ms, quality, embedding):
    return CandidateFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        bbox=(0.0, 0.0, 120.0, 120.0),
        det_score=0.95,
        frontal_score=0.90,
        blur_score=120.0,
        quality_score=quality,
        embedding=embedding,
    )


def test_selection_keeps_best_diverse_frames_across_time():
    frames = [
        candidate("a", 0, 0.95, [1.0, 0.0]),
        candidate("b", 500, 0.94, [0.999, 0.001]),
        candidate("c", 2500, 0.90, [0.8, 0.6]),
        candidate("d", 5000, 0.88, [0.6, 0.8]),
    ]

    selected = select_diverse_candidates(
        frames,
        target_count=3,
        duplicate_similarity=0.98,
        min_candidate_spacing_ms=1000,
    )

    assert [item.frame_id for item in selected] == ["a", "c", "d"]


def test_selection_never_returns_more_than_target_count():
    assert select_diverse_candidates(
        [], target_count=6, duplicate_similarity=0.94, min_candidate_spacing_ms=1000
    ) == []
~~~

- [ ] **Step 2: Verify the tests fail**

Run:

~~~powershell
python -m pytest tests/test_video_registration.py -q
~~~

Expected: import failure because video_registration.py does not exist.

- [ ] **Step 3: Add configuration and data types**

Add this block beneath current vision.registration settings:

~~~yaml
    video:
      max_duration_seconds: 60
      max_file_size_mb: 100
      sample_interval_ms: 500
      max_sample_frames: 120
      min_submit_frames: 3
      max_submit_frames: 8
      target_candidate_frames: 6
      min_candidate_spacing_ms: 1000
      min_blur_variance: 55.0
      max_yaw_ratio: 0.20
      max_pitch_ratio: 0.28
      duplicate_similarity: 0.94
      max_analyze_request_bytes: 20971520
~~~

Create immutable types:

~~~python
@dataclass(frozen=True)
class CandidateFrame:
    frame_id: str
    timestamp_ms: int
    bbox: tuple[float, float, float, float]
    det_score: float
    frontal_score: float
    blur_score: float
    quality_score: float
    embedding: list[float]


@dataclass(frozen=True)
class RejectedFrame:
    frame_id: str
    timestamp_ms: int
    reason: str
~~~

Implement select_diverse_candidates by sorting descending quality_score, then ascending timestamp_ms. Accept a candidate only if cosine similarity to every selected normalized embedding is below duplicate_similarity and its timestamp differs from every selected candidate by at least min_candidate_spacing_ms. Return no more than target_count.

- [ ] **Step 4: Run tests**

Run:

~~~powershell
python -m pytest tests/test_video_registration.py -q
~~~

Expected: 2 passed.

- [ ] **Step 5: Commit**

~~~powershell
git add configs/default.yaml backend/app/services/video_registration.py tests/test_video_registration.py
git commit -m "feat(registration): define video candidate selection rules"
~~~

### Task 2: Analyze frame quality with deterministic rejection reasons

**Files:**

- Modify: backend/app/services/video_registration.py
- Modify: backend/app/services/face_service.py
- Modify: tests/test_video_registration.py

- [ ] **Step 1: Write failing quality-rule tests using FaceResult fixtures**

~~~python
from vision.events import FaceResult
from backend.app.services.video_registration import analyze_face_result


def test_analysis_rejects_zero_or_multiple_faces(image, video_cfg):
    assert analyze_face_result([], image, "f1", 0, video_cfg).reason == "no_face"
    face = FaceResult(bbox=(0, 0, 120, 120), det_score=0.95, kps=valid_kps, embedding=emb)
    assert analyze_face_result([face, face], image, "f1", 0, video_cfg).reason == "multiple_faces"


def test_analysis_rejects_small_or_low_confidence_face(image, video_cfg):
    low = FaceResult(bbox=(0, 0, 120, 120), det_score=0.4, kps=valid_kps, embedding=emb)
    small = FaceResult(bbox=(0, 0, 30, 30), det_score=0.95, kps=valid_kps, embedding=emb)
    assert analyze_face_result([low], image, "f1", 0, video_cfg).reason == "low_detection_score"
    assert analyze_face_result([small], image, "f2", 500, video_cfg).reason == "face_too_small"


def test_analysis_rejects_non_frontal_landmarks(image, video_cfg):
    profile = FaceResult(
        bbox=(0, 0, 120, 120), det_score=0.95, kps=profile_kps, embedding=emb
    )
    assert analyze_face_result([profile], image, "f1", 0, video_cfg).reason == "not_frontal"
~~~

- [ ] **Step 2: Verify the tests fail**

Run:

~~~powershell
python -m pytest tests/test_video_registration.py -q
~~~

Expected: analysis helper is absent.

- [ ] **Step 3: Implement one-face, frontalness, blur, and quality rules**

Add pure helpers:

~~~python
def blur_variance(image: np.ndarray, bbox: tuple[float, float, float, float]) -> float: ...
def frontal_ratios(kps: list[tuple[float, float]]) -> tuple[float, float]: ...
def analyze_face_result(
    faces: list[FaceResult],
    image: np.ndarray,
    frame_id: str,
    timestamp_ms: int,
    cfg: Mapping[str, Any],
) -> CandidateFrame | RejectedFrame: ...
~~~

Use exactly these rejection reasons in this order:

~~~text
no_face
multiple_faces
low_detection_score
face_too_small
missing_landmarks
not_frontal
too_blurry
missing_embedding
~~~

For landmarks ordered left eye, right eye, nose, left mouth, right mouth:

~~~python
inter_eye = max(abs(right_eye[0] - left_eye[0]), 1.0)
eye_mid_x = (left_eye[0] + right_eye[0]) / 2
mouth_mid_x = (left_mouth[0] + right_mouth[0]) / 2
yaw_ratio = abs(nose[0] - eye_mid_x) / inter_eye
pitch_ratio = abs(
    (nose[1] - (left_eye[1] + right_eye[1]) / 2)
    - ((left_mouth[1] + right_mouth[1]) / 2 - nose[1])
) / inter_eye
~~~

Reject values above configured maximums. Crop the face safely to image bounds and calculate blur with cv2.Laplacian(gray_crop, cv2.CV_64F).var(). Reject blur below min_blur_variance. Calculate quality:

~~~python
frontal_score = max(
    0.0,
    1.0 - max(yaw_ratio / max_yaw_ratio, pitch_ratio / max_pitch_ratio),
)
quality = (
    0.45 * det_score
    + 0.35 * frontal_score
    + 0.20 * min(1.0, blur / (2.0 * min_blur_variance))
)
~~~

The FaceService invokes engine.detect exactly once per image and passes the result to this helper. No second model and no server-side video decoder are allowed.

- [ ] **Step 4: Run quality and selection tests**

Run:

~~~powershell
python -m pytest tests/test_video_registration.py -q
~~~

Expected: all tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add backend/app/services/video_registration.py backend/app/services/face_service.py tests/test_video_registration.py
git commit -m "feat(registration): analyze frontal video frame quality"
~~~

### Task 3: Commit selected video frames atomically

**Files:**

- Modify: backend/app/repositories/identity_repo.py
- Modify: backend/app/services/face_service.py
- Modify: tests/test_video_registration.py

- [ ] **Step 1: Write fake-service behavior tests**

~~~python
@pytest.mark.asyncio
async def test_video_commit_requires_at_least_three_valid_embeddings(fake_face_service):
    with pytest.raises(VideoRegistrationError, match="at least 3"):
        await fake_face_service.commit_video_frames(
            mode="create",
            name="Alice",
            notes="",
            identity_id=None,
            frames=[one_valid_frame],
        )


@pytest.mark.asyncio
async def test_video_append_uses_video_source_and_refreshes_once(fake_face_service):
    result = await fake_face_service.commit_video_frames(
        mode="append",
        name=None,
        notes=None,
        identity_id=existing_id,
        frames=three_valid_frames,
    )
    assert result.mode == "append"
    assert fake_face_service.repo.add_many_calls[0].source == "video"
    assert fake_face_service.gallery_refreshes == 1
~~~

- [ ] **Step 2: Verify tests fail**

Run:

~~~powershell
python -m pytest tests/test_video_registration.py -q
~~~

Expected: missing video commit behavior.

- [ ] **Step 3: Add repository value object and transaction methods**

Add:

~~~python
@dataclass(frozen=True)
class EmbeddingInput:
    embedding: list[float]
    source: str
    quality_score: float
~~~

Add:

~~~python
async def create_with_embeddings(
    self, identity: Identity, items: list[EmbeddingInput]
) -> Identity: ...

async def add_many_embeddings(
    self, identity_id: uuid.UUID, items: list[EmbeddingInput]
) -> bool: ...
~~~

Both methods add or append all IdentityEmbedding rows, set source to video and quality_score from CandidateFrame, commit once on success, and roll back on exception. They must never create an identity with zero embeddings. Preserve old public create and add_embedding behavior by delegating with source image and quality score zero.

- [ ] **Step 4: Add FaceService commit contract**

~~~python
async def commit_video_frames(
    self,
    *,
    mode: Literal["create", "append"],
    name: str | None,
    notes: str | None,
    identity_id: uuid.UUID | None,
    frames: list[tuple[np.ndarray, str, int]],
) -> VideoCommitResult: ...
~~~

Re-run analysis during commit. Keep only valid supplied frames, select at most max_submit_frames, require at least min_submit_frames, then create a trimmed non-empty name or append to an existing identity. Refresh gallery exactly once after a successful transaction. On failure, no partial identity, no partial embeddings, and no gallery refresh.

- [ ] **Step 5: Run service tests**

Run:

~~~powershell
python -m pytest tests/test_video_registration.py tests/test_backend_logic.py -q
~~~

Expected: all tests pass.

- [ ] **Step 6: Commit**

~~~powershell
git add backend/app/repositories/identity_repo.py backend/app/services/face_service.py tests/test_video_registration.py
git commit -m "feat(registration): commit video embeddings atomically"
~~~

### Task 4: Add typed analyze and commit endpoints

**Files:**

- Modify: backend/app/schemas/face.py
- Modify: backend/app/api/faces.py
- Modify: tests/test_video_registration.py

- [ ] **Step 1: Write metadata validation tests**

~~~python
def test_parse_video_frame_metadata_requires_one_record_per_file():
    with pytest.raises(HTTPException, match="metadata"):
        parse_video_frame_metadata(
            '[{"frame_id":"a","timestamp_ms":0}]',
            file_count=2,
        )


def test_parse_video_frame_metadata_rejects_negative_timestamp():
    with pytest.raises(HTTPException, match="timestamp_ms"):
        parse_video_frame_metadata(
            '[{"frame_id":"a","timestamp_ms":-1}]',
            file_count=1,
        )
~~~

- [ ] **Step 2: Verify tests fail**

Run:

~~~powershell
python -m pytest tests/test_video_registration.py -q
~~~

Expected: parser is absent.

- [ ] **Step 3: Add schemas and routes**

Define response models:

~~~python
class VideoFrameAnalysisOut(BaseModel):
    frame_id: str
    timestamp_ms: int
    accepted: bool
    reason: str | None
    bbox: list[float] | None
    det_score: float | None
    frontal_score: float | None
    blur_score: float | None
    quality_score: float | None


class VideoRegistrationAnalysisOut(BaseModel):
    sampled_count: int
    accepted_count: int
    recommended_frame_ids: list[str]
    frames: list[VideoFrameAnalysisOut]


class VideoRegistrationCommitOut(BaseModel):
    mode: Literal["create", "append"]
    identity_id: uuid.UUID
    embedding_count_added: int
~~~

Add these routes:

~~~python
@router.post(
    "/faces/video-registration/analyze",
    response_model=VideoRegistrationAnalysisOut,
)
async def analyze_video_registration(
    frames: list[UploadFile] = File(...),
    metadata_json: str = Form(...),
    db: AsyncSession = Depends(get_db),
): ...


@router.post(
    "/faces/video-registration/commit",
    response_model=VideoRegistrationCommitOut,
    status_code=201,
)
async def commit_video_registration(
    mode: Literal["create", "append"] = Form(...),
    name: str | None = Form(None),
    notes: str | None = Form(None),
    identity_id: uuid.UUID | None = Form(None),
    frames: list[UploadFile] = File(...),
    metadata_json: str = Form(...),
    db: AsyncSession = Depends(get_db),
): ...
~~~

Use one shared parser to validate JSON metadata and decode files. Enforce configured count and total-byte limits before service calls. Reject duplicate frame_id or negative timestamp. Analyze must never call a repository write method. Commit returns 400 for invalid input or too few valid frames and 404 for a missing append identity.

- [ ] **Step 4: Run API checks**

Run:

~~~powershell
python -m pytest tests/test_video_registration.py tests/test_backend_logic.py tests/test_vision_core.py -q
python -c "from backend.app.api.faces import router; print([r.path for r in router.routes if 'video-registration' in r.path])"
~~~

Expected: tests pass and both paths are printed.

- [ ] **Step 5: Commit**

~~~powershell
git add backend/app/api/faces.py backend/app/schemas/face.py backend/app/services/face_service.py tests/test_video_registration.py
git commit -m "feat(api): add video face registration endpoints"
~~~

### Task 5: Implement uploaded-video frame source and HTTP client

**Files:**

- Create: frontend/src/video-registration/types.ts
- Create: frontend/src/video-registration/UploadedVideoSource.ts
- Create: frontend/src/api/videoRegistration.ts

- [ ] **Step 1: Define the source seam**

~~~ts
export interface VideoRegistrationSource {
  open(): Promise<void>;
  getDuration(): number;
  extractFrame(timestampMs: number): Promise<Blob>;
  close(): void;
}

export type ExtractedVideoFrame = {
  frameId: string;
  timestampMs: number;
  blob: Blob;
};

export type VideoFrameAnalysis = {
  frame_id: string;
  timestamp_ms: number;
  accepted: boolean;
  reason: string | null;
  bbox: number[] | null;
  det_score: number | null;
  frontal_score: number | null;
  blur_score: number | null;
  quality_score: number | null;
};
~~~

Do not implement CameraVideoSource in this task. The interface alone is the reserved future seam.

- [ ] **Step 2: Implement UploadedVideoSource with serial seeking**

The constructor accepts File, maxDurationSeconds, maxFileSizeMb, output maximum height, and JPEG quality. Open creates one off-DOM HTMLVideoElement and waits for loadedmetadata. Reject a too-long or too-large file before extraction begins.

The extraction sequence is serial:

~~~ts
await seek(video, Math.min(timestampMs / 1000, video.duration));
canvas.width = scaledWidth;
canvas.height = scaledHeight;
canvas.getContext('2d')!.drawImage(video, 0, 0, scaledWidth, scaledHeight);
return await canvasToBlob(canvas, 'image/jpeg', 0.82);
~~~

Close pauses the video, clears src, releases the object URL, and drops Canvas references. It must reject failed seek or unsupported video metadata with a message the modal can display.

- [ ] **Step 3: Send extracted frames, never original video**

Create clients:

~~~ts
export const analyzeVideoFrames = (
  frames: ExtractedVideoFrame[],
) => Promise<AxiosResponse<VideoRegistrationAnalysisOut>>;

export const commitVideoFrames = (input: {
  mode: 'create' | 'append';
  name?: string;
  notes?: string;
  identityId?: string;
  frames: ExtractedVideoFrame[];
}) => Promise<AxiosResponse<VideoRegistrationCommitOut>>;
~~~

Both use FormData with upload filename equal to frameId plus .jpg and metadata_json in the identical frame order. Neither function appends the original video File.

- [ ] **Step 4: Build**

Run in frontend:

~~~powershell
npm run build
~~~

Expected: exit 0.

- [ ] **Step 5: Commit**

~~~powershell
git add frontend/src/video-registration/types.ts frontend/src/video-registration/UploadedVideoSource.ts frontend/src/api/videoRegistration.ts
git commit -m "feat(frontend): add local video frame registration source"
~~~

### Task 6: Build the review-before-commit modal

**Files:**

- Create: frontend/src/components/VideoRegisterModal.tsx
- Modify: frontend/src/pages/FaceLibraryPage.tsx

- [ ] **Step 1: Implement explicit modal stages**

~~~ts
type Stage = 'setup' | 'extracting' | 'review' | 'committing' | 'complete';
type RegistrationMode = 'create' | 'append';
~~~

Required setup:

~~~text
create mode: name required, notes optional, one local video file
append mode: existing identity Select required, one local video file
~~~

Required actions:

~~~text
setup: Analyze video, Cancel
extracting: progress, Cancel
review: candidate cards, remove selected frame, Back, Submit registration
committing: disabled controls and loading state
complete: added embedding count and Close
~~~

- [ ] **Step 2: Implement bounded extraction and candidate review**

Create UploadedVideoSource and sample every configured 500 ms up to 120 frames. Update progress after each frame. Send frames to analyzeVideoFrames, retain only recommended_frame_ids, and require at least three candidates before entering review.

Each candidate card must show timestamp, quality score, detection box overlay, and remove action. Removing a candidate must remove it from final commit only; analysis result remains available if the user goes back.

- [ ] **Step 3: Implement cancellation and cleanup**

Use a monotonic run token and AbortController. On modal close, file replacement, cancel, or unmount:

~~~ts
runTokenRef.current += 1;
sourceRef.current?.close();
sourceRef.current = null;
selectedFrames.forEach((frame) => URL.revokeObjectURL(frame.previewUrl));
setSelectedFrames([]);
~~~

Every async continuation checks the run token before setting state. Closing before submit starts sends no commit. If commit already began, leave the request alive and show its result before allowing a close, so success is never mistaken for cancellation.

- [ ] **Step 4: Add face-library entry point**

In FaceLibraryPage add VideoCameraOutlined, videoRegisterOpen state, a button labelled 视频注册, and:

~~~tsx
<VideoRegisterModal
  open={videoRegisterOpen}
  identities={faces}
  onClose={() => setVideoRegisterOpen(false)}
  onCommitted={() => queryClient.invalidateQueries({ queryKey: ['faces'] })}
/>
~~~

Do not modify existing image registration, image search, batch import, edit, or delete behavior.

- [ ] **Step 5: Build and run manual workflow checks**

Run in frontend:

~~~powershell
npm run build
~~~

Manually verify:

~~~text
1. A short local frontal video shows extraction progress and candidate review.
2. Creating a new identity adds the selected embeddings.
3. Append mode increases an existing identity's embedding count.
4. Multi-person or poor-quality video produces fewer than three valid frames and does not write the database.
5. Cancel during extraction leaves no pending request or leaked preview URL.
~~~

- [ ] **Step 6: Commit**

~~~powershell
git add frontend/src/components/VideoRegisterModal.tsx frontend/src/pages/FaceLibraryPage.tsx
git commit -m "feat(frontend): register faces from local video"
~~~

### Task 7: Perform end-to-end verification and publish operational limits

**Files:**

- Modify: README.md
- Modify: docs/API.md

- [ ] **Step 1: Document privacy and limits**

Document these rules:

~~~text
Only local frontal video files are supported in this release.
The original video remains in the browser and is not persisted by the server.
At least three and at most eight user-approved frames become embeddings.
Zero/multiple faces, low-confidence or small faces, non-frontal faces, blur, and duplicate embeddings are rejected.
Append mode adds video-derived features to an existing identity.
Browser-camera registration will use the reserved CameraVideoSource in a future release.
~~~

- [ ] **Step 2: Run all automatic checks**

Run:

~~~powershell
python -m pytest tests -q
~~~

Run in frontend:

~~~powershell
npm run build
~~~

Expected: both commands exit 0.

- [ ] **Step 3: Record manual acceptance evidence**

Report video duration, extracted count, rejection counts grouped by reason, recommended count, user-approved count, create or append result, identity embedding count before/after, and confirmation that no original video file was written server-side.

- [ ] **Step 4: Commit**

~~~powershell
git add README.md docs/API.md
git commit -m "docs: explain local video face registration"
~~~
