"""services.video_registration — 摄像头/视频帧注册的纯函数质量分析。

设计原则:
    - 纯函数,不依赖 FastAPI/DB/请求对象,单测零 mock;
    - 与实时识别(fast path)解耦:这里允许更重的质量计算(模糊度/正脸度);
    - 质量规则与设计文档 docs/superpowers/plans/2026-08-21-video-face-registration.md 对齐。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from vision.events import FaceResult


@dataclass(frozen=True)
class CandidateFrame:
    """一次高质量检测:可用于登记的候选帧。"""

    frame_id: str
    timestamp_ms: int
    pose: str                    # 该帧采集时的动作标签(frontal/left/right/up/down)
    bbox: Tuple[float, float, float, float]
    det_score: float
    yaw_ratio: float
    pitch_ratio: float
    blur_score: float
    quality_score: float
    embedding: list[float]       # 已归一化 512-d
    kps: Optional[list[Tuple[float, float]]] = None


@dataclass(frozen=True)
class RejectedFrame:
    """被拒绝的候选帧。"""

    frame_id: str
    timestamp_ms: int
    reason: str                  # 拒绝原因(英文 token,前端映射文案)


# 姿态判定区间(与配置 max_yaw_ratio/max_pitch_ratio 联动):
#   frontal: |yaw| <= max_yaw 且 |pitch| <= max_pitch
#   left   : yaw <= -max_yaw          (画面中人脸朝向观察者左侧)
#   right  : yaw >=  max_yaw
#   up     : pitch <= -max_pitch
#   down   : pitch >=  max_pitch


def compute_pose_ratios(kps) -> Tuple[float, float]:
    """5 点关键点 → (yaw_ratio, pitch_ratio)。

    kps 顺序: [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角](insightface 标准输出)。
    inter_eye = max(|右眼x - 左眼x|, 1.0)
    eye_mid_x = (左眼x+右眼x)/2; nose_x 偏离 eye_mid_x 的比例即 yaw
    pitch 用 鼻尖y 与 双眼中点y / 双嘴角中点y 的两段差不对称度
    """
    if kps is None or len(kps) < 5:
        return 0.0, 0.0
    left_eye, right_eye, nose, left_mouth, right_mouth = kps[:5]
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    inter_eye = max(abs(right_eye[0] - left_eye[0]), 1.0)
    eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
    mouth_mid_y = (left_mouth[1] + right_mouth[1]) / 2.0
    # 鼻尖相对双眼横向中点的偏移 → yaw(带符号)
    # 镜像几何:用户面对镜头时其左手在画面右侧;用户向自己的左边转头时,
    # 鼻尖在画面中向右移动(nose_x 增大)。因此约定:
    #   yaw_ratio > 0 = 用户向左转头;< 0 = 用户向右转头
    # (旧实现取 abs 丢失方向,导致前端 left 区间永不命中、right 误命中未翻转的脸)
    yaw_ratio = (nose[0] - eye_mid_x) / inter_eye
    # pitch: 鼻尖到双眼距 与 鼻尖到嘴距 不对称 → pitch(带符号,抬头为正,低头为负)
    # 推导:抬头时鼻尖上移 → up_dist(nose_y-eye_y) 缩小 → pitch=(down-up) 增大(变正);低头反向
    up_mid_y = eye_mid_y
    down_mid_y = mouth_mid_y
    up_dist = nose[1] - up_mid_y
    down_dist = down_mid_y - nose[1]
    pitch_ratio = (down_dist - up_dist) / inter_eye
    return yaw_ratio, pitch_ratio


def classify_pose(
    yaw_ratio: float, pitch_ratio: float, max_yaw: float, max_pitch: float
) -> str:
    """返回 'frontal'|'left'|'right'|'up'|'down'。

    左右约定与 compute_pose_ratios 一致:yaw>0 = 用户向左转头(left)。
    上下约定:抬头 → pitch 为正(up);低头 → pitch 为负(down)。
    """
    if abs(yaw_ratio) <= max_yaw and abs(pitch_ratio) <= max_pitch:
        return "frontal"
    if yaw_ratio >= max_yaw:
        return "left"
    if yaw_ratio <= -max_yaw:
        return "right"
    if pitch_ratio >= max_pitch:
        return "up"
    if pitch_ratio <= -max_pitch:
        return "down"
    return "frontal"


def blur_variance(image: np.ndarray, bbox) -> float:
    """bbox 裁剪(边界安全钳制)→ 灰度 → Laplacian 方差(越清晰越高)。"""
    h, w = image.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _frontal_score(yaw_ratio: float, pitch_ratio: float, cfg: Mapping[str, Any]) -> float:
    """姿态贴合度:越接近目标姿态分越高(区间内=1,farther→0)。"""
    # 实时引导场景:候选帧姿态本身就是目标,这里返回绝对正脸贴合度
    max_yaw = float(cfg.get("max_yaw_ratio", 0.20))
    max_pitch = float(cfg.get("max_pitch_ratio", 0.28))
    return max(
        0.0,
        1.0 - max(abs(yaw_ratio) / max_yaw if max_yaw else 0.0,
                  abs(pitch_ratio) / max_pitch if max_pitch else 0.0),
    )


def analyze_face_result(
    faces: Sequence[FaceResult],
    image: np.ndarray,
    frame_id: str,
    timestamp_ms: int,
    pose: str,
    cfg: Mapping[str, Any],
) -> "CandidateFrame | RejectedFrame":
    """按序拒绝(顺序固定,reason 用英文 token)。

    no_face → multiple_faces → low_detection_score → face_too_small
    → missing_landmarks → missing_embedding → too_blurry
    """
    if not faces:
        return RejectedFrame(frame_id, timestamp_ms, "no_face")
    if len(faces) > 1:
        return RejectedFrame(frame_id, timestamp_ms, "multiple_faces")

    face = faces[0]
    min_score = float(cfg.get("min_det_score", 0.5))
    min_size = float(cfg.get("min_face_size", 48))
    min_blur = float(cfg.get("min_blur_variance", 55.0))

    if face.det_score < min_score:
        return RejectedFrame(frame_id, timestamp_ms, "low_detection_score")
    if min(face.width, face.height) < min_size:
        return RejectedFrame(frame_id, timestamp_ms, "face_too_small")
    if face.kps is None or len(face.kps) < 5:
        return RejectedFrame(frame_id, timestamp_ms, "missing_landmarks")
    if face.embedding is None:
        return RejectedFrame(frame_id, timestamp_ms, "missing_embedding")

    yaw_ratio, pitch_ratio = compute_pose_ratios(face.kps)
    blur = blur_variance(image, face.bbox)
    if blur < min_blur * 0.6:
        return RejectedFrame(frame_id, timestamp_ms, "too_blurry")

    fr = _frontal_score(yaw_ratio, pitch_ratio, cfg)
    quality = (
        0.45 * float(face.det_score)
        + 0.35 * fr
        + 0.20 * min(1.0, blur / (2.0 * min_blur) if min_blur else 1.0)
    )
    return CandidateFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        pose=pose,
        bbox=(float(face.bbox[0]), float(face.bbox[1]), float(face.bbox[2]), float(face.bbox[3])),
        det_score=float(face.det_score),
        yaw_ratio=yaw_ratio,
        pitch_ratio=pitch_ratio,
        blur_score=blur,
        quality_score=quality,
        embedding=[float(v) for v in face.embedding],
        kps=[(float(k[0]), float(k[1])) for k in face.kps],
    )


def select_diverse_candidates(
    frames: Sequence[CandidateFrame],
    target_per_pose: int = 2,
    duplicate_similarity: float = 0.94,
) -> list[CandidateFrame]:
    """按 pose 分桶,每桶质量降序,桶内 embedding 余弦去重,每桶最多 target_per_pose 帧。

    返回顺序: frontal 桶优先,其余按 pose 固定序(left/right/up/down)。
    """
    pose_order = ["frontal", "left", "right", "up", "down"]
    buckets: dict[str, list[CandidateFrame]] = {p: [] for p in pose_order}
    for f in frames:
        if f.pose in buckets:
            buckets[f.pose].append(f)
    out: list[CandidateFrame] = []
    for p in pose_order:
        bucket = sorted(buckets[p], key=lambda c: c.quality_score, reverse=True)
        kept: list[CandidateFrame] = []
        for c in bucket:
            if any(_cos_sim(c.embedding, k.embedding) >= duplicate_similarity for k in kept):
                continue
            kept.append(c)
            if len(kept) >= target_per_pose:
                break
        out.extend(kept)
    return out


def _cos_sim(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))
