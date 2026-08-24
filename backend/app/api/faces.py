"""人脸库:注册 / 检索 / 追加特征 / 头像 / 批量导入 / 旧数据迁移。"""

from __future__ import annotations
import os
import uuid as uuid_mod
from typing import List, Literal

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import resolve_project_path
from ..deps import get_db
from ..schemas.face import (
    FaceSearchOut,
    IdentityListOut,
    IdentityOut,
    IdentityUpdateIn,
    RegistrationAnalyzeOut,
    RegistrationCommitOut,
)
from ..services.face_service import FaceService
from ..services.model_manager import get_engine_pool
from ..services.pipeline_manager import get_pipeline_manager

router = APIRouter()


def _get_service(db: AsyncSession) -> FaceService:
    mgr = get_pipeline_manager()
    return FaceService(db, gallery=mgr.gallery, engine_pool=get_engine_pool())


def _decode_image(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")
    return img


async def _extract_embedding(image: UploadFile, svc: FaceService) -> List[float]:
    img = _decode_image(await image.read())
    embedding = await svc.extract_embedding(img)
    if embedding is None:
        raise HTTPException(400, "未检测到合格人脸(需足够大且置信度达标,请换一张清晰的正脸照)")
    return embedding.astype(np.float32).ravel().tolist()


@router.post("/faces/detect")
async def detect_faces(image: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """上传照片预览检测:返回图片尺寸与人脸框坐标(不入库,GPU 推理)。

    供前端注册弹窗在预览图上绘制检测框;不包含注册质量筛选。
    """
    svc = _get_service(db)
    img = _decode_image(await image.read())
    h, w = img.shape[:2]
    faces = await svc.detect_faces(img)
    return {"width": w, "height": h, "faces": faces}

# ── 摄像头/视频实时注册(analyze 只分析不写库,commit 复验后原子入库) ──

_POSE_TOKENS = {"frontal", "left", "right", "up", "down"}


def _parse_registration_metadata(metadata_json: str, file_count: int) -> list[dict]:
    """校验 metadata_json:JSON 数组,每元素 frame_id(str)/timestamp_ms(int>=0)/pose(五值之一);
    长度必须 == file_count;frame_id 不得重复。非法抛 HTTPException(422)。"""
    import json as _json

    try:
        meta = _json.loads(metadata_json)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, "metadata_json 不是合法 JSON 数组") from exc
    if not isinstance(meta, list):
        raise HTTPException(422, "metadata_json 必须是数组")
    if len(meta) != file_count:
        raise HTTPException(422, "metadata_json 长度必须等于上传帧数")
    seen = set()
    for m in meta:
        frame_id = m.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise HTTPException(422, "frame_id 必须为非空字符串")
        if frame_id in seen:
            raise HTTPException(422, f"frame_id 重复: {frame_id}")
        seen.add(frame_id)
        ts = m.get("timestamp_ms")
        if not isinstance(ts, int) or ts < 0:
            raise HTTPException(422, "timestamp_ms 必须为非负整数")
        pose = m.get("pose")
        if pose not in _POSE_TOKENS:
            raise HTTPException(422, f"非法 pose: {pose}")
    return meta


async def _decode_registration_files(files) -> list[np.ndarray]:
    """逐个解码为 BGR ndarray;非法图片抛 400。"""
    out = []
    for f in files:
        data = await f.read()
        img = _decode_image(bytes(data))
        out.append(img)
    return out


@router.post("/faces/registration/analyze", response_model=RegistrationAnalyzeOut)
async def analyze_registration_frames(
    frames: List[UploadFile] = File(...),
    metadata_json: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """批量质量分析(不写库):摄像头实时采集的候选帧 → 返回各帧质量与推荐帧。"""
    if len(frames) > 30:
        raise HTTPException(422, "单次最多上传 30 帧")
    meta = _parse_registration_metadata(metadata_json, len(frames))
    images = await _decode_registration_files(frames)
    svc = _get_service(db)
    try:
        result = await svc.analyze_registration_frames(
            list(zip(images, [m["frame_id"] for m in meta],
                     [m["timestamp_ms"] for m in meta], [m["pose"] for m in meta]))
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    accepted = result["accepted"]
    from ..services.video_registration import CandidateFrame

    frames_out = []
    for m, img in zip(meta, images):
        # 命中 accepted 则给详细质量,否则查 rejected reason
        acc = next((a for a in accepted if a.frame_id == m["frame_id"]), None)
        if acc is not None:
            frames_out.append({
                "frame_id": acc.frame_id, "timestamp_ms": acc.timestamp_ms,
                "accepted": True, "reason": None, "pose": acc.pose,
                "bbox": list(acc.bbox), "det_score": acc.det_score,
                "yaw_ratio": acc.yaw_ratio, "pitch_ratio": acc.pitch_ratio,
                "blur_score": acc.blur_score, "quality_score": acc.quality_score,
            })
        else:
            rej = next((r for r in result["rejected"] if r["frame_id"] == m["frame_id"]), None)
            frames_out.append({
                "frame_id": m["frame_id"], "timestamp_ms": m["timestamp_ms"],
                "accepted": False, "reason": rej["reason"] if rej else "rejected",
                "pose": m["pose"], "bbox": None, "det_score": None,
                "yaw_ratio": None, "pitch_ratio": None, "blur_score": None, "quality_score": None,
            })
    return {
        "sampled_count": result["sampled"],
        "accepted_count": len(accepted),
        "recommended_frame_ids": result["recommended_frame_ids"],
        "frames": frames_out,
    }


@router.post("/faces/registration/commit", response_model=RegistrationCommitOut, status_code=201)
async def commit_registration_frames(
    mode: Literal["create", "append"] = Form(...),
    name: str | None = Form(None),
    notes: str | None = Form(None),
    identity_id: uuid_mod.UUID | None = Form(None),
    frames: List[UploadFile] = File(...),
    metadata_json: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """提交注册:服务端复验后原子入库(create 新身份 / append 已有身份)。"""
    if len(frames) > 30:
        raise HTTPException(422, "单次最多上传 30 帧")
    if mode == "create" and not name:
        raise HTTPException(422, "create 模式必须提供姓名")
    if mode == "append" and identity_id is None:
        raise HTTPException(422, "append 模式必须提供 identity_id")
    meta = _parse_registration_metadata(metadata_json, len(frames))
    images = await _decode_registration_files(frames)
    svc = _get_service(db)
    try:
        result = await svc.commit_registration_frames(
            mode=mode,
            name=name,
            notes=notes,
            identity_id=identity_id,
            frames=list(zip(images, [m["frame_id"] for m in meta],
                            [m["timestamp_ms"] for m in meta], [m["pose"] for m in meta])),
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "identity_not_found" in msg:
            raise HTTPException(404, "Identity not found") from exc
        raise HTTPException(400, msg) from exc
    return result



@router.get("/faces", response_model=IdentityListOut)
async def list_faces(
    page: int = Query(1, ge=1, description="页码,从 1 起"),
    page_size: int = Query(20, ge=1, le=200, description="每页条数,上限 200"),
    search: str = "",
    db: AsyncSession = Depends(get_db),
):
    svc = _get_service(db)
    identities, total = await svc.list_identities(page, page_size, search)
    items = [
        IdentityOut(
            id=i.id,
            name=i.name,
            avatar_path=i.avatar_path,
            notes=i.notes,
            embedding_count=len(i.embeddings),
            created_at=i.created_at,
        )
        for i in identities
    ]
    return IdentityListOut(items=items, total=total)


@router.get("/faces/{face_id}", response_model=IdentityOut)
async def get_face(face_id: uuid_mod.UUID, db: AsyncSession = Depends(get_db)):
    identity = await _get_service(db).get_identity(face_id)
    if not identity:
        raise HTTPException(404, "Identity not found")
    return IdentityOut(
        id=identity.id,
        name=identity.name,
        avatar_path=identity.avatar_path,
        notes=identity.notes,
        embedding_count=len(identity.embeddings),
        created_at=identity.created_at,
    )


@router.post("/faces", status_code=201)
async def create_face(
    name: str = Form(..., min_length=1, max_length=128),
    notes: str = Form(""),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    svc = _get_service(db)
    embedding = await _extract_embedding(image, svc)
    identity = await svc.create_identity(name, embedding, source="image", notes=notes)
    return {"id": str(identity.id), "name": identity.name}


@router.put("/faces/{face_id}")
async def update_face(face_id: uuid_mod.UUID, body: IdentityUpdateIn, db: AsyncSession = Depends(get_db)):
    identity = await _get_service(db).update_identity(face_id, body.model_dump(exclude_none=True))
    if not identity:
        raise HTTPException(404, "Identity not found")
    return {"id": str(identity.id), "name": identity.name, "notes": identity.notes}


@router.delete("/faces/{face_id}")
async def delete_face(face_id: uuid_mod.UUID, db: AsyncSession = Depends(get_db)):
    ok = await _get_service(db).delete_identity(face_id)
    if not ok:
        raise HTTPException(404, "Identity not found")
    return {"deleted": True}


# ── embedding 管理 ────────────────────────────────────────

@router.post("/faces/{face_id}/embeddings", status_code=201)
async def add_embedding(face_id: uuid_mod.UUID, image: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    svc = _get_service(db)
    embedding = await _extract_embedding(image, svc)
    ok = await svc.add_embedding(face_id, embedding, source="image")
    if not ok:
        raise HTTPException(404, "Identity not found")
    return {"added": True, "identity_id": str(face_id)}


@router.delete("/faces/{face_id}/embeddings/{embedding_id}")
async def delete_embedding(face_id: uuid_mod.UUID, embedding_id: uuid_mod.UUID, db: AsyncSession = Depends(get_db)):
    ok = await _get_service(db).delete_embedding(embedding_id)
    if not ok:
        raise HTTPException(404, "Embedding not found")
    return {"deleted": True}


# ── 检索与批量导入 ────────────────────────────────────────

@router.post("/faces/search", response_model=FaceSearchOut)
async def search_face(image: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    svc = _get_service(db)
    embedding = await _extract_embedding(image, svc)
    identity_id, name, similarity = await svc.search_by_embedding(embedding, threshold=0.4)
    return FaceSearchOut(identity_id=identity_id, name=name, similarity=similarity)


@router.post("/faces/batch-import")
async def batch_import(
    name: str = Form(...),
    notes: str = Form(""),
    images: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    """同一人批量导入多张照片,每张提取一条 embedding。"""
    if len(images) > 32:
        raise HTTPException(422, "单次批量导入最多 32 张图片")
    svc = _get_service(db)
    results = []
    embeddings = []
    for img_file in images:
        try:
            img = _decode_image(await img_file.read())
            emb = await svc.extract_embedding(img)
            if emb is None:
                results.append({"file": img_file.filename, "status": "error", "reason": "未检测到合格人脸(过小/模糊/置信度低)"})
                continue
            embeddings.append(emb.astype(np.float32).ravel().tolist())
            results.append({"file": img_file.filename, "status": "ok"})
        except HTTPException as exc:
            results.append({"file": img_file.filename, "status": "error", "reason": str(exc.detail)})
        except Exception as exc:  # noqa: BLE001
            results.append({"file": img_file.filename, "status": "error", "reason": str(exc)})

    if embeddings:
        from ..models.identity import Identity
        from ..repositories.identity_repo import IdentityRepository

        identity = await IdentityRepository(db).create(Identity(name=name, notes=notes), embeddings=embeddings)
        await svc.refresh_gallery()
        identity_id = str(identity.id)
    else:
        identity_id = None

    return {"name": name, "total": len(images), "identity_id": identity_id, "results": results}


# ── 头像 ─────────────────────────────────────────────────

ALLOWED_AVATAR_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_AVATAR_BYTES = 5 * 1024 * 1024  # 5MB


@router.post("/faces/{face_id}/avatar")
async def upload_avatar(face_id: uuid_mod.UUID, image: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    svc = _get_service(db)
    identity = await svc.get_identity(face_id)
    if not identity:
        raise HTTPException(404, "Identity not found")

    ext = os.path.splitext(image.filename or "avatar.jpg")[1].lower() or ".jpg"
    if ext not in ALLOWED_AVATAR_EXTS:
        raise HTTPException(400, f"不支持的图片类型: {ext},仅允许 {'/'.join(sorted(ALLOWED_AVATAR_EXTS))}")

    data = await image.read()
    if len(data) > MAX_AVATAR_BYTES:
        raise HTTPException(400, "图片超过 5MB 上限")
    _decode_image(data)  # 复用现有校验:非法图片会抛 400

    avatars_dir = resolve_project_path("face_db/avatars")
    os.makedirs(avatars_dir, exist_ok=True)
    filename = f"{face_id}_{uuid_mod.uuid4().hex[:8]}{ext}"
    filepath = os.path.join(avatars_dir, filename)
    with open(filepath, "wb") as f:
        f.write(data)

    identity = await svc.update_identity(face_id, {"avatar_path": f"face_db/avatars/{filename}"})
    return {"avatar_path": identity.avatar_path}


# ── 旧数据迁移(一次性)────────────────────────────────────

@router.post("/faces/import-pickle")
async def import_pickle(body: dict, db: AsyncSession = Depends(get_db)):
    """从旧版 pickle 底库导入(仅允许 face_db/ 目录下的 .pkl 文件)。"""
    pickle_path = str(body.get("path", "face_db/identities.pkl"))
    if not pickle_path.endswith(".pkl"):
        raise HTTPException(400, "仅支持 .pkl 文件")
    try:
        count = await _get_service(db).import_pickle(pickle_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"imported": count}
