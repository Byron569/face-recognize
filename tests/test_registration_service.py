"""摄像头/视频注册 service 层测试(fake 引擎/repo,不连 DB)。"""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from backend.app.services.face_service import FaceService, _VideoRegistrationError


import uuid

import numpy as np
import pytest

from backend.app.services.face_service import FaceService, _VideoRegistrationError


KPS = [(200.0, 180.0), (320.0, 180.0), (260.0, 260.0), (220.0, 340.0), (300.0, 340.0)]


def _emb():
    rng = np.random.default_rng(9)
    e = rng.standard_normal(512).astype(np.float32)
    return (e / np.linalg.norm(e)).ravel()


EMB = _emb().tolist()


class _FakeEngine:
    """固定返回合格正脸的个人脸结果。"""

    def __init__(self, faces=None):
        self.faces = faces if faces is not None else [
            {"bbox": (200, 200, 320, 320), "det_score": 0.95, "kps": KPS, "embedding": EMB}
        ]
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        from vision.events import FaceResult
        return [FaceResult(**f) for f in self.faces]


class _FakePool:
    def __init__(self, engine):
        self._engine = engine
    def get(self, cfg):
        return self._engine


class _FakeGallery:
    def __init__(self):
        self.rebuild_calls = 0
        self.size = 0
    def rebuild(self, rows):
        self.rebuild_calls += 1
        self.size = len(rows)


class _FakeRepo:
    def __init__(self):
        self.create_inputs = []
        self.add_many_inputs = []
        self.exists = True
        self.get_calls = 0
    async def create_with_embeddings(self, identity, items):
        self.create_inputs.append((identity, items))
        identity.id = uuid.uuid4()
        return identity
    async def add_many_embeddings(self, identity_id, items):
        self.add_many_inputs.append((identity_id, items))
        return True
    async def get(self, identity_id):
        self.get_calls += 1
        return object() if self.exists else None
    async def all_embeddings(self):
        return []


def _img():
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _make_service(monkeypatch, repo=None, engine=None):
    repo = repo or _FakeRepo()
    engine = engine or _FakeEngine()
    gallery = _FakeGallery()
    monkeypatch.setattr("backend.app.services.face_service.IdentityRepository", lambda db: repo)
    svc = FaceService(db=None, gallery=gallery, engine_pool=_FakePool(engine))
    return svc, repo, gallery, engine


def _frames(n, pose="frontal"):
    return [(_img(), f"f{i}", i * 100, pose) for i in range(n)]


@pytest.mark.anyio
async def test_commit_create_uses_source_camera_and_refreshes_once(monkeypatch):
    svc, repo, gallery, _ = _make_service(monkeypatch)
    res = await svc.commit_registration_frames(
        mode="create", name=" 测试人 ", notes="", identity_id=None, frames=_frames(3)
    )
    assert res["mode"] == "create"
    assert repo.create_inputs and len(repo.create_inputs) == 1
    identity, items = repo.create_inputs[0]
    assert identity.name == "测试人"
    assert all(i.source == "camera" for i in items)
    assert len(items) == 3
    assert gallery.rebuild_calls == 1


@pytest.mark.anyio
async def test_commit_append_identity_not_found(monkeypatch):
    repo = _FakeRepo()
    repo.exists = False
    svc, repo, gallery, _ = _make_service(monkeypatch, repo=repo)
    with pytest.raises(_VideoRegistrationError, match="identity_not_found"):
        await svc.commit_registration_frames(
            mode="append", name=None, notes=None,
            identity_id=uuid.uuid4(), frames=_frames(3)
        )
    assert gallery.rebuild_calls == 0


@pytest.mark.anyio
async def test_commit_too_few_frames(monkeypatch):
    repo = _FakeRepo()
    svc, repo, gallery, _ = _make_service(monkeypatch, repo=repo)
    with pytest.raises(_VideoRegistrationError, match="too_few_frames"):
        await svc.commit_registration_frames(
            mode="create", name="x", notes=None, identity_id=None, frames=_frames(1)
        )
    assert not repo.create_inputs
    assert gallery.rebuild_calls == 0


@pytest.mark.anyio
async def test_analyze_does_not_touch_repo(monkeypatch):
    repo = _FakeRepo()
    svc, repo, gallery, engine = _make_service(monkeypatch, repo=repo)
    res = await svc.analyze_registration_frames(_frames(3))
    assert res["sampled"] == 3
    assert not repo.create_inputs
    assert not repo.add_many_inputs
    assert gallery.rebuild_calls == 0
    assert engine.calls >= 3  # 逐帧检测
    assert len(res["accepted"]) == 3
