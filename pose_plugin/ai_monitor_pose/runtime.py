"""父端 Runtime 与 Lease（第 6.1 / 6.x）。

每个 runtime_key 一个 PoseRuntime，负责一个 Worker 进程的生命周期：惰性非阻塞启动、
HELLO/READY 握手、崩溃(pipe EOF)→UNAVAILABLE、受限重启(新 epoch)与熔断、old-epoch 结果丢弃。
进程通过注入的 process_factory 创建（生产用 win_job，测试用 fake_worker / StubChild）。
唯一 I/O 读线程拥有阻塞 Pipe；正常/异常退出都清理资源。
"""
from __future__ import annotations

import json
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

from .config import RuntimeConfig
from .health import (
    READY,
    STARTING,
    STOPPED,
    STOPPING,
    UNAVAILABLE,
    CameraFallHealthV1,
    FallRuntimeHealthSnapshotV1,
    WorkerHealthV1,
)
from .ipc import decode_message, encode_message

# journal drain 最小间隔（纳秒）：250ms
_DRAIN_THROTTLE_NS = 250_000_000


class RuntimeOfferOutcome(str, Enum):
    ACCEPTED = "accepted"
    REPLACED_OLDER_FRAME = "replaced_older_frame"
    RATE_LIMITED = "rate_limited"
    WORKER_NOT_READY = "worker_not_ready"
    NO_WRITABLE_SLOT = "no_writable_slot"
    CLOSED = "closed"
    STALE = "stale"
    DUPLICATE = "duplicate"


@dataclass
class PoseRuntimeLease:
    """第 5.7 节 FallRuntimeHandle 的父端实现（Camera lease）。

    实现 host_protocols.FallRuntimeHandle：offer_frame / has_*_ / poll / unregister_camera
    均委托给共享 PoseRuntime；release() 通过注册表回调归还引用计数。
    """

    lease_id: str
    runtime_key: str
    runtime: "PoseRuntime"
    closed: bool = False
    _release_cb: object | None = field(default=None, repr=False)

    def offer(self, camera_id: str, camera_session_id: str, frame_id: int,
              config_revision: str) -> RuntimeOfferOutcome:
        if self.closed:
            return RuntimeOfferOutcome.CLOSED
        return self.runtime.offer(camera_id, camera_session_id, frame_id, config_revision)

    def offer_frame(self, frame, request_meta) -> RuntimeOfferOutcome:
        if self.closed:
            return RuntimeOfferOutcome.CLOSED
        return self.runtime.offer_frame(frame, request_meta)

    def poll(self, camera_id: str | None = None, camera_session_id: str | None = None):
        if camera_id is None or camera_session_id is None:
            return self.runtime.poll()
        return self.runtime.poll(camera_id, camera_session_id)

    def has_latest_result_or_health_change(self, camera_id: str, camera_session_id: str) -> bool:
        if self.closed:
            return False
        return self.runtime.has_latest_result_or_health_change(camera_id, camera_session_id)

    def has_unseen_compatibility_event(self, camera_id: str, camera_session_id: str) -> bool:
        if self.closed:
            return False
        return self.runtime.has_unseen_compatibility_event(camera_id, camera_session_id)

    def unregister_camera(self, camera_id: str, camera_session_id: str) -> None:
        if self.closed:
            return
        self.runtime.unregister_camera(camera_id, camera_session_id)

    def release(self) -> None:
        if self.closed:
            return
        self.closed = True
        cb = self._release_cb
        if cb is not None:
            cb(self)

    def close(self) -> None:
        self.closed = True


class PoseRuntime:
    def __init__(self, config: RuntimeConfig, *, process_factory=None,
                 restart_limit: int = 3, retry_delay_s: float = 0.0,
                 event_sink=None, drain_max_attempts: int = 5,
                 heartbeat_interval_s: float = 0, heartbeat_timeout_s: float = 0,
                 mode: str = "shadow") -> None:
        self.config = config
        # 任务级运行模式（FallTaskConfig.mode：shadow/alert），透传到 health_snapshot
        self._mode = mode
        self._heartbeat_interval_s = max(0.0, float(heartbeat_interval_s))
        self._heartbeat_timeout_s = max(0.0, float(heartbeat_timeout_s))
        self._last_pong_monotonic_ns: int | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._factory = process_factory or (lambda: _null_factory_child())
        self.restart_limit = max(0, restart_limit)
        self.retry_delay_s = retry_delay_s
        self._lock = threading.RLock()
        self._closed = False
        self._state: str = STARTING
        self._child = None
        self._reader: threading.Thread | None = None
        self._epoch: str | None = None
        self._worker_instance_id: str | None = None
        self._worker_pid: int | None = None
        self._device: str | None = None
        self._device_name: str | None = None
        self._precision: str | None = None
        self._model_sha256: str | None = None
        self.start_count = 0
        self.restart_count = 0
        self.circuit_open = False
        self._registered: set[tuple[str, str]] = set()
        self._latest_by_camera: dict[tuple[str, str], dict] = {}
        self._error_code: str | None = None
        self._error_message: str | None = None
        self._event_sink = event_sink
        self._regions: dict[tuple[str, str], object] = {}
        # 每个 (camera, session) 的环形槽写指针：双槽轮转，避免两槽都"活动"后永久 NO_WRITABLE_SLOT
        self._ring_by_camera: dict[tuple[str, str], int] = {}
        self._compatibility_by_camera: dict[tuple[str, str], list] = {}
        self._result_seq_by: dict[str, dict[str, int]] = {}
        self._consumed_seq_by: dict[str, dict[str, int]] = {}
        # health_snapshot 真实计数（按 (camera_id, camera_session_id)，全部在 _lock 内更新）
        self._submitted_total: dict[tuple[str, str], int] = {}
        self._analyzed_total: dict[tuple[str, str], int] = {}
        self._replaced_total: dict[tuple[str, str], int] = {}
        # journal drain 毒丸上限：连续投递失败达到该次数的事件落父端 spool 兜底并 ack
        self._drain_max_attempts = max(1, int(drain_max_attempts))
        # drain 节流：距上次实际 drain 不足 250ms 直接跳过（pending 不丢，顺延到下一时间片）
        self._last_drain_monotonic_ns = 0

    # ---------------- 启动 / 停止 ----------------
    def start(self) -> None:
        with self._lock:
            if self._child is not None or self._closed:
                return
            self._spawn()

    def _spawn(self) -> None:
        with self._lock:
            self.start_count += 1
            child = self._factory()
            self._child = child
            out = getattr(child, "stdout", None)
            self._send_hello()
            if out is not None:
                self._reader = threading.Thread(target=self._read_loop, args=(child, out), daemon=True)
                self._reader.start()
            # 无可用 stdout（StubChild）则不启动读线程，状态保持 STARTING
            if self._heartbeat_interval_s > 0 and (
                self._heartbeat_thread is None or not self._heartbeat_thread.is_alive()
            ):
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop, daemon=True)
                self._heartbeat_thread.start()

    def _spawn_new_epoch(self) -> None:
        self._epoch = None
        # 新 epoch 的 PONG 重新计时，避免旧 PONG 时间戳误判新 worker 超时
        self._last_pong_monotonic_ns = None
        self._worker_instance_id = None
        self._worker_pid = None
        self._close_current_child()
        self._spawn()
        self._state = STARTING

    def _close_current_child(self) -> None:
        c = self._child
        self._child = None
        if c is not None:
            try:
                c.kill()
            except Exception:
                pass
            try:
                c.wait(timeout=2.0)
            except Exception:
                pass
            try:
                c.close()
            except Exception:
                pass

    def stop_and_drain_blocking(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._state = STOPPING
            c = self._child
            self._child = None
        if c is not None:
            try:
                c.write({"message_type": "SHUTDOWN", "worker_epoch": self._epoch,
                         "message_id": uuid.uuid4().hex, "payload": {"reason": "shutdown"}})
            except Exception:
                pass
            try:
                c.terminate()
            except Exception:
                pass
            try:
                c.wait(timeout=2.0)
            except Exception:
                pass
            if getattr(c, "is_alive", lambda: False)():
                try:
                    c.kill()
                except Exception:
                    pass
            try:
                c.close()
            except Exception:
                pass
        r = self._reader
        if r is not None and r.is_alive():
            r.join(timeout=2.0)
        hb = self._heartbeat_thread
        if hb is not None and hb.is_alive():
            hb.join(timeout=2.0)
        with self._lock:
            self._state = STOPPED

    # ---------------- Worker 消息处理 ----------------
    def _send_hello(self) -> None:
        c = self._child
        if c is None:
            return
        try:
            c.write({"message_type": "HELLO", "worker_epoch": None,
                     "message_id": uuid.uuid4().hex,
                     "payload": {
                         "supported_schema_versions": [1],
                         "runtime_key": "",
                         "runtime_config_revision": "",
                         "parent_pid": _parent_pid(),
                         "challenge": uuid.uuid4().hex,
                     }})
        except Exception:
            self._on_worker_exit()

    def _read_loop(self, child, out) -> None:
        try:
            while not self._closed:
                hdr = out.read(4)
                if not hdr or len(hdr) < 4:
                    break
                (n,) = struct.unpack(">I", hdr)
                if n > 64 * 1024 * 1024:
                    break
                body = b""
                while len(body) < n:  # raw FileIO 可能短读,循环读满
                    chunk = out.read(n - len(body))
                    if not chunk:
                        break
                    body += chunk
                if len(body) != n:
                    break
                msg = decode_message(hdr + body)
                self._ingest(msg)
        except Exception:
            pass
        finally:
            if not self._closed:
                try:
                    self._on_worker_exit()
                except Exception:
                    pass

    def _ingest(self, msg: dict) -> None:
        if msg is None or not isinstance(msg, dict):
            return
        mt = msg.get("message_type")
        env_epoch = msg.get("worker_epoch")
        if mt in ("WORKER_STARTING", "WORKER_READY"):
            payload = msg.get("payload") or {}
            ep = payload.get("worker_epoch") or env_epoch
            if mt == "WORKER_STARTING" and ep:
                with self._lock:
                    self._epoch = str(ep)
                    self._worker_instance_id = payload.get("worker_instance_id")
                    self._worker_pid = payload.get("worker_pid")
            if mt == "WORKER_READY":
                with self._lock:
                    if ep:
                        self._epoch = str(ep)
                        self._worker_instance_id = payload.get("worker_instance_id")
                        self._worker_pid = payload.get("worker_pid")
                    self._device = payload.get("device")
                    self._device_name = payload.get("device_name")
                    self._precision = payload.get("precision")
                    self._model_sha256 = payload.get("model_sha256")
                    self._error_code = None
                    self._error_message = None
                    self._state = READY
            elif mt == "WORKER_READY":
                pass
            return
        # 后续消息要求 epoch 与当前一致，否则丢弃（old-epoch）
        if env_epoch is None or env_epoch != self._epoch:
            return
        if mt == "INFERENCE_RESULT":
            payload = msg.get("payload") or {}
            camera_id = payload.get("camera_id")
            camera_session_id = payload.get("camera_session_id")
            if camera_id and camera_session_id and payload.get("request_id"):
                key = (camera_id, camera_session_id)
                had_previous = key in self._latest_by_camera
                self._latest_by_camera[key] = payload
                with self._lock:
                    seq = self._result_seq_by.setdefault(camera_id, {}).get(camera_session_id, 0) + 1
                    self._result_seq_by[camera_id][camera_session_id] = seq
                    self._bump(self._analyzed_total, key)
                    if had_previous:
                        # latest-only 替换：覆盖了尚未消费的旧最新结果
                        self._bump(self._replaced_total, key)
                self._drain_journal()
        elif mt == "PONG":
            # epoch 一致性校验通过后才记录：只认当前 worker 的心跳
            self._last_pong_monotonic_ns = time.monotonic_ns()
        elif mt == "WORKER_ERROR":
            payload = msg.get("payload") or {}
            with self._lock:
                self._state = UNAVAILABLE
                self._error_code = payload.get("error_code")
                self._error_message = payload.get("error_message")

    def _on_worker_exit(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self.restart_limit <= 0 or self.circuit_open or self.restart_count >= self.restart_limit:
                self.circuit_open = True
                self._state = UNAVAILABLE
                self._close_current_child()
                return
            self.restart_count += 1
            self._spawn_new_epoch()

    # ---------------- 对外 API ----------------
    def offer(self, camera_id: str, camera_session_id: str, frame_id: int,
              config_revision: str) -> RuntimeOfferOutcome:
        with self._lock:
            if self._closed:
                return RuntimeOfferOutcome.CLOSED
            if self._state != READY:
                return RuntimeOfferOutcome.WORKER_NOT_READY
            if (camera_id, camera_session_id) not in self._registered:
                self._registered.add((camera_id, camera_session_id))
                try:
                    self._child.write({"message_type": "REGISTER_CAMERA",
                                       "worker_epoch": self._epoch,
                                       "message_id": uuid.uuid4().hex,
                                       "payload": {"camera_id": camera_id,
                                                   "camera_session_id": camera_session_id,
                                                   "camera_config_revision": config_revision}})
                except Exception:
                    return RuntimeOfferOutcome.WORKER_NOT_READY
            try:
                self._child.write({"message_type": "INFER_FRAME",
                                   "worker_epoch": self._epoch,
                                   "message_id": uuid.uuid4().hex,
                                   "payload": {
                                       "request_id": uuid.uuid4().hex,
                                       "camera_id": camera_id,
                                       "camera_session_id": camera_session_id,
                                       "frame_id": frame_id,
                                       "config_revision": config_revision,
                                   }})
            except Exception:
                return RuntimeOfferOutcome.WORKER_NOT_READY
            self._bump(self._submitted_total, (camera_id, camera_session_id))
            return RuntimeOfferOutcome.ACCEPTED

    def offer_frame(self, frame, request_meta) -> RuntimeOfferOutcome:
        camera_id = request_meta.camera_id
        camera_session_id = request_meta.camera_session_id
        key = (camera_id, camera_session_id)
        need_register = False
        with self._lock:
            if self._closed:
                return RuntimeOfferOutcome.CLOSED
            if self._state != READY:
                return RuntimeOfferOutcome.WORKER_NOT_READY
            region = self._regions.get(key)
            if region is None:
                from .shared_frames import FrameShmRegion, new_region_name
                try:
                    region = FrameShmRegion(
                        shm_name=new_region_name("ai-monitor-pose"),
                        max_width=self.config.max_frame_width,
                        max_height=self.config.max_frame_height,
                        slots=2,
                    )
                except Exception:
                    return RuntimeOfferOutcome.WORKER_NOT_READY
                self._regions[key] = region
                if key not in self._registered:
                    self._registered.add(key)
                    need_register = True
            # 环形轮转写槽：双槽上轮流写，避免两槽 activity 常驻导致 SlotBusy
            ri = self._ring_by_camera.get(key, 0)
            self._ring_by_camera[key] = ri + 1
        # 锁外重 I/O：REGISTER_CAMERA 管道写、1080p 帧共享内存 memcpy（毫秒级）。
        # region 引用自 get 起不会被并发删除（unregister_camera 仅在任务关闭时调用），
        # 多摄像头帧提交因此不再被全局锁串行化。
        if need_register:
            self._send_register(camera_id, camera_session_id, request_meta.config_revision)
        try:
            ref = region.submit(
                frame,
                now_ns=request_meta.observed_at_monotonic_ns,
                prefer_slot=ri % region.slots,
            )
        except Exception:
            return RuntimeOfferOutcome.NO_WRITABLE_SLOT
        try:
            self._child.write({
                "message_type": "INFER_FRAME",
                "worker_epoch": self._epoch,
                "message_id": uuid.uuid4().hex,
                "correlation_id": request_meta.request_id,
                "payload": {
                    "request_id": request_meta.request_id,
                    "camera_id": camera_id,
                    "camera_session_id": camera_session_id,
                    "frame_id": request_meta.frame_id,
                    "config_revision": request_meta.config_revision,
                    "observed_at_unix_ns": request_meta.observed_at_unix_ns,
                    "observed_at_monotonic_ns": request_meta.observed_at_monotonic_ns,
                    "frame_ref": ref.to_dict(),
                },
            })
        except Exception:
            return RuntimeOfferOutcome.WORKER_NOT_READY
        with self._lock:
            self._bump(self._submitted_total, (camera_id, camera_session_id))
        return RuntimeOfferOutcome.ACCEPTED

    def _send_register(self, camera_id, camera_session_id, config_revision) -> None:
        c = self._child
        if c is None:
            return
        try:
            c.write({"message_type": "REGISTER_CAMERA",
                     "worker_epoch": self._epoch,
                     "message_id": uuid.uuid4().hex,
                     "payload": {"camera_id": camera_id,
                                 "camera_session_id": camera_session_id,
                                 "camera_config_revision": config_revision}})
        except Exception:
            pass

    def _heartbeat_loop(self) -> None:
        """父端心跳看门狗：按 interval 发 PING，超过 timeout 未收到 PONG 判定 worker 挂死。

        PING 发送失败（管道断开）由 read loop 的 EOF 路径处理，此处 continue 即可；
        判定超时前必须至少收到过一次 PONG（_last_pong_monotonic_ns 非 None）。
        """
        interval = self._heartbeat_interval_s
        timeout_ns = self._heartbeat_timeout_s * 1_000_000_000
        while True:
            time.sleep(interval)
            if self._closed:
                return
            with self._lock:
                child = self._child
                epoch = self._epoch
                state = self._state
            if child is None or epoch is None or state != READY:
                continue
            try:
                child.write({"message_type": "PING", "worker_epoch": epoch,
                             "message_id": uuid.uuid4().hex,
                             "payload": {"ping_id": uuid.uuid4().hex,
                                         "parent_sent_ns": time.time_ns()}})
            except Exception:
                continue
            last_pong = self._last_pong_monotonic_ns
            if last_pong is not None and time.monotonic_ns() - last_pong > timeout_ns:
                self._on_worker_exit()
                return

    def _drain_journal(self) -> None:
        """把 worker journal pending 排空到父端（可靠事件路径）。

        只有 event_sink Future 成功返回 event_id 精确匹配且 persisted=true 的 ACK 才
        mark_parent_acked；失败保留 pending 并记录 attempt_count/last_error，由下轮
        drain 重试。达到 _drain_max_attempts 的毒丸事件落父端 event_spool（durable
        兜底）后再 ack，修复"submit Future 未决即 ack、失败即丢事件"。
        """
        jp = getattr(self.config, "worker_journal_path", "")
        if not jp:
            return
        now_ns = time.monotonic_ns()
        if now_ns - self._last_drain_monotonic_ns < _DRAIN_THROTTLE_NS:
            # 节流：INFERENCE_RESULT 每秒触发 8-32 次，SQLite 打开/查询/关闭只需 4Hz 级；
            # pending 行不丢，顺延到下一个时间片被捡起。
            return
        self._last_drain_monotonic_ns = now_ns
        from .contracts import FallTransitionV1
        for row in self._journal_pending_rows(jp):
            eid = row["event_id"]
            payload_json = row["payload"]
            try:
                tr = FallTransitionV1.from_dict(json.loads(payload_json))
            except Exception as e:
                self._handle_drain_failure(jp, eid, payload_json, f"payload parse failed: {e!r}")
                continue
            key = (tr.camera_id, tr.camera_session_id)
            bucket = self._compatibility_by_camera.setdefault(key, [])
            # 重试期间不向兼容队列重复堆积同一事件（at-least-once 以 event_id 去重）
            if not any(getattr(t, "event_id", None) == eid for t in bucket):
                bucket.append(tr)
            if self._event_sink is None:
                # 无 sink 兼容模式：不做可靠投递，直接 ack（规格 5.8 兼容路径）
                self._journal_mark_acked(jp, eid)
                continue
            self._submit_journal_event(jp, eid, payload_json, tr)

    @staticmethod
    def _bump(counter: dict, key) -> None:
        counter[key] = counter.get(key, 0) + 1

    def _journal_pending_rows(self, jp: str) -> list:
        from .worker.transition_journal import WorkerJournal
        try:
            journal = WorkerJournal(jp, worker_instance_id="parent")
        except Exception:
            return []
        try:
            return list(journal.pending())
        except Exception:
            return []
        finally:
            try:
                journal.close()
            except Exception:
                pass

    def _journal_mark_acked(self, jp: str, eid: str) -> None:
        from .worker.transition_journal import WorkerJournal
        try:
            journal = WorkerJournal(jp, worker_instance_id="parent")
        except Exception:
            return
        try:
            journal.mark_parent_acked(eid)
        except Exception:
            pass
        finally:
            try:
                journal.close()
            except Exception:
                pass

    def _journal_record_failure(self, jp: str, eid: str, error: str):
        from .worker.transition_journal import WorkerJournal
        try:
            journal = WorkerJournal(jp, worker_instance_id="parent")
        except Exception:
            return None
        try:
            return journal.record_attempt_failure(eid, error)
        except Exception:
            return None
        finally:
            try:
                journal.close()
            except Exception:
                pass

    def _submit_journal_event(self, jp: str, eid: str, payload_json: str, tr) -> None:
        from .event_mapper import map_transition_to_vision_event
        from .host_protocols import FrozenHostEventV1
        try:
            ve = map_transition_to_vision_event(tr)
            fut = self._event_sink.submit(FrozenHostEventV1.from_vision_event(ve))
        except Exception as e:
            self._handle_drain_failure(jp, eid, payload_json, f"sink submit raised: {e!r}")
            return
        if fut is None:
            self._handle_drain_failure(jp, eid, payload_json, "sink submit returned None")
            return
        # 不阻塞读线程：Future 完成回调里做 ack/失败计数（已完成的 Future 立即执行）
        fut.add_done_callback(
            lambda f, _jp=jp, _eid=eid, _pj=payload_json:
            self._on_sink_future(_jp, _eid, _pj, f))

    def _on_sink_future(self, jp: str, eid: str, payload_json: str, fut) -> None:
        try:
            ack = fut.result()
        except Exception as e:
            self._handle_drain_failure(jp, eid, payload_json, f"sink future failed: {e!r}")
            return
        except BaseException as e:  # Future 取消（CancelledError）也必须保持 pending
            self._handle_drain_failure(jp, eid, payload_json, f"sink future cancelled: {e!r}")
            return
        if ack is None:
            self._handle_drain_failure(jp, eid, payload_json, "sink ack is None")
            return
        if getattr(ack, "event_id", None) != eid:
            self._handle_drain_failure(
                jp, eid, payload_json, f"ack event_id mismatch: {getattr(ack, 'event_id', None)!r}")
            return
        if not getattr(ack, "persisted", False):
            self._handle_drain_failure(jp, eid, payload_json, "ack persisted=false")
            return
        self._journal_mark_acked(jp, eid)

    def _handle_drain_failure(self, jp: str, eid: str, payload_json: str, error: str) -> None:
        attempts = self._journal_record_failure(jp, eid, error)
        if attempts is None or attempts < self._drain_max_attempts:
            return
        # 毒丸：落父端 spool 兜底（durable），成功落盘后才 ack
        if self._spool_poison_pill(eid, payload_json, error):
            self._journal_mark_acked(jp, eid)

    def _spool_poison_pill(self, eid: str, payload_json: str, error: str) -> bool:
        sp = getattr(self.config, "event_spool_path", "")
        if not sp:
            return False
        from .event_spool import EventSpool
        try:
            spool = EventSpool(
                sp,
                pending_capacity=int(
                    getattr(self.config, "event_spool_pending_capacity", 10000) or 10000))
            try:
                spool.add(_PoisonPillEvent(eid, payload_json, error))
            finally:
                spool.close()
            return True
        except Exception:
            return False

    def poll(self, camera_id=None, camera_session_id=None):
        if camera_id is None or camera_session_id is None:
            return None  # 保留兼容空实现
        key = (camera_id, camera_session_id)
        with self._lock:
            res = self._latest_by_camera.get(key)
            compat = tuple(self._compatibility_by_camera.pop(key, ())) or ()
            self._consumed_seq_by.setdefault(camera_id, {})[camera_session_id] = \
                self._result_seq_by.get(camera_id, {}).get(camera_session_id, 0)
        return SimpleNamespace(latest_result=res, compatibility_events=compat, health=None)

    def has_latest_result_or_health_change(self, camera_id: str, camera_session_id: str) -> bool:
        with self._lock:
            result_pending = self._result_seq_by.get(camera_id, {}).get(camera_session_id, 0) > \
                self._consumed_seq_by.get(camera_id, {}).get(camera_session_id, 0)
            compat_pending = bool(self._compatibility_by_camera.get((camera_id, camera_session_id)))
        return bool(result_pending or compat_pending)

    def has_unseen_compatibility_event(self, camera_id: str, camera_session_id: str) -> bool:
        with self._lock:
            return bool(self._compatibility_by_camera.get((camera_id, camera_session_id)))

    def unregister_camera(self, camera_id: str, camera_session_id: str) -> None:
        with self._lock:
            key = (camera_id, camera_session_id)
            self._registered.discard(key)
            region = self._regions.pop(key, None)
            self._compatibility_by_camera.pop(key, None)
            self._latest_by_camera.pop(key, None)
        if region is not None:
            try:
                region.close(unlink=True)
            except Exception:
                try:
                    region.close()
                except Exception:
                    pass
        c = self._child
        if c is not None and self._epoch:
            try:
                c.write({"message_type": "UNREGISTER_CAMERA", "worker_epoch": self._epoch,
                         "message_id": uuid.uuid4().hex,
                         "payload": {"camera_id": camera_id,
                                     "camera_session_id": camera_session_id}})
            except Exception:
                pass

    def latest_result(self, camera_id: str, camera_session_id: str):
        return self._latest_by_camera.get((camera_id, camera_session_id))

    def register_camera(self, camera_id: str, camera_session_id: str, config_revision: str) -> None:
        self.offer(camera_id, camera_session_id, 0, config_revision)

    def _children_alive(self) -> int:
        c = self._child
        if c is None:
            return 0
        return 1 if getattr(c, "is_alive", lambda: False)() else 0

    def _read_spool_pending(self) -> int | None:
        """读父端 event spool 的 pending 计数；无路径或任何异常返回 None（快照保持 delivery_metrics={}）。

        独立 sqlite 文件、不碰 runtime 锁；调用方应在 self._lock 之外调用（registry 锁内
        持锁做 sqlite 读无死锁风险，但锁外更稳妥）。只读计数，用完即关。
        """
        sp = getattr(self.config, "event_spool_path", "")
        if not sp:
            return None
        from .event_spool import EventSpool
        try:
            spool = EventSpool(sp, pending_capacity=1)
            try:
                return spool.pending_count()
            finally:
                spool.close()
        except Exception:
            return None

    def health_snapshot(self, runtime_key: str | None = None) -> FallRuntimeHealthSnapshotV1:
        # spool 读数在 self._lock 之外完成（避免持 runtime 锁做 sqlite I/O）
        spool_pending = self._read_spool_pending()
        with self._lock:
            wh = WorkerHealthV1(
                schema_version=1,
                state=self._state,
                error_code=self._error_code,
                error_message=self._error_message,
                worker_epoch=self._epoch,
                worker_pid=self._worker_pid,
                cuda_device=self._device,
                cuda_device_name=self._device_name,
                model_sha256=self._model_sha256,
                last_heartbeat_monotonic_ns=self._last_pong_monotonic_ns,
                restart_count=self.restart_count,
            )
            return FallRuntimeHealthSnapshotV1(
                schema_version=1, enabled=True, mode=self._mode, runtime_key=runtime_key,
                worker=wh, gpu_metrics={}, model_metadata={},
                delivery_metrics=({} if spool_pending is None
                                  else {"spool_pending": spool_pending}),
                cameras=tuple(
                    CameraFallHealthV1(camera_id=k[0], camera_session_id=k[1], state=self._state,
                                       submitted_total=self._submitted_total.get(k, 0),
                                       analyzed_total=self._analyzed_total.get(k, 0),
                                       replaced_total=self._replaced_total.get(k, 0),
                                       stale_total=0, effective_fps=0.0, latest_result_age_ms=None,
                                       transition_queue_depth=0, open_incidents=0)
                    for k in list(self._registered)
                ),
            )

    @property
    def state(self) -> str:
        return self._state

    @property
    def worker_epoch(self) -> str | None:
        return self._epoch

    @property
    def worker_pid(self) -> int | None:
        return self._worker_pid

    @property
    def gpu_device_name(self) -> str | None:
        return self._device_name

    @property
    def precision(self) -> str | None:
        return self._precision


class _PoisonPillEvent:
    """drain 毒丸兜底记录：把无法成功投递的原始 journal payload 原样落入父端 spool。

    EventSpool.add 只需要 event_id + to_dict()；这里保留原始 payload 并附上
    poison_pill/last_error 审计标记，供人工排查与后续恢复导入器处理。
    """

    def __init__(self, event_id: str, payload_json: str, error: str) -> None:
        self.event_id = event_id
        self._payload_json = payload_json
        self._error = error

    def to_dict(self) -> dict:
        try:
            d = json.loads(self._payload_json)
            if isinstance(d, dict):
                out = dict(d)
                out.setdefault("poison_pill", True)
                out["poison_last_error"] = self._error
                return out
        except Exception:
            pass
        return {"event_id": self.event_id, "poison_pill": True,
                "poison_last_error": self._error, "raw_payload_json": self._payload_json}


class _NullChild:
    pid = -1

    def is_alive(self) -> bool:
        return False

    def write(self, env: dict) -> None:
        raise BrokenPipeError("no child configured")

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        return -1

    def close(self):
        pass


def _null_factory_child():
    return _NullChild()


def _parent_pid() -> int:
    try:
        import os
        return os.getpid()
    except Exception:
        return -1