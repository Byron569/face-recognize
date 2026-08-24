/**
 * stage10 (M2): analytics wire 协议解析与校验 (纯函数 + 有限缓存)。
 *
 * 解析后端 ``type: analytics`` 消息。校验通过后返回可直接按 preview_pixels
 * 绘制的 fall overlay;任何不合法(尺寸非正 / scale 非有限 / 内外 session 不等 /
 * 未知 pose state)一律返回 null,绝不猜坐标系。缓存按 (camera_session_id,
 * preview_frame_id) 键控,TTL 由调用方负责到期清除。
 */

export type PoseWireState = 'normal' | 'potential' | 'fallen';

const VALID_STATES: ReadonlySet<string> = new Set(['normal', 'potential', 'fallen']);

export interface FallTrack {
  poseTrackId: number | null;
  state: PoseWireState;
  score: number | null;
  bbox: [number, number, number, number] | null; // [x, y, w, h] preview pixels
  keypoints: Array<[number, number]> | null;
}

export interface FallAnalytics {
  schemaVersion: number;
  cameraSessionId: string;
  previewFrameId: number;
  sourceFrameId: number | null;
  previewWidth: number;
  previewHeight: number;
  transform: { scaleX: number; scaleY: number };
  health: string | null;
  resultAgeMs: number | null;
  overlayExpiresInMs: number | null;
  workerEndToEndMs: number | null;
  tracks: FallTrack[];
  color: 'orange' | 'red' | null; // potential=orange, fallen=red, normal=null
}

export type ParsedAnalytics =
  | { ok: true; analytics: FallAnalytics }
  | { ok: false; reason: string };

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function coord(v: unknown, limit: number): number | null {
  if (!isFiniteNumber(v)) return null;
  return Math.min(Math.max(v, 0), limit - 0.001);
}

function parseTransform(t: unknown): { scaleX: number; scaleY: number } | null {
  if (!t || typeof t !== 'object') return null;
  const o = t as Record<string, unknown>;
  const sx = o['scale_x'];
  const sy = o['scale_y'];
  if (!isFiniteNumber(sx) || !isFiniteNumber(sy) || sx <= 0 || sy <= 0) return null;
  return { scaleX: sx, scaleY: sy };
}

/** 解析并校验一条 analytics WS 文本消息。 */
export function parseAnalyticsMessage(raw: string): ParsedAnalytics {
  let msg: unknown;
  try {
    msg = JSON.parse(raw);
  } catch {
    return { ok: false, reason: 'bad json' };
  }
  if (!msg || typeof msg !== 'object') return { ok: false, reason: 'not object' };
  const m = msg as Record<string, unknown>;
  if (m['type'] !== 'analytics') return { ok: false, reason: 'not analytics' };

  const outerSession = typeof m['camera_session_id'] === 'string' ? m['camera_session_id'] : '';
  const previewFrameId = m['preview_frame_id'];
  if (!isFiniteNumber(previewFrameId) || previewFrameId < 0) {
    return { ok: false, reason: 'bad preview_frame_id' };
  }

  const fd = m['fall_detection'];
  if (!fd || typeof fd !== 'object') return { ok: false, reason: 'missing fall_detection' };
  const f = fd as Record<string, unknown>;

  const innerSession = f['camera_session_id'];
  if (typeof innerSession !== 'string' || innerSession !== outerSession) {
    return { ok: false, reason: 'session mismatch' };
  }

  const previewWidth = f['preview_width'];
  const previewHeight = f['preview_height'];
  if (!isFiniteNumber(previewWidth) || !isFiniteNumber(previewHeight) ||
      previewWidth <= 0 || previewHeight <= 0) {
    return { ok: false, reason: 'bad preview size' };
  }
  const scale = parseTransform(f['transform']);
  if (!scale) return { ok: false, reason: 'bad transform' };

  const tracks: FallTrack[] = [];
  const rawTracks = Array.isArray(f['tracks']) ? (f['tracks'] as unknown[]) : [];
  for (const raw of rawTracks) {
    if (!raw || typeof raw !== 'object') continue;
    const t = raw as Record<string, unknown>;
    const stateRaw = t['state'];
    const state = typeof stateRaw === 'string' ? (stateRaw as PoseWireState) : 'normal';
    if (!VALID_STATES.has(state)) return { ok: false, reason: 'unknown pose state' };

    let bbox: [number, number, number, number] | null = null;
    const rawBbox = t['bbox'];
    if (Array.isArray(rawBbox) && rawBbox.length === 4) {
      const x1 = coord(rawBbox[0], previewWidth);
      const y1 = coord(rawBbox[1], previewHeight);
      const w = coord(rawBbox[2], previewWidth);
      const h = coord(rawBbox[3], previewHeight);
      if (x1 === null || y1 === null || w === null || h === null) return { ok: false, reason: 'bad bbox' };
      bbox = [x1, y1, w, h];
    }

    let keypoints: Array<[number, number]> | null = null;
    const rawKps = t['keypoints'];
    if (Array.isArray(rawKps)) {
      const kps: Array<[number, number]> = [];
      for (const kp of rawKps) {
        if (Array.isArray(kp) && kp.length >= 2) {
          const x = coord(kp[0], previewWidth);
          const y = coord(kp[1], previewHeight);
          if (x !== null && y !== null) kps.push([x, y]);
        }
      }
      keypoints = kps.length ? kps : null;
    }

    const scoreRaw = t['score'];
    tracks.push({
      poseTrackId: (isFiniteNumber(t['pose_track_id']) ? t['pose_track_id'] : null),
      state,
      score: isFiniteNumber(scoreRaw) ? scoreRaw : null,
      bbox,
      keypoints,
    });
  }

  const color: FallAnalytics['color'] =
    tracks.some((t) => t.state === 'fallen') ? 'red'
      : tracks.some((t) => t.state === 'potential') ? 'orange' : null;

  const analytics: FallAnalytics = {
    schemaVersion: isFiniteNumber(f['schema_version']) ? f['schema_version'] : 1,
    cameraSessionId: innerSession,
    previewFrameId: Number(previewFrameId),
    sourceFrameId: isFiniteNumber(f['source_frame_id']) ? f['source_frame_id'] : null,
    previewWidth,
    previewHeight,
    transform: scale,
    health: typeof f['health'] === 'string' ? f['health'] : null,
    resultAgeMs: isFiniteNumber(f['result_age_ms']) ? f['result_age_ms'] : null,
    overlayExpiresInMs: isFiniteNumber(f['overlay_expires_in_ms']) ? f['overlay_expires_in_ms'] : null,
    workerEndToEndMs: isFiniteNumber(f['worker_end_to_end_ms']) ? f['worker_end_to_end_ms'] : null,
    tracks,
    color,
  };

  return { ok: true, analytics };
}

/** 缓存条目:该 (session, preview_frame_id) 下的 overlay,带本地 deadline(ns)。 */
export interface FallOverlayCacheEntry {
  analytics: FallAnalytics & { previewFrameId: number };
  deadline: number; // performance.now() 时间轴上的过期时刻
}

/** 以 (camera_session_id, preview_frame_id) 为键的有限 LRU 缓存。 */
export class FallOverlayCache {
  private map = new Map<string, FallOverlayCacheEntry>();
  constructor(private capacity = 64) {}

  private key(session: string, frameId: number): string {
    return `${session}:${frameId}`;
  }

  /** 设置条目;返回是否替换了旧 key。超出容量逐出最旧。 */
  set(session: string, frameId: number, entry: FallOverlayCacheEntry): boolean {
    const k = this.key(session, frameId);
    const existing = this.map.has(k);
    this.map.delete(k);
    this.map.set(k, entry);
    if (this.map.size > this.capacity) {
      const oldest = this.map.keys().next().value;
      if (oldest !== undefined) this.map.delete(oldest);
    }
    return existing;
  }

  get(session: string, frameId: number): FallOverlayCacheEntry | undefined {
    const k = this.key(session, frameId);
    return this.map.get(k);
  }

  /** 清除某个 session 的全部条目(接收新 camera_session_id 时调用)。 */
  clearSession(session: string): void {
    for (const k of Array.from(this.map.keys())) {
      if (k.startsWith(`${session}:`)) this.map.delete(k);
    }
  }

  /** 清除所有已过期条目(惰性回收),返回清除条数。 */
  evictExpired(now = performance.now()): number {
    let removed = 0;
    for (const k of Array.from(this.map.keys())) {
      const e = this.map.get(k);
      if (e && e.deadline <= now) {
        this.map.delete(k);
        removed += 1;
      }
    }
    return removed;
  }

  clear(): void {
    this.map.clear();
  }

  get size(): number {
    return this.map.size;
  }
}