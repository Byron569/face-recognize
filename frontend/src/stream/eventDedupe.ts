/**
 * 可靠 fall 事件去重（阶段 8 / M1）。
 *
 * 规则严格遵循融合实施方案：
 * - 只使用冻结的 WS envelope 顶层 `event_id`，缺失时才回退顶层 `dedupe_key`；
 * - 绝不从 `payload` 或数据库整数 `id` 猜测 key；
 * - 维护最多 4096 项、24 小时 TTL 的 LRU，并把 canonical key/timestamp 保存到有界 sessionStorage；
 * - 可靠 fall 事件缺少两种 ID 时拒绝告警并记录协议错误；
 * - 旧 recognition 事件保持兼容（不参与去重门禁）。
 */

export const DEFAULT_MAX = 4096;
export const DEFAULT_TTL_MS = 24 * 60 * 60 * 1000;
export const DEFAULT_STORAGE_KEY = 'ai-monitor:event-dedupe:v1';

export interface ReliableEventLike {
  event_type?: string | null;
  event_id?: string | null;
  dedupe_key?: string | null;
  id?: number | string | null;
}

export interface ProtocolErrorInfo {
  event_type: string | null;
  reason: string;
}

export interface EventDedupeOptions {
  max?: number;
  ttlMs?: number;
  storage?: Storage | null;
  storageKey?: string;
  now?: () => number;
  onProtocolError?: (info: ProtocolErrorInfo) => void;
}

export type DedupeDecision =
  | { status: 'new'; key: string }
  | { status: 'duplicate'; key: string }
  | { status: 'rejected'; reason: string };

/** 是否为可靠 fall 事件类型（fall_ 前缀）。 */
export function isFallType(eventType: unknown): boolean {
  return typeof eventType === 'string' && eventType.startsWith('fall_');
}

/**
 * 提取稳定去重 key：顶层 event_id > 顶层 dedupe_key。
 * 无两种 ID 时返回 null（调用方据此决定拒绝或兼容放行）。
 */
export function resolveDedupeKey(input: ReliableEventLike): string | null {
  if (typeof input.event_id === 'string' && input.event_id !== '') return input.event_id;
  if (typeof input.dedupe_key === 'string' && input.dedupe_key !== '') return input.dedupe_key;
  return null;
}

interface PersistedShape {
  entries: Array<[string, number]>;
  saved_at?: number;
}

export class EventDedupe {
  private readonly max: number;
  private readonly ttlMs: number;
  private readonly storage: Storage | null;
  private readonly storageKey: string;
  private readonly now: () => number;
  private readonly onProtocolError?: (info: ProtocolErrorInfo) => void;

  // key -> ts(单调/clock ms)。Map 保持插入序,重插即视为最近使用。
  private readonly entries: Map<string, number> = new Map();

  constructor(opts: EventDedupeOptions = {}) {
    this.max = opts.max ?? DEFAULT_MAX;
    this.ttlMs = opts.ttlMs ?? DEFAULT_TTL_MS;
    this.storage = opts.storage ?? null;
    this.storageKey = opts.storageKey ?? DEFAULT_STORAGE_KEY;
    this.now = opts.now ?? (() => Date.now());
    this.onProtocolError = opts.onProtocolError;
    this.load();
  }

  get size(): number {
    return this.entries.size;
  }

  private load(): void {
    if (!this.storage) return;
    try {
      const raw = this.storage.getItem(this.storageKey);
      if (!raw) return;
      const parsed = JSON.parse(raw) as PersistedShape;
      if (!parsed || !Array.isArray(parsed.entries)) return;
      for (const item of parsed.entries) {
        if (!Array.isArray(item)) continue;
        const [key, ts] = item;
        if (typeof key === 'string' && key !== '' && typeof ts === 'number' && Number.isFinite(ts)) {
          this.entries.set(key, ts);
        }
      }
      this.dropExpired();
    } catch {
      // 损坏的 sessionStorage 不抛异常,视为空
      this.entries.clear();
    }
  }

  private persist(): void {
    if (!this.storage) return;
    try {
      const snapshot: Array<[string, number]> = [];
      for (const [key, ts] of this.entries) {
        snapshot.push([key, ts]);
        if (snapshot.length >= this.max) break;
      }
      this.storage.setItem(this.storageKey, JSON.stringify({ entries: snapshot, saved_at: this.now() }));
    } catch {
      // sessionStorage 满/不可用:静默丢弃持久化,内存 LRU 仍正常工作
    }
  }

  private dropExpired(): void {
    const t = this.now();
    for (const [key, ts] of this.entries) {
      if (t - ts >= this.ttlMs) this.entries.delete(key);
    }
  }

  private touch(key: string): void {
    this.entries.delete(key);
    this.entries.set(key, this.now());
  }

  /** 判定一帧事件是否为新事件。可靠 fall 做幂等门禁,旧 recognition 保持兼容放行。 */
  classify(input: ReliableEventLike): DedupeDecision {
    const key = resolveDedupeKey(input);
    if (key === null) {
      if (isFallType(input.event_type)) {
        const reason = 'reliable fall event missing both top-level event_id and dedupe_key';
        this.onProtocolError?.({ event_type: input.event_type ?? null, reason });
        return { status: 'rejected', reason };
      }
      // 旧 recognition/未知事件无可靠 ID,兼容放行(不参与 fall 去重门禁)
      return { status: 'new', key: '' };
    }

    // 先清理过期条目,避免已过期 key 因仍留在 map 中而被误判为重复
    this.dropExpired();

    if (this.entries.has(key)) {
      this.touch(key);
      this.persist();
      return { status: 'duplicate', key };
    }

    this.entries.set(key, this.now());
    // LRU 淘汰最旧
    while (this.entries.size > this.max) {
      const oldest = this.entries.keys().next().value;
      if (oldest === undefined) break;
      this.entries.delete(oldest);
    }
    this.persist();
    return { status: 'new', key };
  }
}

/**
 * 便捷工厂：默认 sessionStorage、默认 4096/24h、可选协议错误回调。
 * sessionStorage 不存在（SSR）时退化为纯内存去重。
 */
export function createEventDedupe(options: Omit<EventDedupeOptions, 'storage' | 'storageKey'> = {}): EventDedupe {
  let storage: Storage | null = null;
  try {
    storage = window.sessionStorage ?? null;
  } catch {
    storage = null;
  }
  return new EventDedupe({ ...options, storage, storageKey: DEFAULT_STORAGE_KEY });
}