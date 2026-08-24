import { describe, it, expect, vi } from 'vitest';
import {
  EventDedupe,
  resolveDedupeKey,
  isFallType,
  DEFAULT_STORAGE_KEY,
} from './eventDedupe';

/** 内存版 Storage,用于精确控制 sessionStorage 内容。 */
class FakeStorage implements Storage {
  private data = new Map<string, string>();
  get length(): number {
    return this.data.size;
  }
  clear(): void {
    this.data.clear();
  }
  getItem(key: string): string | null {
    return this.data.has(key) ? this.data.get(key)! : null;
  }
  key(index: number): string | null {
    return [...this.data.keys()][index] ?? null;
  }
  removeItem(key: string): void {
    this.data.delete(key);
  }
  setItem(key: string, value: string): void {
    this.data.set(key, String(value));
  }
}

function seededStorage(): { storage: FakeStorage } {
  return { storage: new FakeStorage() };
}

describe('resolveDedupeKey', () => {
  it('prefers top-level event_id over dedupe_key', () => {
    expect(
      resolveDedupeKey({ event_id: 'evt-1', dedupe_key: 'dedupe-1', id: 5 }),
    ).toBe('evt-1');
  });

  it('falls back to top-level dedupe_key when event_id is missing', () => {
    expect(resolveDedupeKey({ dedupe_key: 'dedupe-1', id: 5 })).toBe('dedupe-1');
  });

  it('never derives key from the database integer id', () => {
    expect(resolveDedupeKey({ id: 912, event_type: 'fall_detected' })).toBeNull();
  });

  it('returns null when both reliable ids are absent (nested payload ids are ignored)', () => {
    const input = { event_type: 'fall_detected', payload: { event_id: 'x' } } as const;
    expect(resolveDedupeKey(input as unknown as Parameters<typeof resolveDedupeKey>[0])).toBeNull();
  });
});

describe('isFallType', () => {
  it('recognizes fall_ prefixed types', () => {
    expect(isFallType('fall_detected')).toBe(true);
    expect(isFallType('fall_potential')).toBe(true);
    expect(isFallType('fall_recovered')).toBe(true);
  });
  it('rejects unrelated types', () => {
    expect(isFallType('recognition')).toBe(false);
    expect(isFallType(null)).toBe(false);
    expect(isFallType(42)).toBe(false);
  });
});

describe('EventDedupe: key precedence & idempotency', () => {
  it('returns new for a fresh reliable fall event, duplicate for the same key', () => {
    const d = new EventDedupe({ storage: null });
    const evt = { event_type: 'fall_detected', event_id: '9b4d...', dedupe_key: 'k', id: 1 };

    expect(d.classify(evt)).toEqual({ status: 'new', key: '9b4d...' });
    expect(d.classify(evt)).toEqual({ status: 'duplicate', key: '9b4d...' });
    expect(d.size).toBe(1);
  });

  it('dedupes by top-level event_id even when dedupe_key differs', () => {
    const d = new EventDedupe({ storage: null });
    d.classify({ event_type: 'fall_potential', event_id: 'E1', dedupe_key: 'A' });
    expect(d.classify({ event_type: 'fall_potential', event_id: 'E1', dedupe_key: 'A' })).toEqual({
      status: 'duplicate',
      key: 'E1',
    });
  });

  it('rejects a reliable fall event missing both ids and reports a protocol error', () => {
    const onProtocolError = vi.fn();
    const d = new EventDedupe({ storage: null, onProtocolError });
    const decision = d.classify({ event_type: 'fall_detected', id: 912 });

    expect(decision.status).toBe('rejected');
    expect(onProtocolError).toHaveBeenCalledWith({
      event_type: 'fall_detected',
      reason: expect.stringContaining('event_id'),
    });
  });

  it('keeps legacy recognition events compatible (no dedup gate)', () => {
    const d = new EventDedupe({ storage: null });
    const evt = { event_type: 'recognition', id: 5 };
    expect(d.classify(evt)).toEqual({ status: 'new', key: '' });
    expect(d.classify(evt)).toEqual({ status: 'new', key: '' }); // 旧事件不被去重门禁拦截
  });
});

describe('EventDedupe: TTL', () => {
  it('treats a key as new again after its TTL elapses', () => {
    const now = vi.fn(() => 1_000_000);
    const d = new EventDedupe({ storage: null, ttlMs: 1000, now });
    d.classify({ event_type: 'fall_detected', event_id: 'K' });
    expect(d.classify({ event_type: 'fall_detected', event_id: 'K' }).status).toBe('duplicate');

    now.mockReturnValue(1_000_000 + 1001); // 越过 TTL
    expect(d.classify({ event_type: 'fall_detected', event_id: 'K' }).status).toBe('new');
  });
});

describe('EventDedupe: LRU eviction', () => {
  it('evicts the oldest entries beyond max, evicted key re-enterable', () => {
    const now = vi.fn(() => 0);
    const d = new EventDedupe({ storage: null, max: 2, ttlMs: 60_000, now });
    d.classify({ event_type: 'fall_detected', event_id: 'k1' });
    now.mockReturnValue(1);
    d.classify({ event_type: 'fall_detected', event_id: 'k2' });
    now.mockReturnValue(2);
    d.classify({ event_type: 'fall_detected', event_id: 'k3' }); // k1 应被挤出

    expect(d.size).toBe(2);
    // 重新录入被挤出的 k1,会再次淘汰 map 中最旧的 k2,但容量保持有界
    expect(d.classify({ event_type: 'fall_detected', event_id: 'k1' }).status).toBe('new');
    expect(d.size).toBe(2);
    expect(d.classify({ event_type: 'fall_detected', event_id: 'k3' }).status).toBe('duplicate');
    expect(d.classify({ event_type: 'fall_detected', event_id: 'k2' }).status).toBe('new');
  });
});

describe('EventDedupe: sessionStorage 持久化', () => {
  it('persists canonical keys to a bounded sessionStorage entry', () => {
    const { storage } = seededStorage();
    const now = vi.fn(() => 0);
    const d = new EventDedupe({ storage, max: 100, ttlMs: 60_000, now });
    d.classify({ event_type: 'fall_detected', event_id: 'center' });
    d.classify({ event_type: 'fall_detected', event_id: 'right' });

    const raw = storage.getItem(DEFAULT_STORAGE_KEY);
    expect(raw).toBeTruthy();
    const parsed = JSON.parse(raw!);
    // 有界：不超过 max 项
    expect(parsed.entries.length).toBeLessThanOrEqual(100);
    expect(parsed.entries).toContainEqual(['center', 0]);
  });

  it('restores dedup state from sessionStorage on construction', () => {
    const { storage } = seededStorage();
    const seed = new EventDedupe({ storage, ttlMs: 60_000 });
    seed.classify({ event_type: 'fall_detected', event_id: 'K' });

    const restored = new EventDedupe({ storage, ttlMs: 60_000 });
    expect(restored.classify({ event_type: 'fall_detected', event_id: 'K' }).status).toBe('duplicate');
  });

  it('survives corrupted sessionStorage without throwing', () => {
    const { storage } = seededStorage();
    storage.setItem(DEFAULT_STORAGE_KEY, '{{{ not-json');
    const d = new EventDedupe({ storage, ttlMs: 60_000 });
    expect(d.classify({ event_type: 'fall_detected', event_id: 'K' }).status).toBe('new');
  });

  it('ignores malformed persisted entries and recovers cleanly', () => {
    const { storage } = seededStorage();
    storage.setItem(
      DEFAULT_STORAGE_KEY,
      JSON.stringify({ entries: [['ok', 1000], [null, 1], ['bad-ts', 'x'], { nope: true }] }),
    );
    const d = new EventDedupe({ storage, ttlMs: 60_000, now: () => 1000 });
    expect(d.size).toBe(1);
    expect(d.classify({ event_type: 'fall_detected', event_id: 'ok' }).status).toBe('duplicate');
  });
});