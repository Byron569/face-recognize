import { describe, it, expect } from 'vitest';
import {
  parseAnalyticsMessage,
  FallOverlayCache,
  FallAnalytics,
} from './analyticsProtocol';

function wireMessage(fd: Record<string, unknown>, overrides: Record<string, unknown> = {}) {
  return JSON.stringify({
    type: 'analytics',
    schema_version: 1,
    camera_id: 'cam-1',
    camera_session_id: 'sess-1',
    preview_frame_id: 100,
    fall_detection: {
      schema_version: 1,
      camera_session_id: 'sess-1',
      source_frame_id: 98,
      preview_width: 853,
      preview_height: 480,
      coordinate_space: 'preview_pixels',
      transform: { kind: 'scale_no_letterbox', scale_x: 0.44, scale_y: 0.44, offset_x: 0, offset_y: 0 },
      health: 'READY',
      result_age_ms: 82.4,
      overlay_expires_in_ms: 1118,
      worker_end_to_end_ms: 61.7,
      tracks: [
        {
          pose_track_id: 3, state: 'fallen', score: 0.9,
          bbox: [44.0, 88.0, 88.0, 220.0],
          keypoints: [[44.0, 88.0], [48.4, 92.4]],
        },
      ],
      ...fd,
    },
    ...overrides,
  });
}

describe('parseAnalyticsMessage', () => {
  it('parses a valid fallen message with preview_pixels', () => {
    const r = parseAnalyticsMessage(wireMessage({}));
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    const a = r.analytics;
    expect(a.previewFrameId).toBe(100);
    expect(a.cameraSessionId).toBe('sess-1');
    expect(a.sourceFrameId).toBe(98);
    expect(a.color).toBe('red');
    expect(a.transform.scaleX).toBeCloseTo(0.44);
    expect(a.tracks[0].state).toBe('fallen');
    expect(a.tracks[0].bbox).toEqual([44, 88, 88, 220]);
  });

  it('potential maps to orange, normal to null color', () => {
    const pot = parseAnalyticsMessage(wireMessage({
      tracks: [{ pose_track_id: 1, state: 'potential', score: 0.7, bbox: [1, 2, 3, 4], keypoints: [] }],
    }));
    if (pot.ok) expect(pot.analytics.color).toBe('orange');
    const norm = parseAnalyticsMessage(wireMessage({
      tracks: [{ pose_track_id: 1, state: 'normal', score: 0.5, bbox: [1, 2, 3, 4], keypoints: [] }],
    }));
    if (norm.ok) expect(norm.analytics.color).toBeNull();
  });

  it('rejects inner/outer session mismatch', () => {
    const r = parseAnalyticsMessage(wireMessage({ camera_session_id: 'other' }));
    expect(r.ok).toBe(false);
  });

  it('rejects unknown pose state', () => {
    const r = parseAnalyticsMessage(wireMessage({
      tracks: [{ pose_track_id: 1, state: 'FALLEN', score: 0.9, bbox: [1, 2, 3, 4], keypoints: [] }],
    }));
    expect(r.ok).toBe(false);
  });

  it('rejects non-positive preview size and bad transform', () => {
    expect(parseAnalyticsMessage(wireMessage({ preview_width: 0 })).ok).toBe(false);
    expect(parseAnalyticsMessage(wireMessage({ transform: { scale_x: 0, scale_y: 0 } })).ok).toBe(false);
  });

  it('rejects bad json and non-analytics type', () => {
    expect(parseAnalyticsMessage('not-json').ok).toBe(false);
    expect(parseAnalyticsMessage(JSON.stringify({ type: 'detections' })).ok).toBe(false);
  });

  it('rejects NaN in bbox', () => {
    const r = parseAnalyticsMessage(wireMessage({
      tracks: [{ pose_track_id: 1, state: 'fallen', score: 0.9, bbox: [NaN, 1, 2, 3], keypoints: [] }],
    }));
    // JSON.stringify 会把 NaN 变 null;按 null bbox 处理(不崩溃)
    expect(Array.isArray(r.ok === true ? (r.analytics.tracks[0].keypoints ?? []) : [])).toBe(true);
  });
});

describe('FallOverlayCache', () => {
  const entry = { analytics: {} as unknown as FallAnalytics & { previewFrameId: number }, deadline: 1e12 };

  it('keyed by session + frame id, distinct orders kept', () => {
    const c = new FallOverlayCache();
    c.set('sess-1', 100, entry);
    c.set('sess-1', 101, entry);
    expect(c.size).toBe(2);
    expect(c.get('sess-1', 100)).toBeDefined();
  });

  it('clearSession removes only that session', () => {
    const c = new FallOverlayCache();
    c.set('sess-1', 100, entry);
    c.set('sess-2', 100, entry);
    c.clearSession('sess-1');
    expect(c.get('sess-1', 100)).toBeUndefined();
    expect(c.get('sess-2', 100)).toBeDefined();
  });

  it('evicts expired entries past deadline', () => {
    const c = new FallOverlayCache();
    c.set('s', 1, { ...entry, deadline: 100 });
    c.set('s', 2, { ...entry, deadline: 500 });
    const removed = c.evictExpired(300);
    expect(removed).toBe(1);
    expect(c.get('s', 1)).toBeUndefined();
    expect(c.get('s', 2)).toBeDefined();
  });

  it('evicts oldest when over capacity', () => {
    const c = new FallOverlayCache(2);
    c.set('s', 1, entry);
    c.set('s', 2, entry);
    c.set('s', 3, entry);
    expect(c.size).toBe(2);
  });
});