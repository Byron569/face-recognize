import { describe, it, expect, vi } from 'vitest';
import {
  isOverlayCompatible,
  drawFallAnalytics,
  isStale,
} from './fallOverlay';
import { FallAnalytics } from './analyticsProtocol';

function makeAnalytics(overrides: Partial<FallAnalytics> = {}): FallAnalytics {
  return {
    schemaVersion: 1,
    cameraSessionId: 'sess-1',
    previewFrameId: 100,
    sourceFrameId: 98,
    previewWidth: 853,
    previewHeight: 480,
    transform: { scaleX: 853 / 1920, scaleY: 480 / 1080 },
    health: 'READY',
    resultAgeMs: 82.4,
    overlayExpiresInMs: 1118,
    workerEndToEndMs: 61.7,
    tracks: [],
    color: null,
    ...overrides,
  };
}

describe('isOverlayCompatible', () => {
  it('accepts when frame id and bitmap size match preview', () => {
    const r = isOverlayCompatible(makeAnalytics(), 100, 853, 480);
    expect(r.ok).toBe(true);
  });

  it('rejects when preview_frame_id mismatches decoded JPEG frame', () => {
    const r = isOverlayCompatible(makeAnalytics(), 101, 853, 480);
    expect(r.ok).toBe(false);
  });

  it('rejects when bitmap dimensions differ from preview', () => {
    const r = isOverlayCompatible(makeAnalytics(), 100, 1920, 1080);
    expect(r.ok).toBe(false);
  });
});

describe('drawFallAnalytics', () => {
  function mockCtx() {
    return {
      strokeStyle: '',
      lineWidth: 0,
      fillStyle: '',
      font: '',
      strokeRect: vi.fn(),
      fillRect: vi.fn(),
      fillText: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      measureText: vi.fn(() => ({ width: 10 })),
    } as unknown as CanvasRenderingContext2D;
  }

  it('does not draw when incompatible (stale frame bound to wrong jpeg)', () => {
    const ctx = mockCtx();
    const check = drawFallAnalytics(ctx, makeAnalytics(), 101, 853, 480);
    expect(check.ok).toBe(false);
    expect(ctx.strokeRect).not.toHaveBeenCalled();
  });

  it('draws skeletons for valid fallen analytics (no second scaling)', () => {
    const ctx = mockCtx();
    const analytics = makeAnalytics({
      color: 'red',
      tracks: [{
        poseTrackId: 3, state: 'fallen', score: 0.9,
        bbox: [44, 88, 88, 220], keypoints: [[44, 88]],
      }],
    });
    const check = drawFallAnalytics(ctx, analytics, 100, 853, 480);
    expect(check.ok).toBe(true);
    // preview 像素原样绘制,strokeRect(44,88,88,220) 不被外部缩放
    expect(ctx.strokeRect).toHaveBeenCalledWith(44, 88, 88, 220);
    expect(ctx.arc).toHaveBeenCalled();
  });

  it('does nothing without analytics', () => {
    const ctx = mockCtx();
    const check = drawFallAnalytics(ctx, undefined, 100, 853, 480);
    expect(check.ok).toBe(false);
    expect(ctx.strokeRect).not.toHaveBeenCalled();
  });
});

describe('isStale', () => {
  it('true when overlay_expires_in_ms <= 0', () => {
    expect(isStale(makeAnalytics({ overlayExpiresInMs: 0 }))).toBe(true);
    expect(isStale(makeAnalytics({ overlayExpiresInMs: 120 }))).toBe(false);
    expect(isStale(undefined)).toBe(false);
  });
});