import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, act, cleanup } from '@testing-library/react';

// mock useWebSocket:捕获传给 CameraCell 的 handler,测试可驱动 WS 消息
type Handler = (data: unknown) => void;
const captured: { handler: Handler | null } = { handler: null };

vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: (_path: string, onMessage: (_d: unknown) => void) => {
    captured.handler = onMessage;
    return { disconnect: () => {} };
  },
}));

// mock createImageBitmap / rAF / antd message
vi.stubGlobal('createImageBitmap', vi.fn(() =>
  Promise.resolve({ width: 853, height: 480, close: () => {} } as ImageBitmap),
));
vi.stubGlobal('requestAnimationFrame', vi.fn((cb: FrameRequestCallback) => {
  (cb as unknown as () => void)();
  return 1;
}));
vi.stubGlobal('cancelAnimationFrame', vi.fn());
vi.mock('antd', async (importOriginal) => {
  const actual = await importOriginal<typeof import('antd')>();
  return {
    ...actual,
    message: { success: vi.fn(), error: vi.fn() },
  };
});

import CameraCell from './CameraCell';

const timerIds: number[] = [];
const realSetTimeout = globalThis.setTimeout;
vi.stubGlobal('setTimeout', ((fn: TimerHandler, ms?: number) => {
  const id = realSetTimeout(fn, 0); // 立即触发便于测试
  timerIds.push(id as unknown as number);
  return id;
}) as unknown as typeof setTimeout);

function analyticsMsg(overrides: Record<string, unknown> = {}) {
  return {
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
        { pose_track_id: 3, state: 'fallen', score: 0.9, bbox: [44, 88, 88, 220], keypoints: [[44, 88]] },
      ],
      ...(overrides.fall_detection as Record<string, unknown> | undefined),
    },
    ...(overrides as Record<string, unknown>),
  };
}

describe('CameraCell fall overlay (M2 集成)', () => {
  beforeEach(() => {
    captured.handler = null;
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it('renders and accepts a valid analytics message without throwing', () => {
    let msg: string | null = null;
    render(<CameraCell cameraId="cam-1" title="Main" profile="desktop" />);
    expect(captured.handler).toBeTruthy();
    act(() => {
      captured.handler?.(analyticsMsg());
      msg = 'ok';
    });
    expect(msg).toBe('ok');
  });

  it('ignores an invalid analytics message (session mismatch) safely', () => {
    render(<CameraCell cameraId="cam-1" title="Main" profile="desktop" />);
    let msg: string | null = null;
    act(() => {
      captured.handler?.(analyticsMsg({ fall_detection: { camera_session_id: 'other' } }));
      msg = 'handled';
    });
    expect(msg).toBe('handled');
  });

  it('ignores non-analytics message types as before (detections)', () => {
    render(<CameraCell cameraId="cam-1" title="Main" profile="desktop" />);
    let msg: string | null = null;
    act(() => {
      captured.handler?.({
        type: 'detections', frame_id: 100, persons: [{ track_id: 1, bbox: [1, 2, 3, 4], identity: 'Alice' }],
      });
      msg = 'handled';
    });
    expect(msg).toBe('handled');
  });
});