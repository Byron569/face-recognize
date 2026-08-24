import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act, cleanup } from '@testing-library/react';
import { EventDedupe } from '../stream/eventDedupe';
import AlertBanner from './AlertBanner';

/** 最小 WebSocket 假实现:捕获实例、允许测试驱动 onmessage/onclose。 */
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: ((ev: unknown) => void) | null = null;
  readyState = 1;
  url: string;
  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }
  close(): void {
    this.onclose?.({});
  }
  emit(payload: unknown): void {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }
}

const lastSocket = () => FakeWebSocket.instances[FakeWebSocket.instances.length - 1];

function fallFrame(overrides: Record<string, unknown>) {
  return {
    type: 'event',
    event_type: 'fall_detected',
    camera_id: 'cam-1',
    track_id: 7,
    confidence: 0.95,
    id: 900,
    created_at: '2026-08-23T12:34:57.001Z',
    ...overrides,
  };
}

function freshDedupe(onProtocolError?: (info: { event_type: string | null; reason: string }) => void) {
  return { dedupe: new EventDedupe({ storage: null, onProtocolError }) };
}

describe('AlertBanner event 去重（M1）', () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    vi.stubGlobal('WebSocket', FakeWebSocket as unknown as typeof WebSocket);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
    cleanup();
  });

  it('同一 fall WS 帧(同 event_id)只新增一条告警,不递增合并 count', () => {
    render(<AlertBanner {...freshDedupe()} />);
    const ws = lastSocket();

    act(() => ws.emit(fallFrame({ event_id: 'E1', dedupe_key: 'k1' })));
    act(() => ws.emit(fallFrame({ event_id: 'E1', dedupe_key: 'k1' })));

    expect(screen.getByText('摔倒检测 — cam-1')).toBeInTheDocument();
    expect(screen.queryByText(/×2/)).toBeNull();
  });

  it('断线重连重放同 event_id 不新增告警(去重状态在重连间保留)', () => {
    vi.useFakeTimers();
    render(<AlertBanner {...freshDedupe()} />);
    const ws0 = lastSocket();

    act(() => ws0.emit(fallFrame({ event_id: 'E1', dedupe_key: 'k1' })));
    expect(screen.getByText('摔倒检测 — cam-1')).toBeInTheDocument();

    // 连接断开 -> 调度 3s 后重连
    act(() => ws0.close());
    act(() => vi.advanceTimersByTime(3000));
    const ws1 = lastSocket();
    expect(ws1).not.toBe(ws0);

    // outbox at-least-once 重送同一条 fall transition
    act(() => ws1.emit(fallFrame({ event_id: 'E1', dedupe_key: 'k1' })));

    expect(screen.queryByText(/×2/)).toBeNull();
    expect(screen.getAllByText('摔倒检测 — cam-1').length).toBe(1);
  });

  it('不同 event_id 的同摄像头同类型事件才按当前 UI 规则合并/计数', () => {
    render(<AlertBanner {...freshDedupe()} />);
    const ws = lastSocket();

    act(() => ws.emit(fallFrame({ event_id: 'E1', dedupe_key: 'k1' })));
    act(() => ws.emit(fallFrame({ event_id: 'E2', dedupe_key: 'k2' })));

    expect(screen.getByText('摔倒检测 — cam-1 ×2')).toBeInTheDocument();
  });

  it('缺少 event_id/dedupe_key 的可靠 fall 事件被拒绝告警并上报协议错误', () => {
    const onProtocolError = vi.fn();
    render(<AlertBanner {...freshDedupe(onProtocolError)} />);
    const ws = lastSocket();

    act(() => ws.emit(fallFrame({ id: 999 })));

    expect(screen.queryByText(/摔倒检测/)).toBeNull();
    expect(onProtocolError).toHaveBeenCalledWith({
      event_type: 'fall_detected',
      reason: expect.stringContaining('event_id'),
    });
  });

  it('旧 recognition 事件保持兼容,正常展示告警', () => {
    render(<AlertBanner {...freshDedupe()} />);
    const ws = lastSocket();

    act(() =>
      ws.emit({
        type: 'event',
        event_type: 'recognition',
        camera_id: 'cam-0',
        id: 5,
        payload: { name: '张三' },
        created_at: '2026-08-23T12:34:57.001Z',
      }),
    );

    expect(screen.getByText('人脸识别 — cam-0')).toBeInTheDocument();
  });
});