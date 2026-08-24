import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { EventItem, fetchEvents } from '../api/events';
import EventLogPage from './EventLogPage';

vi.mock('../api/events', () => ({
  fetchEvents: vi.fn(),
  acknowledgeEvent: vi.fn(),
  deleteEvents: vi.fn(),
  deleteAllEvents: vi.fn(),
  fetchEventTypes: vi.fn(),
  fetchRecognitionLogs: vi.fn(),
}));

function renderPage(): void {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <EventLogPage />
    </QueryClientProvider>,
  );
}

function fallEvent(overrides: Partial<EventItem>): EventItem {
  return {
    id: 1,
    event_type: 'fall_detected',
    camera_id: 'cam-1',
    track_id: 7,
    identity_id: null,
    identity_name: null,
    confidence: 0.95,
    payload: { score_semantics: 'heuristic_rule_score_not_probability' },
    snapshot_path: null,
    acknowledged: false,
    acknowledged_at: null,
    created_at: '2026-08-23T12:34:57.001Z',
    event_id: '9b4d...',
    incident_id: 'inc-1',
    dedupe_key: 'dedupe-1',
    occurred_at: '2026-08-23T12:34:57.000Z',
    delivery_mode: 'alert',
    ...overrides,
  };
}

describe('EventLogPage fall 分数渲染（M1）', () => {
  beforeEach(() => {
    vi.mocked(fetchEvents).mockReset();
  });
  afterEach(() => {
    cleanup();
  });

  it('fall_ 事件渲染「规则分数<原值>（非概率）」而非百分比', async () => {
    vi.mocked(fetchEvents).mockResolvedValue({
      data: { items: [fallEvent({})], total: 1 },
    } as never);
    renderPage();

    expect((await screen.findAllByText(/规则分数/)).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/0\.95/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/非概率/).length).toBeGreaterThan(0);
    // 绝不渲染成百分比
    expect(screen.queryByText(/0\.95%/)).toBeNull();
    expect(screen.queryByText('95%')).toBeNull();
  });

  it('recognition 行保持现有百分比置信度语义', async () => {
    vi.mocked(fetchEvents).mockResolvedValue({
      data: {
        items: [
          {
            id: 2,
            event_type: 'recognition',
            camera_id: 'cam-0',
            track_id: 3,
            identity_id: null,
            identity_name: null,
            confidence: 0.87,
            payload: {},
            snapshot_path: null,
            acknowledged: false,
            acknowledged_at: null,
            created_at: '2026-08-23T12:34:57.001Z',
          } as EventItem,
        ],
        total: 1,
      },
    } as never);
    renderPage();

    expect(await screen.findByText('87%')).toBeInTheDocument();
  });

  it('缺少/不匹配 score_semantics 时显示协议异常,不猜成概率', async () => {
    vi.mocked(fetchEvents).mockResolvedValue({
      data: { items: [fallEvent({ payload: {} })], total: 1 },
    } as never);
    renderPage();

    expect((await screen.findAllByText('协议异常')).length).toBeGreaterThan(0);
    expect(screen.queryByText(/非概率/)).toBeNull();
  });

  it('API 类型向后兼容:旧 recognition 响应不含幂等字段仍满足 EventItem', () => {
    // 编译期由 npm run build(tsc) 把关;此处补充运行时可选性断言
    const legacy: EventItem = {
      id: 9,
      event_type: 'recognition',
      camera_id: 'cam-0',
      track_id: 1,
      identity_id: null,
      identity_name: null,
      confidence: 0.8,
      payload: {},
      snapshot_path: null,
      acknowledged: false,
      acknowledged_at: null,
      created_at: '2026-08-23T09:00:00.000Z',
    };
    expect(legacy.event_id).toBeUndefined();
    expect(legacy.incident_id).toBeUndefined();
    expect(legacy.dedupe_key).toBeUndefined();
    expect(legacy.occurred_at).toBeUndefined();
    expect(legacy.delivery_mode).toBeUndefined();
  });
});