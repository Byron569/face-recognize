import { useState, useEffect, useRef } from 'react';
import { Alert } from 'antd';
import { eventMeta } from '../config';
import { createReconnectGuard } from '../hooks/reconnectGuard';

interface AlertItem {
  id: string | number;
  event_type: string;
  camera_id: string;
  identity_name: string | null;
  created_at: string;
  count: number;
}

// 全局事件总线 — 由 /ws/events 实时推送
const alertBus = {
  _listeners: new Set<(a: Omit<AlertItem, 'count'>) => void>(),
  push(alert: Omit<AlertItem, 'count'>) {
    this._listeners.forEach((fn) => fn(alert));
  },
  subscribe(fn: (a: Omit<AlertItem, 'count'>) => void) {
    this._listeners.add(fn);
    return () => { this._listeners.delete(fn); };
  },
};

export { alertBus };

// 告警类事件用醒目配色;常规事件(识别等)用低调 info 样式
const alertTypeOf: Record<string, 'error' | 'warning' | 'info'> = {
  fall_detected: 'error',
  fall_potential: 'warning',
  intrusion: 'error',
  loitering: 'warning',
};

const MAX_ALERTS = 3;

/**
 * 事件提示条(页面内嵌、不悬浮遮挡画面):
 * - 普通文档流布局,位于页面内容顶部,不覆盖视频
 * - 同一摄像头 + 同类型事件合并计数,避免刷屏
 */
export default function AlertBanner() {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const reconnectGuard = useRef(createReconnectGuard());
  const idCounter = useRef(0);

  useEffect(() => {
    return alertBus.subscribe((alert) => {
      setAlerts((prev) => {
        const idx = prev.findIndex(
          (x) => x.event_type === alert.event_type && x.camera_id === alert.camera_id
        );
        if (idx >= 0) {
          // 合并同类事件:计数 +1,更新触发时间
          const next = [...prev];
          next[idx] = { ...next[idx], count: next[idx].count + 1, created_at: alert.created_at };
          return next;
        }
        return [{ ...alert, count: 1 }, ...prev].slice(0, MAX_ALERTS);
      });
    });
  }, []);

  useEffect(() => {
    const connect = () => {
      if (!reconnectGuard.current.isActive()) return;
      const generation = reconnectGuard.current.start();
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${protocol}//${location.host}/ws/events`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'ping') return;
          if (data.type === 'event' || data.event_type) {
            // 后端格式: {"type":"event","event_type":"recognition","camera_id":"cam0","payload":{"name":"..."},...}
            alertBus.push({
              id: data.id ?? ++idCounter.current,
              event_type: data.event_type ?? 'unknown',
              camera_id: data.camera_id ?? 'unknown',
              identity_name: data.payload?.name ?? data.identity_name ?? null,
              created_at: data.created_at ?? new Date().toISOString(),
            });
          }
        } catch { /* ignore malformed */ }
      };

      ws.onclose = () => {
        if (!reconnectGuard.current.canReconnect(generation)) return;
        if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
        reconnectTimer.current = setTimeout(() => {
          if (reconnectGuard.current.canReconnect(generation)) connect();
        }, 3000);
      };
    };

    reconnectGuard.current.start();
    connect();
    return () => {
      reconnectGuard.current.stop();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  if (alerts.length === 0) return null;

  return (
    <div style={{ marginBottom: 12, position: 'relative', zIndex: 10 }}>
      {alerts.map((a) => {
        const meta = eventMeta[a.event_type];
        const title = meta ? meta.label : a.event_type;
        return (
          <Alert
            key={a.id}
            message={`${title} — ${a.camera_id}${a.count > 1 ? ` ×${a.count}` : ''}`}
            description={a.identity_name || 'Unknown person'}
            type={alertTypeOf[a.event_type] ?? 'info'}
            banner
            closable
            onClose={() => setAlerts((prev) => prev.filter((x) => x.id !== a.id))}
          />
        );
      })}
    </div>
  );
}
