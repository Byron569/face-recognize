import { useState, useEffect, useRef } from 'react';
import { Alert } from 'antd';
import { eventMeta } from '../config';

interface AlertItem {
  id: string | number;
  event_type: string;
  camera_id: string;
  identity_name: string | null;
  created_at: string;
}

// 全局事件总线 — 由 /ws/events 实时推送
const alertBus = {
  _listeners: new Set<(a: AlertItem) => void>(),
  push(alert: AlertItem) {
    this._listeners.forEach((fn) => fn(alert));
  },
  subscribe(fn: (a: AlertItem) => void) {
    this._listeners.add(fn);
    return () => { this._listeners.delete(fn); };
  },
};

export { alertBus };

/** 顶部实时告警横幅(/ws/events 推送驱动)。 */
export default function AlertBanner() {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const idCounter = useRef(0);

  useEffect(() => {
    return alertBus.subscribe((alert) => {
      setAlerts((prev) => [alert, ...prev].slice(0, 5));
    });
  }, []);

  useEffect(() => {
    const connect = () => {
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
        reconnectTimer.current = setTimeout(connect, 3000);
      };
    };

    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  if (alerts.length === 0) return null;

  return (
    <div style={{ position: 'fixed', top: 48, left: 220, right: 0, zIndex: 1000 }}>
      {alerts.map((a) => {
        const meta = eventMeta[a.event_type];
        const title = meta ? meta.label : a.event_type;
        return (
          <Alert
            key={a.id}
            message={`${title} — ${a.camera_id}`}
            description={a.identity_name || 'Unknown person'}
            type="error"
            banner
            closable
            onClose={() => setAlerts((prev) => prev.filter((x) => x.id !== a.id))}
          />
        );
      })}
    </div>
  );
}
