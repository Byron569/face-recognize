import { useEffect, useRef, useCallback } from 'react';

type MessageHandler = (data: any) => void;

/**
 * WebSocket 通用 hook:自动连接 / 断线重连(指数退避)/ 心跳过滤。
 */
export function useWebSocket(
  path: string | null,
  onMessage: MessageHandler,
  reconnectBaseMs: number = 2000,
  binaryType: BinaryType = 'blob',
) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const attemptsRef = useRef(0);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    if (!path) return;
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${location.host}${path}`);
    ws.binaryType = binaryType;
    wsRef.current = ws;

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        attemptsRef.current = 0;
        onMessageRef.current(event.data);
        return;
      }
      if (event.data instanceof Blob) {
        event.data.arrayBuffer().then((buffer) => {
          attemptsRef.current = 0;
          onMessageRef.current(buffer);
        }).catch(() => { /* ignore unreadable binary */ });
        return;
      }
      if (typeof event.data !== 'string') return;
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'ping') return;
        attemptsRef.current = 0;
        onMessageRef.current(data);
      } catch { /* ignore malformed */ }
    };

    ws.onclose = () => {
      const delay = reconnectBaseMs * Math.min(2 ** attemptsRef.current, 8);
      attemptsRef.current += 1;
      reconnectTimer.current = setTimeout(connect, delay);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [path, reconnectBaseMs, binaryType]);

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return { disconnect };
}
