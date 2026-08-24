import { useCallback, useEffect, useRef, useState } from 'react';
import { Button, Card, Space, Tag, message } from 'antd';
import { CameraOutlined, WifiOutlined } from '@ant-design/icons';
import { useWebSocket } from '../hooks/useWebSocket';
import { snapshotCamera } from '../api/cameras';
import { detectionColors, videoConfig } from '../config';
import { DecodedFramePacket, parseFramePacket } from '../stream/frameProtocol';
import {
  FallAnalytics,
  FallOverlayCache,
  parseAnalyticsMessage,
} from '../stream/analyticsProtocol';
import { drawFallAnalytics } from '../stream/fallOverlay';

interface Detection {
  track_id: number;
  bbox: [number, number, number, number]; // [x, y, w, h]
  identity: string;
  confidence: number;
}

interface Props {
  cameraId: string;
  title: string;
  profile: string;
}

/** 单路摄像头卡片:WebSocket 视频流 + Canvas 检测框叠加 + 抓拍。 */
export default function CameraCell({ cameraId, title, profile }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const detectionsByFrameRef = useRef(new Map<number, Detection[]>());
  const pendingFrameRef = useRef<DecodedFramePacket | null>(null);
  const lastFrameRef = useRef<DecodedFramePacket | null>(null);
  const drawScheduledRef = useRef(false);
  const animationFrameRef = useRef<number | null>(null);
  const canvasSizeRef = useRef({ width: 0, height: 0 });
  const mountedRef = useRef(true);
  const scheduleDrawRef = useRef<() => void>(() => {});
  const overlayCacheRef = useRef<FallOverlayCache | null>(null);
  const activeOverlayRef = useRef<FallAnalytics | null>(null);
  const overlayTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const currentSessionRef = useRef<string | null>(null);
  const [hasVideo, setHasVideo] = useState(false);
  const fallOverlayCache: FallOverlayCache = (() => {
    if (!overlayCacheRef.current) overlayCacheRef.current = new FallOverlayCache(64);
    return overlayCacheRef.current;
  })();

  const clearOverlayTimer = useCallback(() => {
    if (overlayTimerRef.current !== null) {
      clearTimeout(overlayTimerRef.current);
      overlayTimerRef.current = null;
    }
    activeOverlayRef.current = null;
  }, []);

  const drawLatestFrame = useCallback(async () => {
    const frame = pendingFrameRef.current;
    pendingFrameRef.current = null;
    if (!frame) {
      drawScheduledRef.current = false;
      return;
    }

    let bitmap: ImageBitmap | null = null;
    try {
      bitmap = await createImageBitmap(frame.jpeg);
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (!canvas || !ctx) return;

      if (
        canvasSizeRef.current.width !== bitmap.width ||
        canvasSizeRef.current.height !== bitmap.height
      ) {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        canvasSizeRef.current = { width: bitmap.width, height: bitmap.height };
      }
      ctx.drawImage(bitmap, 0, 0);
      for (const detection of detectionsByFrameRef.current.get(frame.frameId) || []) {
        drawDetection(ctx, detection);
      }
      // M2:跌倒骨架叠加(仅在 overlay 与该 JPEG 帧匹配时绘制,按 preview 像素)
      const overlay = activeOverlayRef.current;
      if (overlay) {
        drawFallAnalytics(ctx, overlay, frame.frameId, bitmap.width, bitmap.height);
      }
      for (const id of detectionsByFrameRef.current.keys()) {
        if (id < frame.frameId - 30) detectionsByFrameRef.current.delete(id);
      }
      if (mountedRef.current) setHasVideo(true);
      lastFrameRef.current = frame;
    } catch {
      // Ignore malformed frames and continue with the newest pending packet.
    } finally {
      bitmap?.close();
      drawScheduledRef.current = false;
      animationFrameRef.current = null;
      if (mountedRef.current && pendingFrameRef.current) {
        scheduleDrawRef.current();
      }
    }
  }, []);

  const scheduleDraw = useCallback(() => {
    if (drawScheduledRef.current) return;
    drawScheduledRef.current = true;
    animationFrameRef.current = requestAnimationFrame(() => {
      animationFrameRef.current = null;
      void drawLatestFrame();
    });
  }, [drawLatestFrame]);
  scheduleDrawRef.current = scheduleDraw;

  const handleMessage = useCallback((data: any) => {
    if (data instanceof ArrayBuffer) {
      try {
        pendingFrameRef.current = parseFramePacket(data);
        scheduleDrawRef.current();
      } catch {
        // Ignore unsupported binary messages.
      }
      return;
    }
    if (data.type === 'detections') {
      const frameId = Number(data.frame_id);
      if (!Number.isFinite(frameId)) return;
      detectionsByFrameRef.current.set(frameId, data.persons || []);
      if (lastFrameRef.current?.frameId === frameId && !pendingFrameRef.current) {
        pendingFrameRef.current = lastFrameRef.current;
        scheduleDrawRef.current();
      }
      for (const id of detectionsByFrameRef.current.keys()) {
        if (id < frameId - 30) detectionsByFrameRef.current.delete(id);
      }
      return;
    }
    if (data.type === 'analytics') {
      const parsed = parseAnalyticsMessage(JSON.stringify(data));
      if (!parsed.ok) return; // 协议无效则丢弃该 overlay
      const a = parsed.analytics;
      if (currentSessionRef.current !== a.cameraSessionId) {
        currentSessionRef.current = a.cameraSessionId;
        fallOverlayCache.clearSession(a.cameraSessionId);
        clearOverlayTimer();
      }
      const expiresInMs = a.overlayExpiresInMs;
      if (expiresInMs !== null && expiresInMs <= 0) {
        clearOverlayTimer();
        return;
      }
      const deadline = performance.now() + (expiresInMs ?? 0);
      fallOverlayCache.set(a.cameraSessionId, a.previewFrameId, { analytics: a, deadline });
      activeOverlayRef.current = a;
      if (overlayTimerRef.current !== null) clearTimeout(overlayTimerRef.current);
      overlayTimerRef.current = setTimeout(() => {
        overlayTimerRef.current = null;
        activeOverlayRef.current = null;
        if (mountedRef.current && lastFrameRef.current) {
          pendingFrameRef.current = lastFrameRef.current;
          scheduleDrawRef.current();
        }
      }, expiresInMs ?? 0);
      if (lastFrameRef.current?.frameId === a.previewFrameId && !pendingFrameRef.current) {
        pendingFrameRef.current = lastFrameRef.current;
        scheduleDrawRef.current();
      }
      return;
    }
  }, [clearOverlayTimer, fallOverlayCache]);

  useWebSocket(`/ws/cameras/${cameraId}`, handleMessage, 2000, 'arraybuffer');

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      pendingFrameRef.current = null;
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      drawScheduledRef.current = false;
      clearOverlayTimer();
    };
  }, [clearOverlayTimer]);

  const handleSnapshot = async () => {
    try {
      const res = await snapshotCamera(cameraId);
      const url = URL.createObjectURL(res.data);
      const a = document.createElement('a');
      a.href = url;
      a.download = videoConfig.snapshotDownloadName(cameraId);
      a.click();
      URL.revokeObjectURL(url);
      message.success('抓拍成功');
    } catch {
      message.error('抓拍失败,请确认摄像头已启用');
    }
  };

  return (
    <Card
      variant="borderless"
      style={{ borderRadius: 8, overflow: 'hidden' }}
      styles={{ body: { padding: 0 } }}
      title={
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span>
            <WifiOutlined style={{ color: '#52c41a', marginRight: 8 }} />
            <strong>{title}</strong>
          </span>
          <Space>
            <Button size="small" icon={<CameraOutlined />} onClick={handleSnapshot}>抓拍</Button>
            <Tag color="processing">{profile}</Tag>
          </Space>
        </div>
      }
    >
      <div style={{ background: detectionColors.background, position: 'relative', aspectRatio: videoConfig.aspectRatio }}>
        <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
        {!hasVideo && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#555',
              pointerEvents: 'none',
            }}
          >
            <CameraOutlined style={{ fontSize: 40, opacity: 0.5 }} />
            <span style={{ marginTop: 8, fontSize: 13, opacity: 0.6 }}>等待视频流...</span>
          </div>
        )}
      </div>
    </Card>
  );
}

/** Canvas 检测框绘制(颜色来自集中配置)。 */
function drawDetection(ctx: CanvasRenderingContext2D, d: Detection) {
  const [x, y, w, h] = d.bbox;
  const identified = d.identity && d.identity !== 'Unknown';
  ctx.strokeStyle = identified ? detectionColors.identified : detectionColors.unknown;
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = identified ? detectionColors.identified : detectionColors.unknown;
  ctx.font = '14px sans-serif';
  const label = `${d.identity || 'Unknown'} ${Math.round((d.confidence || 0) * 100)}%`;
  const tw = ctx.measureText(label).width;
  ctx.fillRect(x, y - 20, tw + 8, 20);
  ctx.fillStyle = detectionColors.text;
  ctx.fillText(label, x + 4, y - 5);
}
