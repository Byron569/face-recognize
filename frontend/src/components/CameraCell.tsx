import { useCallback, useEffect, useRef, useState } from 'react';
import { Button, Card, Space, Tag, message } from 'antd';
import { CameraOutlined, WifiOutlined } from '@ant-design/icons';
import { useWebSocket } from '../hooks/useWebSocket';
import { snapshotCamera } from '../api/cameras';
import { detectionColors, videoConfig } from '../config';
import { DecodedFramePacket, parseFramePacket } from '../stream/frameProtocol';

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
  const [hasVideo, setHasVideo] = useState(false);

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
    }
  }, []);

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
    };
  }, []);

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
