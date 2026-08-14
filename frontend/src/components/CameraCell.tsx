import { useCallback, useRef, useState } from 'react';
import { Button, Card, Space, Tag, message } from 'antd';
import { CameraOutlined, WifiOutlined } from '@ant-design/icons';
import { useWebSocket } from '../hooks/useWebSocket';
import { snapshotCamera } from '../api/cameras';
import { detectionColors, videoConfig } from '../config';

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
  const imgRef = useRef<HTMLImageElement>(new Image());
  const detectionsRef = useRef<Detection[]>([]);
  const [hasVideo, setHasVideo] = useState(false);

  const handleMessage = useCallback((data: any) => {
    if (data.type === 'detections') {
      detectionsRef.current = data.persons || [];
    }
    if (data.type === 'frame') {
      setHasVideo(true);
      const img = imgRef.current;
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        for (const d of detectionsRef.current) {
          drawDetection(ctx, d);
        }
      };
      img.src = `data:image/jpeg;base64,${data.data}`;
    }
  }, []);

  useWebSocket(`/ws/cameras/${cameraId}`, handleMessage);

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
