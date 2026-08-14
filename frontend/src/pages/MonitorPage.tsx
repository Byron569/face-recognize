import { useEffect, useState } from 'react';
import { Card, Col, Empty, Row } from 'antd';
import { CameraOutlined } from '@ant-design/icons';
import { fetchCameras, CameraInfo } from '../api/cameras';
import AlertBanner from '../components/AlertBanner';
import CameraCell from '../components/CameraCell';

/** 实时监控页:启用的摄像头按网格排列。 */
export default function MonitorPage() {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);

  useEffect(() => {
    fetchCameras().then((res) => setCameras(res.data));
    const interval = setInterval(() => {
      fetchCameras().then((res) => setCameras(res.data));
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const enabledCameras = cameras.filter((c) => c.enabled);

  if (enabledCameras.length === 0) {
    return (
      <>
        <AlertBanner />
        <Card variant="borderless" style={{ borderRadius: 8, textAlign: 'center', padding: 60 }}>
          <Empty
            image={<CameraOutlined style={{ fontSize: 64, color: '#bbb' }} />}
            description={
              <span style={{ color: '#999', fontSize: 16 }}>
                暂无启用的摄像头<br />
                <span style={{ fontSize: 13 }}>请前往 <strong>系统设置</strong> 启用摄像头</span>
              </span>
            }
          />
        </Card>
      </>
    );
  }

  const cols =
    enabledCameras.length === 1 ? 24 : enabledCameras.length <= 4 ? 12 : 8;

  return (
    <>
      <AlertBanner />
      <Row gutter={[12, 12]}>
        {enabledCameras.map((cam) => (
          <Col key={cam.id} span={cols}>
            <CameraCell cameraId={cam.id} title={cam.name || cam.id} profile={cam.profile} />
          </Col>
        ))}
      </Row>
    </>
  );
}
