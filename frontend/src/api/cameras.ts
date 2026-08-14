import client from './client';

export interface CameraInfo {
  id: string;
  name: string;
  source: string;
  width: number;
  height: number;
  profile: string;
  enabled: boolean;
  config: Record<string, unknown>;
  status: string;
  metrics: {
    alive: boolean;
    frames: number;
    tracks: number;
    uptime_seconds: number;
    stage_ms: Record<string, number>;
  } | null;
  created_at: string | null;
  updated_at: string | null;
}

export const fetchCameras = () => client.get<CameraInfo[]>('/cameras');
export const createCamera = (data: Record<string, unknown>) => client.post('/cameras', data);
export const updateCamera = (id: string, data: Record<string, unknown>) => client.put(`/cameras/${id}`, data);
export const deleteCamera = (id: string) => client.delete(`/cameras/${id}`);
export const startCamera = (id: string) => client.post(`/cameras/${id}/start`);
export const stopCamera = (id: string) => client.post(`/cameras/${id}/stop`);

/** 抓拍当前帧,返回 JPEG blob(用于 <img> 或下载)。 */
export const snapshotCamera = (id: string) =>
  client.post(`/cameras/${id}/snapshot`, undefined, { responseType: 'blob' });

/** 运行时切换部署档位(会重启该摄像头 pipeline)。 */
export const switchCameraProfile = (id: string, profile: string) =>
  client.put(`/cameras/${id}/profile`, { profile });

/** 设置采集/推理分辨率与推流分辨率(0 = 原生/不缩放;运行中会自动重启流水线生效)。 */
export const updateCameraResolution = (
  id: string,
  data: { capture_width: number; capture_height: number; stream_max_height: number },
) => client.put(`/cameras/${id}/resolution`, data);
