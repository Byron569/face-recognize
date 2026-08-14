import client from './client';

export interface SystemStatus {
  cpu_percent: number;
  memory_percent: number;
  gpu_name: string | null;
  gpu_utilization: number | null;
  gpu_memory_percent: number | null;
  camera_count: number;
  active_camera_count: number;
  engine_count: number;
  gallery_size: number;
}

export const fetchSystemStatus = () => client.get<SystemStatus>('/system/status');

/** 性能指标(per-camera 帧数/阶段延迟/跟踪数)。 */
export interface CameraMetrics {
  fps: number;
  frames: number;
  tracks: number;
  uptime_seconds: number;
  stage_ms: Record<string, number>;
}
export interface SystemMetrics {
  cameras: Record<string, CameraMetrics>;
  global: { camera_count: number; engine_count: number; gallery_size: number };
}
export const fetchSystemMetrics = () => client.get<SystemMetrics>('/system/metrics');

/** 部署档位列表(后端返回)。 */
export interface ProfileInfo {
  name: string;
  device: string;
  det_size: string;
  desc: string;
}
export const fetchProfiles = () => client.get<{ profiles: ProfileInfo[] }>('/system/profiles');

/** 任务清单(可插拔任务注册表)。 */
export interface TaskInfo {
  name: string;
  enabled: boolean;
  class_path: string | null;
  loaded: boolean;
}
export const fetchTasks = () => client.get<{ items: TaskInfo[] }>('/tasks');
