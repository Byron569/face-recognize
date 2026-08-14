/**
 * 前端集中配置 — 主题 / 菜单 / 档位元数据。
 *
 * 修改配色、菜单、档位描述等只需改本目录,不触碰页面代码。
 */

import type { ThemeConfig } from 'antd';

/** Ant Design 主题 token(全局唯一配色入口)。 */
export const themeConfig: ThemeConfig = {
  token: {
    colorPrimary: '#1677ff',
    borderRadius: 8,
    fontSize: 14,
  },
  components: {
    Layout: {
      siderBg: '#001529',
      headerBg: '#ffffff',
      bodyBg: '#f0f2f5',
    },
    Menu: {
      darkItemBg: '#001529',
      darkItemSelectedBg: '#1677ff',
    },
    Card: {
      borderRadiusLG: 8,
    },
  },
};

/** 监控页检测框配色(Canvas 绘制)。 */
export const detectionColors = {
  identified: '#52c41a',
  unknown: '#faad14',
  text: '#ffffff',
  background: '#1a1a2e',
} as const;

/** 视频区域参数。 */
export const videoConfig = {
  aspectRatio: '4 / 3',
  snapshotDownloadName: (cameraId: string) => `snapshot_${cameraId}_${Date.now()}.jpg`,
} as const;

/** 侧边栏菜单(增删页面只需改这里,AppLayout 自动渲染)。 */
export interface MenuEntry {
  key: string;
  icon: string;
  label: string;
}

export const menuConfig: MenuEntry[] = [
  { key: '/monitor', icon: 'video', label: '实时监控' },
  { key: '/faces', icon: 'user', label: '人脸库' },
  { key: '/events', icon: 'file', label: '事件记录' },
  { key: '/settings', icon: 'setting', label: '系统设置' },
];

/** 部署档位元数据(优先展示后端 /api/system/profiles 返回,此处为兜底)。 */
export interface ProfileMeta {
  color: string;
  device: string;
  desc: string;
}

export const profileMeta: Record<string, ProfileMeta> = {
  desktop: { color: '#1677ff', device: 'CUDA GPU', desc: '640px / 高性能工作站' },
  balanced: { color: '#fa8c16', device: 'CUDA GPU', desc: '480px / 均衡模式' },
  edge_minimal: { color: '#52c41a', device: 'CPU', desc: '320px / 边缘设备' },
};

export const profileOptions = Object.keys(profileMeta).map((value) => ({
  value,
  label: value === 'desktop' ? 'Desktop' : value === 'balanced' ? 'Balanced' : 'Edge Minimal',
}));

/** 事件类型展示元数据(颜色/中文名)。 */
export interface EventMeta {
  color: string;
  label: string;
  icon: string;
}

export const eventMeta: Record<string, EventMeta> = {
  recognition: { color: '#1677ff', label: '人脸识别', icon: 'user' },
  fall_detected: { color: '#ff4d4f', label: '摔倒检测', icon: 'warning' },
  fall_potential: { color: '#fa8c16', label: '疑似摔倒', icon: 'alert' },
  fall_recovered: { color: '#52c41a', label: '摔倒恢复', icon: 'check' },
  intrusion: { color: '#eb2f96', label: '闯入告警', icon: 'alert' },
  loitering: { color: '#722ed1', label: '徘徊告警', icon: 'clock' },
};
