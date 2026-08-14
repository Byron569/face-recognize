import { useState } from 'react';
import { Card, Row, Col, Switch, Select, message, Tag, Space, Divider, Progress, Button, Modal, Form, Input, Popconfirm, Radio, InputNumber } from 'antd';
import {
  ThunderboltOutlined,
  DashboardOutlined,
  DesktopOutlined,
  ApiOutlined,
  PlayCircleOutlined,
  CameraOutlined,
  PlusOutlined,
  DeleteOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchSystemStatus, fetchSystemMetrics, CameraMetrics } from '../api/system';
import {
  fetchCameras,
  startCamera,
  stopCamera,
  switchCameraProfile,
  createCamera,
  deleteCamera,
  updateCameraResolution,
  CameraInfo,
} from '../api/cameras';
import { profileMeta, profileOptions } from '../config';

const STREAM_HEIGHT_OPTIONS = [
  { value: 0, label: '不缩放(原样推流)' },
  { value: 360, label: '360p' },
  { value: 480, label: '480p' },
  { value: 720, label: '720p' },
  { value: 1080, label: '1080p' },
];

/** 从摄像头 config 读取实际生效的采集分辨率(0 = 原生)。 */
function effectiveCaptureSize(cam: { width: number; height: number; config?: Record<string, unknown> }) {
  const defaults = (cam.config?.camera_defaults ?? {}) as Record<string, number | undefined>;
  const w = defaults.width ?? cam.width;
  const h = defaults.height ?? cam.height;
  return w > 0 && h > 0 ? `${w} × ${h}` : '跟随源(原生)';
}

/** 从摄像头 config 读取推流最大高度(0 = 不缩放)。 */
function effectiveStreamHeight(cam: { config?: Record<string, unknown> }) {
  const stream = (cam.config?.stream ?? {}) as Record<string, number | undefined>;
  const h = stream.max_height ?? 0;
  return h > 0 ? `≤ ${h}p` : '不缩放';
}

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const [createOpen, setCreateOpen] = useState(false);
  const [form] = Form.useForm();

  const { data: status } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: () => fetchSystemStatus().then((r) => r.data),
    refetchInterval: 5000,
  });

  const { data: cameras } = useQuery({
    queryKey: ['cameras'],
    queryFn: () => fetchCameras().then((r) => r.data),
  });

  const toggleMutation = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      enabled ? startCamera(id) : stopCamera(id),
    onSuccess: (_, vars) => {
      message.success(`摄像头 ${vars.id} ${vars.enabled ? '已启用' : '已停用'}`);
      queryClient.invalidateQueries({ queryKey: ['cameras'] });
    },
    onError: (err: any) => message.error(err?.response?.data?.detail || '操作失败'),
  });

  const profileMutation = useMutation({
    mutationFn: ({ id, profile }: { id: string; profile: string }) => switchCameraProfile(id, profile),
    onSuccess: () => {
      message.success('推理配置已更新,摄像头 pipeline 正在重启');
      queryClient.invalidateQueries({ queryKey: ['cameras'] });
    },
    onError: (err: any) => message.error(err?.response?.data?.detail || '更新失败'),
  });

  const createMutation = useMutation({
    mutationFn: (data: Record<string, unknown>) => createCamera(data),
    onSuccess: () => {
      message.success('摄像头已添加');
      setCreateOpen(false);
      form.resetFields();
      queryClient.invalidateQueries({ queryKey: ['cameras'] });
    },
    onError: (err: any) => message.error(err?.response?.data?.detail || '添加失败'),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteCamera(id),
    onSuccess: () => {
      message.success('摄像头已删除');
      queryClient.invalidateQueries({ queryKey: ['cameras'] });
    },
    onError: (err: any) => message.error(err?.response?.data?.detail || '删除失败'),
  });

  // ── 分辨率设置 ──
  const [resTarget, setResTarget] = useState<CameraInfo | null>(null);
  const [resMode, setResMode] = useState<'native' | 'custom'>('native');
  const [resWidth, setResWidth] = useState(640);
  const [resHeight, setResHeight] = useState(480);
  const [resStream, setResStream] = useState(0);

  const openResModal = (cam: CameraInfo) => {
    const defaults = (cam.config?.camera_defaults ?? {}) as Record<string, number | undefined>;
    const w = defaults.width ?? cam.width;
    const h = defaults.height ?? cam.height;
    const stream = (cam.config?.stream ?? {}) as Record<string, number | undefined>;
    setResTarget(cam);
    setResMode(w > 0 && h > 0 ? 'custom' : 'native');
    setResWidth(w > 0 ? w : 640);
    setResHeight(h > 0 ? h : 480);
    setResStream(stream.max_height ?? 0);
  };

  const resMutation = useMutation({
    mutationFn: (vars: {
      id: string;
      capture_width: number;
      capture_height: number;
      stream_max_height: number;
    }) => updateCameraResolution(vars.id, vars),
    onSuccess: () => {
      message.success(resTarget?.enabled ? '分辨率已应用,流水线已自动重启' : '分辨率已保存');
      setResTarget(null);
      queryClient.invalidateQueries({ queryKey: ['cameras'] });
    },
    onError: (err: any) => message.error(err?.response?.data?.detail || '更新失败'),
  });

  const { data: metrics } = useQuery({
    queryKey: ['systemMetrics'],
    queryFn: () => fetchSystemMetrics().then((r) => r.data),
    refetchInterval: 3000,
  });

  const activeCount = cameras?.filter((c) => c.enabled).length ?? 0;
  const totalCount = cameras?.length ?? 0;

  return (
    <div>
      {/* ── 系统资源卡片 ── */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card variant="borderless" style={{ borderRadius: 8, height: '100%' }}>
            <Space align="start">
              <ThunderboltOutlined style={{ fontSize: 28, color: '#1677ff', marginTop: 4 }} />
              <div>
                <div style={{ color: '#888', fontSize: 13 }}>CPU 使用率</div>
                <div style={{ fontSize: 28, fontWeight: 700 }}>{status?.cpu_percent ?? 0}%</div>
                <Progress percent={status?.cpu_percent ?? 0} showInfo={false} size="small" strokeColor="#1677ff" />
              </div>
            </Space>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card variant="borderless" style={{ borderRadius: 8, height: '100%' }}>
            <Space align="start">
              <DashboardOutlined style={{ fontSize: 28, color: '#52c41a', marginTop: 4 }} />
              <div>
                <div style={{ color: '#888', fontSize: 13 }}>内存使用率</div>
                <div style={{ fontSize: 28, fontWeight: 700 }}>{status?.memory_percent ?? 0}%</div>
                <Progress percent={status?.memory_percent ?? 0} showInfo={false} size="small" strokeColor="#52c41a" />
              </div>
            </Space>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card variant="borderless" style={{ borderRadius: 8, height: '100%' }}>
            <Space align="start">
              <DesktopOutlined style={{ fontSize: 28, color: '#722ed1', marginTop: 4 }} />
              <div>
                <div style={{ color: '#888', fontSize: 13 }}>
                  GPU {status?.gpu_name ? '· ' + status.gpu_name : '(未检测到)'}
                </div>
                <div style={{ fontSize: 28, fontWeight: 700 }}>
                  {status?.gpu_utilization != null ? `${status.gpu_utilization}%` : 'N/A'}
                </div>
                {status?.gpu_memory_percent != null ? (
                  <Progress percent={Math.round(status.gpu_memory_percent)} showInfo={false} size="small" strokeColor="#722ed1" />
                ) : (
                  <div style={{ height: 8 }} />
                )}
              </div>
            </Space>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card variant="borderless" style={{ borderRadius: 8, height: '100%' }}>
            <Space align="start">
              <ApiOutlined style={{ fontSize: 28, color: '#fa8c16', marginTop: 4 }} />
              <div>
                <div style={{ color: '#888', fontSize: 13 }}>活跃摄像头</div>
                <div style={{ fontSize: 28, fontWeight: 700 }}>
                  {activeCount} <span style={{ fontSize: 16, color: '#aaa' }}>/ {totalCount}</span>
                </div>
                <div style={{ height: 8 }} />
                <Tag color={activeCount > 0 ? 'green' : 'default'}>{activeCount > 0 ? '运行中' : '无活跃'}</Tag>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* ── 摄像头配置卡片 ── */}
      <div style={{ marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
        <CameraOutlined style={{ fontSize: 20, color: '#1677ff' }} />
        <span style={{ fontSize: 18, fontWeight: 600 }}>摄像头配置</span>
        <Tag color="blue">{totalCount} 台设备</Tag>
        <Button size="small" icon={<PlusOutlined />} onClick={() => setCreateOpen(true)}>
          添加摄像头
        </Button>
      </div>

      <Row gutter={[16, 16]}>
        {(cameras || []).map((cam) => {
          const meta = profileMeta[cam.profile] || profileMeta.desktop;
          return (
            <Col key={cam.id} xs={24} sm={12} lg={8}>
              <Card
                variant="borderless"
                style={{ borderRadius: 8, height: '100%' }}
                title={
                  <Space>
                    <Switch
                      checked={cam.enabled}
                      loading={toggleMutation.isPending}
                      onChange={(checked) => toggleMutation.mutate({ id: cam.id, enabled: checked })}
                      size="small"
                    />
                    <span style={{ fontWeight: 600 }}>{cam.name || cam.id}</span>
                    <Tag color={cam.enabled ? 'success' : 'default'} style={{ fontSize: 11 }}>
                      {cam.enabled ? 'ON' : 'OFF'}
                    </Tag>
                  </Space>
                }
                extra={
                  <Space>
                    <Button
                      size="small"
                      type="text"
                      icon={<SettingOutlined />}
                      title="分辨率设置"
                      onClick={() => openResModal(cam)}
                    />
                    <Popconfirm
                      title="确定删除该摄像头?"
                      onConfirm={() => deleteMutation.mutate(cam.id)}
                      okButtonProps={{ danger: true }}
                    >
                      <Button size="small" type="text" danger icon={<DeleteOutlined />} />
                    </Popconfirm>
                  </Space>
                }
              >
                <Row gutter={[12, 12]}>
                  <Col span={12}>
                    <div style={{ color: '#888', fontSize: 12 }}>信号源</div>
                    <code style={{ fontSize: 13 }}>{cam.source}</code>
                  </Col>
                  <Col span={12}>
                    <div style={{ color: '#888', fontSize: 12 }}>推理分辨率</div>
                    <div style={{ fontSize: 13 }}>{effectiveCaptureSize(cam)}</div>
                  </Col>
                  <Col span={12}>
                    <div style={{ color: '#888', fontSize: 12 }}>推流分辨率</div>
                    <div style={{ fontSize: 13 }}>{effectiveStreamHeight(cam)}</div>
                  </Col>
                  <Col span={24}>
                    <Divider style={{ margin: '8px 0' }} />
                    <div style={{ color: '#888', fontSize: 12, marginBottom: 4 }}>推理配置</div>
                    <Select
                      size="small"
                      value={cam.profile}
                      style={{ width: '100%' }}
                      options={profileOptions}
                      onChange={(profile) => profileMutation.mutate({ id: cam.id, profile })}
                    />
                  </Col>
                  <Col span={24}>
                    <div style={{ marginTop: 4 }}>
                      <Tag color={meta.color} style={{ borderRadius: 4 }}>
                        <PlayCircleOutlined /> {meta.device}
                      </Tag>
                      <span style={{ fontSize: 12, color: '#999', marginLeft: 8 }}>{meta.desc}</span>
                    </div>
                  </Col>
                </Row>
              </Card>
            </Col>
          );
        })}
      </Row>

      {/* ── 添加摄像头 Modal ── */}
      <Modal
        title="添加摄像头"
        open={createOpen}
        onCancel={() => setCreateOpen(false)}
        onOk={() => form.validateFields().then((values) => createMutation.mutate(values))}
        confirmLoading={createMutation.isPending}
        okText="添加"
      >
        <Form form={form} layout="vertical" initialValues={{ width: 640, height: 480, profile: 'desktop' }}>
          <Form.Item name="id" label="摄像头 ID" rules={[{ required: true, message: '请输入唯一 ID' }]}>
            <Input placeholder="例如 cam0" />
          </Form.Item>
          <Form.Item name="name" label="名称">
            <Input placeholder="显示名称(可选)" />
          </Form.Item>
          <Form.Item name="source" label="信号源" rules={[{ required: true, message: '请输入信号源' }]}>
            <Input placeholder="0(本地摄像头)或 rtsp://..." />
          </Form.Item>
          <Form.Item name="profile" label="推理档位">
            <Select options={profileOptions} />
          </Form.Item>
        </Form>
      </Modal>

      {/* ── 分辨率设置 Modal ── */}
      <Modal
        title={`分辨率设置 — ${resTarget?.name || resTarget?.id || ''}`}
        open={!!resTarget}
        onCancel={() => setResTarget(null)}
        onOk={() =>
          resTarget &&
          resMutation.mutate({
            id: resTarget.id,
            capture_width: resMode === 'custom' ? resWidth : 0,
            capture_height: resMode === 'custom' ? resHeight : 0,
            stream_max_height: resStream,
          })
        }
        confirmLoading={resMutation.isPending}
        okText="应用"
      >
        {resTarget && (
          <Form layout="vertical">
            <Form.Item label="推理/采集分辨率">
              <Radio.Group value={resMode} onChange={(e) => setResMode(e.target.value)}>
                <Radio value="native">跟随源分辨率(原生)</Radio>
                <Radio value="custom">自定义</Radio>
              </Radio.Group>
              {resMode === 'custom' && (
                <Space style={{ marginTop: 8 }}>
                  <InputNumber
                    min={1}
                    max={7680}
                    value={resWidth}
                    onChange={(v) => setResWidth(v ?? 0)}
                    addonAfter="宽"
                    style={{ width: 150 }}
                  />
                  <InputNumber
                    min={1}
                    max={7680}
                    value={resHeight}
                    onChange={(v) => setResHeight(v ?? 0)}
                    addonAfter="高"
                    style={{ width: 150 }}
                  />
                </Space>
              )}
              <div style={{ color: '#999', fontSize: 12, marginTop: 6 }}>
                提示:仅本地 USB 摄像头可强制分辨率;RTSP / 视频文件由源决定,「原生」即源本身分辨率。
              </div>
            </Form.Item>
            <Form.Item label="推流到前端分辨率">
              <Select options={STREAM_HEIGHT_OPTIONS} value={resStream} onChange={setResStream} style={{ width: 220 }} />
              <div style={{ color: '#999', fontSize: 12, marginTop: 6 }}>
                推流前按高度等比缩放(宽度自适应),仅降低不放大。
              </div>
            </Form.Item>
          </Form>
        )}
      </Modal>

      {/* ── 性能指标卡片 ── */}
      {metrics && Object.keys(metrics.cameras).length > 0 && (
        <>
          <div style={{ marginTop: 32, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
            <DashboardOutlined style={{ fontSize: 20, color: '#1677ff' }} />
            <span style={{ fontSize: 18, fontWeight: 600 }}>性能指标</span>
            <Tag color="blue">实时</Tag>
          </div>
          <Row gutter={[16, 16]}>
            {Object.entries(metrics.cameras).map(([cid, m]: [string, CameraMetrics]) => (
              <Col key={cid} xs={24} sm={12} lg={8}>
                <Card variant="borderless" style={{ borderRadius: 8 }} title={<span>{cid}</span>}>
                  <Row gutter={[8, 8]}>
                    <Col span={12}>
                      <div style={{ color: '#888', fontSize: 12 }}>FPS</div>
                      <div style={{ fontSize: 22, fontWeight: 700, color: '#52c41a' }}>{m.fps.toFixed(1)}</div>
                    </Col>
                    <Col span={12}>
                      <div style={{ color: '#888', fontSize: 12 }}>活跃目标</div>
                      <div style={{ fontSize: 22, fontWeight: 700 }}>{m.tracks}</div>
                    </Col>
                    <Col span={24}>
                      <Divider style={{ margin: '8px 0' }} />
                      <div style={{ color: '#888', fontSize: 12, marginBottom: 4 }}>各阶段延迟 (ms)</div>
                      <Space size={[8, 4]} wrap>
                        <Tag>采集 {m.stage_ms.capture?.toFixed(0) ?? '-'}</Tag>
                        <Tag>检测 {m.stage_ms.detect?.toFixed(0) ?? '-'}</Tag>
                        <Tag>跟踪 {m.stage_ms.track?.toFixed(0) ?? '-'}</Tag>
                        <Tag>任务 {m.stage_ms.tasks?.toFixed(0) ?? '-'}</Tag>
                      </Space>
                    </Col>
                    <Col span={24}>
                      <div style={{ color: '#888', fontSize: 12, marginBottom: 4 }}>运行状态</div>
                      <Space size={[8, 4]} wrap>
                        <Tag color="blue">帧 {m.frames}</Tag>
                        <Tag color="green">运行 {m.uptime_seconds}s</Tag>
                      </Space>
                    </Col>
                  </Row>
                </Card>
              </Col>
            ))}
          </Row>
        </>
      )}
    </div>
  );
}
