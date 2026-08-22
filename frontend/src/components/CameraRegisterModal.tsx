import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Modal, Form, Input, Select, Button, Space, Steps, Tag, message, Spin,
} from 'antd';
import { VideoCameraOutlined, CameraOutlined } from '@ant-design/icons';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { CameraCaptureSource } from '../registration/CameraCaptureSource';
import { SystemCameraSource } from '../registration/SystemCameraSource';
import { POSE_STEPS, checkPose, shouldAdvanceStep } from '../registration/poseEngine';
import type {
  CapturedFrame, PoseName, AnalyzeResult, CommitResult, SelectedSource,
} from '../registration/types';
import { detectFace } from '../api/faces';
import { analyzeRegistrationFrames, commitRegistrationFrames } from '../api/registration';
import { fetchCameras, startCamera, type CameraInfo } from '../api/cameras';
import { useWebSocket } from '../hooks/useWebSocket';
import { registrationConfig } from '../config';
import type { Identity } from '../api/faces';

type Stage = 'setup' | 'capturing' | 'review' | 'committing' | 'complete';
type Mode = 'create' | 'append';

interface Props {
  open: boolean;
  identities: Identity[];
  onClose: () => void;
}

interface DetState {
  bbox: [number, number, number, number];
  detScore: number;
  yawRatio: number;
  pitchRatio: number;
  imgW: number;   // 送检帧宽度(overlay 按真实尺寸换算,勿用硬编码)
  imgH: number;
}

/** 摄像头实时注册弹窗:视频源选择 → 实时取景 → 五步动作引导自动采帧 → 审核 → 原子入库。
 * 视频源支持两类:本机摄像头(getUserMedia)与系统监控摄像头(后端 WS 流 + snapshot 采帧)。
 */
export default function CameraRegisterModal({ open, identities, onClose }: Props) {
  const [mode, setMode] = useState<Mode>('create');
  const [stage, setStage] = useState<Stage>('setup');
  const [stepIndex, setStepIndex] = useState(0);
  const [captured, setCaptured] = useState<CapturedFrame[]>([]);
  const [perPose, setPerPose] = useState<Record<PoseName, number>>({
    frontal: 0, left: 0, right: 0, up: 0, down: 0,
  });
  const [det, setDet] = useState<DetState | null>(null);
  const [hint, setHint] = useState('');
  const [analysis, setAnalysis] = useState<AnalyzeResult | null>(null);
  const [result, setResult] = useState<CommitResult | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [sysCameras, setSysCameras] = useState<CameraInfo[]>([]);
  const [selected, setSelected] = useState<SelectedSource>({
    kind: 'device', deviceId: '__default__', cameraId: '',
  });
  const [wsPreviewUrl, setWsPreviewUrl] = useState<string | null>(null);
  const [form] = Form.useForm();
  const queryClient = useQueryClient();

  const videoRef = useRef<HTMLVideoElement>(null);
  const sourceRef = useRef<CameraCaptureSource | null>(null);
  const systemSourceRef = useRef<SystemCameraSource | null>(null);
  const detectTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastCaptureRef = useRef(0);
  const runTokenRef = useRef(0);
  const capturedRef = useRef<CapturedFrame[]>([]);
  const perPoseRef = useRef<Record<PoseName, number>>({ frontal: 0, left: 0, right: 0, up: 0, down: 0 });
  const stepIndexRef = useRef(0);
  const detRef = useRef<DetState | null>(null);
  const selectedRef = useRef<SelectedSource>(selected);

  capturedRef.current = captured;
  perPoseRef.current = perPose;
  stepIndexRef.current = stepIndex;
  detRef.current = det;
  selectedRef.current = selected;

  // ── 系统摄像头 WS 连接(hook 顶层无条件调用,path 为 null 则不连) ──
  const wsPath =
    open && stage === 'capturing' && selectedRef.current.kind === 'system' && selectedRef.current.cameraId
      ? `/ws/cameras/${selectedRef.current.cameraId}` : null;

  useWebSocket(
    wsPath,
    (data: unknown) => {
      if (!(data instanceof ArrayBuffer)) return; // detections JSON 忽略(姿态由 detectFace 算)
      const src = systemSourceRef.current;
      if (!src) return;
      src.offer(data);
      // 预览:最新帧 → objectURL(函数式更新里 revoke 旧 URL,避免闭包旧值)
      const f = src.latestFrame;
      if (f) {
        setWsPreviewUrl((old) => {
          if (old) URL.revokeObjectURL(old);
          return URL.createObjectURL(f.blob);
        });
      }
    },
    2000,
    'arraybuffer',
  );

  // ── setup 挂载:加载设备/系统摄像头/恢复上次选择 ──
  useEffect(() => {
    if (!open) return;
    CameraCaptureSource.listDevices().then(setDevices).catch(() => {});
    const onDeviceChange = () => CameraCaptureSource.listDevices().then(setDevices).catch(() => {});
    navigator.mediaDevices?.addEventListener?.('devicechange', onDeviceChange);
    fetchCameras().then((r) => setSysCameras(r.data)).catch(() => {});
    try {
      const saved = localStorage.getItem('camreg.source');
      if (saved) setSelected(JSON.parse(saved));
    } catch { /* 忽略损坏数据 */ }
    return () => {
      navigator.mediaDevices?.removeEventListener?.('devicechange', onDeviceChange);
    };
  }, [open]);

  // ── 供两个分支共用的检测帧获取 ──
  const getDetectBlob = useCallback((): Blob | null => {
    if (selectedRef.current.kind === 'system') {
      return systemSourceRef.current?.getDetectFrame() ?? null;
    }
    return sourceRef.current?.captureFrame(registrationConfig.maxHeight, registrationConfig.jpegQuality) ?? null;
  }, []);

  // ── 供两个分支共用的高质量采帧(系统模式走 snapshot 原始帧) ──
  const grabRegFrame = useCallback(async (): Promise<Blob | null> => {
    if (selectedRef.current.kind === 'system') {
      const src = systemSourceRef.current;
      if (!src) return null;
      try { return await src.captureFrame(); } catch { return null; } // snapshot 404 静默跳过本轮
    }
    return sourceRef.current?.captureFrame(registrationConfig.maxHeight, registrationConfig.jpegQuality) ?? null;
  }, []);

  const finishCapture = useCallback((token: number) => {
    if (detectTimerRef.current) {
      clearInterval(detectTimerRef.current);
      detectTimerRef.current = null;
    }
    sourceRef.current?.close();
    sourceRef.current = null;
    systemSourceRef.current?.close();
    systemSourceRef.current = null;
    setWsPreviewUrl((old) => { if (old) URL.revokeObjectURL(old); return null; });
    if (token !== runTokenRef.current) return;
    runAnalyze(capturedRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const maybeCapture = useCallback(async (token: number, pose: PoseName) => {
    const now = Date.now();
    if (now - lastCaptureRef.current < registrationConfig.captureIntervalMs) return;
    const total = capturedRef.current.length;
    if (total >= registrationConfig.maxCapturedFrames) {
      finishCapture(token);
      return;
    }
    const blob = await grabRegFrame();
    if (!blob) return;
    lastCaptureRef.current = now;
    const frame: CapturedFrame = {
      frameId: `c${now}_${Math.random().toString(36).slice(2, 6)}`,
      timestampMs: now,
      pose,
      blob,
      previewUrl: URL.createObjectURL(blob),
    };
    const next = [...capturedRef.current, frame];
    setCaptured(next);
    setPerPose((prev) => ({ ...prev, [pose]: (prev[pose] || 0) + 1 }));
    if (shouldAdvanceStep(next.filter((nf) => nf.pose === pose).length)) {
      const nextIdx = stepIndexRef.current + 1;
      if (nextIdx >= POSE_STEPS.length) {
        finishCapture(token);
        return;
      }
      setStepIndex(nextIdx);
      setHint(POSE_STEPS[nextIdx].instruction);
    }
  }, [grabRegFrame, finishCapture]);

  const startDetectLoop = useCallback((token: number) => {
    detectTimerRef.current = setInterval(async () => {
      const tokenNow = runTokenRef.current;
      const frame = getDetectBlob();
      if (!frame) return; // 系统模式无新帧 / 本机模式未就绪 → 直接返回,不改 hint
      const fd = new FormData();
      fd.append('image', frame, 'cam.jpg');
      try {
        const res = await detectFace(fd);
        if (tokenNow !== runTokenRef.current) return;
        const faces = res.data?.faces ?? [];
        if (faces.length === 0) {
          setDet(null);
          setHint('未检测到人脸,请正对摄像头');
          return;
        }
        const f = faces[0];
        const d: DetState = {
          bbox: [f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3]],
          detScore: f.det_score,
          yawRatio: f.yaw_ratio ?? 0,
          pitchRatio: f.pitch_ratio ?? 0,
          imgW: res.data.width || 1,
          imgH: res.data.height || 1,
        };
        setDet(d);
        const step = POSE_STEPS[stepIndexRef.current];
        const chk = checkPose(step, { yawRatio: d.yawRatio, pitchRatio: d.pitchRatio, detScore: d.detScore, bbox: d.bbox });
        setHint(chk.hint);
        if (chk.ok && chk.capture) void maybeCapture(tokenNow, step.pose);
      } catch {
        if (tokenNow === runTokenRef.current) setHint('检测请求失败,请重试');
      }
    }, registrationConfig.detectIntervalMs);
  }, [getDetectBlob, maybeCapture]);

  const startCapture = useCallback(async (keepFrames = false) => {
    runTokenRef.current += 1;
    const token = runTokenRef.current;
    if (!keepFrames) {
      setCaptured([]);
      setPerPose({ frontal: 0, left: 0, right: 0, up: 0, down: 0 });
      setStepIndex(0);
      setAnalysis(null);
      setResult(null);
    }
    setDet(null);
    setHint('正在启动摄像头…');
    // 先切到 capturing 阶段让 <video>/<img> 元素挂载,双 rAF 等 React 渲染提交后再取 ref
    setStage('capturing');
    await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
    if (token !== runTokenRef.current) return;

    // 分支一:系统监控摄像头(浏览器无法直连,走后端 WS + snapshot)
    if (selectedRef.current.kind === 'system') {
      const cam = sysCameras.find((c) => c.id === selectedRef.current.cameraId);
      if (!cam) {
        if (token === runTokenRef.current) { message.error('请先选择系统摄像头'); setStage('setup'); }
        return;
      }
      setHint('正在连接摄像头…');
      // 未运行则自动启动(不自动停止——系统摄像头归设置页管)
      if (cam.status !== 'running') {
        try { await startCamera(cam.id); } catch (e: any) {
          if (token === runTokenRef.current) {
            message.error(e?.response?.data?.detail || '摄像头启动失败');
            setStage('setup');
          }
          return;
        }
        await new Promise((r) => setTimeout(r, 1500)); // 等推流起来
        if (token !== runTokenRef.current) return;
      }
      systemSourceRef.current = new SystemCameraSource(cam.id);
      sourceRef.current = null;
      setHint(POSE_STEPS[0].instruction);
      startDetectLoop(token);
      return;
    }

    // 分支二:本机摄像头(getUserMedia)
    const video = videoRef.current;
    if (!video) {
      if (token === runTokenRef.current) { message.error('视频组件初始化失败,请重试'); setStage('setup'); }
      return;
    }
    const src = new CameraCaptureSource(video, selectedRef.current.deviceId);
    sourceRef.current = src;
    try {
      await src.open();
    } catch (err: any) {
      if (token === runTokenRef.current) {
        const name = (err as any)?.name;
        const msg =
          name === 'NotAllowedError' ? '摄像头权限被拒绝,请在浏览器地址栏允许后重试'
          : name === 'NotFoundError' ? '未找到所选摄像头,可能已被拔出,请重新选择'
          : name === 'OverconstrainedError' ? '所选摄像头不支持当前参数,请换一个'
          : ((err as Error)?.message || '无法访问摄像头');
        message.error(msg);
        if (name === 'NotFoundError' || name === 'OverconstrainedError') {
          CameraCaptureSource.listDevices().then(setDevices).catch(() => {});
        }
        setStage('setup');
      }
      return;
    }
    // open 成功 = 已授权,重枚举刷新 label
    CameraCaptureSource.listDevices().then(setDevices).catch(() => {});
    if (token !== runTokenRef.current) return;
    setHint(POSE_STEPS[0].instruction);
    startDetectLoop(token);
  }, [sysCameras, startDetectLoop]);

  const runAnalyze = useCallback(async (frames: CapturedFrame[]) => {
    if (frames.length === 0) {
      setHint('未采集到帧');
      return;
    }
    try {
      const res = await analyzeRegistrationFrames(frames);
      setAnalysis(res.data);
      const okCount = res.data.accepted_count;
      if (okCount < 3) {
        message.warning(`有效帧不足(${okCount}),已保留已采集帧,继续采集`);
        void startCapture(true);
        return;
      }
      const recIds = new Set(res.data.recommended_frame_ids);
      setCaptured(frames.filter((f) => recIds.has(f.frameId)));
      setStage('review');
    } catch (err: any) {
      message.error(err?.response?.data?.detail || '分析失败');
      void startCapture(true);
    }
  }, [startCapture]);

  const commitMutation = useMutation({
    mutationFn: (input: { mode: Mode; name?: string; notes?: string; identityId?: string; frames: CapturedFrame[] }) =>
      commitRegistrationFrames(input),
    onSuccess: (res) => {
      setResult(res.data);
      setStage('complete');
      message.success(`已注册 ${res.data.embedding_count_added} 个角度特征`);
      queryClient.invalidateQueries({ queryKey: ['faces'] });
    },
    onError: (err: any) => message.error(err?.response?.data?.detail || '提交失败'),
  });

  const handleSubmit = () => {
    const frames = capturedRef.current;
    if (frames.length < 3) {
      message.warning('至少保留 3 帧');
      return;
    }
    if (mode === 'create') {
      form.validateFields().then((values) => {
        commitMutation.mutate({ mode, name: values.name, notes: values.notes || '', frames });
      });
    } else {
      const id = form.getFieldValue('identityId');
      if (!id) {
        message.warning('请选择要追加的身份');
        return;
      }
      commitMutation.mutate({ mode, identityId: id, frames });
    }
  };

  // 组件卸载/关闭清理
  useEffect(() => {
    if (!open) return;
    return () => {
      runTokenRef.current += 1;
      if (detectTimerRef.current) clearInterval(detectTimerRef.current);
      systemSourceRef.current?.close();
      systemSourceRef.current = null;
      sourceRef.current?.close();
      sourceRef.current = null;
      capturedRef.current.forEach((f) => URL.revokeObjectURL(f.previewUrl));
      setWsPreviewUrl((old) => { if (old) URL.revokeObjectURL(old); return null; });
    };
  }, [open]);

  const detect = det as DetState | null;

  const sourceSelectValue = selected.kind === 'device' ? selected.deviceId : `sys:${selected.cameraId}`;
  const onSourceChange = (v: string) => {
    if (v.startsWith('sys:')) {
      setSelected({ kind: 'system', deviceId: '__default__', cameraId: v.slice(4) });
      localStorage.setItem('camreg.source', JSON.stringify({ kind: 'system', deviceId: '__default__', cameraId: v.slice(4) }));
    } else {
      setSelected({ kind: 'device', deviceId: v, cameraId: '' });
      localStorage.setItem('camreg.source', JSON.stringify({ kind: 'device', deviceId: v, cameraId: '' }));
    }
  };

  return (
    <Modal
      title={<Space><VideoCameraOutlined />摄像头实时注册</Space>}
      open={open}
      width={560}
      footer={null}
      onCancel={onClose}
      destroyOnClose
    >
      {stage === 'setup' && (
        <div>
          <Form form={form} layout="vertical">
            <Form.Item label="视频源">
              <Select
                value={sourceSelectValue}
                onChange={onSourceChange}
                options={[
                  {
                    label: '本机摄像头',
                    options: [
                      { value: '__default__', label: '默认摄像头' },
                      ...devices.map((d, i) => ({
                        value: d.deviceId,
                        label: d.label || `摄像头 ${i + 1}`,
                      })),
                    ],
                  },
                  {
                    label: '系统监控摄像头(含远程)',
                    options: sysCameras.map((c) => ({
                      value: `sys:${c.id}`,
                      label: `${c.name}(${c.status === 'running' ? '运行中' : '未启动'})`,
                    })),
                  },
                ]}
              />
              <div style={{ color: '#999', fontSize: 12, marginTop: 4 }}>
                本机摄像头需浏览器授权;系统监控摄像头走后端推流,未启动会自动尝试启动
              </div>
            </Form.Item>
            <Form.Item label="注册方式">
              <Select
                value={mode}
                onChange={(v: Mode) => setMode(v)}
                options={[
                  { value: 'create', label: '新建身份' },
                  { value: 'append', label: '给已有身份追加' },
                ]}
              />
            </Form.Item>
            {mode === 'create' ? (
              <>
                <Form.Item name="name" label="姓名" rules={[{ required: true, message: '请输入姓名' }]}>
                  <Input placeholder="输入姓名" />
                </Form.Item>
                <Form.Item name="notes" label="备注">
                  <Input.TextArea rows={2} />
                </Form.Item>
              </>
            ) : (
              <Form.Item name="identityId" label="选择身份" rules={[{ required: true, message: '请选择' }]}>
                <Select
                  options={(identities || []).map((i) => ({ value: i.id, label: i.name }))}
                  placeholder="选择要追加特征的身份"
                />
              </Form.Item>
            )}
          </Form>
          <Button type="primary" icon={<VideoCameraOutlined />} onClick={() => void startCapture()} block>
            开始采集
          </Button>
          <div style={{ color: '#999', fontSize: 12, marginTop: 8 }}>
            请依次完成:正脸 → 向左转头 → 向右转头 → 抬头 → 低头,系统会自动在每步达标时采帧。
          </div>
        </div>
      )}

      {stage === 'capturing' && (
        <div>
          <div style={{ position: 'relative', background: '#1a1a2e', aspectRatio: '4 / 3', borderRadius: 8, overflow: 'hidden' }}>
            {selected.kind === 'system' ? (
              wsPreviewUrl ? (
                <img src={wsPreviewUrl} alt="预览" style={{ width: '100%', display: 'block' }} />
              ) : (
                <div style={{ color: '#888', textAlign: 'center', paddingTop: '40%' }}>等待视频流…</div>
              )
            ) : (
              <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', display: 'block' }} />
            )}
            {detect && (
              <div
                style={{
                  position: 'absolute',
                  left: `${(detect.bbox[0] / detect.imgW) * 100}%`,
                  top: `${(detect.bbox[1] / detect.imgH) * 100}%`,
                  width: `${((detect.bbox[2] - detect.bbox[0]) / detect.imgW) * 100}%`,
                  height: `${((detect.bbox[3] - detect.bbox[1]) / detect.imgH) * 100}%`,
                  border: '2px solid #52c41a',
                  boxSizing: 'border-box',
                  pointerEvents: 'none',
                }}
              />
            )}
            <div style={{ position: 'absolute', top: 8, left: 0, right: 0, textAlign: 'center', pointerEvents: 'none' }}>
              <Tag color="blue" style={{ fontSize: 14, padding: '4px 12px' }}>{hint}</Tag>
            </div>
          </div>
          <Steps
            current={stepIndex}
            items={POSE_STEPS.map((s) => ({
              title: s.instruction.split(',')[0],
              description: `${(perPose[s.pose] || 0)} 帧`,
            }))}
            size="small"
            style={{ marginTop: 12 }}
          />
          <Space style={{ marginTop: 12 }} wrap>
            <Button
              icon={<CameraOutlined />}
              onClick={() => void maybeCapture(runTokenRef.current, POSE_STEPS[stepIndexRef.current].pose)}
            >
              手动抓拍
            </Button>
            <Button onClick={() => {
              const idx = stepIndexRef.current;
              if (idx + 1 < POSE_STEPS.length) setStepIndex(idx + 1);
              else finishCapture(runTokenRef.current);
            }}>
              跳过此动作
            </Button>
            <Button danger onClick={() => {
              if (detectTimerRef.current) clearInterval(detectTimerRef.current);
              sourceRef.current?.close();
              sourceRef.current = null;
              systemSourceRef.current?.close();
              systemSourceRef.current = null;
              setWsPreviewUrl((old) => { if (old) URL.revokeObjectURL(old); return null; });
              setCaptured([]);
              setStage('setup');
            }}>
              取消
            </Button>
            <Button type="primary" onClick={() => finishCapture(runTokenRef.current)}>
              完成采集({captured.length})
            </Button>
          </Space>
        </div>
      )}

      {stage === 'review' && (
        <div>
          <div style={{ marginBottom: 12, color: '#888' }}>已选 {captured.length} 帧,至少 3 帧可提交</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {captured.map((f) => (
              <div key={f.frameId} style={{ position: 'relative', width: 150 }}>
                <img src={f.previewUrl} alt={f.pose} style={{ width: 150, height: 150, objectFit: 'cover', borderRadius: 6 }} />
                <Tag color="green" style={{ position: 'absolute', top: 4, left: 4 }}>{f.pose}</Tag>
                <Button
                  size="small"
                  danger
                  style={{ position: 'absolute', top: 4, right: 4 }}
                  onClick={() => {
                    setCaptured((prev) => prev.filter((x) => x.frameId !== f.frameId));
                    URL.revokeObjectURL(f.previewUrl);
                  }}
                >
                  删除
                </Button>
              </div>
            ))}
          </div>
          <Space style={{ marginTop: 16 }}>
            <Button onClick={() => { setCaptured([]); setStage('setup'); }}>重新采集</Button>
            <Button
              type="primary"
              disabled={captured.length < 3}
              loading={commitMutation.isPending}
              onClick={handleSubmit}
            >
              提交注册
            </Button>
          </Space>
        </div>
      )}

      {stage === 'committing' && (
        <div style={{ textAlign: 'center', padding: 24 }}>
          <Spin /> 正在提交...
        </div>
      )}

      {stage === 'complete' && result && (
        <div style={{ textAlign: 'center', padding: 24 }}>
          <div style={{ fontSize: 40 }}>✓</div>
          <div style={{ fontSize: 18, marginTop: 8 }}>
            注册成功,新增 {result.embedding_count_added} 个角度特征
          </div>
          <Button type="primary" style={{ marginTop: 16 }} onClick={onClose}>完成</Button>
        </div>
      )}
    </Modal>
  );
}
