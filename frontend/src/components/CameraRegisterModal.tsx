import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Modal, Form, Input, Select, Button, Space, Steps, Tag, message, Spin,
} from 'antd';
import { VideoCameraOutlined, CameraOutlined } from '@ant-design/icons';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { CameraCaptureSource } from '../registration/CameraCaptureSource';
import { SystemCameraSource } from '../registration/SystemCameraSource';
import { POSE_STEPS, checkPose, shouldAdvanceStep, REASON_LABELS } from '../registration/poseEngine';
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

/** 摄像头实时注册弹窗:视频源选择 → 实时取景 → 五步动作引导自动采帧 → 审核(可单方向重采)→ 原子入库。
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
  /** 单方向重采模式:非 null 表示当前只对某 pose 采集单帧(审核页进入)。 */
  const [recapturePose, setRecapturePose] = useState<PoseName | null>(null);
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
  const recapturePoseRef = useRef<PoseName | null>(null);

  capturedRef.current = captured;
  perPoseRef.current = perPose;
  stepIndexRef.current = stepIndex;
  detRef.current = det;
  selectedRef.current = selected;
  recapturePoseRef.current = recapturePose;

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

  const getDetectBlob = useCallback((): Blob | null => {
    if (selectedRef.current.kind === 'system') {
      return systemSourceRef.current?.getDetectFrame() ?? null;
    }
    return sourceRef.current?.captureFrame(registrationConfig.maxHeight, registrationConfig.jpegQuality) ?? null;
  }, []);

  const grabRegFrame = useCallback(async (): Promise<Blob | null> => {
    if (selectedRef.current.kind === 'system') {
      const src = systemSourceRef.current;
      if (!src) return null;
      try { return await src.captureFrame(); } catch { return null; }
    }
    return sourceRef.current?.captureFrame(registrationConfig.maxHeight, registrationConfig.jpegQuality) ?? null;
  }, []);

  /** 按内核清理定时器与音视频源(不含状态重置)。 */
  const cleanupSources = useCallback(() => {
    if (detectTimerRef.current) {
      clearInterval(detectTimerRef.current);
      detectTimerRef.current = null;
    }
    sourceRef.current?.close();
    sourceRef.current = null;
    systemSourceRef.current?.close();
    systemSourceRef.current = null;
    setWsPreviewUrl((old) => { if (old) URL.revokeObjectURL(old); return null; });
  }, []);

  /** 打开当前选中源(device/system 双分支),返回是否成功。失败已自行回退 UI。 */
  const openActiveSource = useCallback(async (token: number): Promise<boolean> => {
    if (selectedRef.current.kind === 'system') {
      const cam = sysCameras.find((c) => c.id === selectedRef.current.cameraId);
      if (!cam) {
        if (token === runTokenRef.current) { message.error('请先选择系统摄像头'); setStage('setup'); }
        return false;
      }
      if (cam.status !== 'running') {
        try { await startCamera(cam.id); } catch (e: any) {
          if (token === runTokenRef.current) {
            message.error(e?.response?.data?.detail || '摄像头启动失败');
            setStage('setup');
          }
          return false;
        }
        await new Promise((r) => setTimeout(r, 1500));
        if (token !== runTokenRef.current) return false;
      }
      systemSourceRef.current = new SystemCameraSource(cam.id);
      sourceRef.current = null;
      return true;
    }

    const video = videoRef.current;
    if (!video) {
      if (token === runTokenRef.current) { message.error('视频组件初始化失败,请重试'); setStage('setup'); }
      return false;
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
      return false;
    }
    CameraCaptureSource.listDevices().then(setDevices).catch(() => {});
    if (token !== runTokenRef.current) return false;
    return true;
  }, [sysCameras]);

  /** 停止检测循环并仅关闭源,不触发 runAnalyze(重采用到)。 */
  const stopCaptureAndClose = useCallback(() => {
    cleanupSources();
  }, [cleanupSources]);

  const finishCapture = useCallback((token: number) => {
    cleanupSources();
    if (token !== runTokenRef.current) return;
    runAnalyze(capturedRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cleanupSources]);

  const maybeCapture = useCallback(async (token: number, pose: PoseName) => {
    const now = Date.now();
    if (now - lastCaptureRef.current < registrationConfig.captureIntervalMs) return;

    // 重采模式:跳过同姿态去重与 maxCapturedFrames 上限(用户明确要替换旧帧)
    const isRecapture = recapturePoseRef.current !== null;
    if (!isRecapture) {
      const total = capturedRef.current.length;
      if (total >= registrationConfig.maxCapturedFrames) {
        finishCapture(token);
        return;
      }
      const curDet = detRef.current;
      if (curDet) {
        const dup = capturedRef.current.some((f) =>
          f.pose === pose
          && f.yawRatio !== undefined
          && Math.abs(f.yawRatio - curDet.yawRatio) < 0.06
          && Math.abs((f.pitchRatio ?? 0) - curDet.pitchRatio) < 0.06,
        );
        if (dup) return;
      }
    }

    const blob = await grabRegFrame();
    if (!blob) return;
    lastCaptureRef.current = now;
    const curDetBack = detRef.current;
    const frame: CapturedFrame = {
      frameId: `c${now}_${Math.random().toString(36).slice(2, 6)}`,
      timestampMs: now,
      pose,
      blob,
      previewUrl: URL.createObjectURL(blob),
      yawRatio: curDetBack?.yawRatio,
      pitchRatio: curDetBack?.pitchRatio,
    };
    const next = [...capturedRef.current, frame];
    setCaptured(next);
    setPerPose((prev) => ({ ...prev, [pose]: (prev[pose] || 0) + 1 }));

    if (isRecapture) {
      cleanupSources();
      if (token !== runTokenRef.current) return;
      await commitRecapturedFrame(token, frame);
      return;
    }

    if (shouldAdvanceStep(next.filter((nf) => nf.pose === pose).length)) {
      const nextIdx = stepIndexRef.current + 1;
      if (nextIdx >= POSE_STEPS.length) {
        finishCapture(token);
        return;
      }
      setStepIndex(nextIdx);
      setHint(POSE_STEPS[nextIdx].instruction);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [grabRegFrame, finishCapture, cleanupSources]);

  /** 重采模式:单帧分析 → 替换该方向旧帧 → 回审核页。 */
  const commitRecapturedFrame = useCallback(async (token: number, frame: CapturedFrame) => {
    const pose = recapturePoseRef.current;
    try {
      const res = await analyzeRegistrationFrames([frame]);
      const fr = res.data.frames?.[0];
      frame.accepted = fr?.accepted ?? false;
      frame.reason = fr?.reason ?? null;
      frame.qualityScore = fr?.quality_score ?? null;
    } catch {
      frame.accepted = false;
      frame.reason = null;
      frame.qualityScore = null;
    }
    if (token !== runTokenRef.current) return;
    if (!pose) { setRecapturePose(null); setStage('review'); return; }

    const removed = capturedRef.current.filter((f) => f.pose === pose && f.frameId !== frame.frameId);
    removed.forEach((f) => URL.revokeObjectURL(f.previewUrl));
    const others = capturedRef.current.filter((f) => f.pose !== pose);
    setCaptured([...others, frame]);
    setRecapturePose(null);
    setStage('review');
    if (frame.accepted !== true) {
      message.warning(`该帧未通过校验(${REASON_LABELS[frame.reason ?? ''] ?? '原因未知'}),可再次重采`);
    }
  }, []);

  const startDetectLoop = useCallback((token: number) => {
    detectTimerRef.current = setInterval(async () => {
      const tokenNow = runTokenRef.current;
      const frame = getDetectBlob();
      if (!frame) return;
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
        // 重采模式固定目标步骤;正常模式按 stepIndex
        const step = recapturePoseRef.current
          ? POSE_STEPS.find((s) => s.pose === recapturePoseRef.current)!
          : POSE_STEPS[stepIndexRef.current];
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
    const ok = await openActiveSource(token);
    if (!ok || token !== runTokenRef.current) return;
    setHint(POSE_STEPS[0].instruction);
    startDetectLoop(token);
  }, [openActiveSource, startDetectLoop]);

  /** 审核页进入单方向重采:复用 capturing 状态机,固定该 pose 采一帧。 */
  const startRecapture = useCallback(async (pose: PoseName) => {
    runTokenRef.current += 1;
    const token = runTokenRef.current;
    setRecapturePose(pose);
    setDet(null);
    setHint('正在启动摄像头…');
    setStage('capturing');
    await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
    if (token !== runTokenRef.current) { setRecapturePose(null); return; }
    const ok = await openActiveSource(token);
    if (!ok || token !== runTokenRef.current) { setRecapturePose(null); return; }
    setHint(POSE_STEPS.find((s) => s.pose === pose)!.instruction);
    startDetectLoop(token);
  }, [openActiveSource, startDetectLoop]);

  const runAnalyze = useCallback(async (frames: CapturedFrame[]) => {
    if (frames.length === 0) {
      setHint('未采集到帧');
      return;
    }
    try {
      const res = await analyzeRegistrationFrames(frames);
      // 合并分析结论到每帧,全量保留(不再按 recommended 过滤——审核页要展示全部方向)
      const byId = new Map(res.data.frames.map((fr) => [fr.frame_id, fr]));
      const merged = frames.map((f) => {
        const fr = byId.get(f.frameId);
        return {
          ...f,
          accepted: fr?.accepted ?? false,
          reason: fr?.reason ?? null,
          qualityScore: fr?.quality_score ?? null,
        };
      });
      setCaptured(merged);
      setStage('review');
      const ok = merged.filter((f) => f.accepted === true).length;
      if (ok < 3) message.warning(`通过质量校验 ${ok}/3 帧,请对未通过的方向重新采集`);
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
    const frames = capturedRef.current.filter((f) => f.accepted === true);
    if (frames.length < 3) {
      message.warning('至少 3 帧通过质量校验');
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
  const recapturing = recapturePose !== null;
  const acceptedCount = captured.filter((f) => f.accepted === true).length;

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
      width={640}
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
            {/* 镜像 wrapper:仅本机自拍模式做水平镜像(照镜子习惯);系统监控模式不镜像。
                镜像只是显示层,送检测/采帧的数据帧始终不镜像,否则左右判定语义会颠倒。 */}
            <div style={selected.kind === 'device' ? { width: '100%', height: '100%', transform: 'scaleX(-1)' } : { width: '100%', height: '100%' }}>
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
            </div>
            <div style={{ position: 'absolute', top: 8, left: 12, right: 12, textAlign: 'center', pointerEvents: 'none' }}>
              {recapturing && (
                <Tag color="orange" style={{ marginRight: 4 }}>重采:{POSE_STEPS.find((s) => s.pose === recapturePose)?.shortLabel}</Tag>
              )}
              <Tag color="blue" style={{ fontSize: 14, padding: '4px 12px', maxWidth: '100%', whiteSpace: 'normal' }}>{hint}</Tag>
            </div>
            {/* 姿态实时调试数值(在镜像 wrapper 外,避免被翻转成反字) */}
            {detect && (
              <div style={{
                position: 'absolute', bottom: 8, right: 12, pointerEvents: 'none',
                fontFamily: 'monospace', fontSize: 12, color: 'rgba(255,255,255,0.85)',
                background: 'rgba(0,0,0,0.45)', padding: '2px 8px', borderRadius: 4,
              }}>
                yaw {detect.yawRatio >= 0 ? '+' : ''}{detect.yawRatio.toFixed(2)}  pitch {detect.pitchRatio >= 0 ? '+' : ''}{detect.pitchRatio.toFixed(2)}
              </div>
            )}
          </div>
          {!recapturing && (
            <Steps
              current={stepIndex}
              items={POSE_STEPS.map((s) => ({
                title: s.shortLabel,
                description: `${(perPose[s.pose] || 0)} 帧`,
              }))}
              size="small"
              style={{ marginTop: 12 }}
            />
          )}
          <Space style={{ marginTop: 12 }} wrap>
            <Button
              icon={<CameraOutlined />}
              onClick={() => void maybeCapture(
                runTokenRef.current,
                recapturePose ?? POSE_STEPS[stepIndexRef.current].pose,
              )}
            >
              手动抓拍
            </Button>
            {!recapturing && (
              <Button onClick={() => {
                const idx = stepIndexRef.current;
                if (idx + 1 < POSE_STEPS.length) setStepIndex(idx + 1);
                else finishCapture(runTokenRef.current);
              }}>
                跳过此动作
              </Button>
            )}
            <Button danger onClick={() => {
              cleanupSources();
              if (recapturing) {
                setRecapturePose(null);
                setStage('review');   // 重采取消:回审核页,不清空已采帧
              } else {
                setCaptured([]);
                setStage('setup');
              }
            }}>
              取消
            </Button>
            {!recapturing && (
              <Button type="primary" onClick={() => finishCapture(runTokenRef.current)}>
                完成采集({captured.length})
              </Button>
            )}
          </Space>
        </div>
      )}

      {stage === 'review' && (
        <div>
          <div style={{ marginBottom: 12, color: '#888' }}>
            通过质量校验 {acceptedCount}/3 帧 · 共 {captured.length} 帧 · 每个方向可单独重采/补拍
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
            {POSE_STEPS.map((step) => {
              const framesOf = captured.filter((f) => f.pose === step.pose);
              return (
                <div key={step.pose} style={{ width: 180, textAlign: 'center' }}>
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>{step.shortLabel}</div>
                  {framesOf.length === 0 ? (
                    <div style={{ width: '100%', height: 110, borderRadius: 6, background: '#f5f5f5',
                      display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#bbb' }}>
                      未采集
                    </div>
                  ) : (
                    <div style={{ display: 'flex', gap: 4 }}>
                      {framesOf.map((f) => (
                        <div key={f.frameId} style={{ position: 'relative', flex: 1, minWidth: 0 }}>
                          <img src={f.previewUrl} alt={f.pose}
                            style={{ width: '100%', height: 110, objectFit: 'cover', borderRadius: 6 }} />
                          <Tag color={f.accepted === true ? 'green' : 'red'}
                            style={{ position: 'absolute', top: 2, left: 2, fontSize: 11 }}>
                            {f.accepted === true ? `✓${(f.qualityScore ?? 0).toFixed(2)}` : `✗${REASON_LABELS[f.reason ?? ''] ?? '未通过'}`}
                          </Tag>
                          <Button size="small" danger type="text" style={{ position: 'absolute', top: 2, right: 2 }}
                            onClick={() => {
                              setCaptured((prev) => prev.filter((x) => x.frameId !== f.frameId));
                              URL.revokeObjectURL(f.previewUrl);
                            }}>×</Button>
                        </div>
                      ))}
                    </div>
                  )}
                  <Button size="small" block style={{ marginTop: 4 }}
                    onClick={() => void startRecapture(step.pose)}>
                    {framesOf.length ? '重新采集' : '补拍'}
                  </Button>
                </div>
              );
            })}
          </div>
          <Space style={{ marginTop: 16 }}>
            <Button onClick={() => { setCaptured([]); setStage('setup'); }}>全部重采</Button>
            <Button type="primary" disabled={acceptedCount < 3} loading={commitMutation.isPending} onClick={handleSubmit}>
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
