import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Modal, Form, Input, Select, Button, Space, Steps, Tag, message, Spin,
} from 'antd';
import { VideoCameraOutlined, CameraOutlined } from '@ant-design/icons';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { CameraCaptureSource } from '../registration/CameraCaptureSource';
import { POSE_STEPS, checkPose, shouldAdvanceStep } from '../registration/poseEngine';
import type {
  CapturedFrame, PoseName, AnalyzeResult, CommitResult,
} from '../registration/types';
import { detectFace } from '../api/faces';
import { analyzeRegistrationFrames, commitRegistrationFrames } from '../api/registration';
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
}

/** 摄像头实时注册弹窗:实时取景 → 五步动作引导自动采帧 → 审核 → 原子入库。 */
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
  const [form] = Form.useForm();
  const queryClient = useQueryClient();

  const videoRef = useRef<HTMLVideoElement>(null);
  const sourceRef = useRef<CameraCaptureSource | null>(null);
  const detectTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastCaptureRef = useRef(0);
  const runTokenRef = useRef(0);
  const capturedRef = useRef<CapturedFrame[]>([]);
  const perPoseRef = useRef<Record<PoseName, number>>({ frontal: 0, left: 0, right: 0, up: 0, down: 0 });
  const stepIndexRef = useRef(0);
  const detRef = useRef<DetState | null>(null);

  capturedRef.current = captured;
  perPoseRef.current = perPose;
  stepIndexRef.current = stepIndex;
  detRef.current = det;

  const startCapture = useCallback(async () => {
    runTokenRef.current += 1;
    const token = runTokenRef.current;
    const video = videoRef.current;
    if (!video) return;
    sourceRef.current = new CameraCaptureSource(video);
    setCaptured([]);
    setPerPose({ frontal: 0, left: 0, right: 0, up: 0, down: 0 });
    setStepIndex(0);
    setDet(null);
    setHint('');
    try {
      await sourceRef.current.open();
    } catch (err: any) {
      if (token === runTokenRef.current) {
        message.error(err?.message || '无法访问摄像头');
        setStage('setup');
      }
      return;
    }
    if (token !== runTokenRef.current) return;
    setStage('capturing');
    setHint(POSE_STEPS[0].instruction);

    // 实时检测节流循环
    detectTimerRef.current = setInterval(async () => {
      const tokenNow = runTokenRef.current;
      const src = sourceRef.current;
      if (!src) return;
      const frame = src.captureFrame(registrationConfig.maxHeight, registrationConfig.jpegQuality);
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
        };
        setDet(d);
        const step = POSE_STEPS[stepIndexRef.current];
        const chk = checkPose(step, { yawRatio: d.yawRatio, pitchRatio: d.pitchRatio, detScore: d.detScore, bbox: d.bbox });
        setHint(chk.hint);
        if (chk.ok && chk.capture) maybeCapture(tokenNow, step.pose);
      } catch {
        if (tokenNow === runTokenRef.current) setHint('检测请求失败,请重试');
      }
    }, registrationConfig.detectIntervalMs);
  }, []);

  const maybeCapture = useCallback((token: number, pose: PoseName) => {
    const now = Date.now();
    if (now - lastCaptureRef.current < registrationConfig.captureIntervalMs) return;
    const total = capturedRef.current.length;
    if (total >= registrationConfig.maxCapturedFrames) {
      finishCapture(token);
      return;
    }
    const src = sourceRef.current;
    if (!src) return;
    const blob = src.captureFrame(registrationConfig.maxHeight, registrationConfig.jpegQuality);
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
  }, []);

  const finishCapture = useCallback((token: number) => {
    // 停止检测循环,进入批量分析
    if (detectTimerRef.current) {
      clearInterval(detectTimerRef.current);
      detectTimerRef.current = null;
    }
    sourceRef.current?.close();
    sourceRef.current = null;
    if (token !== runTokenRef.current) return;
    runAnalyze(capturedRef.current);
  }, []);

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
        message.warning(`有效帧不足(${okCount}),已保留已采集帧,可继续采集`);
        setStage('capturing');
        // 重新开检测
        return;
      }
      // 只保留推荐帧
      const recIds = new Set(res.data.recommended_frame_ids);
      setCaptured(frames.filter((f) => recIds.has(f.frameId)));
      setStage('review');
    } catch (err: any) {
      message.error(err?.response?.data?.detail || '分析失败');
      setStage('capturing');
    }
  }, []);

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
      sourceRef.current?.close();
      sourceRef.current = null;
      capturedRef.current.forEach((f) => URL.revokeObjectURL(f.previewUrl));
    };
  }, [open]);

  const detect = det as DetState | null;

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
          <Button type="primary" icon={<VideoCameraOutlined />} onClick={startCapture} block>
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
            <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', display: 'block' }} />
            {detect && (
              <div
                style={{
                  position: 'absolute',
                  left: `${(detect.bbox[0] / 640) * 100}%`,
                  top: `${(detect.bbox[1] / 480) * 100}%`,
                  width: `${((detect.bbox[2] - detect.bbox[0]) / 640) * 100}%`,
                  height: `${((detect.bbox[3] - detect.bbox[1]) / 480) * 100}%`,
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
              onClick={() => maybeCapture(runTokenRef.current, POSE_STEPS[stepIndexRef.current].pose)}
            >
              手动抓拍
            </Button>
            <Button onClick={() => { // 跳过当前动作
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
              setCaptured([]);
              setStage('setup');
            }}>
              取消
            </Button>
            <Button type="primary" loading={false} onClick={() => finishCapture(runTokenRef.current)}>
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
