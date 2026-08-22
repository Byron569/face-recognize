import client from './client';
import type { AnalyzeResult, CommitResult, CapturedFrame } from '../registration/types';

/** 实时注册批量分析(不写库)。 */
export const analyzeRegistrationFrames = (frames: CapturedFrame[]) => {
  const fd = new FormData();
  const meta = frames.map((f) => ({ frame_id: f.frameId, timestamp_ms: f.timestampMs, pose: f.pose }));
  fd.append('metadata_json', JSON.stringify(meta));
  frames.forEach((f) => fd.append('frames', f.blob, `${f.frameId}.jpg`));
  return client.post<AnalyzeResult>('/faces/registration/analyze', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

/** 提交注册(服务端复验后原子入库)。 */
export const commitRegistrationFrames = (input: {
  mode: 'create' | 'append';
  name?: string;
  notes?: string;
  identityId?: string;
  frames: CapturedFrame[];
}) => {
  const fd = new FormData();
  fd.append('mode', input.mode);
  if (input.mode === 'create') {
    fd.append('name', input.name ?? '');
    fd.append('notes', input.notes ?? '');
  } else {
    fd.append('identity_id', input.identityId ?? '');
  }
  const meta = input.frames.map((f) => ({ frame_id: f.frameId, timestamp_ms: f.timestampMs, pose: f.pose }));
  fd.append('metadata_json', JSON.stringify(meta));
  input.frames.forEach((f) => fd.append('frames', f.blob, `${f.frameId}.jpg`));
  return client.post<CommitResult>('/faces/registration/commit', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};
