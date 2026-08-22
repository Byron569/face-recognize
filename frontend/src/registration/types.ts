/** 摄像头注册域类型。 */

/** 摄像头采集帧(浏览器侧持有,含预览 URL)。 */
export interface CapturedFrame {
  frameId: string;
  timestampMs: number;
  pose: PoseName;            // 采集时所属动作步骤
  blob: Blob;                // JPEG
  previewUrl: string;        // Object URL(审核页缩略图)
  /** 采帧时的姿态比率(review 卡片显示 + 同姿态重复帧拦截)。 */
  yawRatio?: number;
  pitchRatio?: number;
  /** analyze 质量结论(审核页展示/提交门槛用)。 */
  accepted?: boolean;
  reason?: string | null;          // 拒绝原因 token
  qualityScore?: number | null;
}

export type PoseName = 'frontal' | 'left' | 'right' | 'up' | 'down';

/** 后端单帧分析结果(与 RegistrationFrameAnalysisOut 对齐)。 */
export interface FrameAnalysis {
  frame_id: string;
  timestamp_ms: number;
  accepted: boolean;
  reason: string | null;
  pose: string | null;
  bbox: number[] | null;
  det_score: number | null;
  yaw_ratio: number | null;
  pitch_ratio: number | null;
  blur_score: number | null;
  quality_score: number | null;
}

export interface AnalyzeResult {
  sampled_count: number;
  accepted_count: number;
  recommended_frame_ids: string[];
  frames: FrameAnalysis[];
}

export interface CommitResult {
  mode: 'create' | 'append';
  identity_id: string;
  embedding_count_added: number;
}

/** 视频源类型。 */
export type SourceKind = 'device' | 'system';

/** 选中的视频源描述(序列化进 localStorage)。 */
export interface SelectedSource {
  kind: SourceKind;
  /** kind=device:设备 deviceId,'__default__' 表示不指定(浏览器默认) */
  deviceId: string;
  /** kind=system:后端摄像头 id */
  cameraId: string;
}
