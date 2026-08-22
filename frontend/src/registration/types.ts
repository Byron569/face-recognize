/** 摄像头注册域类型。 */

/** 摄像头采集帧(浏览器侧持有,含预览 URL)。 */
export interface CapturedFrame {
  frameId: string;
  timestampMs: number;
  pose: PoseName;            // 采集时所属动作步骤
  blob: Blob;                // JPEG
  previewUrl: string;        // Object URL(审核页缩略图)
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
