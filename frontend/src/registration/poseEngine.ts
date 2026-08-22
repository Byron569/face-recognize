/** 五步动作引导状态机:正脸 → 左转头 → 右转头 → 抬头 → 低头。
 * 判定输入来自后端 detect 返回的关键点/姿态比率(实时 detect 节流 ~400ms)。
 */
import type { PoseName } from './types';

export interface PoseCheckInput {
  yawRatio: number;      // 后端实时计算
  pitchRatio: number;
  detScore: number;
  bbox: [number, number, number, number];  // x1y1x2y2(原图坐标)
}

export type PoseCheckResult =
  | { ok: true; capture: true; hint: string }     // 动作达标,本轮可采帧
  | { ok: true; capture: false; hint: string }    // 姿态在容忍区,继续引导
  | { ok: false; capture: false; hint: string };  // 姿态不符/质量不足,提示纠正

export interface PoseStepConfig {
  pose: PoseName;
  instruction: string;        // 中文指令
  hintOk: string;             // 达标时反馈
  hintAdjust: string;         // 纠正提示
  targetYaw: [number, number];    // [min, max]
  targetPitch: [number, number];
}

export const POSE_STEPS: PoseStepConfig[] = [
  { pose: 'frontal', instruction: '请正对摄像头', hintOk: '很好,保持', hintAdjust: '请正对屏幕,双眼平视',
    targetYaw: [-0.2, 0.2], targetPitch: [-0.28, 0.28] },
  { pose: 'left',   instruction: '请缓缓向左转头', hintOk: '角度到位', hintAdjust: '再向左转一点',
    targetYaw: [-1.0, -0.22], targetPitch: [-0.5, 0.5] },
  { pose: 'right',  instruction: '请缓缓向右转头', hintOk: '角度到位', hintAdjust: '再向右转一点',
    targetYaw: [0.22, 1.0], targetPitch: [-0.5, 0.5] },
  { pose: 'up',     instruction: '请微微抬头', hintOk: '角度到位', hintAdjust: '再抬高一点',
    targetYaw: [-0.35, 0.35], targetPitch: [-1.0, -0.3] },
  { pose: 'down',   instruction: '请微微低头', hintOk: '角度到位', hintAdjust: '再低一点',
    targetYaw: [-0.35, 0.35], targetPitch: [0.3, 1.0] },
];

export function checkPose(
  step: PoseStepConfig,
  input: PoseCheckInput,
  minDetScore = 0.6,
): PoseCheckResult {
  if (input.detScore < minDetScore) {
    return { ok: false, capture: false, hint: '未检测到人脸,请靠近并正对镜头' };
  }
  const inYaw = input.yawRatio >= step.targetYaw[0] && input.yawRatio <= step.targetYaw[1];
  const inPitch = input.pitchRatio >= step.targetPitch[0] && input.pitchRatio <= step.targetPitch[1];
  if (inYaw && inPitch) {
    return { ok: true, capture: true, hint: step.hintOk };
  }
  const yawSpan = step.targetYaw[1] - step.targetYaw[0];
  const pitchSpan = step.targetPitch[1] - step.targetPitch[0];
  const nearYaw = yawSpan > 0 && Math.abs(input.yawRatio - (step.targetYaw[0] + step.targetYaw[1]) / 2) <= yawSpan * 0.5;
  const nearPitch = pitchSpan > 0 && Math.abs(input.pitchRatio - (step.targetPitch[0] + step.targetPitch[1]) / 2) <= pitchSpan * 0.5;
  if (nearYaw && nearPitch) {
    return { ok: true, capture: false, hint: step.hintAdjust };
  }
  return { ok: false, capture: false, hint: step.hintAdjust };
}

/** 动作步骤进度:每步需达标采集 required 帧数(默认 1 帧即采,留扩展)。 */
export function shouldAdvanceStep(capturedCount: number, required = 1): boolean {
  return capturedCount >= required;
}
