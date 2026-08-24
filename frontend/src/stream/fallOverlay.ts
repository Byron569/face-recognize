/**
 * stage10 (M2): 跌倒姿态叠加。
 *
 * 仅按后端给出的 preview 像素坐标直接绘制,不二次缩放。点击前先校验 analytics
 * 的 preview_frame_id 与已解码 JPEG 帧号一致、bitmap 尺寸与 preview 尺寸一致,
 * 不一致则拒绝绘制(防止把 source 帧结果绑到错误 preview 帧)。Canvas overlay
 * 的 TTL 到期由调用方以单一可取消 timer 清除。
 */
import { FallAnalytics, FallTrack } from './analyticsProtocol';

/** COCO-17 骨骼连接(左右各一次;数字为 keypoints 下标)。 */
const COCO17_BONES: ReadonlyArray<[number, number]> = [
  [0, 1], [0, 2], // 鼻→左/右眼
  [1, 3], [2, 4], // 眼→耳
  [0, 5], [0, 6], // 鼻→肩
  [5, 7], [7, 9], // 左臂
  [6, 8], [8, 10], // 右臂
  [5, 11], [6, 12], // 肩→髋
  [11, 12], // 髋
  [11, 13], [13, 15], // 左腿
  [12, 14], [14, 16], // 右腿
];

/** 各姿态状态的人体框/骨骼配色:正常(青)、潜在(橙)、跌倒(红)。 */
function stateColor(state: FallTrack['state']): string {
  if (state === 'fallen') return '#ff4d4f';
  if (state === 'potential') return '#fa8c16';
  return '#00c2a8';
}

export interface OverlayCompatibility {
  ok: boolean;
  reason: string | null;
}

/** 校验 analytics 是否可绘制到给定帧与 bitmap 尺寸上。 */
export function isOverlayCompatible(
  analytics: Pick<FallAnalytics, 'previewFrameId' | 'previewWidth' | 'previewHeight'>,
  frameId: number,
  bitmapWidth: number,
  bitmapHeight: number,
): OverlayCompatibility {
  if (analytics.previewFrameId !== frameId) {
    return { ok: false, reason: `frame id mismatch ${analytics.previewFrameId} != ${frameId}` };
  }
  if (bitmapWidth !== analytics.previewWidth || bitmapHeight !== analytics.previewHeight) {
    return {
      ok: false,
      reason: `bitmap size (${bitmapWidth}x${bitmapHeight}) != preview (${analytics.previewWidth}x${analytics.previewHeight})`,
    };
  }
  return { ok: true, reason: null };
}

/** 绘制一条 fall track 到 canvas(数据坐标 = preview 像素)。 */
export function drawFallTrack(
  ctx: CanvasRenderingContext2D,
  track: FallTrack,
  previewWidth: number,
): void {
  const color = stateColor(track.state);
  const kps = track.keypoints;
  // 骨骼连线(COCO-17 下标),跳过缺失或落在原点的未激活关键点
  if (kps && kps.length > 1) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    for (const [a, b] of COCO17_BONES) {
      const pa = kps[a];
      const pb = kps[b];
      if (!pa || !pb) continue;
      if ((pa[0] === 0 && pa[1] === 0) || (pb[0] === 0 && pb[1] === 0)) continue;
      ctx.beginPath();
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
      ctx.stroke();
    }
  }
  const bbox = track.bbox;
  if (bbox) {
    const [x, y, w, h] = bbox;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    // 仅显著状态(潜在/跌倒)打标签,避免 normal 画面信息过载
    if (track.poseTrackId !== null && track.state !== 'normal') {
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      const label = `#${track.poseTrackId} ${track.state}`;
      ctx.font = '13px sans-serif';
      const tw = ctx.measureText(label).width;
      ctx.fillRect(x, y - 18, tw + 8, 18);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + 4, y - 4);
    }
  }
  if (kps) {
    ctx.fillStyle = color;
    for (const [kx, ky] of kps) {
      if (kx === 0 && ky === 0) continue;
      ctx.beginPath();
      ctx.arc(kx, ky, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

/**
 * 以整块方式绘制 analytics 的全部骨骼;兼容通过后按 preview 像素直接绘制。
 * @returns false 表示该 analytics 不该画(返回原因),true 表示已绘制。
 */
export function drawFallAnalytics(
  ctx: CanvasRenderingContext2D | null,
  analytics: FallAnalytics | undefined,
  frameId: number,
  bitmapWidth: number,
  bitmapHeight: number,
): OverlayCompatibility {
  if (!analytics) return { ok: false, reason: 'no analytics' };
  const check = isOverlayCompatible(analytics, frameId, bitmapWidth, bitmapHeight);
  if (!check.ok || ctx === null) return check;
  for (const track of analytics.tracks) {
    drawFallTrack(ctx, track, analytics.previewWidth);
  }
  return check;
}

/** 检测是否达到"延迟/不可用"展示条件。 */
export function isStale(analytics: Pick<FallAnalytics, 'overlayExpiresInMs'> | undefined): boolean {
  return analytics !== undefined && analytics.overlayExpiresInMs !== null && analytics.overlayExpiresInMs <= 0;
}