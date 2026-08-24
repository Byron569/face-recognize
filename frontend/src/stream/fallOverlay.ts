/**
 * stage10 (M2): 跌倒姿态叠加。
 *
 * 仅按后端给出的 preview 像素坐标直接绘制,不二次缩放。点击前先校验 analytics
 * 的 preview_frame_id 与已解码 JPEG 帧号一致、bitmap 尺寸与 preview 尺寸一致,
 * 不一致则拒绝绘制(防止把 source 帧结果绑到错误 preview 帧)。Canvas overlay
 * 的 TTL 到期由调用方以单一可取消 timer 清除。
 */
import { FallAnalytics, FallTrack } from './analyticsProtocol';

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
  const color = track.state === 'fallen' ? '#ff4d4f' : track.state === 'potential' ? '#fa8c16' : 'transparent';
  const bbox = track.bbox;
  if (bbox) {
    const [x, y, w, h] = bbox;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    if (track.poseTrackId !== null) {
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      const label = `#${track.poseTrackId} ${track.state}`;
      ctx.font = '13px sans-serif';
      const tw = ctx.measureText(label).width;
      ctx.fillRect(x, y - 18, tw + 8, 18);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + 4, y - 4);
    }
  }
  if (track.keypoints) {
    ctx.fillStyle = color;
    for (const [kx, ky] of track.keypoints) {
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