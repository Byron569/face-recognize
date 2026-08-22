/** 系统监控摄像头帧源:WS 推流帧(预览/检测) + snapshot 原始帧(注册采帧)。
 * WS 连接由调用方(useWebSocket hook)建立,通过 offer() 喂入二进制帧。
 */
import { parseFramePacket } from '../stream/frameProtocol';
import { snapshotCamera } from '../api/cameras';

export class SystemCameraSource {
  private cameraId: string;
  /** 最新一帧(WS 推流,可能被缩放/压缩,用于预览与姿态检测)。 */
  latestFrame: { frameId: number; blob: Blob } | null = null;
  /** 上一次已送检的 frameId,避免同一帧重复 detect。 */
  lastAnalyzedFrameId = -1;
  private closed = false;

  constructor(cameraId: string) {
    this.cameraId = cameraId;
  }

  /** 由 WS onMessage 调用:解析二进制包并保存最新帧。 */
  offer(data: ArrayBuffer): void {
    if (this.closed) return;
    try {
      const pkt = parseFramePacket(data);
      this.latestFrame = { frameId: pkt.frameId, blob: pkt.jpeg };
    } catch {
      // 忽略畸形包
    }
  }

  /** 取一帧用于姿态检测(WS 最新帧);与上次送检相同帧返回 null。 */
  getDetectFrame(): Blob | null {
    const f = this.latestFrame;
    if (!f || f.frameId === this.lastAnalyzedFrameId) return null;
    this.lastAnalyzedFrameId = f.frameId;
    return f.blob;
  }

  /** 高质量采帧(原始分辨率 JPEG,quality 90),用于注册入库。摄像头未运行抛错。 */
  async captureFrame(): Promise<Blob | null> {
    if (this.closed) return null;
    const res = await snapshotCamera(this.cameraId);
    return res.data as Blob;
  }

  close(): void {
    this.closed = true;
    this.latestFrame = null;
  }
}
