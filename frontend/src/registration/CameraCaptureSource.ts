/** 浏览器摄像头采集源:实时取景 + 定时抓 JPEG 帧。
 * 约束:视频流仅驻留浏览器内存,close() 必须停流并释放。
 */
export class CameraCaptureSource {
  private videoEl: HTMLVideoElement;
  private stream: MediaStream | null = null;
  private canvas: HTMLCanvasElement | null = null;

  constructor(videoEl: HTMLVideoElement) {
    this.videoEl = videoEl;
  }

  /** 请求摄像头权限并开始实时取景。失败抛错由调用方展示。 */
  async open(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('当前浏览器不支持摄像头,请使用 Chrome/Edge 并通过 https 或 localhost 访问');
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    this.stream = stream;
    // 等 video 元数据就绪
    await new Promise<void>((resolve, reject) => {
      const onReady = () => {
        this.videoEl.removeEventListener('loadedmetadata', onReady);
        resolve();
      };
      const onErr = () => {
        this.videoEl.removeEventListener('error', onErr);
        reject(new Error('无法读取摄像头画面'));
      };
      this.videoEl.addEventListener('loadedmetadata', onReady);
      this.videoEl.addEventListener('error', onErr);
    });
    this.videoEl.srcObject = stream;
    await this.videoEl.play().catch(() => { /* autoPlay 拦截会有 muted playsInline,见调用方 */ });
  }

  /** 抓取一帧 JPEG(等比缩至 maxHeight),未就绪返回 null。 */
  captureFrame(maxHeight = 720, quality = 0.85): Blob | null {
    const vw = this.videoEl.videoWidth;
    const vh = this.videoEl.videoHeight;
    if (!vw || !vh) return null;
    const scale = Math.min(1, maxHeight > 0 ? maxHeight / vh : 1);
    const w = Math.max(1, Math.round(vw * scale));
    const h = Math.max(1, Math.round(vh * scale));
    if (!this.canvas) this.canvas = document.createElement('canvas');
    this.canvas.width = w;
    this.canvas.height = h;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(this.videoEl, 0, 0, w, h);
    const dataUrl = this.canvas.toDataURL('image/jpeg', quality);
    // toDataURL → Blob
    const bin = atob(dataUrl.split(',')[1]);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: 'image/jpeg' });
  }

  /** 停流并释放资源。 */
  close(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.videoEl.srcObject = null;
    this.canvas = null;
  }
}
