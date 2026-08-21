export const FRAME_PACKET_TYPE = 0x01;
export const FRAME_HEADER_SIZE = 9;

export type DecodedFramePacket = {
  frameId: number;
  jpeg: Blob;
};

export function parseFramePacket(data: ArrayBuffer): DecodedFramePacket {
  if (data.byteLength < FRAME_HEADER_SIZE) {
    throw new Error('frame packet header is incomplete');
  }

  const view = new DataView(data);
  if (view.getUint8(0) !== FRAME_PACKET_TYPE) {
    throw new Error('unsupported frame packet type');
  }

  return {
    frameId: Number(view.getBigUint64(1, false)),
    jpeg: new Blob([data.slice(FRAME_HEADER_SIZE)], { type: 'image/jpeg' }),
  };
}
