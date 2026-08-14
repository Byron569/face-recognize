import client from './client';

export interface Identity {
  id: string;
  name: string;
  avatar_path: string | null;
  notes: string;
  embedding_count: number;
  created_at: string;
}

export const fetchFaces = (page: number, search: string) =>
  client.get<{ items: Identity[]; total: number }>('/faces', { params: { page, search } });
export const deleteFace = (id: string) => client.delete(`/faces/${id}`);
export const updateFace = (id: string, data: Record<string, unknown>) => client.put(`/faces/${id}`, data);
export const addFaceEmbedding = (id: string, formData: FormData) =>
  client.post(`/faces/${id}/embeddings`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
export const createFace = (formData: FormData) =>
  client.post('/faces', formData, { headers: { 'Content-Type': 'multipart/form-data' } });

/** 上传照片预览检测:返回图片尺寸与人脸框坐标(不入库,GPU 推理)。 */
export const detectFace = (formData: FormData) =>
  client.post<{
    width: number;
    height: number;
    faces: Array<{ bbox: number[]; det_score: number }>;
  }>('/faces/detect', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

/** 图片识别搜索(返回最匹配的身份)。 */
export const searchFace = (formData: FormData) =>
  client.post<{ identity_id: string | null; name: string | null; similarity: number }>('/faces/search', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

/** 批量导入人脸(同一个人多张图)。 */
export const batchImportFaces = (formData: FormData) =>
  client.post<{ name: string; total: number; identity_id: string | null; results: Array<{ file: string; status: string; reason?: string }> }>(
    '/faces/batch-import', formData, { headers: { 'Content-Type': 'multipart/form-data' } }
  );

/** 删除单条 embedding。 */
export const deleteFaceEmbedding = (faceId: string, embId: string) =>
  client.delete(`/faces/${faceId}/embeddings/${embId}`);

/** 上传头像(不重新提取 embedding)。 */
export const uploadAvatar = (faceId: string, formData: FormData) =>
  client.post<{ avatar_path: string }>(`/faces/${faceId}/avatar`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
