import client from './client';

export interface EventItem {
  id: number;
  event_type: string;
  camera_id: string;
  track_id: number | null;
  identity_id: string | null;
  identity_name: string | null;
  confidence: number;
  payload: Record<string, unknown>;
  snapshot_path: string | null;
  acknowledged: boolean;
  acknowledged_at: string | null;
  created_at: string;
}

export const fetchEvents = (params: Record<string, unknown>) =>
  client.get<{ items: EventItem[]; total: number }>('/events', { params });
export const acknowledgeEvent = (id: number) => client.post(`/events/${id}/acknowledge`);

/** 批量删除事件记录(ids 传数组)。 */
export const deleteEvents = (ids: number[]) =>
  client.delete<{ deleted: number }>('/events', { params: { ids } });

/** 事件类型枚举(用于筛选下拉框)。 */
export const fetchEventTypes = () => client.get<{ types: string[] }>('/events/types');

export interface RecognitionLog {
  id: number;
  camera_id: string;
  identity_id: string | null;
  track_id: number;
  similarity: number;
  latency_ms: number | null;
  created_at: string;
}

export const fetchRecognitionLogs = (params: Record<string, unknown>) =>
  client.get<{ items: RecognitionLog[]; total: number }>('/recognition-logs', { params });
