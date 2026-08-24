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
  /** 可靠 fall 事件幂等字段(可选;旧 recognition 事件为 null,保持向后兼容) */
  event_id?: string | null;
  incident_id?: string | null;
  dedupe_key?: string | null;
  occurred_at?: string | null;
  delivery_mode?: string | null;
}

export const fetchEvents = (params: Record<string, unknown>) =>
  client.get<{ items: EventItem[]; total: number }>('/events', { params });
export const acknowledgeEvent = (id: number) => client.post(`/events/${id}/acknowledge`);

/** 批量删除事件记录(ids 逗号分隔,避免 axios 数组序列化问题)。 */
export const deleteEvents = (ids: number[]) =>
  client.delete<{ deleted: number }>('/events', { params: { ids: ids.join(',') } });

/** 删除当前筛选条件下的全部事件。 */
export const deleteAllEvents = (filters: { event_type?: string; acknowledged?: boolean }) =>
  client.delete<{ deleted: number }>('/events', { params: { all: true, ...filters } });

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
