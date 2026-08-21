# AI Monitor 接口文档(API Reference)

> 运行时交互式文档(OpenAPI 自动生成):启动后端后访问 `http://localhost:8000/docs`(Swagger UI)或 `/redoc`。
> 本文档为语义说明与示例;字段级校验以 OpenAPI 为准。

## 约定

- 前缀:除 WebSocket 与根路径外,全部以 `/api` 开头
- 响应:JSON;错误统一为 `{"detail": "<message>"}` + 对应 HTTP 状态码
- 时间:ISO 8601(UTC,`2026-02-18T12:00:00Z`)
- 分页参数:`page`(从 1 起)、`page_size`;响应 `{items, total}`

## 1. 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 服务存活探测,返回 `{"status":"ok"}` |

## 2. 摄像头管理 `/api/cameras`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/cameras` | 全部摄像头(含运行状态与性能指标) |
| POST | `/api/cameras` | 添加摄像头 |
| GET | `/api/cameras/{id}` | 详情 |
| PUT | `/api/cameras/{id}` | 修改配置(名称/信号源/分辨率/档位/启用/个性化 config) |
| DELETE | `/api/cameras/{id}` | 删除(运行中先停) |
| POST | `/api/cameras/{id}/start` | 启动推理流水线(配置 = default → profile → 摄像头 config 级联) |
| POST | `/api/cameras/{id}/stop` | 停止流水线 |
| POST | `/api/cameras/{id}/snapshot` | 抓拍当前帧(响应 `image/jpeg`) |
| PUT | `/api/cameras/{id}/profile` | 运行时切换档位(自动重启流水线),body `{"profile": "desktop"}` |
| PUT | `/api/cameras/{id}/resolution` | 设置采集/推理分辨率与推流分辨率(运行中自动重启生效) |

**创建示例**

```json
POST /api/cameras
{
  "id": "cam0",
  "name": "大门",
  "source": "0",
  "width": 640,
  "height": 480,
  "profile": "desktop",
  "enabled": false,
  "config": {
    "vision": {
      "det_interval": 3,
      "recognition": { "threshold": 0.45 }
    }
  }
}
```

`config` 为 JSONB 个性化参数,可覆盖 default/profile 配置的任意键(级联合并)。

**分辨率设置示例**(采集/推理 + 推流,`0` = 原生/不缩放):

```json
PUT /api/cameras/cam0/resolution
{
  "capture_width": 0,          // 0 = 跟随源原生分辨率(仅本地 USB 摄像头可强制,RTSP/文件由源决定)
  "capture_height": 0,
  "stream_max_height": 480     // 推流前按高度等比缩放(0 = 不缩放)
}
```

响应:`{"updated": true, "restarted": true, "capture": "native", "stream_max_height": 480}`
`restarted` 表示该摄像头流水线已自动重启并应用新配置;`capture` 为 `"WxH"` 或 `"native"`。

预览分辨率只影响 WebSocket JPEG,不改变 InsightFace 的采集/推理分辨率。默认 480p 适合普通局域网；局域网内需要更清晰的画面时,可将 `stream.max_height` 设为 720。已有摄像头配置中的 `stream.max_height: 0`(原生)不会被自动覆盖,需在系统设置中手动调整。

## 3. 人脸库 `/api/faces`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/faces?page=&page_size=&search=` | 身份列表(分页 + 姓名模糊搜索) |
| POST | `/api/faces` | 注册新人脸(multipart:`name`、`notes`、`image` 照片) |
| GET | `/api/faces/{id}` | 身份详情 |
| PUT | `/api/faces/{id}` | 修改姓名/备注(JSON) |
| DELETE | `/api/faces/{id}` | 删除身份及其全部特征 |
| POST | `/api/faces/{id}/embeddings` | 追加特征图片(multipart `image`) |
| DELETE | `/api/faces/{id}/embeddings/{emb_id}` | 删除单条特征 |
| POST | `/api/faces/search` | 以图搜人(multipart `image` → `{identity_id, name, similarity}`) |
| POST | `/api/faces/batch-import` | 批量导入同一人多图(multipart `name` + `images[]`) |
| POST | `/api/faces/{id}/avatar` | 上传头像(multipart `image`,仅展示不提取特征) |
| POST | `/api/faces/import-pickle` | 旧版 pickle 底库一次性导入,body `{"path": "face_db/identities.pkl"}` |

**注册流程**:上传图片 → 后端用共享 InsightFace 引擎提取最大脸的 512-d embedding → 入库 → 自动刷新内存底库快照(识别热路径即时生效)。

## 4. 识别记录 `/api/recognition-logs`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/recognition-logs?page=&page_size=&camera_id=&identity_id=&start=&end=` | 分页查询识别记录 |
| GET | `/api/recognition-logs/{id}` | 单条详情 |

识别流水线在身份变化时自动写入一条记录(相似度、耗时、track_id)。

## 5. 事件 `/api/events`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/events?page=&page_size=&event_type=&camera_id=&acknowledged=&start=&end=` | 事件列表(分页 + 筛选) |
| GET | `/api/events/types` | 事件类型枚举 |
| GET | `/api/events/{id}` | 事件详情 |
| POST | `/api/events/{id}/acknowledge` | 确认告警 |

事件类型:`recognition`(识别)、`fall_detected` / `fall_potential` / `fall_recovered`(跌倒,**预留**)、`intrusion`(闯入,预留)、`loitering`(徘徊,预留)。

## 6. 系统 `/api/system`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/system/status` | CPU/内存/GPU(nvidia-smi)/摄像头数/引擎数/底库大小 |
| GET | `/api/system/metrics` | 各摄像头推理 FPS、帧数、活跃目标、各阶段耗时及预览流指标 |
| GET | `/api/system/profiles` | 部署档位清单(desktop/balanced/edge_minimal) |
| GET | `/api/system/config?profile=` | 指定档位的运行时合并配置 |
| PUT | `/api/system/config` | **预留**:运行时热更新全局配置 |

## 7. 任务注册表 `/api/tasks`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/tasks?profile=` | 配置中登记的可插拔任务(启用状态 + 实现路径) |

扩展任务(如跌倒检测)在此可见,接入方式见 `docs/PLUGIN_GUIDE.md`。

## 8. WebSocket

### `/ws/cameras/{camera_id}` — 实时视频流

**服务端 → 客户端:**

视频帧不再使用 Base64 JSON,而是使用版本化二进制协议 `binary_jpeg_v1`:

```text
byte 0     : packet type = 0x01
byte 1..8  : frame_id, unsigned 64-bit big-endian
byte 9..N  : raw JPEG bytes (`image/jpeg`)
```

随后发送同一个 `frame_id` 的检测文本消息:

```json
{ "type": "detections", "frame_id": 1234, "persons": [
    { "track_id": 1, "bbox": [x, y, w, h], "identity": "Byron", "confidence": 0.87 }
] }
{ "type": "ping" }
```

- 客户端需要将摄像头 WebSocket 的 `binaryType` 设为 `arraybuffer`,按上述头部解析二进制帧;
- JPEG 已按 `stream.max_height`(默认 480)缩放并按 `stream.jpeg_quality`(默认 70)编码;
- `detections.bbox` 为 `[x, y, w, h]`(左上角 + 宽高),前端 Canvas 直接可用;
- 推流频率由 `stream.push_fps` 控制(默认 20 FPS,服务端上限 30);
- 服务端每路摄像头只编码一次 JPEG;每个订阅者只有一个待发送槽位,慢客户端会丢弃旧帧而不会拖慢其他订阅者;
- `/api/system/metrics` 的每路 `stream` 包含 `preview_enqueue_fps`、`encoded_fps`、`sent_fps`、`encode_dropped_frames`、`subscriber_dropped_frames` 和 `avg_jpeg_bytes`;
- 无订阅者时不会执行预览 JPEG 编码,因此 `encoded_fps` 应为 0,但摄像头推理流水线仍可正常运行。

### `/ws/events` — 全局事件通道

```json
{ "type": "event", "event_type": "recognition", "camera_id": "cam0",
  "track_id": 1, "confidence": 0.87, "payload": { "name": "Byron", "identity_id": "..." }, "timestamp": 1712345678.1 }
```

## 9. 数据保留

`events` 与 `recognition_logs` 默认保留 30 天,每日 `cleanup.cron_hour`(默认 03:00)自动清理;`AIM_EVENT_RETENTION_DAYS` 可调。
