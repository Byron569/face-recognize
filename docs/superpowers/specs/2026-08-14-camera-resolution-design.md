# 摄像头分辨率设置功能 — 设计文档

日期:2026-08-14
状态:已批准(用户确认总体方案)

## 目标

1. 推理/采集分辨率可直接设置为摄像头原生分辨率(不固定 640×480),也可自定义。
2. 推流到前端的分辨率可配置(现为全局固定 `stream_max_height=480`)。
3. 前端提供设置入口,修改保存后后端立即生效(自动重启该摄像头流水线)。

## 现状(改动前的三条链路)

| 链路 | 位置 | 当前行为 |
|---|---|---|
| 采集/推理分辨率 | `pipeline_manager.start_camera` 读 `config["camera_defaults"]["width/height"]`(default.yaml 默认 640×480)→ `OpenCVFrameSource.open()` 调 `cap.set()` | 本地 USB 摄像头有效;RTSP/视频文件无效(分辨率由源决定)。另有 `max_width=960` 兜底强制缩放大帧 |
| 推流分辨率 | `_push_frame`:帧高 > `settings.stream_max_height`(默认 480)则等比缩放 | 全局 .env 配置,无前端入口 |
| DB `cameras.width/height` 列 | 仅展示用 | 前端表单未渲染这两个字段,对推理零影响 |

## 数据存储:复用 `cameras.config` JSONB(零迁移)

每摄像头 config 新增约定键(级联 default → profile → config 天然支持):

```json
{
  "camera_defaults": { "width": 0, "height": 0, "max_width": 0 },
  "stream": { "max_height": 480 }
}
```

| 键 | 含义 | 0 的语义 |
|---|---|---|
| `camera_defaults.width/height` | 采集/推理分辨率 | `0` = 跟随源原生分辨率(不调 `cap.set`) |
| `camera_defaults.max_width` | 采集兜底上限 | `0` = 不缩放(否则现有 960 兜底会缩掉 1080p 帧,原生模式失效) |
| `stream.max_height` | 推流最大高度 | `0` = 原样推流不缩放 |

## 后端改动

1. **`vision/camera.py`**:`open()` 仅当 `width > 0 && height > 0` 才调 `cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT)`;`max_width` 由调用方传入(0 = 不缩放)。
2. **`backend/app/services/pipeline_manager.py`**:
   - `start_camera`:`max_width` 从 `camera_defaults` 读取传入 frame_source。
   - `_push_frame`:缩放上限从全局 `settings.stream_max_height` 改为 per-camera `stream.max_height`(读 `config.get("stream", {}).get("max_height")`,fallback 全局默认)。`_push_frame` 是闭包,直接捕获 config。
3. **`backend/app/api/cameras.py`** 新增 `PUT /api/cameras/{id}/resolution`:
   - body:`{capture_width, capture_height, stream_max_height}`,均允许 0(原生/不缩放)。
   - 校验:0 ≤ 值 ≤ 7680;`stream_max_height` 允许 0(不缩放)。
   - 行为:合并进 `cameras.config` JSONB → 更新 DB → 若正在运行,自动 `stop → start`(与 `switch_profile` 同模式),一次完成。
4. **`backend/app/schemas/camera.py`**:新增 `CameraResolutionUpdate`。

## 前端改动

1. **`frontend/src/api/cameras.ts`**:加 `updateCameraResolution(id, data)` → `PUT /cameras/{id}/resolution`。
2. **`frontend/src/pages/SettingsPage.tsx`**:每张摄像头卡片加"分辨率"设置按钮 → Modal:
   - 推理分辨率:Radio「跟随源分辨率(原生)/ 自定义」,自定义时填 width × height。
   - 推流分辨率:Select 预设(不缩放 / 360 / 480 / 720 / 1080)。
   - 保存 → 调端点 → message 提示"已应用,流水线已自动重启"。
   - 提示文案:RTSP/视频文件分辨率由源决定,设置仅对本地 USB 摄像头生效。
   - 卡片"分辨率"行改为显示实际生效配置(从 `config` 读,带 fallback)。

## 边界与说明

- RTSP/视频文件:采集分辨率设置无效(源决定),原生模式即文件/流本身分辨率。
- 原生高分辨率时,检测输入仍走 `det_size 640×640`,推理开销基本不变;track/裁剪在高分辨率下工作,延迟略升。
- 重启期间该摄像头画面短暂中断(与现有"切换档位"一致)。
- `configs/default.yaml` 的 `camera_defaults.max_width: 0` 加入基线(文档说明),避免行为不一致。

## 验证

1. 起 backend,用视频文件摄像头:设"原生" + 推流 360 → 检查 WS 帧实际尺寸与 metrics。
2. pytest 回归(改动了 vision/camera.py)。
3. 前端 `npm run build`。
