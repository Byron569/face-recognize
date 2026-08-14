# 扩展任务开发指南(Plugin Guide)

系统采用**可插拔视觉任务**架构:任何视觉分析能力(跌倒检测、徘徊检测、人数统计……)
只需实现一个任务类并登记配置,无需改动主循环、路由与前端。

## 1. 任务接口

任务基类定义在 `vision/tasks.py`:

```python
class VisionTask(ABC):
    name: str = "unnamed"
    enabled: bool = True
    interval: int = 1

    def __init__(self, config: dict = None): ...          # 配置注入
    def should_run(self, frame_id: int, context: PipelineContext) -> bool: ...
    def run(self, frame, context: PipelineContext) -> list[VisionEvent]: ...
    def close(self) -> None: ...                          # 释放资源(线程/模型)
```

`PipelineContext` 提供:`camera_id`、`frame_id`、`frame`(BGR ndarray)、`tracks`(活跃 TrackResult 列表,含 bbox/identity/embedding)。

## 2. 三步接入(以跌倒检测为例)

### 第一步:写任务类

```python
# backend/app/tasks/builtin/fall_detection_task.py
from vision.tasks import VisionTask
from vision.events import VisionEvent, PipelineContext

class FallDetectionTask(VisionTask):
    name = "fall_detection"

    def __init__(self, config=None, full_config=None, **kwargs):
        super().__init__(config)
        # 从 full_config 读取任务专属参数(如模型路径/阈值),全部走配置
        self._model_path = (full_config or {}).get("fall_detection", {}).get("model_path")

    def should_run(self, frame_id, context):
        return frame_id % self.interval == 0 and bool(context.tracks)

    def run(self, frame, context):
        events = []
        # ...对 frame/context.tracks 做姿态推理与判断...
        events.append(VisionEvent(
            event_type="fall_detected",
            camera_id=context.camera_id,
            track_id=context.tracks[0].track_id,
            confidence=0.87,
            payload={"name": context.tracks[0].identity},
        ))
        return events
```

### 第二步:登记配置

```yaml
# configs/default.yaml
tasks:
  face_recognition:
    enabled: true
    class_path: "app.tasks.builtin.face_recognition_task.FaceRecognitionTask"
  fall_detection:
    enabled: true                                    # ← 开启
    class_path: "app.tasks.builtin.fall_detection_task.FallDetectionTask"
    interval: 5

fall_detection:                                      # ← 任务专属配置节
  model_path: "models/yolov8n-pose.onnx"
  threshold: 0.5
```

### 第三步:重启流水线

切换摄像头档位或重启服务,`TaskRegistry` 按 `class_path` 动态加载任务。

**主循环、REST 路由、前端页面零改动。** 事件自动进入:
- `/ws/events` 实时推送(前端 AlertBanner 已订阅,eventMeta 配置中补一条元数据即可显示);
- `events` 表持久化(EventBridge 通用落库);
- 事件类型枚举已预留 `fall_detected / fall_potential / fall_recovered / intrusion / loitering`。

## 3. 任务开发约定

1. **不阻塞主循环**:重型推理放子线程/进程,任务只提交与收割(参照识别任务的冷却调度思路);
2. **状态按摄像头隔离**:每个摄像头有独立任务实例,状态存实例内,勿用模块级全局;
3. **参数只从 config 读**:`full_config` 是 default → profile → camera.config 级联后的完整配置;
4. **事件去重**:状态变化才发事件(如跌倒状态 Normal → FALL),避免刷屏;
5. **异常隔离**:主循环已对任务 try/except,单任务崩溃不影响流水线。

## 4. 任务注入参数参考

`TaskRegistry.load()` 会按构造器签名注入以下依赖(任务按需声明):

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | dict | 任务自身配置节(如 tasks.fall_detection) |
| `full_config` | dict | 摄像头完整级联配置 |
| `gallery` | FaceGallery | 内存人脸底库(供识别类任务复用) |
| `tracker` | ByteTracker | 跟踪器引用(可写回身份等) |

> 参照 `backend/app/tasks/builtin/face_recognition_task.py` —— 它是完整的最佳实践示例。
