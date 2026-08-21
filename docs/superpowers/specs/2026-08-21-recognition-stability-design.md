# 视频流人脸识别稳定性优化设计

## 目标

在保持 `buffalo_l`、现有人脸库、前端接口、阈值和事件去重行为不变的前提下，避免低质量视频帧直接参与识别，并让同一条人脸轨迹通过多帧真实相似度聚合后再确认身份。

最终事件的 `confidence` 必须是同一身份候选的 Top-K 单帧相似度平均值，不通过乘系数、前端百分比或降低阈值制造更高分。

## 非目标

- 不切换 InsightFace 模型包。
- 不删除、重建或迁移现有人脸库 embedding。
- 不增加姿态估计、超分辨率或其他重型模型。
- 不修改前端 API、WebSocket 消息结构或数据库 schema。
- 不重构与稳定识别无关的任务、跟踪器或事件持久化代码。

## 现有链路

当前流水线为：

```text
视频帧 → InsightFaceEngine.detect → ByteTracker → PipelineContext.tracks
     → FaceRecognitionTask.gallery.search(单帧) → tracker.set_identity
     → VisionEvent → WebSocket / 数据库
```

`FaceRecognitionTask` 目前按冷却策略对单帧最新 embedding 检索，命中阈值后立即写回轨迹并产生事件。`det_interval` 非检测帧仍会调用任务，因此必须区分“新检测产生的 embedding”和跟踪预测帧中的旧 embedding，不能把同一向量重复计数为多帧样本。

## 设计

### 1. 配置

在 `vision.recognition` 下增加：

```yaml
quality:
  min_det_score: 0.60
  min_face_size: 80

temporal:
  min_valid_samples: 3
  max_samples_per_track: 8
  top_k: 3
```

`backend.app.config._deep_merge` 已支持递归字典合并，因此新增嵌套字段天然支持 `default.yaml → profiles/*.yaml → camera.config` 级联覆盖。`RecognitionConfig.from_dict` 负责读取嵌套字段并提供相同默认值，现有平铺识别配置保持兼容。

### 2. 新 embedding 标记

给 `TrackResult` 增加可选的 `embedding_frame_id`。`STrack` 在检测结果实际更新 embedding 时记录当前 tracker 帧号；`skip()` 预测帧只预测位置，不更新该标记。识别任务只处理尚未消费过的 embedding 帧号。

这样既保留当前任务每帧运行的接口，又避免因检测降频而重复搜索相同 embedding。没有 embedding 或没有新的 embedding 帧号时，不执行 gallery 检索，也不计入失败冷却或样本数量。

### 3. 质量筛选

识别任务在 gallery 检索前检查：

1. `track.score >= recognition.quality.min_det_score`；
2. `min(track.width, track.height) >= recognition.quality.min_face_size`；
3. `track.embedding` 存在且是新的检测帧。

任一质量条件不满足时，不检索、不加入候选分数、不确认身份、不产生识别事件。只累计对应轨迹的跳过次数和原因：`low_det_score`、`face_too_small`。本次不实现模糊度判断。

### 4. 轨迹多帧聚合

`FaceRecognitionTask` 内的 `_TrackRecState` 维护每条轨迹的临时状态：

- 现有单轨迹身份、稳定相似度、失败次数和冷却帧字段；
- 每个候选 identity 独立的分数有界队列；
- 轨迹总有效样本数不超过 `max_samples_per_track`；
- 已消费 embedding 帧号；
- 有效样本数、跳过数、跳过原因计数。

gallery 命中后，分数只写入该命中的 identity 队列；不同 identity 永远不共享队列。达到 `min_valid_samples` 的候选身份取该队列最高的 `top_k` 条分数，实际取值数量为 `min(top_k, 队列长度)`，计算算术平均值作为稳定分数。

当稳定分数低于阈值时不确认、不写回身份、不产生事件；当稳定分数达到阈值时：

- 把候选 identity 和稳定分数写回 tracker；
- 事件 `confidence` 写入稳定分数；
- payload 中保留 `similarity` 为稳定分数，并补充候选单帧分数和 Top-K 分数，供日志/调试追踪；
- 清空该轨迹临时样本窗口，后续重验证从新的窗口开始。

初始稳定窗口允许连续处理新的合格 embedding，以便在正常检测帧上快速达到最小样本数。确认后保留现有 `recognized_cooldown_frames`；没有候选样本的 Unknown 轨迹保留现有失败退避和 `max_attempts`；身份变化的事件仍由现有 `changed` 标志和后端持久化过滤控制。

轨迹从当前 `context.tracks` 消失后，任务清理该轨迹状态并输出一次摘要日志，避免 `_states` 无界增长。由于 tracker 的 lost 轨迹会在缓冲期内继续出现在快照中，清理发生在 tracker 真正移除该轨迹之后。

### 5. 可观测性与模型安全

引擎启动日志记录：

- 配置的 `model_pack`；
- 检测模型 ONNX 路径；
- 识别模型 ONNX 路径；
- 请求设备、实际设备和 Provider。

模型目录或检测/识别模型文件缺失时输出 warning，明确说明 InsightFace 可能触发自动下载；不会静默换用另一个模型包。日志不输出 embedding 向量或人脸图像。

未来可为 embedding 增加 `model_pack`、模型指纹和维度字段，防止模型切换后混用特征空间；本次只在 README/设计记录中保留该后续事项，不修改 schema 或现有数据。

### 6. 测试设计

新增/补充纯逻辑测试覆盖：

- 检测置信度低于 `min_det_score` 时不调用 gallery；
- 人脸短边低于 `min_face_size` 时不调用 gallery；
- 候选少于 `min_valid_samples` 时不确认；
- 同一身份取最高 `top_k` 分数计算平均值；
- 不同身份的样本不混合；
- 事件 `confidence` 等于稳定分数；
- 旧阈值、Unknown 失败冷却、已识别冷却和 `changed` 去重保持行为；
- 配置嵌套字段能按 default/profile/camera 级联；
- 预测帧不会重复消费同一个 embedding。

实现后运行项目全量 pytest。若环境具备 `D:\test6.mp4`、模型、数据库和后端依赖，再启动 `cam-0` 跑一轮实际视频，记录日志与事件 confidence，并显式停止摄像头；不删除历史事件或人脸库数据。

## 验收标准

1. 低质量帧不进入 gallery，不参与轨迹样本聚合，不触发识别事件。
2. 识别事件只由同一轨迹、同一候选身份的多帧真实相似度聚合确认。
3. `confidence` 可由日志中的 Top-K 分数复算。
4. 现有 API、前端、人脸库和模型包保持兼容。
5. 状态有界且轨迹消失后清理。
6. 全量单元测试通过；实际视频测试后 `cam-0` 停止。
