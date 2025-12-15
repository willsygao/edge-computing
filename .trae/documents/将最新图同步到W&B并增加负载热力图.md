## 目标
- 在每个 summary_interval 生成的最新折线图同步到 W&B，并增加负载均衡热力图以直观展示服务器间的负载分布。

## 数据来源
- 负载数据：`server.state.num_offload`（每时隙环境中已计算），队列长度：`get_queue_status()` 的 `high/medium/low` 合计。
- 时间戳：`time_slot` 由可视化器的历史缓存记录并在渲染时使用。

## 改动点
- 文件：`onpolicy/envs/my/visualize.py`
  - 扩展 `QueueVisualizer.__init__(..., use_wandb: bool = True)` 与内部状态 `self._last_time`。
  - 在 `update(servers, time_slot)` 中附加记录 `num_offload` 至 `self.history[sid]['offload']`，更新 `self._last_time = time_slot`。
  - 在 `render_summary()`：
    - 生成并覆盖保存：`queues_over_time.png`、`completion_failure_over_time.png`。
    - 新增热力图：`load_heatmap.png`（Y轴服务器ID，X轴时间，值为每时隙 `num_offload` 或其滑动平均）。为降低尺寸，仅绘制最近 `M` 个时隙（默认 `M=500`）或按 `summary_interval` 取样。
    - 若 `use_wandb` 为真，使用 `wandb.log({'queues_over_time': wandb.Image(path1), 'completion_failure_over_time': wandb.Image(path2), 'load_heatmap': wandb.Image(path3)}, step=self._last_time)` 同步到 W&B。
  - 图标题附带简要均衡性指标（如最近窗口的服务器间 `num_offload` 方差或变异系数）。

## 输出频率与覆盖策略
- 仅在 `time_slot % summary_interval == 0` 时生成图并覆盖同名文件；停止训练时自动保留最新图并已同步到 W&B。

## 兼容性
- 不改动队列调度与 Runner；保持现有折线图输出逻辑，仅增加 W&B 同步与热力图。

## 验证
- 运行若干时隙后，W&B 中应出现三张最新图片；本地仅保留固定文件名的最新图；热力图能直观显示各服务器随时间的负载分布与均衡程度。