## 目标
- 为每个服务器生成单独的时序折线图，展示低/中/高三类队列长度随时间变化；本地覆盖保存并同步到 W&B。保留负载热力图与本地保存。

## 输出
- 每服务器：`visual_out/server_{sid}_queues_levels.png`，三条线：`high/medium/low`。
- 全局：保留 `visual_out/load_heatmap.png`，展示各服务器的 `num_offload` 时序分布（最近窗口）。
- W&B：在每个 `summary_interval` 时，将上述图以 `wandb.Image` 上传，并使用 `step = last_time_slot`。

## 数据来源
- 队列数据：`get_queue_status()` 的 `high/medium/low`。
- 负载数据：`server.state.num_offload`（环境中每步已写入）。

## 修改点
- 文件：`onpolicy/envs/my/visualize.py`
  - 在 `render_summary()` 中遍历服务器，生成并覆盖保存每服务器三线折线图；将路径加入 W&B 日志。
  - 保持现有热力图输出与本地保存；保留既有聚合折线图（不强制显示，可保留以便总览）。

## 触发与频率
- 仍由 `summary_interval` 控制；每到间隔时更新所有图片并上传，覆盖旧图，磁盘占用恒定。

## 兼容性
- 不改动队列算法与 Runner；仅在可视化层扩展输出。