## 目标
- 统一控制队列可视化的 W&B 上传逻辑，使其受 `config.use_wandb` 开关控制，关闭时仅本地保存不上传。

## 修改点
- 在 `onpolicy/envs/my/scenarios/simple_spread.py` 中创建 `MecWorld` 后，重建 `QueueVisualizer` 并传入 `use_wandb=args.use_wandb`，覆盖默认构造。
- 其他逻辑不变：本地折线图与热力图仍覆盖保存；W&B 上传仅在开关为真时发生。

## 验证
- 设置 `--use_wandb` 为 False：仅本地生成并覆盖保存图像，不上传。
- 设置 `--use_wandb` 为 True：按间隔上传最新图到 W&B，并覆盖保存本地图。