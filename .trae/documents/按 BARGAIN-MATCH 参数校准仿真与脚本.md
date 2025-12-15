## 目标
- 参照 BARGAIN-MATCH 在 VEC 场景中的常用仿真设置，对代码中的关键参数做统一校准，使任务规模、服务器算力、时隙长度与时延容忍度处在合理量级，避免“秒失败”。
- 同步更新演示脚本的任务分布与时隙配置，使图片中的 near/权重/完成率更贴近论文设定下的直觉表现。

## 参考依据
- BARGAIN-MATCH 公开版本（arXiv/IEEE Xplore），采用 RSU 侧 VEC 服务器、LTE/5G 常见带宽与噪声模型、车辆发射功率量级；仿真设置通常在 10–20 MHz 带宽、-114 dBm 噪声、0.1s 或更细的时隙尺度，以及边缘服务器 10–20 GHz 计算能力范围。
- 参考文献：
  - [1] BARGAIN-MATCH: A Game Theoretical Approach for Resource Allocation and Task Offloading in Vehicular Edge Computing Networks, arXiv:2203.14064（HTML/PDF）

## 拟修改项（参数映射）
- 时间与算力：
  - `PriorityQueueServer.slot_time` → 0.05 s（更细时隙，提升推进粒度）
  - `MecServer.freq` → 20e9 cycles/s（服务器算力上调，匹配重负载场景）
- 通信：
  - `MecServer.bandwidth`（或 `MecWorld.bandwidth`）→ 20e6 Hz（20 MHz 带宽）
  - `MecWorld.noise_in_DBm` → -114 dBm（与 10–20 MHz 下常见噪声量级一致）
  - `AgentState.trans_pow` → 0.2 W（车辆发射功率在 0.1–1 W 量级）
- 任务生成与单位：
  - 任务输入与执行规模：保持与 `core.Task` 分布一致，但将提交到队列的 `computation_requirement` 做比例缩放（例如乘以 `alpha_comp`），以对齐 `cur_comp` 公式推进量级；
  - 截止期：`delay_tol ~ U(3.0, 10.0)` 秒 → `max_tolerance_delay = ceil(delay_tol / slot_time)`（保证 60–200 时隙窗口，避免“秒失败”）。
- 权重与优先级：
  - 保持“仅紧急度分数”进行优先级分类不变；
  - 继续使用近截止与积压联合的权重切分（闭环编排），以体现动态资源调度。

## 代码改动位置
- `onpolicy/envs/my/task_queue.py`
  - `PriorityQueueServer.slot_time`（常量）
- `onpolicy/envs/my/core.py`
  - `MecServer.freq`、`MecServer.bandwidth`、`AgentState.trans_pow`、`MecWorld.noise_in_DBm`
  - `MecServer.submit_offload_task`：为 `computation_requirement` 增加缩放因子 `alpha_comp`
- `onpolicy/envs/my/multi_server_demo.py`
  - 任务 `delay_tol` 分布上调为 `U(3.0,10.0)`；
  - 根据新 `slot_time` 自动换算 `max_tolerance_delay`

## 验证
- 运行 `multi_server_demo.py`：3–5 台服务器，50–100 时隙；检查 `visual_out/slot_*.png` 中：
  - near 与权重随负载的变化；
  - 完成/失败累计在合理区间（例如完成率 50–80%），若偏差则微调 `alpha_comp`、`freq`、`delay_tol` 分布。

## 说明
- 若后续希望严格复刻论文中的精确数值（如特定车速、路径损耗模型参数、任务到达率等），可在得到具体章节参数后继续细化；当前按论文通行量级做一致性校准，能显著改善演示与动态图片的可解释性。