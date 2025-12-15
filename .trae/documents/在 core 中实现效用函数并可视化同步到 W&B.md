## 映射与定义
- 目标：在 `core.py` 按 `jiagou.tex` 的 OG(t) 公式，建立三类效用并记录/作图：移动机器人效用 U_i^a(t)、服务器效用 U_j^i(t)、总目标 OG(t)。
- 公式映射：
  - 卸载决策 φ_i^a(t)：`a∈{0,E_j}`，本地 `a=0`；卸载到服务器 j 则 `a=E_j`。来源：`agent.action.offload`，本地为全 0；卸载为 one-hot 指向 j。
  - 权重/功率 p_i^b(t)：使用当前实现中的 `p_weight` 映射：本地 `p_weight=1.0`，边缘 `p_weight=agent.state.trans_pow`。与 `environment.py:111-114` 的 `objective_cur=p_weight*(E+λT)` 一致。
  - U_i^a(t)：采用“满意度−成本”的规范化组合：`U_i^a(t)=θ·S_norm − (1−θ)·C_norm`，θ∈[0,1]。
  - U_j^i(t)：服务器侧对任务 i 的效用，结合完成/失败增量与负载/紧急度。

## 移动机器人效用 U_i^a(t)
- 满意度 S_norm：基于截止约束的时延满意度
  - `S_norm = clamp(1 − T_cur / τ, 0, 1)`
  - 变量：`T_cur=agent.state.time_cur`，`τ=agent.state.task_delay_tol`；若 `τ≤0`，则 `S_norm=0`。
- 成本 C_norm：按用户要求“归一化成本为每时隙服务器分配给该任务的计算资源”
  - 边缘：`C_norm_edge = allocated_resource(task_id)`，直接取 `PriorityQueueServer` 中该任务的 `task.current_comp_resource`（已是 [0,1] 权重，反映本时隙分配比例）。
  - 本地：`C_norm_local = (v_alloc · f_UE) / world.max_local_frequency`，其中 `v_alloc=agent.action.v_resource_alloc`，`f_UE=agent.freq`，归一化到 [0,1]。
- 组合：`U_i^a(t) = θ·S_norm − (1−θ)·C_norm`，θ 默认 0.5，可在 `MecWorld` 中添加 `sat_weight` 字段。
- 任务资源查询：在 `core.py` 中为 `PriorityQueueServer` 以遍历队列条目查找当前 `task_id` 的 `current_comp_resource`（无需新增文件，直接在使用处实现一个小辅助函数）。

## 服务器效用 U_j^i(t)
- 使用队列状态增量与紧急度：
  - 从 `get_queue_status()` 得到：`completed_tasks`, `failed_tasks`, `high/medium/low`；从 `_compute_queue_metrics()` 得到 `near_deadline`。
  - 用上一时隙值计算增量：`ΔC_s(t)=completed_t−completed_{t−1}`，`ΔF_s(t)=failed_t−failed_{t−1}`。
  - `Q_s(t)=high+medium+low`，`N_s(t)=near_deadline_total`。
- 定义：`U_server_s(t) = a·ΔC_s(t) − b·ΔF_s(t) − c·Q_s(t) − d·N_s(t)`，默认 `a=1,b=1,c=0.1,d=0.1`，在 `MecWorld` 中提供可调系数。
- 对应到 `U_j^i(t)`：若 agent 当前选择服务器 j，则该服务器的 `U_server_j(t)` 作为该任务的服务器侧效用贡献（其它服务器对该任务的贡献视为 0）。

## 总目标 OG(t)
- 聚合：`OG(t) = Σ_i Σ_{a∈{0,E}} φ_i^a(t) · p_i^b(t) · ( U_i^a(t) + U_j^i(t) )`
- 实现：对每个 agent 计算其当前 a（本地或选定 j），对应 `φ_i^a=1`，`p_i^b=p_weight`；累加 `U_i^a` 与该 agent 的服务器 `U_server_j`（本地时取 0）。

## 代码落点
- `onpolicy/envs/my/core.py`
  - 在 `MecWorld` 中新增：`sat_weight`、`server_util_a,b,c,d`；新增方法：`agent_utility(agent)`, `server_utility(server, last_status)`, `total_objective(agents, servers, last_statuses)`。
  - 在 `step()` 中：
    - 维护上一时隙服务器状态快照用于增量计算；
    - 计算每个 agent 的 `U_i^a`（含资源查询），每个服务器的 `U_server_s`，以及 `OG(t)`；
    - 将结果传递给 `visualizer.update(servers, time, agents, metrics)`（扩展一个 `metrics` 字典）。
- 资源查询辅助：在 `edge_cost()` 或 `step()` 内部提供函数 `get_allocated_resource(server, task_id)`，从 `server.priority_queues` 三个队列中遍历匹配 `task_id`，返回 `task.current_comp_resource`，未找到返回 0。

## 可视化与 W&B
- 扩展 `onpolicy/envs/my/visualize.py`：
  - `update(...)` 增加历史字段：`agent_utility_mean`、`server_utility[sid]`、`og_total`。
  - `render_summary()` 新增三张图：`agent_utility_over_time.png`、`server_utility_over_time.png`、`og_total_over_time.png`；加入现有 `wandb.log` 批次，受 `use_wandb` 控制。

## 验证
- 短程运行：检查本地 `visual_out/` 三图是否随 `summary_interval` 刷新；W&B 是否出现对应条目；通过切换本地/边缘动作观察 `S_norm` 与 `C_norm_edge` 行为（边缘分配增加时，成本项升高，效用下降）。