## 目标
- 按 `word/jiagou.tex` 中的架构，在服务器侧实现“多级优先级队列 + 复合优先级分数 + 闭环动态资源编排”。
- 将 `onpolicy/envs/my/task_queue.py` 的优先级队列与 `onpolicy/envs/my/core.py` 的 MEC 服务器、卸载与执行流程打通，实现端边协同下的任务入队、分配与执行反馈。

## 现有逻辑梳理
- `task_queue.py` 提供多级队列与按计算量比例的资源分配：
  - 任务数据结构 `ComputeTask`（`onpolicy/envs/my/task_queue.py:23`）
  - 紧迫度评分与优先级判定 `TaskPriorityEvaluator`（`onpolicy/envs/my/task_queue.py:42-84`）
  - 多级队列与时隙处理 `PriorityQueueServer`（`onpolicy/envs/my/task_queue.py:86-349`）
- `core.py` 中的 MEC 环境：
  - 服务器 `MecServer` 已持有 `PriorityQueueServer` 实例（`onpolicy/envs/my/core.py:97-98`）
  - 智能体任务生成与卸载时延/能耗计算（`onpolicy/envs/my/core.py:245-314`）
  - 连接关系与带宽共享（`onpolicy/envs/my/core.py:317-338`）

## 改造思路
- 接入点：将智能体的“边缘卸载”改为“提交任务到服务器优先级队列 + 服务器按时隙推进执行”。
- 资源编排：引入跨队列的动态权重 `w_H/w_M/w_L`（高/中/低）与复合优先级分数，兼顾紧迫度与资源需求；根据服务器负载与队列饥饿情况闭环调整下一时隙的分配比例。
- 反馈通路：服务器执行结束后，更新任务完成/失败状态，并向环境（agent/server）写回统计，用于下一轮策略与资源分配。

## 具体改动
### 1. 复合优先级分数
- 在 `TaskPriorityEvaluator.determine_priority` 中将 `composite_score = α·urgency + β·norm(resource_demand)`，预留 `γ·server_load/CSI` 扩展接口（默认 γ=0）。
- 资源需求取 `remaining_computation` 归一化；阈值仍按三档（高/中/低）。

### 2. 跨队列动态配比
- 在 `PriorityQueueServer._allocate_resources` 改为：
  - 计算队列指标：各队列任务数、近截止任务比例、高优先级积压、上一时隙完成率。
  - 生成 `w_H/w_M/w_L`，例如：`w_H ∝ 近截止比例 + backlog`，`w_M`、`w_L`做归一化并保底最小权重，避免饥饿。
  - 将 `total_computation_power` 按 `w_*` 切分成队列级资源池；对每个队列内部继续用现有“按计算量比例”分配。

### 3. 任务提交（core → queue）
- 在 `MecServer` 增加 `submit_offload_task(agent: MecAgent)`：把 `agent.task` 映射为 `ComputeTask`：
  - `task_id`=`f"{agent.id}-{world.time}"`
  - `agent_id`=`agent.id`
  - `computation_requirement`：由 `task.exe_data` 映射（与 `task_queue` 的单位一致，建议 `exe_data`→“待执行数据量”）
  - `max_tolerance_delay`：由 `task.delay_tol` 换算为时隙数（`ceil(delay_tol / slot_time)`）
  - `creation_slot`=`priority_server.current_slot`
  - 调用 `priority_server.add_task(task)`

### 4. 世界时步推进服务器执行
- 在 `MecWorld.step` 最后为每个 `server` 调用 `server.priority_server.process_time_slot()`，并将 `slot_time` 与环境步长对齐（或保持 0.1s，一步推进一个时隙）。
- 将完成/失败列表用于更新统计与调试输出；必要时把结果写回 `server.state.num_offload` 或 agent 侧记录。

### 5. 卸载成本与传输
- 在 `edge_cost` 中保留传输计算（`trans_t/trans_e`）；执行时间由队列调度决定，不再用 `exe_t = task.exe_data / (...)` 的闭式公式。
- `edge_cost` 返回的 `t_cost/e_cost` 可改为“仅传输成本 + 上下文转换”，把计算时延交由服务器在任务完成时累计产出；训练阶段可选择用“期望执行时间估计”占位。

### 6. 指标与调试
- 增加 `PriorityQueueServer.get_queue_status()` 的扩展信息：各队列近截止任务数、动态权重；打印时显示当前权重与分配结果。
- 在 `MecServer` 将 `state.num_offload` 与队列长度关联，便于策略观测。

## 数据与单位对齐
- `PriorityQueueServer._process_tasks` 当前用 `cur_comp = (time * total_computation_power / (8 * 10e6)) * allocated_resource` 推进进度；保持该公式，统一把 `ComputeTask.computation_requirement` 视为同一“任务量单位”。
- 将 `agent.task.exe_data` 按同样单位喂给 `computation_requirement`（建议使用简单比例映射，初版先线性映射，后续可校准）。

## 代码挂点（便于修改）
- `onpolicy/envs/my/task_queue.py`
  - 复合分数与三档阈值：`TaskPriorityEvaluator.determine_priority`（`onpolicy/envs/my/task_queue.py:71-83`）
  - 跨队列权重与资源切分：`PriorityQueueServer._allocate_resources`（`onpolicy/envs/my/task_queue.py:181-201`）
- `onpolicy/envs/my/core.py`
  - 服务器持有队列：`MecServer.__init__`（`onpolicy/envs/my/core.py:97-98`）
  - 任务提交：新增 `MecServer.submit_offload_task(...)`（建议放在 `MecServer` 内）
  - 卸载成本：`MecWorld.edge_cost`（`onpolicy/envs/my/core.py:268-314`）改为提交任务并仅计算传输相关成本
  - 时步推进：`MecWorld.step`（`onpolicy/envs/my/core.py:197-206`）末尾处理服务器时隙

## 验证与可视化
- 构造若干代理在多服务器环境下，固定/随机生成任务，观察队列权重变化与完成率、超时率。
- 关键指标：任务超时率、平均时延、平均能耗、队列饥饿度；对比“FIFO/仅紧迫度/复合分数+闭环”的消融。

## 交付内容
- 更新两处文件的实现与接口，保持最小侵入式变更。
- 提供一个最小模拟脚本或环境运行入口，打印每时隙的队列权重与完成任务列表，用于检查资源分配逻辑生效。