## 问题

* 报错源于 `agent.state.task_e_s` 为 None，在首次卸载时调用 `submit_offload_task` 取 `float(None)` 崩溃。

* 原因：`MecAgent` 在 `__init__` 时已创建 `agent.task` 且 `_state=0`，导致 `update_agent_task_state` 的条件未触发，未把任务参数同步到 `agent.state`。

## 方案

* 修改 `update_agent_task_state`：无论是否新建任务，只要缺少状态字段或任务处于初始状态，均将 `agent.task` 的 `input_data/exe_data/delay_tol/type` 同步到 `agent.state`。

* 保持已有的“任务结束后才计算效用与总目标”的逻辑不变。

## 实施点

* `onpolicy/envs/my/core.py`：

  * 将 `update_agent_task_state(agent)` 改为：

    1. 若任务为空或已结束，则新建任务；
    2. 始终把 `agent.task` 的字段写入 `agent.state`（`task_i_s/task_e_s/task_delay_tol/task_type/task_q_left`）。

## 验证

* 运行训练首步卸载，确保不再出现 NoneType 转换报错。

* 检查 `agent.state.task_e_s` 在首次 step 前后均为有效数值。

## 影响

* 仅初始化与同步任务状态，不改变奖励/效用与可视化逻辑；避免首次卸载崩溃。

