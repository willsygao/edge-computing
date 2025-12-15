## 目标
- 在MAPPO训练的每一步依据 offload 动作选择规则计算OG(t)，并将相关指标上传到W&B。

## 动作选择规则
- 依据用户指定：`offload_s_id = np.argmax(agent.action.offload) + 1`（`core.py:297`）确定卸载目标服务器。
- 辅助判定：若 `sum(agent.action.offload) == 0` 则为本地执行；否则为边缘到 `offload_s_id`。

## OG(t) 定义与映射
- 每智能体 i 的当前步目标：
  - 选择指示 φ_i^a(t)：
    - local：`φ=1, a=local`；edge：`φ=1, a=edge_{offload_s_id}`。
  - 权重 p_i^b(t)：local=1；edge=`agent.state.trans_pow`。
  - 成本 U_i^a(t)：`U = energy_cur + λ·time_cur`，其中 `λ = world.trade_lambda`（`simple_spread.py:15`）。
  - 目标：`OG_i(t) = φ · p · (energy_cur + λ·time_cur)`。
- 全局目标：`OG(t) = Σ_i OG_i(t)`。

## 代码改动点
- 环境层（产出OG与动作选择信息）：
  - 文件：`onpolicy/envs/my/environment.py`
  - 位置：`step()` 中每个智能体完成 `edge_cost` 或 `local_cost` 后（`environment.py:96-107`）。
  - 改动：
    - 使用 `sum(offload)==0` 判定local；否则 `offload_s_id = np.argmax(offload)+1` 判定edge服务器。
    - 计算 `p` 与 `U`，得到 `objective_cur`。
    - 在 `info_n[i]` 中加入：`objective_cur`、`action_type`（`local`或`edge_{offload_s_id}`）、`offload_s_id`、`energy_cur`、`time_cur`、`trans_rate`（若为edge）。
- Runner层（每步上传W&B）：
  - 文件：`onpolicy/runner/separated/my_runner.py`
  - 位置：训练循环内 `envs.step(...)` 之后（`my_runner.py:36-41`）。
  - 改动：
    - 从 `infos` 提取每智能体的 `objective_cur` 与组成项。
    - 记录到 W&B：
      - `objective/OG_global`
      - `objective/OG_agent_{i}`
      - `cost/energy_agent_{i}`、`cost/time_agent_{i}`
      - `action/agent_{i}`、`action/offload_s_id_{i}`
      - `net/trans_rate_agent_{i}`（edge时）
    - 使用 `global_step = episode * episode_length + step` 作为 step。

## 现有数据来源
- 本地/边缘成本：`core.py:280-293`、`core.py:294-337`。
- λ：`onpolicy/envs/my/scenarios/simple_spread.py:15`。
- W&B初始化与结束：`onpolicy/scripts/train/train_my.py:95-117, 164-168`。

## 验证
- 运行训练后，检查W&B中：
  - `objective/OG_global` 与 `objective/OG_agent_i` 每步更新；
  - 当 `sum(offload)==0` 时 `action=local`；当 `sum(offload)>0` 时 `action=edge_{offload_s_id}` 且 `trans_rate>0`；
  - 能耗与时延曲线随动作切换合理。

## 备注与扩展
- 若需将服务器执行时延纳入OG，可在服务器优先队列完成反馈后扩展`time_cur`（需要新增队列回传接口）。
- 若需自定义 `p_i^b(t)`，可替换为 `agent.action.v_resource_alloc` 或基于 `server.state.num_conn` 的负载惩罚项。