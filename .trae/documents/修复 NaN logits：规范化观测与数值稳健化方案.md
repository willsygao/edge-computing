## 问题定位
- NaN logits 通常由观测向量含有极端值（Inf/NaN/巨大量级）导致网络前向出现非数。
- 我们刚把任务量级调整到 bit 与 cycles 的真实区间（最高到 1.6e8 bits、5e10 cycles），未做观测归一化，会让网络输入过大。

## 修复思路
- 统一对观测进行归一化/限幅，保证数值在可训练范围。
- 保留已做的 SNR/速率裁剪与路径损耗稳健处理，进一步在观测层加保护。

## 修改点
### 1) 归一化观测（simple_spread.py）
- 在 `Scenario.observation(agent, world)` 将原始值转换为尺度化特征：
  - 输入数据：`i_mb = agent.state.task_i_s / 1e6`（MB）并 `i_mb = clip(i_mb, 0, 200)`；
  - 计算量：`e_gcy = agent.state.task_e_s / 1e9`（G cycles）并 `e_gcy = clip(e_gcy, 0, 100)`；
  - 截止时延：`tau_s = agent.state.task_delay_tol`（秒），归一化 `tau_norm = clip(tau_s / 2.0, 0, 1)`；
  - 队列剩余：若为 None 则 0；可直接使用或做 `clip(q, 0, 1e6)`；
  - 功率增益：`pg_db = 10*log10(pg + 1e-12)`；线性缩放到 [0,1]：`pg_norm = clip((pg_db + 120) / 100, 0, 1)`；
- 拼接前对所有值做 `np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)`。

### 2) 观测数值保护（environment.py）
- 在 `_get_obs(agent)` 返回前做一次 `np.nan_to_num`，确保任何异常值不会进入 buffer。

### 3) 进一步稳健（core.py）
- 已完成：距离、功率增益、SNR、速率裁剪；传输时延回退。保留这些更改。
- 若仍有异常：将 `agent.state.trans_rate` 与 `agent.state.time_cur` 在写入时也做 `np.nan_to_num`。

## 验证
- 运行训练，打印前 10 步观测的 min/max 与是否有非有限值（仅调试）。
- 观察是否仍出现 `FixedCategorical(logits)` NaN 错误。若有，则进一步收缩归一化上下界。

## 影响与兼容
- 只改观测与数值保护，不影响奖励/效用/可视化逻辑；训练更稳定。
- 归一化尺度选取与车辆边缘场景相符，后续可按论文具体数据再微调。