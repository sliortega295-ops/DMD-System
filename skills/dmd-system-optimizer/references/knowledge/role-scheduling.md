# 多角色调度与流水的适用边界

独立阶段可重叠，但拆分资源会改变每个角色的效率，并引入传输、等待和填充排空。

区分物理执行计划和逻辑训练合同。G1/C5 与 G1/C1 的最佳布局不必相同。
流水收益取决于阶段平衡、样本数、microbatch 和阶段效率；不能把独立测得的最快阶段拼成可实现总时间。
保留顺序执行作为对照。流水相比上一轮流水变快，不说明它超过了最好的顺序方案。
检查版本、梯度、随机数和更新顺序；验证允许的 batching/fusion 时按逻辑工作比较，不能机械固定物理 launch 数。
依赖已证明独立才讨论重叠；拓扑变化后重算通信与容量。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- logical_work_preserved: true
- versions_and_dependencies_validated: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- independent_work_available: true
- overlap_exceeds_partition_cost: true

最小验证：按 G/C/Teacher/EMA 的真实频率建最小依赖图，比较相同资源的顺序方案；先测阶段服务时间和跨角色传输。

## 已整理案例

### 早期顺序与流水：改进流水不等于胜过顺序

- ID：helios-pipeline-c10-lead
- 状态：unverified / UNVERIFIED_CONVERSATION_LEAD
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：本地会话报告：C10 相比前一版流水有改进，仍慢于顺序执行；这里未重新取得原运行回执，不发布该比值为实测结论。
- 经验：同时保留当前流水对照和最佳顺序对照，避免只看局部迭代进步。
- 范围：早期 G1/C5/EMA0、16卡与8G+8Score探索的会话线索；不等于当前 W1/W2。
- 条件：{"update_ratio": "G1/C5/EMA0", "ranks": 16}
- 失败分类：unverified_net_slowdown
- 重试条件：阶段比例、角色效率、microbatch 或拓扑变化后，先比较服务时间及填充排空成本。
- 来源：仅本地会话线索，未公开原始会话；缺少可核实的公开实验回执。

### 同布局、同 B16 的 Critic 流水有三对正向结果

- ID：helios-matched-pipeline-positive
- 状态：positive_observation / HISTORICAL_THREE_PAIRED_RUNS
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：固定 post-Step10-A 状态，8G+8C 的顺序/流水三次独立交错配对：中位 354.893050→258.778970 秒；六次正式运行均通过。
- 经验：流水可以有效，但该结论只覆盖共同布局上的 Critic 窗口，不包含冷构造、Step10-A、布局过渡、保存或数据加载。
- 范围：早期 SYN-FEATURE、384x640、DP8 MB1/GAS2、五次 Critic/EMA。
- 条件：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "global_batch": 16, "comparison": "same_layout_critic_window", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / rb16_f5_pipeline_20260809.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_f5_pipeline_20260809.md)；定位：结果、合同、限制与失败记录

### 另一固定内存分区的流水三对结果明显慢于顺序

- ID：helios-fixed-memory-pipeline-negative
- 状态：negative_observation / HISTORICAL_THREE_PAIRED_RUNS
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：同 global B16、G1/C5/EMA1，顺序三次 656.125/596.234/658.338 秒，固定内存流水 1139.194/1043.215/1170.197 秒。
- 经验：流水收益取决于分配给各角色的资源和效率；早期 same-layout Critic 正例不能外推到这种全系统分区。
- 范围：GAN systems 20260811 的 matched-work contract，runtime fairness 通过不代表训练收敛。
- 条件：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "global_batch": 16, "gan_enabled": true, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：resource_partition_net_regression
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / gan_systems_experiment_20260811.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/docs/status/gan_systems_experiment_20260811.md)；定位：结果与证据边界
