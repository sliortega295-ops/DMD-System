# 重复 forward 的参数保留与驻留

同一权重版本反复消费时，阶段内保留可能减少重复聚合或搬运；收益由关键路径上的可消除开销决定。

适合检查：反复 rollout、cond/uncond、冻结角色调用；同一模块名不代表同一权重版本。
保留区间必须结束于权重更新之前，或者有经过验证的刷新协议。检查 backward、reduce-scatter 和所有退出路径。
时间收益只来自关键路径或资源竞争的改善；已被计算掩盖的 gather 不能全算作可节省时间。
显存成本会改变其他阶段的重算、offload 和可行 batch。不同 workload 上收益可以反转。
全量保留不适合时，可研究局部保留；一个 OOM 案例只限定当时的保留范围和容量。
不要把 n=1 的正向描述升级成默认开启，也不要据一次变慢永久封禁驻留。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- weights_stable_during_reuse: true
- state_release_validated: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- repeated_forwards: true
- exposed_gather: true
- memory_headroom: true

最小验证：先数实际 gather/释放次数和版本边界；测一个代表性完整周期的暴露通信、峰值和更新时间。

## 已整理案例

### W2：参数保留有小幅正向观察，同时增加显存

- ID：helios-w2-retention-r36
- 状态：positive_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：原生 C2 83.970328451→83.023064643 秒；peak allocated 26.418→30.911 GiB；16 ranks，单配对，numerical REVIEW。
- 经验：重复使用可以产生小收益；时间和容量一起评估，尚不能据此默认启用。
- 范围：W2 Helios14B No-GAN；该源、该保留范围、两周期的单配对。
- 条件：{"workload": "W2", "update_ratio": "G1/C5/EMA5", "ranks": 16}
- 失败分类：not_established
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r36/conclusion.json)；定位：control_C2_s, candidate_C2_s, peak_allocated_GiB, numerical_status

### W3：冻结 replay 保留只带来小增量

- ID：helios-w3-retention-r1
- 状态：positive_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：C2 117.074207036→116.673757426 秒，约 0.400 秒；numerical REVIEW。
- 经验：保留收益按当前 GAN 合同计算；不能与另一提交的最好成绩叠加。
- 范围：W3 r1 该源同源增量，非长期等价证明。
- 条件：{"workload": "W3", "ranks": 16}
- 失败分类：not_established
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json)；定位：rows[workload=W3]

### W6：同类保留尝试变慢

- ID：helios-w6-retention-r1
- 状态：negative_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：C2 255.363757909→260.582584122 秒，增加约 5.219 秒；未晋级。
- 经验：其他 workload 的保留正例不决定 W6 的收益。净减速已观察，具体成本来源仍需局部测量。
- 范围：W6 r1 的该实现和工作量，numerical REVIEW。
- 条件：{"workload": "W6", "ranks": 16}
- 失败分类：net_slowdown_cause_not_isolated
- 重试条件：保留范围、重复消费次数、暴露通信或容量条件改变后，先做成本探针再重新配对。
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json)；定位：rows[workload=W6]

### W1：扩大到 whole-block tail8 出现 OOM

- ID：helios-w1-tail8-oom
- 状态：implementation_failure / HISTORICAL_CUDA_OOM
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：公开回执记为 CUDA_OOM_NOT_PROMOTED；候选没有有效 C2。
- 经验：容量失败限定当时保留范围；没有候选速度值，不能把它记作慢或零收益。
- 范围：W1 当时 tail8 实现与 A100-40GB 容量。
- 条件：{"workload": "W1", "gpu_memory_gib": 40}
- 失败分类：capacity
- 重试条件：更小保留集合、额外内存释放或不同容量/拓扑后可重新做容量验证；不直接重跑原组合。
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json)；定位：followups.wan_tail8; cleanup[0].unpaired_tail8_source_control

### 只在五次 no-grad rollout 内保留 41 个 FSDP 单元

- ID：helios-bounded-rollout-residency
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：40 blocks+root 保留，进入 Critic 前全部显式 reshard；129.441379→125.814004 秒，峰值 reserved 约 38.594 GiB。
- 经验：保留边界要落在所有权/更新边界，不能溢出到下一角色；更改驻留后旧 profile 已不是当前瓶颈证据。
- 范围：GAN-on C21、A100-40GB、该五次 B1 rollout。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration376_c21_generator_persistent_unshard.md](https://github.com/sliortega295-ops/DMD-System/blob/59ae41c6a6a3d295117737ed6b26ef3e9a4102a9/docs/experiments/gan_cuda_baseline_opt_20260829/iteration376_c21_generator_persistent_unshard.md)；定位：结果、合同、限制与失败记录

### H100 W4 teacher resident 单次差值只有约 0.23%

- ID：h100-w4-teacher-resident-small
- 状态：positive_observation / HISTORICAL_H100_SINGLE_TRIAL_REVIEW
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：已测 4d7c589 为 45.361398→45.257227 秒；后续 audit 修复 1b8236 没有新增 GPU 时间。
- 经验：teacher 保留需先看原先暴露的搬运；小差值缺独立重复时不应标成可靠默认收益。
- 范围：W4 LiveWan/Self-Forcing H100，单配对、full-state parity NOT_RUN。
- 条件：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W4", "gpu_family": "H100", "gpu_memory_gb": 80}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / result.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W4/artifacts/w4_h100_teacher_resident_20261015_r1/reports/completed_single_pair_r1/result.json)；定位：该结果及 qualification / NOT_RUN 边界
- 来源：[sliortega295-ops/DMD-System / timing_audit_committed_source_r1.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W4/artifacts/w4_h100_teacher_resident_20261015_r1/manifests/timing_audit_committed_source_r1.json)；定位：该结果及 qualification / NOT_RUN 边界

### ResearchBoost：跨多次 forward 保留需固定权重版本与后向语义

- ID：rb-fsdp-stable-multiforward
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：其 training card 讨论重复 forward 触发重新聚合，以及保留参数时对最后一次 forward/backward 的处理。
- 经验：先统计实际 all-gather 和调用边界；在 optimizer 更新、梯度归约或异常退出前验证释放，不能只设置一个开关。
- 范围：外部 FSDP 多 forward 方法建议；不是本仓库的一次新测量。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-fsdp-multi-forward/SKILL.md)；定位：重复 forward、reshard 与 backward 约束

### H100 HSDP primitive 证明生命周期可行，不证明 14B 容量

- ID：h100-hsdp-retention-primitive-boundary
- 状态：source_contract / HISTORICAL_PRIMITIVE_ONLY
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：2x4 mesh、五次 tiny forwards：实际 all-gather 205→41，bitwise 与 scope 后 gradient/update parity 通过，41 单元恢复；该报告时 real14B NOT_RUN。
- 经验：可以用真实原语试验缩小旧全局禁令，但 admission 仅覆盖已测 mesh/B1/硬件；tiny 通过不能填补真实模型峰值和速度。
- 范围：H100 early no-grad retention primitive；后续同源完整 fence 结果另有记录。
- 条件：{"workload": "W2", "training_mode": "training", "physical_gpus": 8, "mesh": "replicate2/shard4", "physical_rows_per_rank": 1, "validation_scope": "tiny_primitive", "gpu_family": "H100", "gpu_memory_gb": 80}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / HELIOS_H100_NOGRAD_RETENTION.md](https://github.com/sliortega295-ops/DMD-System/blob/a065830e550151c56934ec83771ea506708d0e14/docs/experiments/HELIOS_H100_NOGRAD_RETENTION.md)；定位：结果、合同、限制与失败记录

### 历史线索：block-major gather 次数大降也可能只省很少时间

- ID：helios-h3-blockmajor-lead
- 状态：unverified / UNVERIFIED_CONVERSATION_LEAD
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：本地 H3 摘要记录过 block-major materialization 降低 gather 次数、端到端变化很小的尝试；公开原始回执仍待补齐。
- 经验：通信次数减少只说明机制；收益还取决于是否暴露、重排开销与共享资源。不能把聊天数字当作性能证据。
- 范围：未核实 H3 block-major 历史方向。
- 条件：{"training_mode": "training", "workload": "W5"}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：仅本地会话线索，未公开原始会话；缺少可核实的公开实验回执。
