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
- 历史上下文：{"workload": "W2", "update_ratio": "G1/C5/EMA5", "ranks": 16}
- 正确性前提：{"weights_stable_during_reuse": true, "state_release_validated": true}
- 收益条件：{"repeated_forwards": true, "exposed_gather": true, "memory_headroom": true}
- 失败分类：not_established
- 重试信号：当前 gather 暴露时间或可用容量有证据支持驻留
- 最小验证：先查完整单元 census、权重版本与 scope 释放，再补同源交错配对和 numerical REVIEW
- 不应重复：不能因一次约 0.95 秒差值就默认晋级
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-w2-retention-r36`
- 关联实现：[FSDP2 完整单元的受控参数驻留](implementations.md#fsdp-phase-retention)；reusable_candidate；同机制的当前复用候选；须检查 guards、生效路径、依赖和验证，不代表历史实现身份或直接迁移通过。
- 函数入口：[src/distillschedule/runtime/fsdp2_variable_forward.py::fsdp2_variable_forward_phase](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_variable_forward.py#L98-L142)
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r36/conclusion.json#L6)；定位：观察中的数值文本锚点 L6；同时阅读同节合同、状态与限制。原定位：control_C2_s, candidate_C2_s, peak_allocated_GiB, numerical_status

### 只在五次 no-grad rollout 内保留 41 个 FSDP 单元

- ID：helios-bounded-rollout-residency
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：40 blocks+root 保留，进入 Critic 前全部显式 reshard；129.441379→125.814004 秒，峰值 reserved 约 38.594 GiB。
- 经验：保留边界要落在所有权/更新边界，不能溢出到下一角色；更改驻留后旧 profile 已不是当前瓶颈证据。
- 范围：GAN-on C21、A100-40GB、该五次 B1 rollout。
- 历史上下文：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"weights_stable_during_reuse": true, "state_release_validated": true, "grad_mode": "no_grad"}
- 收益条件：{"repeated_forwards": true, "exposed_gather": true, "memory_headroom": true}
- 失败分类：not_established
- 重试信号：重复 no-grad rollout gather 仍暴露且内存可容纳
- 最小验证：检查 40 blocks+root、权重稳定、进入 Critic 前 reshard 与峰值
- 不应重复：不把 residency scope 延伸到更新或不相关阶段
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-bounded-rollout-residency`
- 关联实现：[Helios 单次 no-grad rollout 的参数驻留](implementations.md#helios-rollout-residency)；reusable_candidate；同机制的当前复用候选；须检查 guards、生效路径、依赖和验证，不代表历史实现身份或直接迁移通过。
- 函数入口：[src/distillschedule/runtime/fsdp2_nograd_persistence.py::generator_nograd_persistent_unshard_scope](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_nograd_persistence.py#L43-L162)
- 来源：[sliortega295-ops/DMD-System / iteration376_c21_generator_persistent_unshard.md](https://github.com/sliortega295-ops/DMD-System/blob/59ae41c6a6a3d295117737ed6b26ef3e9a4102a9/docs/experiments/gan_cuda_baseline_opt_20260829/iteration376_c21_generator_persistent_unshard.md#L20)；定位：观察中的数值文本锚点 L20；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### H100 W4 teacher resident 单次差值只有约 0.23%

- ID：h100-w4-teacher-resident-small
- 状态：positive_observation / HISTORICAL_H100_SINGLE_TRIAL_REVIEW
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：已测 4d7c589 为 45.361398→45.257227 秒；后续 audit 修复 1b8236 没有新增 GPU 时间。
- 经验：teacher 保留需先看原先暴露的搬运；小差值缺独立重复时不应标成可靠默认收益。
- 范围：W4 LiveWan/Self-Forcing H100，单配对、full-state parity NOT_RUN。
- 历史上下文：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W4", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"weights_stable_during_reuse": true, "state_release_validated": true}
- 收益条件：{"repeated_teacher_forwards": true, "exposed_parameter_transfer": true, "memory_headroom": true}
- 失败分类：not_established
- 重试信号：当前 teacher 调用/容量产生可测增量
- 最小验证：在当前源码复查数值并交错配对，区分旧 4d7c589 与后续 audit
- 不应重复：后续源码修复没有自动获得旧 GPU timing
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case h100-w4-teacher-resident-small`
- 来源：[sliortega295-ops/DMD-System / result.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W4/artifacts/w4_h100_teacher_resident_20261015_r1/reports/completed_single_pair_r1/result.json#L4)；定位：观察中的数值文本锚点 L4；同时阅读同节合同、状态与限制。原定位：该结果及 qualification / NOT_RUN 边界
- 来源：[sliortega295-ops/DMD-System / timing_audit_committed_source_r1.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W4/artifacts/w4_h100_teacher_resident_20261015_r1/manifests/timing_audit_committed_source_r1.json#L12)；定位：观察中的数值文本锚点 L12；同时阅读同节合同、状态与限制。原定位：该结果及 qualification / NOT_RUN 边界

### ResearchBoost：跨多次 forward 保留需固定权重版本与后向语义

- ID：rb-fsdp-stable-multiforward
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：其 training card 讨论重复 forward 触发重新聚合，以及保留参数时对最后一次 forward/backward 的处理。
- 经验：先统计实际 all-gather 和调用边界；在 optimizer 更新、梯度归约或异常退出前验证释放，不能只设置一个开关。
- 范围：外部 FSDP 多 forward 方法建议；不是本仓库的一次新测量。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"weights_stable_during_reuse": true, "state_release_validated": true}
- 收益条件：{"repeated_forwards": true, "exposed_gather": true, "memory_headroom": true}
- 失败分类：not_established
- 重试信号：目标 FSDP 版本的重复 forward 确实重复聚合
- 最小验证：核对最后 forward/backward、权重版本与释放；具体 no-grad 示例仅作参考
- 不应重复：不要把外部方法或本库 Helios wrapper 当通用可微 FSDP 实现
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case rb-fsdp-stable-multiforward`
- 关联实现：[Helios 单次 no-grad rollout 的参数驻留](implementations.md#helios-rollout-residency)；adaptation_example；机制参考或旧版本示例；需核对实现差异并适配，未证明是历史计时所用源码。
- 函数入口：[src/distillschedule/runtime/fsdp2_nograd_persistence.py::generator_nograd_persistent_unshard_scope](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_nograd_persistence.py#L43-L162)
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-fsdp-multi-forward/SKILL.md#L1)；定位：报告范围 L1–L88；L6: FSDP: several forwards in one iteration；L24: 1. All-gather once per iteration；L33: 2. Cast frozen models to the compute dtype before wrapping in FSDP；L45: 3. Keep the frozen encoder's parameters resident；L54: 4. Cross-attention K/V cache；L63: 5. Remove host syncs；L72: 6. Spend the freed memory on speed；L83: What ships with it；原定位：重复 forward、reshard 与 backward 约束

### H100 HSDP primitive 证明生命周期可行，不证明 14B 容量

- ID：h100-hsdp-retention-primitive-boundary
- 状态：source_contract / HISTORICAL_PRIMITIVE_ONLY
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：2x4 mesh、五次 tiny forwards：实际 all-gather 205→41，bitwise 与 scope 后 gradient/update parity 通过，41 单元恢复；该报告时 real14B NOT_RUN。
- 经验：可以用真实原语试验缩小旧全局禁令，但 admission 仅覆盖已测 mesh/B1/硬件；tiny 通过不能填补真实模型峰值和速度。
- 范围：H100 early no-grad retention primitive；后续同源完整 fence 结果另有记录。
- 历史上下文：{"workload": "W2", "training_mode": "training", "physical_gpus": 8, "mesh": "replicate2/shard4", "physical_rows_per_rank": 1, "validation_scope": "tiny_primitive", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"weights_stable_during_reuse": true, "state_release_validated": true}
- 收益条件：{"repeated_forwards": true, "exposed_gather": true, "memory_headroom": true}
- 失败分类：not_established
- 重试信号：真实 14B 的容量/梯度/生命周期检查可执行
- 最小验证：先保留 tiny 205→41 gather 证据，再测实际模型完整 scope 与更新
- 不应重复：primitive bitwise 不能填补 real14B NOT_RUN
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case h100-hsdp-retention-primitive-boundary`
- 关联实现：[FSDP2 完整单元的受控参数驻留](implementations.md#fsdp-phase-retention)；reusable_candidate；同机制的当前复用候选；须检查 guards、生效路径、依赖和验证，不代表历史实现身份或直接迁移通过。
- 函数入口：[src/distillschedule/runtime/fsdp2_variable_forward.py::fsdp2_variable_forward_phase](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_variable_forward.py#L98-L142)
- 来源：[sliortega295-ops/DMD-System / HELIOS_H100_NOGRAD_RETENTION.md](https://github.com/sliortega295-ops/DMD-System/blob/a065830e550151c56934ec83771ea506708d0e14/docs/experiments/HELIOS_H100_NOGRAD_RETENTION.md#L1)；定位：报告范围 L1–L31；L1: Bounded H100 Helios no-grad Generator retention experiment；原定位：结果、合同、限制与失败记录

### 历史线索：block-major gather 次数大降也可能只省很少时间

- ID：helios-h3-blockmajor-lead
- 状态：unverified / UNVERIFIED_CONVERSATION_LEAD
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：本地 H3 摘要记录过 block-major materialization 降低 gather 次数、端到端变化很小的尝试；公开原始回执仍待补齐。
- 经验：通信次数减少只说明机制；收益还取决于是否暴露、重排开销与共享资源。不能把聊天数字当作性能证据。
- 范围：未核实 H3 block-major 历史方向。
- 历史上下文：{"training_mode": "training", "workload": "W5"}
- 正确性前提：{"weights_stable_during_reuse": true, "state_release_validated": true}
- 收益条件：{"repeated_forwards": true, "exposed_gather": true, "memory_headroom": true}
- 失败分类：not_established
- 重试信号：公开 gather/计时回执齐全且 gather 真在关键路径
- 最小验证：对齐 block-major 的调用顺序、计数、replay 与整周期
- 不应重复：减少 gather 次数的会话线索不等于已证实端到端收益
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-h3-blockmajor-lead`
- 来源：仅本地会话线索，未公开原始会话；缺少可核实的公开实验回执。
