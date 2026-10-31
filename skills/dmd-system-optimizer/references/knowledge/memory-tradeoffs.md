# offload、激活保留与容量交换

释放内存可以解锁更有价值的驻留或减少重算；新增传输、同步和峰值也可能抵消收益。

分开三个问题：是否放得下、是否更快、是否解锁另一项优化。
capacity enabler 可以没有直接速度收益，但必须说明下游收益和新增成本，避免重复记账。
allocated、reserved、driver free 和 host cgroup 是不同指标；结束边界的余量不代表全程最小余量。
HBM 富余机器与紧张机器可以选择相反策略。CPU 带宽、PCIe/NVLink 和 NUMA 改变后要重估。
offload 造成变慢时先定位暴露传输、同步或重算；失败原因未查明时保留未知。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- state_lifetime_preserved: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- capacity_limits_candidate: true
- enabled_gain_exceeds_transfer: true

最小验证：先做逐阶段容量账：常驻、激活、临时 gather、workspace、pinned 和 checkpoint；再量传输与计算重叠。

## 已整理案例

### W4：完成运行与全程容量宽裕是不同结论

- ID：helios-w4-capacity-r3
- 状态：inconclusive / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：完整仅B臂 C2 为 42.123914 秒，末边界 driver free 4.375 MiB；回执明确它不是全程最小余量。
- 经验：边界采样不能证明峰值余量；allocated/reserved/driver 指标不能互换。
- 范围：W4 C51 R3；旧 baseline/full 来自父源，新臂 n=1，数值 REVIEW。
- 历史上下文：{"workload": "W4", "ranks": 16}
- 正确性前提：{"state_lifetime_preserved": true}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：peak_headroom_not_established
- 重试信号：获得全程最小 driver-free 或更保守的容量策略
- 最小验证：监测构造、各角色、checkpoint 和下一周期峰值，区分末边界与全程最小值
- 不应重复：不能把末边界 4.375 MiB 当作安全余量
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case helios-w4-capacity-r3`
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/livewan_r3/reports/conclusion.json#L62)；定位：观察中的数值文本锚点 L62；同时阅读同节合同、状态与限制。原定位：results[arm=only_B_full_fsdp16_v1]; capacity_boundary; source_boundary

### parameter reserve 与 driver free 不能混用

- ID：helios-allocator-policy-r36
- 状态：source_contract / HISTORICAL_POLICY
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：r36 的容量口径是 allocated highwater 留 2 GiB，明确不是 driver free 留 2 GiB。
- 经验：沿用实验原容量合同，新增硬件按自己的真实资源约束判断。
- 范围：r36 回执的原始容量政策；不是推荐所有 workload 使用 2 GiB。
- 历史上下文：{"workload": "W2"}
- 正确性前提：{"state_lifetime_preserved": true}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：not_established
- 重试信号：当前容量准则明确采用新的口径
- 最小验证：同时报告 allocated highwater、reserved、driver free 并注明 gate 对应哪项
- 不应重复：不把 allocated 留 2 GiB 改述为 driver free 留 2 GiB
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case helios-allocator-policy-r36`
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r36/conclusion.json)；定位：allocator_reserve_policy; host_reserve_GiB

### 孤立 G/S/T 容量搜索不能代替连续 Full5

- ID：helios-isolated-offload-search
- 状态：positive_observation / HISTORICAL_CAPACITY_ONLY
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：固定目标图的 all40 到 g2 多个点通过孤立 Step15-A；g4 被选为保守点，g1 未启动，g0、连续过渡和正式计时 NOT_RUN。
- 经验：先确认目标图与真实隐藏状态；减少搬运计数是容量机制证据，不能据此宣称最小 offload 集合或训练加速。
- 范围：早期 8G+4S+4T、G3/C11、DP8 MB1/GAS2 的孤立步骤。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"state_lifetime_preserved": true}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：not_established
- 重试信号：已知孤立可行点具备完整阶段容量证据
- 最小验证：从 g4 等已验点跑连续 G/C/EMA 与下一阶段释放，不先扩大搜索
- 不应重复：孤立 Step15-A 通过不能替代 Full5
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-isolated-offload-search`
- 关联实现：[旧 Route-A 不可变 frozen-block streaming](implementations.md#legacy-frozen-streaming)；historical_reference；保留的旧 Route-A 实现参考；不是新 workload 的现成接入或当前默认配置。
- 函数入口：[runtime/frozen_block_streaming.py::install_frozen_block_streamer](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/frozen_block_streaming.py#L1095-L1099)
- 来源：[sliortega295-ops/DMD-System / n2_target_g_offload_20260808.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/n2_target_g_offload_20260808.md#L1)；定位：报告范围 L1–L97；L1: N2 target G/S/T Generator offload search；L3: Iteration preflight；L23: Prior-path disposition；L33: Implemented target executor；L59: Next bounded execution；L68: Bounded execution result；原定位：结果、合同、限制与失败记录

### g8 确实释放了容量，连续 checkpoint backward 仍 OOM

- ID：helios-full5-pr8-oom
- 状态：implementation_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：相对 g4，pre-backward allocation 精确减少 2,811,154,432 字节；随后 216 MiB 临时量分配失败，Full5 和计时未完成。
- 经验：优化生效与容量够用是两个问题；不把 OOM 记为无效开关，也不靠继续清缓存绕过已复现的存活容量瓶颈。
- 范围：早期 Route-A PR8、ZeRO-1 CPU optimizer、该激活策略。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"state_lifetime_preserved": true}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：checkpoint_backward_capacity
- 重试信号：失败点的临时分配空间增加或活跃峰值下降
- 最小验证：重测 Critic checkpoint backward 的分配轨迹与完整 Full5
- 不应重复：不只凭少了 2.81 GB 参数就重跑同一 OOM 点
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-full5-pr8-oom`
- 关联实现：[旧 Route-A 不可变 frozen-block streaming](implementations.md#legacy-frozen-streaming)；historical_reference；保留的旧 Route-A 实现参考；不是新 workload 的现成接入或当前默认配置。
- 函数入口：[runtime/frozen_block_streaming.py::install_frozen_block_streamer](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/frozen_block_streaming.py#L1095-L1099)
- 来源：[sliortega295-ops/DMD-System / ra16_f5_pr8_20260809.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/ra16_f5_pr8_20260809.md#L1)；定位：报告范围 L1–L71；L1: RA16-F5-PR8 continuous Full5 experiment；L3: Iteration preflight；L24: Planned evidence；L39: Measured result；L63: Gate conclusion；原定位：结果、合同、限制与失败记录

### g12 通过 Generator，瓶颈迁移到 Critic

- ID：helios-phase-shift-after-g12
- 状态：implementation_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：修正 tracing 字段冲突后，16 ranks 完成 G backward/update；随后 resident Score 的 checkpoint backward OOM。此时 G 冻结块已离开 GPU。
- 经验：按失败阶段重新建容量账；继续增大 Generator streaming 不会直接释放已无 Generator 冻结块的 Critic 峰值。
- 范围：Route-A PR12 recovery002，G1/C5/EMA5 连续 Full5。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"state_lifetime_preserved": true}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：critic_capacity
- 重试信号：Score backward 峰值有针对性下降
- 最小验证：先从已通过的 G 边界进入 Score 并记录 checkpoint replay 峰值
- 不应重复：不要继续优化已经卸载的 G 来解释 Score OOM
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-phase-shift-after-g12`
- 关联实现：[旧 Route-A 不可变 frozen-block streaming](implementations.md#legacy-frozen-streaming)；historical_reference；保留的旧 Route-A 实现参考；不是新 workload 的现成接入或当前默认配置。
- 函数入口：[runtime/frozen_block_streaming.py::install_frozen_block_streamer](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/frozen_block_streaming.py#L1095-L1099)
- 来源：[sliortega295-ops/DMD-System / ra16_f5_pr12_20260809.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/ra16_f5_pr12_20260809.md#L1)；定位：报告范围 L1–L79；L1: RA16-F5-PR12 experiment — 2026-08-09；L3: Preflight decision；L17: Required evidence；L25: Attempt 001: instrumentation failure after the capacity gate advanced；L43: Bounded recovery 002；L56: Recovery 002 result: bounded Critic-backward capacity failure；原定位：结果、合同、限制与失败记录

### 隐藏 ZeRO-1 flat owner 可精确释放，但不足以完成 backward

- ID：helios-lp-flat-release
- 状态：implementation_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：typed LP offload 使 Critic pre-backward allocation 精确减少 2,502,885,376 字节；checkpoint 重算仍耗尽容量，16 ranks 均未完成 Critic backward。
- 经验：释放量可证明存储所有权与机制成立；不能把当前失败分配量当作完成后续所有工作的足够余量。
- 范围：Route-A g12、resident Score、A32 activation offload。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"state_lifetime_preserved": true}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：remaining_critic_peak
- 重试信号：剩余 Critic replay 峰值被明确削减
- 最小验证：核对 LP flat 的所有者/释放并跑完整 Critic backward
- 不应重复：不能把减少 2.50 GB 分配当作整个 backward 可行
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-lp-flat-release`
- 来源：[sliortega295-ops/DMD-System / ra16_f5_glp_offload_20260809.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/ra16_f5_glp_offload_20260809.md#L1)；定位：报告范围 L1–L80；L1: RA16-F5-GLP-OFFLOAD: typed Generator LP-state phase offload；L6: Question and frozen contract；L22: Gates；L34: Measured causal result；L47: Failure；L61: Interpretation and next boundary；L75: Evidence；原定位：结果、合同、限制与失败记录

### 只给 Critic 所需的四个 Score 块加 streaming 后通过 Full5

- ID：helios-score4-streaming-admission
- 状态：positive_observation / HISTORICAL_FUNCTIONAL_PASS_DIAGNOSTIC_TIME
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：在 LP offload 基础上，仅 Score 尾四块 window1：16 ranks 完成 G1/C5/EMA5，生命周期归零并保存 checkpoint；404.042 秒只是一次带诊断的窗口。
- 经验：容量优化应瞄准实际失败阶段；四块足够不等于四块最少，也不能迁移到不同隐藏状态的 split-role executor。
- 范围：Route-A g12+LP offload+Score4，官方 384x640 B16。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"state_lifetime_preserved": true}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：not_established
- 重试信号：当前 Score 冻结块与 window 能保持完整生命周期
- 最小验证：先复核 G1/C5/EMA5、各 scope 归零及保存，再做低干扰配对
- 不应重复：404 秒诊断窗口不是正式优化收益
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-score4-streaming-admission`
- 关联实现：[旧 Route-A 不可变 frozen-block streaming](implementations.md#legacy-frozen-streaming)；historical_reference；保留的旧 Route-A 实现参考；不是新 workload 的现成接入或当前默认配置。
- 函数入口：[runtime/frozen_block_streaming.py::install_frozen_block_streamer](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/frozen_block_streaming.py#L1095-L1099)
- 来源：[sliortega295-ops/DMD-System / ra16_f5_glp_score4_20260809.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/ra16_f5_glp_score4_20260809.md#L88)；定位：观察中的数值文本锚点 L88；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### 10/40 checkpoint bypass OOM，4/40 能跑但收益极小

- ID：helios-checkpoint-bypass-capacity
- 状态：implementation_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：No-GAN 对两角色或仅 Critic 的 10/40 bypass 均 OOM；Critic-only 4/40 完成，96.529020→96.341895 秒，仅约 0.19%。
- 经验：多保留激活要同时满足容量与重算成本价值；密度调整不应在失败后无边界继续扫。
- 范围：Helios No-GAN node-HSDP、当时显存与算子组合。
- 历史上下文：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": false, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"state_lifetime_preserved": true}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：checkpoint_retention_capacity
- 重试信号：目标 bypass 子集有完整周期容量余量
- 最小验证：从小子集验证 replay/梯度和峰值，再比较净收益
- 不应重复：不要重复 10/40 OOM 或因完成就推定 4/40 有显著收益
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-checkpoint-bypass-capacity`
- 关联实现：[Helios FA2 selective checkpoint 保留](implementations.md#helios-selective-checkpoint)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/fsdp2_static.py::_selective_flash_checkpoint_contexts](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L84-L111)
- 函数入口：[src/distillschedule/runtime/fsdp2_static.py::configure_fsdp2_checkpointing](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L171-L290)
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/111e293dc79978132af91ab5515a096d941b94fe/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md#L34)；定位：观察中的数值文本锚点 L34；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### 保留真实 FA2 输出可省重算，密度受最坏显存限制

- ID：helios-fa2-selective-retention
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：No-GAN 固定 local FA2 后 stride2 retention：95.725157→94.461700 秒；GAN-on 独立配对 143.206713→142.295342 秒，但最小 end driver free 仅约 12 MiB。
- 经验：核对原始 saves/replay reuses 与零物理调用 bypass；小余量的成功不能自动支持 stride1 或新 workload。
- 范围：两个独立 Helios 合同、Critic-only FA2 replay retention、均单次计时。
- 历史上下文：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"state_lifetime_preserved": true, "attention_backend": "FA2"}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：not_established
- 重试信号：当前 FA2 调用可重放且容量覆盖全周期
- 最小验证：固定 backend 身份，再检查选择性 cache 的梯度和全程 driver-free
- 不应重复：仅末边界约 12 MiB free 不足以说明稳健容量
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-fa2-selective-retention`
- 关联实现：[Helios FA2 selective checkpoint 保留](implementations.md#helios-selective-checkpoint)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/fsdp2_static.py::_selective_flash_checkpoint_contexts](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L84-L111)
- 函数入口：[src/distillschedule/runtime/fsdp2_static.py::configure_fsdp2_checkpointing](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L171-L290)
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/111e293dc79978132af91ab5515a096d941b94fe/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md#L1)；定位：报告范围 L1–L146；L1: Helios14B node-HSDP optimization closure (Iterations 368-382)；L3: Status and claim boundary；L27: New experiment ladder；L52: Effective methods；L54: Explicit pinned local FlashAttention-2；L70: Critic selective FA2 checkpoint-output retention；L87: Rejected and bounded-negative methods；L105: Post-winner Nsight result；原定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / iteration355_selective_flash_checkpoint.md](https://github.com/sliortega295-ops/DMD-System/blob/7613dcedcbdca1428c0a484894bf2e421ac5f2c8/docs/experiments/gan_cuda_baseline_opt_20260829/iteration355_selective_flash_checkpoint.md#L6)；定位：观察中的数值文本锚点 L6；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### 冻结权重共享不等于 trainable/optimizer 状态共享

- ID：dmd-frozen-alias-construction-peak
- 状态：source_contract / SOURCE_CONTRACT_ONLY
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：仓库能力契约要求冻结张量值、dtype、layout 兼容，LoRA、Adam 等可训练状态独立；所有角色构造完成后才 alias 也不会降低先前构造峰值。
- 经验：从真实存储对象和生命周期推导共享范围；计划中的相同模型名不能代替权重相等验证。
- 范围：DMD 源码能力边界，CPU/source-only，不是新 GPU 结果。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"state_lifetime_preserved": true}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：not_established
- 重试信号：冻结值/dtype/layout 兼容且 alias 可在峰值前建立
- 最小验证：验证训练状态独立并记录角色构造各阶段峰值
- 不应重复：所有角色已构造后 alias 不会追溯消除早期峰值
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case dmd-frozen-alias-construction-peak`
- 来源：[sliortega295-ops/DMD-System / optimization_capabilities.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/docs/guides/optimization_capabilities.md#L1)；定位：报告范围 L1–L239；L1: Physical capabilities and executable plans；L20: Reuse the existing runtime；L37: Four uses of one executable policy；L59: Model adaptation, not optimization-script copying；L76: Scoped native owner example: DMD2 UNet checkpointing；L113: Official Self-Forcing: Fake-only checkpoint policy；L135: Modern native flow planning is a separately qualified CPU slice；L171: Direct single-CUDA immutable sharing；原定位：结果与证据边界

### H100 W3 已完成的是 official role-swap 对照组合

- ID：h100-w3-role-swap-combination
- 状态：positive_observation / HISTORICAL_H100_SINGLE_TRIAL_REVIEW
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：624515e7 对应的 192.9860455→117.3435074 秒是 G1/C5/EMA1 的组合观察；先前 full-resident OOM 仍保留。
- 经验：模型能在大显存上跑不意味着整套多角色完全驻留；必须保留具体物理策略和源版本。
- 范围：W3 H100、16 logical slots/8 physical GPUs，不能拿另一 A100 batch 作分母。
- 历史上下文：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W3", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"state_lifetime_preserved": true}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：not_established
- 重试信号：组合方案在目标容量/角色比例下可运行
- 最小验证：先测 full-resident 与组合的容量，再明确需要哪些同源消融
- 不应重复：组合 192.99→117.34 秒不能分配给单个组件
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case h100-w3-role-swap-combination`
- 来源：[sliortega295-ops/DMD-System / w3_official_swap_matched_r1.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W3/reports/w3_official_swap_matched_r1.json#L6)；定位：观察中的数值文本锚点 L6；同时阅读同节合同、状态与限制。原定位：该结果及 qualification / NOT_RUN 边界

### 名为 Resident 的 fallback 实际仍搬运 CPU checkpoint boundaries

- ID：h100-w5-resident-cpu-fallback
- 状态：source_contract / HISTORICAL_H100_SINGLE_TRIAL_REVIEW
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：另一个 d81 source 的 fallback 记录 637.204307 秒，每 rank 600 次 boundary copies；不能称为纯 GPU resident。
- 经验：按实际状态和传输描述策略；逻辑 copy 字节不是已经量到的 PCIe 流量，不与另一 source 作同源因果比较。
- 范围：W5 corrected three-arm audit 中的 resident-with-CPU-fallback。
- 历史上下文：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W5", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"state_lifetime_preserved": true}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：not_established
- 重试信号：CPU fallback 的拷贝与 host 预算可接受
- 最小验证：计数每 rank 600 boundary copies，记录 device/host 峰值和完整周期
- 不应重复：不能把使用 CPU fallback 的 637 秒方案称为纯 GPU resident
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case h100-w5-resident-cpu-fallback`
- 关联实现：[H3 checkpoint 边界 CPU offload](implementations.md#h3-boundary-offload)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/backends/training/minimax_h3/checkpoint_boundary_offload.py::install_checkpoint_boundary_cpu_offload](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/checkpoint_boundary_offload.py#L187-L336)
- 来源：[sliortega295-ops/DMD-System / report.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W5/reports/completed_W5_three_arm_source_qualified_r2/report.md#L7)；定位：观察中的数值文本锚点 L7；同时阅读同节合同、状态与限制。原定位：该结果及 qualification / NOT_RUN 边界

### H100 W5 保存 attention 输出/LSE 换取重算减少

- ID：h100-w5-attention-cache-memory-trade
- 状态：positive_observation / HISTORICAL_H100_SINGLE_TRIAL_REVIEW
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：同 9bfd source，FSDP 638.522138→GPU-cache 539.718383 秒；allocated 41.603→64.116 GiB，reserved 45.516→71.082 GiB。
- 经验：有剩余显存时可换取计算，但必须纳入训练状态、24 个 arm/rank 回执与所有更新；没有 full-state parity 仍不能晋级为正式加速。
- 范围：W5 Grid1000 approximate、Student1/Fake5、96 sample identities、单次三臂。
- 历史上下文：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W5", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"state_lifetime_preserved": true}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：not_established
- 重试信号：目标设备能容纳新增 GPU cache 且少重算有净收益
- 最小验证：同源检查 cache/replay 梯度、64 GiB 级 allocated 与周期
- 不应重复：把时间改善迁移前必须重查显存，不能只看 H100 型号
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case h100-w5-attention-cache-memory-trade`
- 关联实现：[H3 FA3 replay cache 与 staged 路径](implementations.md#h3-attention-replay)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py::install_fa3_replay_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py#L880-L1131)
- 函数入口：[src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py::prepare_staged_cache_before_backward](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py#L854-L872)
- 来源：[sliortega295-ops/DMD-System / result.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W5/reports/completed_W5_three_arm_source_qualified_r2/result.json#L28)；定位：观察中的数值文本锚点 L28；同时阅读同节合同、状态与限制。原定位：该结果及 qualification / NOT_RUN 边界

### H100 W6 三臂比较绑定同 0419 source 与 FA2

- ID：h100-w6-same-source-matrix
- 状态：positive_observation / HISTORICAL_H100_SINGLE_TRIAL_REVIEW
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：同源 C2：resident-with-CPU-fallback 308.670470 秒、FSDP 320.666055 秒、candidate 278.886442 秒。
- 经验：先锁定 native AR8、continuous sigma、LoRA64、FA2 与 G1/F1；不能继承旧 FA3 或早期 C1 的数值结论。
- 范围：W6 同源矩阵，当前 C2 full-state parity NOT_RUN。
- 历史上下文：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W6", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"state_lifetime_preserved": true}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：not_established
- 重试信号：在同源同 C2 下可重建三臂比较
- 最小验证：核对 FSDP、CPU-fallback resident、candidate 的实际调用和峰值
- 不应重复：不能跨源码或将 C1/C2 混在同一矩阵
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case h100-w6-same-source-matrix`
- 来源：[sliortega295-ops/DMD-System / result.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W6/artifacts/w6_resident_adaln16_staging_20261015_r1/reports/completed_same0419_matrix_r1/result.json#L8)；定位：观察中的数值文本锚点 L8；同时阅读同节合同、状态与限制。原定位：该结果及 qualification / NOT_RUN 边界

### ResearchBoost/SGLang：大显存能驻留时 offload 可能只剩开销

- ID：rb-inference-offload-vram-dependent
- 状态：positive_observation / EXTERNAL_REPORTED_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：其 inference card 报告 LingBot 在 B300 全模型可驻留时关闭不必要 offload 有收益；这不适用于容纳不了模型的设备。
- 经验：先核对实测峰值和必要状态；推理省掉的 offload 不自动转移到含梯度、Adam、replay 的训练。
- 范围：外部 LingBot diffusion inference、B300 容量条件。
- 历史上下文：{"training_mode": "inference", "model_family": "LingBot", "source_project": "SGLang", "gpu_family": "B300"}
- 正确性前提：{"state_lifetime_preserved": true, "training_mode": "inference"}
- 收益条件：{"capacity_limits_candidate": true, "enabled_gain_exceeds_transfer": true}
- 失败分类：not_established
- 重试信号：目标推理设备可容纳全部活跃权重及工作区
- 最小验证：关闭 offload 前测完整 shape/并发容量与净时间
- 不应重复：B300 的全驻留收益不适用于容纳不了模型的卡
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case rb-inference-offload-vram-dependent`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/sglang-diffusion-performance/SKILL.md#L1)；定位：报告范围 L1–L565；L6: SGLang Diffusion Performance Tuning；L10: Start from the model's cookbook page；L38: Native Backend Gate；L51: Section 1: Lossless Optimizations；L73: Choosing what goes in `--layerwise-offload-components`；L105: Prefetch depth has a knee；L118: Host memory is part of the placement decision；L135: When the transfer knobs do nothing；原定位：LingBot offload on B300
