# 从 profile 找归因并纠正错误假设

源码调用次数和区间名称只能形成假设；关键路径判断需要相关联的 CPU/GPU、rank 与状态证据。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- observer_preserves_execution_path: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- capture_covers_target_phases_and_ranks: true

最小验证：先用轻量计时定位，再采集必要 rank/phase；核对 kernel 命中与 observer 扰动，不把未采到活动当作可消除空闲。

## 已整理案例

### saved-tensor census 确认存储类别，不等于全部 memcpy 归因

- ID：helios-saved-tensor-census
- 状态：source_contract / HISTORICAL_DIAGNOSTIC_CENSUS
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：pass-through hooks 不保留 Tensor，确认三种 BF16 boundary shape 各 80 次；1,302 packs 与 1,298 unpacks 差异被保留，没有人为补齐。
- 经验：先用不改变策略的所有权 census 形成候选；没有 runtime correlation 时不能宣称所有同尺寸 D2H 都来自这些句柄。
- 范围：strict-v4 8 Generator ranks，详细观测时间仅 diagnostic。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"observer_preserves_execution_path": true}
- 收益条件：{"capture_covers_target_phases_and_ranks": true}
- 失败分类：not_established
- 重试信号：保存张量形状/数量或 replay 结构改变
- 最小验证：用不保留 tensor 的 pass-through hooks 重做 shape/lifetime census
- 不应重复：不得人为补齐 pack/unpack 差值或让观察器延长生命周期
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-saved-tensor-census`
- 关联实现：[保存张量的无策略改变 census](implementations.md#legacy-saved-tensor-census)；historical_reference；保留的旧 Route-A 实现参考；不是新 workload 的现成接入或当前默认配置。
- 函数入口：[runtime/saved_tensor_census.py::install_saved_tensor_census_hooks](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/saved_tensor_census.py#L164-L193)
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_saved_tensor_census_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_saved_tensor_census_result_20260810.md#L1)；定位：报告范围 L1–L40；L1: Iteration 302: Step10-A saved-tensor ownership census；L35: Evidence；原定位：结果、合同、限制与失败记录

### 30 个 subgroup 的源码计数没有解释 50 秒 transition

- ID：helios-group-count-hypothesis-refuted
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：源码审计曾怀疑 group/barrier 主导；分段实测 group creation 仅 0.666 秒，而 switching Critic H2D 达 33.169 秒。
- 经验：源代码次数用于提出假设，计时用于推翻假设；等待很长的 retained rank 可能只是等另一 rank 的 materialization。
- 范围：Iteration291→293、49.839 秒 transition 的带观测运行。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"observer_preserves_execution_path": true}
- 收益条件：{"capture_covers_target_phases_and_ranks": true}
- 失败分类：source_only_attribution_refuted
- 重试信号：新 profile 显示 group/barrier 占比确实变大
- 最小验证：分段量化 group creation 与角色 H2D 的暴露时间
- 不应重复：源码调用次数不能推翻已测 0.666 对 33.169 秒差异
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-group-count-hypothesis-refuted`
- 关联实现：[Nsight SQLite 区间与 rank 覆盖分析](implementations.md#profile-attribution)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/search/profile.py::parse_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L140-L211)
- 函数入口：[src/distillschedule/search/profile.py::validate_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L214-L232)
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_transition_optimization_audit_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_transition_optimization_audit_20260810.md#L1)；定位：报告范围 L1–L189；L1: Iteration 291: strict-v4 Step10-A and representation-transition optimization audit；L3: Status and scope；L17: Frozen contract；L31: 1. Step10-A: what the existing evidence says；L82: Step10-A optimization opportunities, ranked；L107: 2. Live representation transition: what is long；L137: Transition optimization opportunities, ranked；L159: 3. Quantitative ceiling and bounded next experiment；原定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / rb16_transition_causal_profile_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_transition_causal_profile_20260810.md#L25)；定位：观察中的数值文本锚点 L25；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### 训练结束而 Nsight teardown 卡住，不能声称完成全 rank profile

- ID：helios-profile-teardown-incomplete
- 状态：inconclusive / HISTORICAL_INCOMPLETE_PROFILE
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：固定内存流水的 node0 八个 rank 停在 cudaProfilerStop/teardown，最终报告不完整。低开销计时记录与缺失 profile 分开保留。
- 经验：进程算完训练不等于采集产物可用；确认最终文件、rank 覆盖和时间关联后再解释 overlap。
- 范围：该 GAN pipeline Nsight capture，不能从残片推出全系统归因。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"observer_preserves_execution_path": true}
- 收益条件：{"capture_covers_target_phases_and_ranks": true}
- 失败分类：not_established
- 重试信号：profiler teardown 与 rank 收尾可可靠完成
- 最小验证：缩小 capture 验证所有 rank/阶段和文件完整性
- 不应重复：不把缺失 trace 补成全程观察或否定另存的低开销 timer
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-profile-teardown-incomplete`
- 关联实现：[Nsight SQLite 区间与 rank 覆盖分析](implementations.md#profile-attribution)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/search/profile.py::parse_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L140-L211)
- 函数入口：[src/distillschedule/search/profile.py::validate_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L214-L232)
- 来源：[sliortega295-ops/DMD-System / gan_systems_experiment_20260811.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/docs/status/gan_systems_experiment_20260811.md#L1)；定位：报告范围 L1–L161；L1: Systems-only GAN Baseline/Pipeline execution report；L6: Environment；L21: Systems-only GAN bootstrap；L43: PR #4 Sequential GAN baseline；L59: PR #5 persistent 8G+8Score pipeline；L77: Fairness；L91: Formal timing；L107: Nsight Sequential；原定位：结果与证据边界

### W5 晚期 backward 的完整 rank/copy 覆盖是归因前提

- ID：h100-partial-profile-rank-coverage
- 状态：source_contract / HISTORICAL_PROFILE_GAP
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：handoff 中需要核对全部八个 rank 的晚期阶段及每 rank 600 次 boundary copies；部分 profile 不能替代完整覆盖。
- 经验：采样到一个快片段不代表整个训练周期；把覆盖缺口作为 unknown，不能直接推总传输或最终收益。
- 范围：W5 H100 的 checkpoint-boundary copy 归因待办，不代表本次启动了采集。
- 历史上下文：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W5", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"observer_preserves_execution_path": true}
- 收益条件：{"capture_covers_target_phases_and_ranks": true}
- 失败分类：not_established
- 重试信号：八个 rank 的晚期阶段与 boundary receipts 齐全
- 最小验证：逐 rank 对齐 600 次 copy 与阶段结束标记
- 不应重复：部分 profile 不能替代完整 rank 覆盖
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case h100-partial-profile-rank-coverage`
- 关联实现：[Nsight SQLite 区间与 rank 覆盖分析](implementations.md#profile-attribution)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/search/profile.py::parse_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L140-L211)
- 函数入口：[src/distillschedule/search/profile.py::validate_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L214-L232)
- 来源：[sliortega295-ops/DMD-System / h100_campaign_handoff_20261016_r47.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/campaign/h100_campaign_handoff_20261016_r47.md#L1)；定位：报告范围 L1–L33；L1: R47 — SSH recovered; W2 completed; W1 canary import repaired; foreign GPU use blocks launch；L5: New measured W2 result；L17: W1 actual cheap failure, not new model/OOM/numeric failure；L24: Actual resource blocker after reconnect；L28: Next authorized serial work；原定位：该结果及 qualification / NOT_RUN 边界

### ResearchBoost：backward hook 名称可能把上游 kernel 算到错误模块

- ID：rb-profile-backward-labels
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：tiered profiling card 提醒 view/in-place/compiled regions 的 full backward hook 范围可能跨模块。
- 经验：需要时用 forward/backward sequence 关系与 enclosing ranges 交叉定位；编译区内部不可凭标签拆出虚构精确归因。
- 范围：外部 PyTorch profiling 方法，具体 profiler 字段需本地确认。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"observer_preserves_execution_path": true}
- 收益条件：{"capture_covers_target_phases_and_ranks": true}
- 失败分类：not_established
- 重试信号：标签与 autograd 实际跨度能够对齐
- 最小验证：用少量原始 op/correlation 检查 view/in-place/compiled 区域
- 不应重复：不能把 full backward hook 标签直接当作单模块成本
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case rb-profile-backward-labels`
- 关联实现：[Nsight SQLite 区间与 rank 覆盖分析](implementations.md#profile-attribution)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/search/profile.py::parse_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L140-L211)
- 函数入口：[src/distillschedule/search/profile.py::validate_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L214-L232)
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-tiered-profiling/SKILL.md#L1)；定位：报告范围 L1–L93；L6: Tiered profiling；L10: Split by training phase first, then by the parts of one step；L17: The parts of one training step；L54: Three kinds of hotspot；L64: Identifying the bottleneck type: by ablation, not by nvidia-smi；L80: Accounting conventions；L86: Kernel level (Nsight Compute)；原定位：Do not attribute backward with a module full backward hook

### ResearchBoost：详细 observer 本身会改变 compile 路径

- ID：rb-observer-may-change-execution
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：其 profiling card 要求 plain 与 detailed trace 核对 kernel 类别/次数；dispatch 拦截可能触发 eager。
- 经验：先验证观测器没有换执行路径；profile 数值与 unprofiled 正式时间承担不同用途。
- 范围：外部 compile training 的观察方法；不强制安装某个 profiler。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"observer_preserves_execution_path": true}
- 收益条件：{"capture_covers_target_phases_and_ranks": true}
- 失败分类：not_established
- 重试信号：详细观察器不会改变被测执行路径
- 最小验证：对照 plain/detailed 的 kernel 类别和调用次数，再讨论归因
- 不应重复：不得用被 dispatch 改成 eager 的 trace 代表正常性能
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case rb-observer-may-change-execution`
- 关联实现：[Nsight SQLite 区间与 rank 覆盖分析](implementations.md#profile-attribution)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/search/profile.py::parse_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L140-L211)
- 函数入口：[src/distillschedule/search/profile.py::validate_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L214-L232)
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-tiered-profiling/SKILL.md#L1)；定位：报告范围 L1–L93；L6: Tiered profiling；L10: Split by training phase first, then by the parts of one step；L17: The parts of one training step；L54: Three kinds of hotspot；L64: Identifying the bottleneck type: by ablation, not by nvidia-smi；L80: Accounting conventions；L86: Kernel level (Nsight Compute)；原定位：Fine-grained observation

### H3 实测大拷贝可对应 boundary storage，但暴露区间不是可直接省掉的时间

- ID：helios-h3-copy-interval-attribution
- 状态：source_contract / HISTORICAL_PARTIAL_MARKER_DIAGNOSTIC
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：16-rank Nsight 将 300 D2H+300 H2D/rank 与 boundary receipts 对齐；NCCL 与 copy 的暴露区间有交集，两者不能直接相加。两个 rank 的 Student marker 还需推断。
- 经验：分清 pooled GPU-seconds、最坏 rank proxy、应用 forward 与 checkpoint replay；partial marker provenance 和 UNKNOWN_NO_GPU_ACTIVITY 必须保留。
- 范围：A100 MiniMax-H3 768x1344/124 frames、Student1/Fake5、该诊断 profile，无新正式时间。
- 历史上下文：{"model_family": "MiniMax-H3", "training_mode": "training", "resolution": "768x1344", "trace": null, "workload": "W5", "frames": 124, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"observer_preserves_execution_path": true}
- 收益条件：{"capture_covers_target_phases_and_ranks": true}
- 失败分类：not_established
- 重试信号：全部 rank 的 copy、NCCL 与 phase 边界可对齐
- 最小验证：按区间并集/交集测暴露时间，标记推断 Student 边界
- 不应重复：不把重叠 NCCL/copy 时间直接相加
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-h3-copy-interval-attribution`
- 关联实现：[H3 checkpoint 边界 CPU offload](implementations.md#h3-boundary-offload)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/backends/training/minimax_h3/checkpoint_boundary_offload.py::install_checkpoint_boundary_cpu_offload](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/checkpoint_boundary_offload.py#L187-L336)
- 来源：[sliortega295-ops/DMD-System / minimax_h3_native_nsys_attribution_20260818.md](https://github.com/sliortega295-ops/DMD-System/blob/17ab1ac8cfc21684b1eaa5cfa83fed7fa70261d8/docs/experiments/minimax_h3_native_nsys_attribution_20260818.md#L1)；定位：报告范围 L1–L147；L1: Iteration 301 — MiniMax-H3 Native Nsight Attribution；L3: Status and claim boundary；L12: Workload identity；L34: Correctness / capture gates；L42: Native cycle decomposition (overlap-aware)；L61: NCCL collective breakdown；L73: Boundary-offload evidence；L85: Stage decomposition；原定位：结果、合同、限制与失败记录

### GEMM/FA2 occupancy 低，不代表有同等比例的系统提速空间

- ID：helios-ncu-low-occupancy-not-low-throughput
- 状态：source_contract / HISTORICAL_KERNEL_DIAGNOSTIC
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：代表性 GEMM occupancy 约 12.5%，Tensor 活跃却约 88.3%；FA2 occupancy 约 12.36%，Tensor 活跃约 70%，DRAM 不足 3%。这些仅是各一个目标 kernel 的 NCU 诊断。
- 经验：结合实际 pipeline 利用、调用权重和关键路径估算上限；不要把 occupancy、issue-slot 建议或 sum-duration 直接转换为 E2E 收益。
- 范围：GAN-on 两个不同 substrate 的 A100 NCU 单 kernel，replay wall 不可作正式计时。
- 历史上下文：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"observer_preserves_execution_path": true}
- 收益条件：{"capture_covers_target_phases_and_ranks": true}
- 失败分类：not_established
- 重试信号：代表性目标 kernel 和限制因素有新证据
- 最小验证：联合看 Tensor 活跃、DRAM、shape 和关键路径，不只看 occupancy
- 不应重复：单 kernel 约 12% occupancy 不证明整模型吞吐低
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-ncu-low-occupancy-not-low-throughput`
- 关联实现：[Nsight SQLite 区间与 rank 覆盖分析](implementations.md#profile-attribution)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/search/profile.py::parse_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L140-L211)
- 函数入口：[src/distillschedule/search/profile.py::validate_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L214-L232)
- 来源：[sliortega295-ops/DMD-System / iteration358_c7_ncu.md](https://github.com/sliortega295-ops/DMD-System/blob/d5ff09623a1f57a8194619b24fc633fd51569fff/docs/experiments/gan_cuda_baseline_opt_20260829/iteration358_c7_ncu.md#L1)；定位：报告范围 L1–L82；L1: Iteration 358: C7 dominant Critic GEMM NCU diagnostic；L3: Claim boundary；L22: Kernel result；L56: Decision；L69: Remote evidence；原定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / iteration374_c19_flash_ncu.md](https://github.com/sliortega295-ops/DMD-System/blob/91ca0e007f64cfb857089fd73957669397754a8c/docs/experiments/gan_cuda_baseline_opt_20260829/iteration374_c19_flash_ncu.md#L1)；定位：报告范围 L1–L72；L1: Iteration 374: C19 production FlashAttention NCU diagnostic；L3: Frozen identity and validity；L28: Hardware result；L53: Claim boundary and decision；原定位：结果、合同、限制与失败记录
