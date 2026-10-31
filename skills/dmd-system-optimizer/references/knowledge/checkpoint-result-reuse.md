# checkpoint 中间计算结果复用

在合法 forward/replay 生命周期内复用选定算子的结果；保留参数、保留整块激活和缓存算子输出分别建模。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- cache_key_complete: true
- invalidation_validated: true
- cached_gradients_preserved: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- replay_reuses_saved_output: true
- saved_compute_exceeds_cache_cost: true
- memory_headroom: true

最小验证：检查实际 save/reuse 计数、梯度路径、版本键与释放；同时测节省的重算和新增峰值。

## 已整理案例

### W3：Critic 尾两块冻结 FFN-up 结果复用的小幅历史增量

- ID：helios-w3-retention-r1
- 状态：positive_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：C2 117.074207036→116.673757426 秒，约 0.400 秒；numerical REVIEW。
- 经验：保存 Critic blocks 38/39 的冻结 FFN-up addmm 结果，减少 checkpoint 重算；不是缓存参数，也不以 exposed_gather 解释收益。约 0.400 秒只属该源单配对。
- 范围：W3 r1 该源同源增量，非长期等价证明。
- 历史上下文：{"workload": "W3", "ranks": 16}
- 正确性前提：{"cache_key_complete": true, "invalidation_validated": true, "cached_gradients_preserved": true, "frozen_base_only": true, "grad_mode": "with_grad"}
- 收益条件：{"replay_reuses_saved_output": true, "saved_compute_exceeds_cache_cost": true, "memory_headroom": true}
- 失败分类：not_established
- 重试信号：选定 FFN-up 重算占比或可用激活容量发生变化
- 最小验证：核对 blocks38/39 的 original/replay 输出复用、输入梯度、权重版本和 graph-drain，然后补同源重复配对
- 不应重复：不按参数驻留或 gather 开销选路线；不把 0.400 秒推广到其他合同
- 代码状态：linked；关系按具体机制和代码职责标注；源码匹配不等于当前 workload 已验证。
- 定位命令：`dmd_knowledge.py locate --case helios-w3-retention-r1`
- 结构化事实：{"retained_object": "frozen_ffn_up_output", "role": "Critic", "blocks": [38, 39], "numerical_status": "REVIEW", "paired_runs": 1}
- 关联实现：[W3 r1 冻结 FFN-up checkpoint 结果复用](implementations.md#w3-frozen-ffn-up-replay)；historical_implementation；绑定该实验训练源 155ffbb5 的 up_blocks 分支；当前同名 down-only 文件不能替代此历史变体。
- 函数入口：[src/distillschedule/runtime/helios_gan_manual/down_replay.py::NativeDownSAC.contexts](https://github.com/sliortega295-ops/DMD-System/blob/155ffbb585356a29b544573a391eb456b790a15a/src/distillschedule/runtime/helios_gan_manual/down_replay.py#L66-L115)
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json#L45)；定位：观察中的数值文本锚点 L45；同时阅读同节合同、状态与限制。原定位：rows[workload=W3]
- 来源：[sliortega295-ops/DMD-System / down_replay.py](https://github.com/sliortega295-ops/DMD-System/blob/155ffbb585356a29b544573a391eb456b790a15a/src/distillschedule/runtime/helios_gan_manual/down_replay.py#L41)；定位：up_blocks in ((),(38,39))

### W6 r1：block49计算结果从CPU缓存改为GPU保留后变慢

- ID：helios-w6-retention-r1
- 状态：negative_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：C2 255.363757909→260.582584122 秒，增加约 5.219 秒；未晋级。
- 经验：保存的是原生selected-output return trees：block49由pinned CPU改留GPU，其他块保持原路径。不是参数驻留，也不减少参数gather；该次约5.219秒回退未隔离原因。
- 范围：W6 r1 的该实现和工作量，numerical REVIEW。
- 历史上下文：{"workload": "W6", "ranks": 16}
- 正确性前提：{"cache_key_complete": true, "invalidation_validated": true, "cached_gradients_preserved": true}
- 收益条件：{"saved_cache_transfer_exceeds_overhead": true, "memory_headroom": true}
- 失败分类：net_slowdown_cause_not_isolated
- 重试信号：CPU/GPU结果存储的传输代价或容量条件改变
- 最小验证：检查block49原输出树、consumer计数、autograd、版本与完整释放；量化传输差异及同源完整周期
- 不应重复：不把GPU保留计算结果当成参数驻留，不用gather解释此结果
- 代码状态：linked；关系按具体机制和代码职责标注；源码匹配不等于当前 workload 已验证。
- 定位命令：`dmd_knowledge.py locate --case helios-w6-retention-r1`
- 结构化事实：{"retained_object": "selected_operation_outputs", "block": 49, "storage_change": "pinned_cpu_to_gpu", "numerical_status": "REVIEW"}
- 归因边界：{"status": "MECHANISM_UNRESOLVED", "claim": "单次完整周期回退；未证明由哪一项开销导致。"}
- 关联实现：[W6 r1 block49 GPU结果缓存](implementations.md#h3-block49-hybrid-output-cache)；historical_implementation；固定历史source实际GPU-tail1输出缓存，不是FSDP参数驻留。
- 函数入口：[src/distillschedule/strategies/h3_ar_hybrid_cache.py::pool_class](https://github.com/sliortega295-ops/DMD-System/blob/45938af88d65ddb15b96ec1a181b65aa4e92de70/src/distillschedule/strategies/h3_ar_hybrid_cache.py#L13-L124)
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json#L89)；定位：观察中的数值文本锚点 L89；同时阅读同节合同、状态与限制。原定位：rows[workload=W6]
- 来源：[sliortega295-ops/DMD-System / h3_ar_hybrid_cache.py](https://github.com/sliortega295-ops/DMD-System/blob/45938af88d65ddb15b96ec1a181b65aa4e92de70/src/distillschedule/strategies/h3_ar_hybrid_cache.py#L4)；定位：Only block49

### 保留真实 FA2 输出可省重算，密度受最坏显存限制

- ID：helios-fa2-selective-retention
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：No-GAN 固定 local FA2 后 stride2 retention：95.725157→94.461700 秒；GAN-on 独立配对 143.206713→142.295342 秒，但最小 end driver free 仅约 12 MiB。
- 经验：核对原始 saves/replay reuses 与零物理调用 bypass；小余量的成功不能自动支持 stride1 或新 workload。
- 范围：两个独立 Helios 合同、Critic-only FA2 replay retention、均单次计时。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"cache_key_complete": true, "invalidation_validated": true, "cached_gradients_preserved": true, "attention_backend": "FA2"}
- 收益条件：{"replay_reuses_saved_output": true, "saved_compute_exceeds_cache_cost": true, "memory_headroom": true}
- 失败分类：not_established
- 重试信号：当前 FA2 调用可重放且容量覆盖全周期
- 最小验证：固定 backend 身份，再检查选择性 cache 的梯度和全程 driver-free
- 不应重复：仅末边界约 12 MiB free 不足以说明稳健容量
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-fa2-selective-retention`
- 独立子观察 `w2`：No-GAN local FA2：95.725156637→94.461699722秒；末边界driver free至少1.762085 GiB。
  - 上下文：{"model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16, "workload": "W2", "gan_enabled": false, "update_ratio": "G1/C5/EMA5"}
  - 状态：positive_observation / HISTORICAL_SINGLE_TIMING_NOT_REPRODUCED
  - 来源：[docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/111e293dc79978132af91ab5515a096d941b94fe/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md#L1)
- 独立子观察 `w3`：GAN-on：143.206713110→142.295342秒；最小末边界driver free约12MiB，不是全程容量保证。
  - 上下文：{"model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16, "workload": "W3", "gan_enabled": true, "update_ratio": "G1/C5/EMA1"}
  - 状态：positive_observation / HISTORICAL_SINGLE_TIMING_NOT_REPRODUCED
  - 来源：[docs/experiments/gan_cuda_baseline_opt_20260829/iteration355_selective_flash_checkpoint.md](https://github.com/sliortega295-ops/DMD-System/blob/7613dcedcbdca1428c0a484894bf2e421ac5f2c8/docs/experiments/gan_cuda_baseline_opt_20260829/iteration355_selective_flash_checkpoint.md#L6)
- 关联实现：[Helios FA2 selective checkpoint 保留](implementations.md#helios-selective-checkpoint)；reusable_candidate；同机制的当前复用候选；须检查 guards、生效路径、依赖和验证，不代表历史实现身份或直接迁移通过。
- 函数入口：[src/distillschedule/runtime/fsdp2_static.py::_selective_flash_checkpoint_contexts](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L84-L111)
- 函数入口：[src/distillschedule/runtime/fsdp2_static.py::configure_fsdp2_checkpointing](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L171-L290)
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/111e293dc79978132af91ab5515a096d941b94fe/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md#L1)；定位：报告范围 L1–L146；L1: Helios14B node-HSDP optimization closure (Iterations 368-382)；L3: Status and claim boundary；L27: New experiment ladder；L52: Effective methods；L54: Explicit pinned local FlashAttention-2；L70: Critic selective FA2 checkpoint-output retention；L87: Rejected and bounded-negative methods；L105: Post-winner Nsight result；原定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / iteration355_selective_flash_checkpoint.md](https://github.com/sliortega295-ops/DMD-System/blob/7613dcedcbdca1428c0a484894bf2e421ac5f2c8/docs/experiments/gan_cuda_baseline_opt_20260829/iteration355_selective_flash_checkpoint.md#L6)；定位：观察中的数值文本锚点 L6；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### H100 W5 保存 attention 输出/LSE 换取重算减少

- ID：h100-w5-attention-cache-memory-trade
- 状态：positive_observation / HISTORICAL_H100_SINGLE_TRIAL_REVIEW
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：同 9bfd source，FSDP 638.522138→GPU-cache 539.718383 秒；allocated 41.603→64.116 GiB，reserved 45.516→71.082 GiB。
- 经验：有剩余显存时可换取计算，但必须纳入训练状态、24 个 arm/rank 回执与所有更新；没有 full-state parity 仍不能晋级为正式加速。
- 范围：W5 Grid1000 approximate、Student1/Fake5、96 sample identities、单次三臂。
- 历史上下文：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W5", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"cache_key_complete": true, "invalidation_validated": true, "cached_gradients_preserved": true}
- 收益条件：{"replay_reuses_saved_output": true, "saved_compute_exceeds_cache_cost": true, "memory_headroom": true}
- 失败分类：not_established
- 重试信号：目标设备能容纳新增 GPU cache 且少重算有净收益
- 最小验证：同源检查 cache/replay 梯度、64 GiB 级 allocated 与周期
- 不应重复：把时间改善迁移前必须重查显存，不能只看 H100 型号
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case h100-w5-attention-cache-memory-trade`
- 关联实现：[H3 FA3 replay cache 与 staged 路径](implementations.md#h3-attention-replay)；reusable_candidate；同机制的当前复用候选；须检查 guards、生效路径、依赖和验证，不代表历史实现身份或直接迁移通过。
- 函数入口：[src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py::install_fa3_replay_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py#L880-L1131)
- 函数入口：[src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py::prepare_staged_cache_before_backward](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py#L854-L872)
- 来源：[sliortega295-ops/DMD-System / result.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W5/reports/completed_W5_three_arm_source_qualified_r2/result.json#L28)；定位：观察中的数值文本锚点 L28；同时阅读同节合同、状态与限制。原定位：该结果及 qualification / NOT_RUN 边界
