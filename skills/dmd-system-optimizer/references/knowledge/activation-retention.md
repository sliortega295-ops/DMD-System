# 激活保留与 checkpoint 重算取舍

保留激活减少重算，增加存活显存；不表示参数驻留或减少 all-gather。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- recompute_semantics_preserved: true
- activation_lifetime_validated: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- recomputation_on_critical_path: true
- memory_headroom: true

最小验证：核对哪些 checkpoint 被跳过、减少的重算与完整周期激活峰值；再验证梯度和更新。

## 已整理案例

### W1：扩大到 whole-block tail8 出现 OOM

- ID：helios-w1-tail8-oom
- 状态：implementation_failure / HISTORICAL_CUDA_OOM
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：公开回执记为 CUDA_OOM_NOT_PROMOTED；候选没有有效 C2。
- 经验：容量失败限定当时保留范围；没有候选速度值，不能把它记作慢或零收益。 whole-block tail8 保留的是激活，收益来自减少重算；新增激活峰值决定容量，不是少做 gather。
- 范围：W1 当时 tail8 实现与 A100-40GB 容量。
- 历史上下文：{"workload": "W1", "gpu_memory_gib": 40}
- 正确性前提：{"recompute_semantics_preserved": true, "activation_lifetime_validated": true}
- 收益条件：{"recomputation_on_critical_path": true, "memory_headroom": true}
- 失败分类：capacity
- 重试信号：真实峰值下降或容量增加，足以覆盖 tail8 留存
- 最小验证：先在目标全周期记录逐阶段峰值与失败分配，再试更小留存范围
- 不应重复：不以 allocated 单点或微测通过替代完整周期容量
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-w1-tail8-oom`
- 结构化事实：{"retained_object": "whole_block_activations", "performance_status": "NOT_RUN", "cuda_oom": true}
- 关联实现：[选择完整 block 保留 autograd 激活](implementations.md#block-activation-retention)；reusable_candidate；同机制的当前复用候选；须检查 guards、生效路径、依赖和验证，不代表历史实现身份或直接迁移通过。
- 函数入口：[src/distillschedule/runtime/checkpoint_block_retention.py::install](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/checkpoint_block_retention.py#L11-L62)
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json)；定位：followups.wan_tail8; cleanup[0].unpaired_tail8_source_control
- 来源：[sliortega295-ops/DMD-System / wan_tail8_retention_v1.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/docs/contracts/wan_tail8_retention_v1.md#L1)；定位：tail8

### 10/40 checkpoint bypass OOM，4/40 能跑但收益极小

- ID：helios-checkpoint-bypass-capacity
- 状态：implementation_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：No-GAN 对两角色或仅 Critic 的 10/40 bypass 均 OOM；Critic-only 4/40 完成，96.529020→96.341895 秒，仅约 0.19%。
- 经验：多保留激活要同时满足容量与重算成本价值；密度调整不应在失败后无边界继续扫。
- 范围：Helios No-GAN node-HSDP、当时显存与算子组合。
- 历史上下文：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": false, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"recompute_semantics_preserved": true, "activation_lifetime_validated": true}
- 收益条件：{"recomputation_on_critical_path": true, "memory_headroom": true}
- 失败分类：checkpoint_retention_capacity
- 重试信号：目标 bypass 子集有完整周期容量余量
- 最小验证：从小子集验证 replay/梯度和峰值，再比较净收益
- 不应重复：不要重复 10/40 OOM 或因完成就推定 4/40 有显著收益
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-checkpoint-bypass-capacity`
- 关联实现：[Helios FA2 selective checkpoint 保留](implementations.md#helios-selective-checkpoint)；reusable_candidate；同机制的当前复用候选；须检查 guards、生效路径、依赖和验证，不代表历史实现身份或直接迁移通过。
- 函数入口：[src/distillschedule/runtime/fsdp2_static.py::_selective_flash_checkpoint_contexts](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L84-L111)
- 函数入口：[src/distillschedule/runtime/fsdp2_static.py::configure_fsdp2_checkpointing](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L171-L290)
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/111e293dc79978132af91ab5515a096d941b94fe/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md#L34)；定位：观察中的数值文本锚点 L34；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录
