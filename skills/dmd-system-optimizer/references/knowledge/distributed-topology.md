# 分布式布局、ZeRO leaf 与通信粒度

分片域与参数获取粒度会改变通信和峰值；必须保留实际训练图与所有 rank 的集合语义。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- collective_membership_order_validated: true
- logical_state_mapping_preserved: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- topology_costs_measured: true
- grouping_fits_capacity: true

最小验证：用真实模型类、dtype 和各角色频率清点分组；先做非成员 rank 与阶段切换测试，再比较目标拓扑。

## 已整理案例

### ZeRO-3 default-world 假设会让部分 rank 初始化或 forward 卡住

- ID：helios-zero3-subgroup-default-world
- 状态：implementation_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：DP8 引擎置于更大 world16 内时，隐式 world barrier 与后续 forward 均暴露非成员问题；缩小门验证后改为独立 default worlds，再显式重建目标 world。
- 经验：参数/optimizer subgroup 配置正确不代表库内部所有 collective 都遵从它；测试非成员 rank 与真实阶段切换。
- 范围：固定 DeepSpeed 0.18.4 与该多角色运行器；不是所有 ZeRO-3 subgroup 都不可用。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"collective_membership_order_validated": true, "logical_state_mapping_preserved": true}
- 收益条件：{"topology_costs_measured": true, "grouping_fits_capacity": true}
- 失败分类：default_world_collective_assumption
- 重试信号：engine/default-world 成员关系已有可验证修复
- 最小验证：先用最小多组用例覆盖 barrier、forward、销毁和重建顺序
- 不应重复：不在未修成员关系的 world16 中重复挂起全模型
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-zero3-subgroup-default-world`
- 来源：[sliortega295-ops/DMD-System / rb16_full5_2g6c_continuous_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_full5_2g6c_continuous_20260810.md#L1)；定位：报告范围 L1–L220；L1: RB16 continuous Full5: 8G4S4T to per-node 2G6C；L3: Scope；L17: Preflight decision；L31: Status；L57: Preserved transition attempts；L81: Iteration 281 correction: Step10-A backend parity；L113: Iteration 281 subgroup-initialization correction；L135: Iteration 282 control and process-world correction；原定位：结果、合同、限制与失败记录

### 小参数 persistence 与同步 prefetch 有收益，但未达到当次目标

- ID：helios-persistence-window
- 状态：positive_observation / HISTORICAL_STEP_ONLY_BOUNDARY_PASS
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：严格 Step10-A 中位 123.115506→116.829453 秒；扩大到约一块 prefetch/两块 live 后 115.489513 秒，后一步边际约 1.15%。正式 Full5 配对未完成。
- 经验：保持“观察到改善”与“是否达到探索目标”分离；默认 stream 上的预取不等于与计算重叠。
- 范围：strict-v4、ZeRO-3、2M persistence、overlap_comm=false。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"collective_membership_order_validated": true, "logical_state_mapping_preserved": true}
- 收益条件：{"topology_costs_measured": true, "grouping_fits_capacity": true}
- 失败分类：not_established
- 重试信号：当前 live/prefetch 窗口和容量允许更多复用
- 最小验证：记录 gather 次数与峰值，跨 G/C/EMA 验证后做 Full5 配对
- 不应重复：Step10-A 的边际改善不能直接外推 Full5
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-persistence-window`
- 关联实现：[ZeRO-3 模块粒度与 parent-first leaf 选择](implementations.md#legacy-zero3-leaf)；adaptation_example；机制参考或旧版本示例；需核对实现差异并适配，未证明是历史计时所用源码。
- 函数入口：[runtime/zero3_module_granularity.py::census_zero3_module_granularity](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/zero3_module_granularity.py#L25-L86)
- 函数入口：[runtime/zero3_module_granularity.py::selected_preorder_leaf_records](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/zero3_module_granularity.py#L89-L111)
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_2m_persistence_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_2m_persistence_result_20260810.md#L9)；定位：观察中的数值文本锚点 L9；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_prefetch_live_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_prefetch_live_result_20260810.md#L13)；定位：观察中的数值文本锚点 L13；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### 打开异步通信后，比同步窗口更慢且占更多显存

- ID：helios-overlap-comm-regression
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：同一窗口将 overlap_comm 打开，Generator median 115.489513→120.465448 秒，峰值从 31,762 MiB 上升至约 33,314–33,514 MiB；边界身份一致。
- 经验：单独 CUDA stream 是实现条件，实际净重叠才是收益；额外 buffers、依赖与竞争需一起测。
- 范围：该 strict-v4 Step10-A、两波 checkpoint backward。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"collective_membership_order_validated": true, "logical_state_mapping_preserved": true}
- 收益条件：{"topology_costs_measured": true, "grouping_fits_capacity": true}
- 失败分类：overlap_net_regression
- 重试信号：通信可重叠空间或内存压力有实测变化
- 最小验证：同窗口比较 compute/comm 暴露区间、峰值和总周期
- 不应重复：开启 overlap_comm 不自动等于更快
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-overlap-comm-regression`
- 关联实现：[ZeRO-3 模块粒度与 parent-first leaf 选择](implementations.md#legacy-zero3-leaf)；adaptation_example；机制参考或旧版本示例；需核对实现差异并适配，未证明是历史计时所用源码。
- 函数入口：[runtime/zero3_module_granularity.py::census_zero3_module_granularity](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/zero3_module_granularity.py#L25-L86)
- 函数入口：[runtime/zero3_module_granularity.py::selected_preorder_leaf_records](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/zero3_module_granularity.py#L89-L111)
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_overlap_comm_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_overlap_comm_result_20260810.md#L9)；定位：观察中的数值文本锚点 L9；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### 按真实 Transformer 类分组优于粗粒度阈值混选

- ID：helios-block-leaf-grouping
- 状态：positive_observation / HISTORICAL_THREE_RUN_TARGET_MET
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：8M 标量阈值误选 mixed-dtype embedding 而失败；仅 40 个 Transformer blocks 成为 ZeRO leaves 后 exact Full5 通过，联合优化三轮 Step10-A world max 为 100.689–104.775 秒。
- 经验：分组要保留 dtype 和调用图；用显式类/实例 inventory 防止连带选择，联合结果不能拆成未测的单项贡献。
- 范围：strict-v4 block-only fetch；正式联合方案还包含 persistent pinned transition cache。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"collective_membership_order_validated": true, "logical_state_mapping_preserved": true}
- 收益条件：{"topology_costs_measured": true, "grouping_fits_capacity": true}
- 失败分类：not_established
- 重试信号：新的模块 census 能排除 mixed-dtype 和重叠 leaf
- 最小验证：先校验父子去重与 dtype，再做完整 Full5 和通信粒度对比
- 不应重复：不重复用粗标量阈值把 mixed-dtype embedding 纳入
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-block-leaf-grouping`
- 关联实现：[ZeRO-3 模块粒度与 parent-first leaf 选择](implementations.md#legacy-zero3-leaf)；adaptation_example；机制参考或旧版本示例；需核对实现差异并适配，未证明是历史计时所用源码。
- 函数入口：[runtime/zero3_module_granularity.py::census_zero3_module_granularity](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/zero3_module_granularity.py#L25-L86)
- 函数入口：[runtime/zero3_module_granularity.py::selected_preorder_leaf_records](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/zero3_module_granularity.py#L89-L111)
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_explicit_block_leaf_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_explicit_block_leaf_result_20260810.md#L1)；定位：报告范围 L1–L47；L1: Iteration 307: explicit Transformer-block ZeRO-3 leaves；L11: Exact continuous Full5；L31: Low-overhead Step10 diagnostic；原定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_transition_combined_formal_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_transition_combined_formal_result_20260810.md#L34)；定位：观察中的数值文本锚点 L34；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### No-GAN 的 node-HSDP 有小幅历史改善

- ID：helios-hsdp-nogan-positive
- 状态：positive_observation / HISTORICAL_UNEQUAL_REPEAT_COUNTS
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：历史 world16 FSDP2 三次中位 118.546757 秒，node-HSDP 两次中位 117.304613 秒；重复次数与采样不同。
- 经验：节点内分片可以缩小通信域，但该比较不能当作当前新实现的三次匹配证据。
- 范围：Helios14B No-GAN，replicate2/shard8，B1/global16。
- 历史上下文：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": false, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"collective_membership_order_validated": true, "logical_state_mapping_preserved": true}
- 收益条件：{"topology_costs_measured": true, "grouping_fits_capacity": true}
- 失败分类：not_established
- 重试信号：当前拓扑带宽/内存与 No-GAN 调用图支持 HSDP
- 最小验证：核对 batch/reduction 和实际 mesh，同等次数交错配对
- 不应重复：不同重复次数的历史中位数不是新机器的保证
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-hsdp-nogan-positive`
- 关联实现：[Helios world16 FSDP/node-HSDP 配置](implementations.md#helios-fsdp-layout)；reusable_candidate；同机制的当前复用候选；须检查 guards、生效路径、依赖和验证，不代表历史实现身份或直接迁移通过。
- 函数入口：[src/distillschedule/runtime/fsdp2_p0.py::create_device_mesh](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_p0.py#L273-L288)
- 函数入口：[src/distillschedule/runtime/fsdp2_p0.py::apply_blockwise_fsdp2_policy](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_p0.py#L291-L398)
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_LADDER_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/c93c2736e0fc528e72e63229998d21d87181d346/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_LADDER_20260825.md#L7)；定位：观察中的数值文本锚点 L7；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### GAN-on node-HSDP 反而变慢；构造修复需单独保留

- ID：helios-hsdp-gan-negative
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：同 GAN-on C5 substrate，world16 FSDP 142.295342→node-HSDP 144.716707 秒。先前构造 OOM 由 CPU restore+blockwise sharding 修复。
- 经验：保留有效的构造容量修复，不把它与最终并行布局收益混为一谈；GAN 与 No-GAN 的结论可以相反。
- 范围：GAN-on 384x640 B1、local FA2、selective checkpoint retention。
- 历史上下文：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"collective_membership_order_validated": true, "logical_state_mapping_preserved": true}
- 收益条件：{"topology_costs_measured": true, "grouping_fits_capacity": true}
- 失败分类：topology_runtime_regression
- 重试信号：GAN 通信重叠和分区开销有实测变化
- 最小验证：先确认构造峰值修复，再同 G1/C5/EMA1 比较 FSDP/HSDP
- 不应重复：不能因 No-GAN 有利就重复 GAN 的慢配置
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-hsdp-gan-negative`
- 关联实现：[Helios world16 FSDP/node-HSDP 配置](implementations.md#helios-fsdp-layout)；reusable_candidate；同机制的当前复用候选；须检查 guards、生效路径、依赖和验证，不代表历史实现身份或直接迁移通过。
- 函数入口：[src/distillschedule/runtime/fsdp2_p0.py::create_device_mesh](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_p0.py#L273-L288)
- 函数入口：[src/distillschedule/runtime/fsdp2_p0.py::apply_blockwise_fsdp2_policy](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_p0.py#L291-L398)
- 来源：[sliortega295-ops/DMD-System / iteration357_c6_hsdp.md](https://github.com/sliortega295-ops/DMD-System/blob/6d88e4d45ab07e3fd54bc9edacc3954ee6168937/docs/experiments/gan_cuda_baseline_opt_20260829/iteration357_c6_hsdp.md#L1)；定位：报告范围 L1–L54；L1: Iteration 357: C6 node-HSDP placement result；L3: Claim boundary；L13: Construction result；L29: Gates and formal result；L42: Decision；原定位：结果、合同、限制与失败记录

### 强制 NCCL protocol 的 canary 未发现优势

- ID：helios-nccl-protocol-canary
- 状态：negative_observation / HISTORICAL_CANARY_NEGATIVE
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：16 A100 的指定 shard/reduce-scatter/GEMM 混合，Simple 和 LL128 相对默认均未改善 overlap 时间；未做全模型运行。
- 经验：看通信与计算混合后的时间，而不只看孤立 collective；版本、消息大小与拓扑改变后才重新校准。
- 范围：Torch2.10/NCCL2.27.5、44.8MiB shard、128MiB reduce-scatter。
- 历史上下文：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"collective_membership_order_validated": true, "logical_state_mapping_preserved": true}
- 收益条件：{"topology_costs_measured": true, "grouping_fits_capacity": true}
- 失败分类：no_measured_overlap_advantage
- 重试信号：实际消息规模/拓扑或 compute overlap 改变
- 最小验证：先测对应 collective+GEMM mix，再决定是否值得全模型
- 不应重复：不要把协议名字或孤立带宽当成训练加速
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-nccl-protocol-canary`
- 来源：[sliortega295-ops/DMD-System / iteration371_c16_nccl_protocol.md](https://github.com/sliortega295-ops/DMD-System/blob/5fab44949b0d9184fe7fb868ee91c79b91ebf174/docs/experiments/gan_cuda_baseline_opt_20260829/iteration371_c16_nccl_protocol.md#L1)；定位：报告范围 L1–L44；L1: Iteration 371: C16 NCCL protocol canary；L3: Question and boundary；L15: Result；L36: Decision；原定位：结果、合同、限制与失败记录
