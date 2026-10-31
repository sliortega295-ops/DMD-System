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
- 条件：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：default_world_collective_assumption
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / rb16_full5_2g6c_continuous_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_full5_2g6c_continuous_20260810.md)；定位：结果、合同、限制与失败记录

### 小参数 persistence 与同步 prefetch 有收益，但未达到当次目标

- ID：helios-persistence-window
- 状态：positive_observation / HISTORICAL_STEP_ONLY_BOUNDARY_PASS
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：严格 Step10-A 中位 123.115506→116.829453 秒；扩大到约一块 prefetch/两块 live 后 115.489513 秒，后一步边际约 1.15%。正式 Full5 配对未完成。
- 经验：保持“观察到改善”与“是否达到探索目标”分离；默认 stream 上的预取不等于与计算重叠。
- 范围：strict-v4、ZeRO-3、2M persistence、overlap_comm=false。
- 条件：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_2m_persistence_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_2m_persistence_result_20260810.md)；定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_prefetch_live_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_prefetch_live_result_20260810.md)；定位：结果、合同、限制与失败记录

### 打开异步通信后，比同步窗口更慢且占更多显存

- ID：helios-overlap-comm-regression
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：同一窗口将 overlap_comm 打开，Generator median 115.489513→120.465448 秒，峰值从 31,762 MiB 上升至约 33,314–33,514 MiB；边界身份一致。
- 经验：单独 CUDA stream 是实现条件，实际净重叠才是收益；额外 buffers、依赖与竞争需一起测。
- 范围：该 strict-v4 Step10-A、两波 checkpoint backward。
- 条件：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：overlap_net_regression
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_overlap_comm_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_overlap_comm_result_20260810.md)；定位：结果、合同、限制与失败记录

### 按真实 Transformer 类分组优于粗粒度阈值混选

- ID：helios-block-leaf-grouping
- 状态：positive_observation / HISTORICAL_THREE_RUN_TARGET_MET
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：8M 标量阈值误选 mixed-dtype embedding 而失败；仅 40 个 Transformer blocks 成为 ZeRO leaves 后 exact Full5 通过，联合优化三轮 Step10-A world max 为 100.689–104.775 秒。
- 经验：分组要保留 dtype 和调用图；用显式类/实例 inventory 防止连带选择，联合结果不能拆成未测的单项贡献。
- 范围：strict-v4 block-only fetch；正式联合方案还包含 persistent pinned transition cache。
- 条件：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_explicit_block_leaf_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_explicit_block_leaf_result_20260810.md)；定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_transition_combined_formal_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_transition_combined_formal_result_20260810.md)；定位：结果、合同、限制与失败记录

### No-GAN 的 node-HSDP 有小幅历史改善

- ID：helios-hsdp-nogan-positive
- 状态：positive_observation / HISTORICAL_UNEQUAL_REPEAT_COUNTS
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：历史 world16 FSDP2 三次中位 118.546757 秒，node-HSDP 两次中位 117.304613 秒；重复次数与采样不同。
- 经验：节点内分片可以缩小通信域，但该比较不能当作当前新实现的三次匹配证据。
- 范围：Helios14B No-GAN，replicate2/shard8，B1/global16。
- 条件：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": false, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_LADDER_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/c93c2736e0fc528e72e63229998d21d87181d346/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_LADDER_20260825.md)；定位：结果、合同、限制与失败记录

### GAN-on node-HSDP 反而变慢；构造修复需单独保留

- ID：helios-hsdp-gan-negative
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：同 GAN-on C5 substrate，world16 FSDP 142.295342→node-HSDP 144.716707 秒。先前构造 OOM 由 CPU restore+blockwise sharding 修复。
- 经验：保留有效的构造容量修复，不把它与最终并行布局收益混为一谈；GAN 与 No-GAN 的结论可以相反。
- 范围：GAN-on 384x640 B1、local FA2、selective checkpoint retention。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：topology_runtime_regression
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration357_c6_hsdp.md](https://github.com/sliortega295-ops/DMD-System/blob/6d88e4d45ab07e3fd54bc9edacc3954ee6168937/docs/experiments/gan_cuda_baseline_opt_20260829/iteration357_c6_hsdp.md)；定位：结果、合同、限制与失败记录

### 强制 NCCL protocol 的 canary 未发现优势

- ID：helios-nccl-protocol-canary
- 状态：negative_observation / HISTORICAL_CANARY_NEGATIVE
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：16 A100 的指定 shard/reduce-scatter/GEMM 混合，Simple 和 LL128 相对默认均未改善 overlap 时间；未做全模型运行。
- 经验：看通信与计算混合后的时间，而不只看孤立 collective；版本、消息大小与拓扑改变后才重新校准。
- 范围：Torch2.10/NCCL2.27.5、44.8MiB shard、128MiB reduce-scatter。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：no_measured_overlap_advantage
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration371_c16_nccl_protocol.md](https://github.com/sliortega295-ops/DMD-System/blob/5fab44949b0d9184fe7fb868ee91c79b91ebf174/docs/experiments/gan_cuda_baseline_opt_20260829/iteration371_c16_nccl_protocol.md)；定位：结果、合同、限制与失败记录
