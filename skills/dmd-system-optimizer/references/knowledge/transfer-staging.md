# Pinned staging、NUMA 与搬运的完整成本

搬运变快取决于源存储、并发、注册与清理成本；增加异步和缓冲不保证关键路径变短。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- buffer_ownership_validated: true
- copy_completion_before_consumer: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- transfer_exposed: true
- host_capacity_and_bandwidth_available: true

最小验证：先确认拷贝的所有者和生命周期；比较节点并发下的准备、DMA、等待与清理，另报持续多周期成本。

## 已整理案例

### 同步 pinned checkpoint boundaries 正确但变慢

- ID：helios-sync-pinned-boundaries-regress
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：三种已确认 boundary 改为同步 pinned pack/unpack，Full5 correctness 通过；Generator phase median 115.489513→133.831677 秒，额外 pinned storage 约 14 GiB/rank。
- 经验：新增同步 pin/copy 未必替换原有传输；同时容纳 target cache 与 boundary storage 还可能超过 host 限制。
- 范围：Iteration303 的同步 boundary offload；不否定不同实现的异步预取。
- 条件：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：additional_synchronous_staging
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_boundary_pinned_sync_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_boundary_pinned_sync_result_20260810.md)；定位：结果、合同、限制与失败记录

### 减少 G3 collective 数量后，广播关键路径反而变长

- ID：helios-bounded-pin-bucket-regression
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：pinned64x2+128MiB buckets 使 H2D 改善，但 G3 broadcast 10.868→15.400 秒；退回逐张量广播仍未达到 transition 的目标。
- 经验：bucket 数量减少不等于通信更快；节点并发 materialization、rendezvous 与临时存储会改变净效果。
- 范围：strict-v4 transition 的两个 diagnostic 候选，未做正式重复。
- 条件：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：coalescing_net_cost
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / rb16_transition_pinned_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_transition_pinned_result_20260810.md)；定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / rb16_transition_pinned_legacy_g3_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_transition_pinned_legacy_g3_result_20260810.md)；定位：结果、合同、限制与失败记录

### persistent pinned target 明显缩短传输，也带来准备与清理成本

- ID：helios-persistent-pin-cold-cost
- 状态：positive_observation / HISTORICAL_DIAGNOSTIC_POSITIVE
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：transition 49.839→30.887 秒；direct H2D 最大 2.770 秒，但同步 pinned trim 达 11.277 秒。cold pin 准备每 switching rank 31–42 秒，需大额 host storage。
- 经验：把 pin 的时机从热路径移出可有用；必须保留冷准备、host 峰值、清理和复用周期的成本。
- 范围：十个预构造 target、strict-v4 固定角色转换，n=1 diagnostic。
- 条件：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / rb16_transition_persistent_pinned_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_transition_persistent_pinned_result_20260810.md)；定位：结果、合同、限制与失败记录

### 清理放到后台会干扰通信；晚点清理又会干扰 Critic

- ID：helios-background-cleanup-contention
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：立即后台 trim：transition 30.887→40.980 秒；等布局 ready 再 trim：transition 23.838 秒达标，但 Full5 350.064 秒，比同步清理方案更慢。
- 经验：异步任务即使最终 join 几乎不等，也可能争用主路径资源；局部阶段达标不能覆盖端到端倒退。
- 范围：两种 cleanup 依赖点，exact Full5 均通过、正式重复 NOT_RUN。
- 条件：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：host_cleanup_interference
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / rb16_transition_immediate_background_cleanup_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_transition_immediate_background_cleanup_result_20260810.md)；定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / rb16_transition_late_cleanup_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_transition_late_cleanup_result_20260810.md)；定位：结果、合同、限制与失败记录

### 缓存保留至语义边界后，仍需报告后续串行清理

- ID：helios-persistent-cache-inclusive-time
- 状态：positive_observation / HISTORICAL_REPEAT_TARGETS_WITH_SCOPE
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：单次 exact 运行 Full5 306.015 秒，边界后 trim 12.605 秒；随后联合方案三次 Full5 中位 279.934 秒，含清理中位 293.016 秒。
- 经验：合法计时窗口外的成本不应消失；持续运行用户应看 inclusive makespan。batch12 资源点也不能与 batch16 直接宣称同工作量加速。
- 范围：strict-v4 4G12C，Critic batch12，联合方案含 block leaves。
- 条件：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / rb16_transition_persistent_cache_post_full5_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_transition_persistent_cache_post_full5_result_20260810.md)；定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_transition_combined_formal_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_transition_combined_formal_result_20260810.md)；定位：结果、合同、限制与失败记录

### CPU staged H3 候选卡住，证据不足以称为 OOM

- ID：h100-w5-cpu-stage-stall
- 状态：inconclusive / HISTORICAL_STALL_UNRESOLVED
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：公开 stall 回执没有 CUDA 或 host OOM 证据；futex 等待根因未确认，candidate timing NOT_RUN。
- 经验：保留“卡住、原因未知”的分类；不能把诊断受限猜成内存不足，也不能为失败填一个速度。
- 范围：W5 CPU-staged candidate stall r1。
- 条件：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W5", "gpu_family": "H100", "gpu_memory_gb": 80}
- 失败分类：unknown_stall
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / w5_cpu_staged_candidate_stall_r1.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W5/reports/w5_cpu_staged_candidate_stall_r1.json)；定位：该结果及 qualification / NOT_RUN 边界

### 历史线索：整进程 NUMA 绑定可能把 staging 优化变成 host 容量问题

- ID：helios-numa-whole-process-lead
- 状态：unverified / UNVERIFIED_CONVERSATION_LEAD
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：本地历史摘要提到整进程绑定把权重、Adam 和激活一起限制到单 NUMA 域；尚未为此记录找到并公开可复核运行回执。
- 经验：区分整进程内存策略与有界 staging 分配策略；在当前节点先检查实际拓扑、页位置和容量。
- 范围：未核实的 Helios 历史线索；不发布性能数值或已证实根因。
- 条件：{"training_mode": "training", "numa_topology": null, "host_memory_policy": null}
- 失败分类：not_publicly_verified
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：仅本地会话线索，未公开原始会话；缺少可核实的公开实验回执。

### 历史线索：单卡传输收益可能在节点并发时缩水

- ID：helios-single-gpu-staging-scale-lead
- 状态：unverified / UNVERIFIED_CONVERSATION_LEAD
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：本地摘要描述单卡与多 rank 同时 staging 收益差异；没有本轮公开的 source-qualified 比较，竞争原因未证实。
- 经验：先保留可检验假设：共享内存带宽/NUMA/PCIe 竞争可能限制扩展；用目标并发和最坏 rank 实测。
- 范围：未核实跨规模经验，不给出加速倍数。
- 条件：{"training_mode": "training", "concurrent_transfers": null}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：仅本地会话线索，未公开原始会话；缺少可核实的公开实验回执。
