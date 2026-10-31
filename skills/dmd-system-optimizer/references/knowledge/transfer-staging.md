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
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"buffer_ownership_validated": true, "copy_completion_before_consumer": true}
- 收益条件：{"transfer_exposed": true, "host_capacity_and_bandwidth_available": true}
- 失败分类：additional_synchronous_staging
- 重试信号：省出容量能启用新收益或传输有真实重叠窗口
- 最小验证：在完整并发下测 boundary copy、约 14 GiB/rank host 占用与周期
- 不应重复：correctness 通过不足以重复同步 offload 的慢点
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-sync-pinned-boundaries-regress`
- 关联实现：[旧 Route-A 精确 saved-tensor offload](implementations.md#legacy-saved-tensor-offload)；adaptation_example；机制参考或旧版本示例；需核对实现差异并适配，未证明是历史计时所用源码。
- 函数入口：[runtime/activation_offload.py::install_activation_offload_hooks](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/activation_offload.py#L362-L407)
- 函数入口：[runtime/activation_offload.py::AuditedSavedTensorOffloader](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/activation_offload.py#L61-L347)
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_boundary_pinned_sync_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_boundary_pinned_sync_result_20260810.md#L23)；定位：观察中的数值文本锚点 L23；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### 减少 G3 collective 数量后，广播关键路径反而变长

- ID：helios-bounded-pin-bucket-regression
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：pinned64x2+128MiB buckets 使 H2D 改善，但 G3 broadcast 10.868→15.400 秒；退回逐张量广播仍未达到 transition 的目标。
- 经验：bucket 数量减少不等于通信更快；节点并发 materialization、rendezvous 与临时存储会改变净效果。
- 范围：strict-v4 transition 的两个 diagnostic 候选，未做正式重复。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"buffer_ownership_validated": true, "copy_completion_before_consumer": true}
- 收益条件：{"transfer_exposed": true, "host_capacity_and_bandwidth_available": true}
- 失败分类：coalescing_net_cost
- 重试信号：pin/bucket 方案能降低合计 transition 而非单项
- 最小验证：联合测 H2D、broadcast、host staging 和完成等待
- 不应重复：不因 H2D 改善就忽略 G3 broadcast 回退
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-bounded-pin-bucket-regression`
- 关联实现：[旧 Route-A pinned target 与有界 staging](implementations.md#legacy-pinned-materialization)；adaptation_example；机制参考或旧版本示例；需核对实现差异并适配，未证明是历史计时所用源码。
- 函数入口：[runtime/bounded_materialization.py::pin_module_cpu_registered_state](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L189-L238)
- 函数入口：[runtime/bounded_materialization.py::materialize_module_from_persistent_pinned_cpu](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L242-L331)
- 函数入口：[runtime/bounded_materialization.py::materialize_module_from_cpu_with_bounded_pinned_staging](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L335-L480)
- 来源：[sliortega295-ops/DMD-System / rb16_transition_pinned_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_transition_pinned_result_20260810.md#L14)；定位：观察中的数值文本锚点 L14；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / rb16_transition_pinned_legacy_g3_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_transition_pinned_legacy_g3_result_20260810.md#L11)；定位：观察中的数值文本锚点 L11；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### persistent pinned target 明显缩短传输，也带来准备与清理成本

- ID：helios-persistent-pin-cold-cost
- 状态：positive_observation / HISTORICAL_DIAGNOSTIC_POSITIVE
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：transition 49.839→30.887 秒；direct H2D 最大 2.770 秒，但同步 pinned trim 达 11.277 秒。cold pin 准备每 switching rank 31–42 秒，需大额 host storage。
- 经验：把 pin 的时机从热路径移出可有用；必须保留冷准备、host 峰值、清理和复用周期的成本。
- 范围：十个预构造 target、strict-v4 固定角色转换，n=1 diagnostic。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"buffer_ownership_validated": true, "copy_completion_before_consumer": true}
- 收益条件：{"transfer_exposed": true, "host_capacity_and_bandwidth_available": true}
- 失败分类：not_established
- 重试信号：足够多的复用次数可摊薄 cold pin 且 host 预算成立
- 最小验证：分别记录 cold pin、H2D、trim 和含准备/清理的完整周期
- 不应重复：不隐藏 31–42 秒准备及 11 秒清理
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-persistent-pin-cold-cost`
- 关联实现：[旧 Route-A pinned target 与有界 staging](implementations.md#legacy-pinned-materialization)；adaptation_example；机制参考或旧版本示例；需核对实现差异并适配，未证明是历史计时所用源码。
- 函数入口：[runtime/bounded_materialization.py::pin_module_cpu_registered_state](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L189-L238)
- 函数入口：[runtime/bounded_materialization.py::materialize_module_from_persistent_pinned_cpu](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L242-L331)
- 函数入口：[runtime/bounded_materialization.py::materialize_module_from_cpu_with_bounded_pinned_staging](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L335-L480)
- 来源：[sliortega295-ops/DMD-System / rb16_transition_persistent_pinned_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_transition_persistent_pinned_result_20260810.md#L10)；定位：观察中的数值文本锚点 L10；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### 清理放到后台会干扰通信；晚点清理又会干扰 Critic

- ID：helios-background-cleanup-contention
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：立即后台 trim：transition 30.887→40.980 秒；等布局 ready 再 trim：transition 23.838 秒达标，但 Full5 350.064 秒，比同步清理方案更慢。
- 经验：异步任务即使最终 join 几乎不等，也可能争用主路径资源；局部阶段达标不能覆盖端到端倒退。
- 范围：两种 cleanup 依赖点，exact Full5 均通过、正式重复 NOT_RUN。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"buffer_ownership_validated": true, "copy_completion_before_consumer": true}
- 收益条件：{"transfer_exposed": true, "host_capacity_and_bandwidth_available": true}
- 失败分类：host_cleanup_interference
- 重试信号：清理调度能够避开关键路径且完成可见
- 最小验证：比较 transition 与下一 Full5、CPU/内存竞争和最终 drain
- 不应重复：不以 transition 达标忽略 Full5 变慢
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-background-cleanup-contention`
- 关联实现：[旧 Route-A pinned target 与有界 staging](implementations.md#legacy-pinned-materialization)；adaptation_example；机制参考或旧版本示例；需核对实现差异并适配，未证明是历史计时所用源码。
- 函数入口：[runtime/bounded_materialization.py::pin_module_cpu_registered_state](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L189-L238)
- 函数入口：[runtime/bounded_materialization.py::materialize_module_from_persistent_pinned_cpu](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L242-L331)
- 函数入口：[runtime/bounded_materialization.py::materialize_module_from_cpu_with_bounded_pinned_staging](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L335-L480)
- 来源：[sliortega295-ops/DMD-System / rb16_transition_immediate_background_cleanup_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_transition_immediate_background_cleanup_result_20260810.md#L10)；定位：观察中的数值文本锚点 L10；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / rb16_transition_late_cleanup_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_transition_late_cleanup_result_20260810.md#L10)；定位：观察中的数值文本锚点 L10；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### 缓存保留至语义边界后，仍需报告后续串行清理

- ID：helios-persistent-cache-inclusive-time
- 状态：positive_observation / HISTORICAL_REPEAT_TARGETS_WITH_SCOPE
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：单次 exact 运行 Full5 306.015 秒，边界后 trim 12.605 秒；随后联合方案三次 Full5 中位 279.934 秒，含清理中位 293.016 秒。
- 经验：合法计时窗口外的成本不应消失；持续运行用户应看 inclusive makespan。batch12 资源点也不能与 batch16 直接宣称同工作量加速。
- 范围：strict-v4 4G12C，Critic batch12，联合方案含 block leaves。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"buffer_ownership_validated": true, "copy_completion_before_consumer": true}
- 收益条件：{"transfer_exposed": true, "host_capacity_and_bandwidth_available": true}
- 失败分类：not_established
- 重试信号：缓存寿命和运行次数足以覆盖创建与清理
- 最小验证：报告 steady 与含 trim 两种分母，并验证最终清理完成
- 不应重复：不把 timer 外 cleanup 当成免费
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-persistent-cache-inclusive-time`
- 关联实现：[旧 Route-A pinned target 与有界 staging](implementations.md#legacy-pinned-materialization)；adaptation_example；机制参考或旧版本示例；需核对实现差异并适配，未证明是历史计时所用源码。
- 函数入口：[runtime/bounded_materialization.py::pin_module_cpu_registered_state](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L189-L238)
- 函数入口：[runtime/bounded_materialization.py::materialize_module_from_persistent_pinned_cpu](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L242-L331)
- 函数入口：[runtime/bounded_materialization.py::materialize_module_from_cpu_with_bounded_pinned_staging](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L335-L480)
- 来源：[sliortega295-ops/DMD-System / rb16_transition_persistent_cache_post_full5_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_transition_persistent_cache_post_full5_result_20260810.md#L36)；定位：观察中的数值文本锚点 L36；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / rb16_step10a_transition_combined_formal_result_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_step10a_transition_combined_formal_result_20260810.md#L36)；定位：观察中的数值文本锚点 L36；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### CPU staged H3 候选卡住，证据不足以称为 OOM

- ID：h100-w5-cpu-stage-stall
- 状态：inconclusive / HISTORICAL_STALL_UNRESOLVED
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：公开 stall 回执没有 CUDA 或 host OOM 证据；futex 等待根因未确认，candidate timing NOT_RUN。
- 经验：保留“卡住、原因未知”的分类；不能把诊断受限猜成内存不足，也不能为失败填一个速度。
- 范围：W5 CPU-staged candidate stall r1。
- 历史上下文：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W5", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"buffer_ownership_validated": true, "copy_completion_before_consumer": true}
- 收益条件：{"transfer_exposed": true, "host_capacity_and_bandwidth_available": true}
- 失败分类：unknown_stall
- 重试信号：等待点及完成协议有可验证进展
- 最小验证：先取得各 rank host/device 栈与 stage/wait 事件，再缩小重现
- 不应重复：没有 OOM 证据时不要按 OOM 重跑或填造 candidate timing
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case h100-w5-cpu-stage-stall`
- 关联实现：[H3 FA3 replay cache 与 staged 路径](implementations.md#h3-attention-replay)；reusable_candidate；同机制的当前复用候选；须检查 guards、生效路径、依赖和验证，不代表历史实现身份或直接迁移通过。
- 函数入口：[src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py::install_fa3_replay_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py#L880-L1131)
- 函数入口：[src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py::prepare_staged_cache_before_backward](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py#L854-L872)
- 来源：[sliortega295-ops/DMD-System / w5_cpu_staged_candidate_stall_r1.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W5/reports/w5_cpu_staged_candidate_stall_r1.json)；定位：该结果及 qualification / NOT_RUN 边界

### 历史线索：整进程 NUMA 绑定可能把 staging 优化变成 host 容量问题

- ID：helios-numa-whole-process-lead
- 状态：unverified / UNVERIFIED_CONVERSATION_LEAD
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：本地历史摘要提到整进程绑定把权重、Adam 和激活一起限制到单 NUMA 域；尚未为此记录找到并公开可复核运行回执。
- 经验：区分整进程内存策略与有界 staging 分配策略；在当前节点先检查实际拓扑、页位置和容量。
- 范围：未核实的 Helios 历史线索；不发布性能数值或已证实根因。
- 历史上下文：{"training_mode": "training", "numa_topology": null, "host_memory_policy": null}
- 正确性前提：{"buffer_ownership_validated": true, "copy_completion_before_consumer": true}
- 收益条件：{"transfer_exposed": true, "host_capacity_and_bandwidth_available": true}
- 失败分类：not_publicly_verified
- 重试信号：找到原回执或当前 NUMA 证据指向具体分配
- 最小验证：先测权重/Adam/激活的 NUMA 分布与带宽竞争，再尝试局部绑定
- 不应重复：未核实会话不支持把整个进程绑定到单域
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-numa-whole-process-lead`
- 来源：仅本地会话线索，未公开原始会话；缺少可核实的公开实验回执。

### 历史线索：单卡传输收益可能在节点并发时缩水

- ID：helios-single-gpu-staging-scale-lead
- 状态：unverified / UNVERIFIED_CONVERSATION_LEAD
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：本地摘要描述单卡与多 rank 同时 staging 收益差异；没有本轮公开的 source-qualified 比较，竞争原因未证实。
- 经验：先保留可检验假设：共享内存带宽/NUMA/PCIe 竞争可能限制扩展；用目标并发和最坏 rank 实测。
- 范围：未核实跨规模经验，不给出加速倍数。
- 历史上下文：{"training_mode": "training", "concurrent_transfers": null}
- 正确性前提：{"buffer_ownership_validated": true, "copy_completion_before_consumer": true}
- 收益条件：{"transfer_exposed": true, "host_capacity_and_bandwidth_available": true}
- 失败分类：not_established
- 重试信号：具备目标 rank 并发下的 staging 回执
- 最小验证：同时记录共享 host/PCIe 带宽、队列与完整周期
- 不应重复：不能把单卡收益乘以卡数或预判竞争根因
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-single-gpu-staging-scale-lead`
- 来源：仅本地会话线索，未公开原始会话；缺少可核实的公开实验回执。
