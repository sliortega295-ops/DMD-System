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
