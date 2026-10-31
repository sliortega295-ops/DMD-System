# 六项结果汇总补充勘误：W6 r46 的同步方向

阅读[2026-10-19 六项结果汇总](SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md)时，
请先应用本勘误。旧汇总是source catalog固定保存的历史证据，原始字节和实验回执保留。
本页纠正机制描述，没有重新计时或完成新的数值验收。

旧表将W6 r46写成“消除metadata barrier”，方向有误。正确记录如下：

| 项目 | Control | Candidate |
| --- | --- | --- |
| 每rank、每周期的metadata读取次数 | 0 | 350 |
| 原生C2时间（秒） | 262.384847224 | 253.740367298 |

原[机器回执](../../artifacts/publications/20261019/evidence/state_retention/r46/conclusion.json)
使用`actual_native_metadata_reads_per_rank_cycle`记录0→350。
[归档实现](../../artifacts/publications/20261019/manual_packets/state_retention/r46/scripts/native_r2/native_metadata_barrier.py)
启用`restore-native-cu-metadata-barrier.v1`后执行`cu_q.tolist()`；
[worker](../../artifacts/publications/20261019/manual_packets/state_retention/r46/scripts/native_r2/metadata_barrier_worker.py)
使用`--restore-native-barrier`。

因此这是同FA3实现下**恢复原生metadata读取/同步**的单次配对，观察到约8.644秒改善。
改善机制尚未隔离，数值状态仍为REVIEW；不能据此建议删除同步，也不证明恢复同步普遍更快。
更早SDPA历史时间不属于这一配对分母。

这里的barrier指host可见的metadata读取，不是新增跨rank的distributed barrier。
完整的经验分类、实现范围及内容回归见
[Optimizer内容修正记录](OPTIMIZER_EXPERIENCE_CONTENT_REVIEW_20261031.md)。
