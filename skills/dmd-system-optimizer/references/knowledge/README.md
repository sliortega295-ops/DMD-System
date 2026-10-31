# 条件化优化经验库

给没有历史会话的新 Codex 使用。先读当前 workload，再按机制检索；成功、失败和未知共同限定适用范围。

本版 21 个主题、109 条案例与方法条目。它们包含实测报告、源码约束、外部方法建议和未核实线索，不能当成同等数量的独立成功实验。

见 [覆盖范围与未完成项](coverage.md)。来源固定到提交与文件 hash；原始私人会话没有上传。

| 主题 | 条目数 |
| --- | ---: |
| [重复 forward 的参数保留与驻留](parameter-residency.md) | 6 |
| [offload、激活保留与容量交换](memory-tradeoffs.md) | 7 |
| [多角色调度与流水的适用边界](role-scheduling.md) | 0 |
| [metadata 同步策略与 CPU 提交次序](host-synchronization.md) | 5 |
| [Attention backend 与真实调用覆盖](attention-routing.md) | 4 |
| [融合、精度边界与 no-grad 复用](fusion-numerics.md) | 15 |
| [图捕获、EMA 与固定状态假设](graphs-and-ema.md) | 6 |
| [缓存、预计算与状态失效](state-caching.md) | 8 |
| [对照身份、计时范围与因果归因](measurement-identity.md) | 6 |
| [按风险验证数值、状态与训练生命周期](validation-lifecycle.md) | 15 |
| [先核对已有实现，再决定移植或重写](implementation-reuse.md) | 5 |
| [小收益、失败成本和重试条件](experiment-decisions.md) | 4 |
| [Pinned staging、NUMA 与搬运的完整成本](transfer-staging.md) | 1 |
| [分布式布局、ZeRO leaf 与通信粒度](distributed-topology.md) | 3 |
| [分配器、构造峰值与运行模式](allocator-policy.md) | 5 |
| [从 profile 找归因并纠正错误假设](profiling-attribution.md) | 5 |
| [Planner 成本转移、搜索空间与实际执行](planner-calibration.md) | 3 |
| [验收器的独立性、灵敏度与覆盖范围](acceptance-design.md) | 3 |
| [数据、冷启动、checkpoint 与持续运行成本](data-lifecycle-costs.md) | 2 |
| [激活保留与 checkpoint 重算取舍](activation-retention.md) | 2 |
| [checkpoint 中间计算结果复用](checkpoint-result-reuse.md) | 4 |

使用 `dmd_knowledge.py recall` 返回条件与同机制全部正反案例；使用 `sources` 搜索尚未整理完的历史来源。详见 [检索规则](../reuse-memory.md)。

使用 `locate --case CASE_ID` 从经验进入 [实现索引](implementations.md)：固定源码、函数、接入、测试与回退。`linked` 是相关参考，`method_only` 是纯方法，`unresolved` 表示对应实现尚未定位。

H100、A100 和外部推理案例都保留硬件/模式边界。关键词命中不等于当前适用，外部报告也不是本系统的验收结果。

维护者编辑 JSON pack 后运行 `scripts/render_knowledge.py` 同步这些卡片；机制说明保留，案例部分由结构化记录生成。
