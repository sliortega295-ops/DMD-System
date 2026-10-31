# Helios 条件化优化经验库

给没有历史会话的新 Codex 使用。每次先读当前 workload，再按机制检索；成功、失败和未知共同限定适用范围。

本版有 12 个机制主题、24 个案例。原始会话留在本地；公开来源固定到源码快照，保留 REVIEW、FAIL、NOT_RUN。

| 主题 | 案例数 |
| --- | --- |
| [重复 forward 的参数保留与驻留](parameter-residency.md) | 4 |
| [offload、激活保留与容量交换](memory-tradeoffs.md) | 2 |
| [多角色调度与流水的适用边界](role-scheduling.md) | 1 |
| [消除 metadata 同步与 CPU 提交空隙](host-synchronization.md) | 1 |
| [Attention backend 与真实调用覆盖](attention-routing.md) | 2 |
| [融合、精度边界与 no-grad 复用](fusion-numerics.md) | 2 |
| [图捕获、EMA 与固定状态假设](graphs-and-ema.md) | 1 |
| [缓存、预计算与状态失效](state-caching.md) | 2 |
| [对照身份、计时范围与因果归因](measurement-identity.md) | 3 |
| [按风险验证数值、状态与训练生命周期](validation-lifecycle.md) | 3 |
| [先核对已有实现，再决定移植或重写](implementation-reuse.md) | 1 |
| [小收益、失败成本和重试条件](experiment-decisions.md) | 2 |

用 dmd_knowledge.py recall 返回条件和同机制全部正反案例；关键词命中不等于当前适用。详见 [检索规则](../reuse-memory.md)。

机制卡可以跨模型提供假设；历史案例的数值只适用于自身上下文。源码实现的 shape/dtype/grad guard 是更窄的一层。
