# 缓存、预计算与状态失效

重复计算可以被缓存替换，但 key、版本、随机性和维护成本决定复用是否正确且合算。

冻结参数不等于输出恒定：输入、噪声、条件和 timestep 都可能变化。
缓存优化器更新前的结果供更新后使用会改变训练；需要证明生命周期。
缓存可能减少计算却增加带宽、内存、同步或索引成本，完整周期可能变慢。
模式相同但 reuse 次数不同可以翻转收益；因缓存维护成本失败的案例可在复用次数显著增加后重试。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- cache_key_complete: True
- invalidation_validated: True
- cached_gradients_preserved: True

收益条件（不满足会降低收益预期，不等于语义错误）：

- cache_reuse_sufficient: True
- saved_compute_exceeds_cache_cost: True

最小验证：先列 value 依赖的输入、shape、dtype、conditioning、timestep、权重、adapter、模式与随机性；验证命中和失效。

## 已整理案例

### W6：AdaLN bank 尝试没有带来时间收益

- ID：helios-w6-adaln-r41
- 状态：negative_observation / HISTORICAL_REVIEW
- 观察：C2 255.602136769→256.312003678 秒；exact cache input gate 通过，参数未删除、权重未合并。
- 经验：缓存输入条件通过不保证净收益；需要区分计算节省和维护/访问成本。
- 范围：W6 r41 单周期配对，完整状态或轨迹 REVIEW。
- 重试条件：reuse 数量、缓存布局或访问成本改变后，用可判别的小测量重新估算。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r41/conclusion.json)

### W1：RETAIN4 与 frozen_down 的条件边际不可相加

- ID：helios-w1-frozen-down-r1
- 状态：negative_observation / HISTORICAL_REVIEW
- 观察：20.028682893→20.169802765 秒，MEASURED_NO_C2_GAIN；范围标为 RETAIN4_VS_FROZEN_DOWN_ONLY。
- 经验：两个替代策略的比较不支持叠加它们的收益。
- 范围：r1 followup 对应源和该对照，不是所有 frozen_down 路径都无效。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json)
