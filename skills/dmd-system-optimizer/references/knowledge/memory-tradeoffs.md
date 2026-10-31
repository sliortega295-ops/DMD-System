# offload、激活保留与容量交换

释放内存可以解锁更有价值的驻留或减少重算；新增传输、同步和峰值也可能抵消收益。

分开三个问题：是否放得下、是否更快、是否解锁另一项优化。
capacity enabler 可以没有直接速度收益，但必须说明下游收益和新增成本，避免重复记账。
allocated、reserved、driver free 和 host cgroup 是不同指标；结束边界的余量不代表全程最小余量。
HBM 富余机器与紧张机器可以选择相反策略。CPU 带宽、PCIe/NVLink 和 NUMA 改变后要重估。
offload 造成变慢时先定位暴露传输、同步或重算；失败原因未查明时保留未知。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- state_lifetime_preserved: True

收益条件（不满足会降低收益预期，不等于语义错误）：

- capacity_limits_candidate: True
- enabled_gain_exceeds_transfer: True

最小验证：先做逐阶段容量账：常驻、激活、临时 gather、workspace、pinned 和 checkpoint；再量传输与计算重叠。

## 已整理案例

### W4：完成运行与全程容量宽裕是不同结论

- ID：helios-w4-capacity-r3
- 状态：inconclusive / HISTORICAL_REVIEW
- 观察：完整仅B臂 C2 为 42.123914 秒，末边界 driver free 4.375 MiB；回执明确它不是全程最小余量。
- 经验：边界采样不能证明峰值余量；allocated/reserved/driver 指标不能互换。
- 范围：W4 C51 R3；旧 baseline/full 来自父源，新臂 n=1，数值 REVIEW。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/livewan_r3/reports/conclusion.json)

### parameter reserve 与 driver free 不能混用

- ID：helios-allocator-policy-r36
- 状态：source_contract / HISTORICAL_POLICY
- 观察：r36 的容量口径是 allocated highwater 留 2 GiB，明确不是 driver free 留 2 GiB。
- 经验：沿用实验原容量合同，新增硬件按自己的真实资源约束判断。
- 范围：r36 回执的原始容量政策；不是推荐所有 workload 使用 2 GiB。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r36/conclusion.json)
