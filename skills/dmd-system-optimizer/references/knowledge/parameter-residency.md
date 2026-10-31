# 重复 forward 的参数保留与驻留

同一权重版本反复消费时，阶段内保留可能减少重复聚合或搬运；收益由关键路径上的可消除开销决定。

适合检查：反复 rollout、cond/uncond、冻结角色调用；同一模块名不代表同一权重版本。
保留区间必须结束于权重更新之前，或者有经过验证的刷新协议。检查 backward、reduce-scatter 和所有退出路径。
时间收益只来自关键路径或资源竞争的改善；已被计算掩盖的 gather 不能全算作可节省时间。
显存成本会改变其他阶段的重算、offload 和可行 batch。不同 workload 上收益可以反转。
全量保留不适合时，可研究局部保留；一个 OOM 案例只限定当时的保留范围和容量。
不要把 n=1 的正向描述升级成默认开启，也不要据一次变慢永久封禁驻留。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- weights_stable_during_reuse: True
- state_release_validated: True

收益条件（不满足会降低收益预期，不等于语义错误）：

- repeated_forwards: True
- exposed_gather: True
- memory_headroom: True

最小验证：先数实际 gather/释放次数和版本边界；测一个代表性完整周期的暴露通信、峰值和更新时间。

## 已整理案例

### W2：参数保留有小幅正向观察，同时增加显存

- ID：helios-w2-retention-r36
- 状态：positive_observation / HISTORICAL_REVIEW
- 观察：原生 C2 83.970328451→83.023064643 秒；peak allocated 26.418→30.911 GiB；16 ranks，单配对，numerical REVIEW。
- 经验：重复使用可以产生小收益；时间和容量一起评估，尚不能据此默认启用。
- 范围：W2 Helios14B No-GAN；该源、该保留范围、两周期的单配对。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r36/conclusion.json)

### W3：冻结 replay 保留只带来小增量

- ID：helios-w3-retention-r1
- 状态：positive_observation / HISTORICAL_REVIEW
- 观察：C2 117.074207036→116.673757426 秒，约 0.400 秒；numerical REVIEW。
- 经验：保留收益按当前 GAN 合同计算；不能与另一提交的最好成绩叠加。
- 范围：W3 r1 该源同源增量，非长期等价证明。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json)

### W6：同类保留尝试变慢

- ID：helios-w6-retention-r1
- 状态：negative_observation / HISTORICAL_REVIEW
- 观察：C2 255.363757909→260.582584122 秒，增加约 5.219 秒；未晋级。
- 经验：其他 workload 的保留正例不决定 W6 的收益。净减速已观察，具体成本来源仍需局部测量。
- 范围：W6 r1 的该实现和工作量，numerical REVIEW。
- 重试条件：保留范围、重复消费次数、暴露通信或容量条件改变后，先做成本探针再重新配对。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json)

### W1：扩大到 whole-block tail8 出现 OOM

- ID：helios-w1-tail8-oom
- 状态：implementation_failure / HISTORICAL_CUDA_OOM
- 观察：公开回执记为 CUDA_OOM_NOT_PROMOTED；候选没有有效 C2。
- 经验：容量失败限定当时保留范围；没有候选速度值，不能把它记作慢或零收益。
- 范围：W1 当时 tail8 实现与 A100-40GB 容量。
- 重试条件：更小保留集合、额外内存释放或不同容量/拓扑后可重新做容量验证；不直接重跑原组合。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json)
