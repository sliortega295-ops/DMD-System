# 多角色调度与流水的适用边界

独立阶段可重叠，但拆分资源会改变每个角色的效率，并引入传输、等待和填充排空。

区分物理执行计划和逻辑训练合同。G1/C5 与 G1/C1 的最佳布局不必相同。
流水收益取决于阶段平衡、样本数、microbatch 和阶段效率；不能把独立测得的最快阶段拼成可实现总时间。
保留顺序执行作为对照。流水相比上一轮流水变快，不说明它超过了最好的顺序方案。
检查版本、梯度、随机数和更新顺序；验证允许的 batching/fusion 时按逻辑工作比较，不能机械固定物理 launch 数。
依赖已证明独立才讨论重叠；拓扑变化后重算通信与容量。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- logical_work_preserved: true
- versions_and_dependencies_validated: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- independent_work_available: true
- overlap_exceeds_partition_cost: true

最小验证：按 G/C/Teacher/EMA 的真实频率建最小依赖图，比较相同资源的顺序方案；先测阶段服务时间和跨角色传输。

## 已整理案例
