# 消除 metadata 同步与 CPU 提交空隙

频繁读取 GPU 上的 metadata 可能阻断 CPU 提交；可推导的 host 状态可减少这些同步。

CPU 时间段很长不等于 GPU 空闲；先关联实际 GPU activity 和依赖。
只有已知且不依赖异步结果的状态可以放在 host 管理；数据决定的分支不能直接猜测。
跨 wrapper 调用要核对 host 状态实际持有者，避免副本更新后调用方仍读旧值。
同一 attention backend 的单因素对照可回答 metadata 的增量；更换 backend 的旧计时不能当其分母。
收益可能随 batch、角色数和图捕获改变。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- metadata_semantics_preserved: True
- host_state_lifetime_validated: True

收益条件（不满足会降低收益预期，不等于语义错误）：

- host_sync_on_critical_path: True

最小验证：定位 item/DtoH/同步调用及其依赖；用开关只改变 metadata 路径，确认调用、版本和缓存输入一致。

## 已整理案例

### W6：同 FA3 对照下 metadata barrier 消除有正向观察

- ID：helios-w6-metadata-r46
- 状态：positive_observation / HISTORICAL_REVIEW
- 观察：两臂同 FA3：C2 262.384847224→253.740367298 秒，节省 8.644479926 秒；单配对 REVIEW。
- 经验：这是 metadata 变化的该次增量；不能把更早 SDPA 的 255.528 秒代入分母。
- 范围：W6 r46 相同训练源加外部 packet；非长期数值或最终全系统收益。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r46/conclusion.json)
