# 小收益、失败成本和重试条件

探索优先级、实测收益保留和默认启用是三个决定；失败记录应缩小搜索而不封死可迁移机制。

已确认的 1% 收益可以保留；不足以分辨的差异保留 inconclusive，不为了追求大目标抹去有效小改动。
默认启用还需适用条件、组合和回退。n=1 REVIEW 小正值可以保留作候选，不能伪称已确认。
实现 bug、条件不满足、净开销、瓶颈转移、噪声、未知根因分别记。
失败/中断也记实际资源。owner phase、run、cycle、GPU-hour 是不同单位。
跨会话继承预算与失败信息；新 workload 可重新判断，不照搬旧任务的失败名单。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- cost_and_evidence_scope_defined: True

收益条件（不满足会降低收益预期，不等于语义错误）：

- 无统一性能门槛；根据当前目标判断。

最小验证：为实验写一个可否证问题、成本和停止条件；记录预计与实测差异以及改变判断所需的新证据。

## 已整理案例

### 累计 owner phases 不是完整训练 run 数

- ID：helios-budget-units-r36
- 状态：source_contract / HISTORICAL_ACCOUNTING_SCOPE
- 观察：r36 campaign_charges=92，回执明确是累计 owner phases，包含失败，不是92次world16训练。
- 经验：把成本单位和失败消耗写清，不能跨会话重置，也不能错把 phase 当 run。
- 范围：该历史实验记账口径；新项目使用自己的明确成本定义。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r36/conclusion.json)

### 小正向时间与数值 REVIEW 可以同时存在

- ID：helios-small-gain-review-r14
- 状态：inconclusive / HISTORICAL_REVIEW
- 观察：W4 C2 42.488375→42.334359 秒；Critic Adam sketch nL2=0.2127849，未建立 symmetric noise floor，未默认晋级。
- 经验：不因收益小就丢弃线索，也不因时间变快就升级数值结论；保留候选状态和剩余问题。
- 范围：r14 单配对和原始统计范围，非严格等价或已确认小收益。
- 重试条件：用预先确定的对照和验证解决状态差异，再判断小收益是否稳定。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r14/conclusion.json)
