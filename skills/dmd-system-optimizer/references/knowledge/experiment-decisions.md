# 小收益、失败成本和重试条件

探索优先级、实测收益保留和默认启用是三个决定；失败记录应缩小搜索而不封死可迁移机制。

已确认的 1% 收益可以保留；不足以分辨的差异保留 inconclusive，不为了追求大目标抹去有效小改动。
默认启用还需适用条件、组合和回退。n=1 REVIEW 小正值可以保留作候选，不能伪称已确认。
实现 bug、条件不满足、净开销、瓶颈转移、噪声、未知根因分别记。
失败/中断也记实际资源。owner phase、run、cycle、GPU-hour 是不同单位。
跨会话继承预算与失败信息；新 workload 可重新判断，不照搬旧任务的失败名单。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- cost_and_evidence_scope_defined: true

收益条件（不满足会降低收益预期，不等于语义错误）：

由当前 workload 的目标和测量确定。

最小验证：为实验写一个可否证问题、成本和停止条件；记录预计与实测差异以及改变判断所需的新证据。

## 已整理案例

### 累计 owner phases 不是完整训练 run 数

- ID：helios-budget-units-r36
- 状态：source_contract / HISTORICAL_ACCOUNTING_SCOPE
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：r36 campaign_charges=92，回执明确是累计 owner phases，包含失败，不是92次world16训练。
- 经验：把成本单位和失败消耗写清，不能跨会话重置，也不能错把 phase 当 run。
- 范围：该历史实验记账口径；新项目使用自己的明确成本定义。
- 历史上下文：{}
- 正确性前提：{"cost_and_evidence_scope_defined": true}
- 收益条件：{}
- 失败分类：not_established
- 重试信号：账本计量单位或资源分配改变
- 最小验证：逐项对齐 owner phases、失败成本与设备时长，转换单位时保留依据
- 不应重复：不把 92 owner phases 当作 92 次 world16 训练
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case helios-budget-units-r36`
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r36/conclusion.json)；定位：campaign_charges; ledger; charged_stage_phases

### 小正向时间与数值 REVIEW 可以同时存在

- ID：helios-small-gain-review-r14
- 状态：inconclusive / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：W4 C2 42.488375→42.334359 秒；Critic Adam sketch nL2=0.2127849，未建立 symmetric noise floor，未默认晋级。
- 经验：不因收益小就丢弃线索，也不因时间变快就升级数值结论；保留候选状态和剩余问题。
- 范围：r14 单配对和原始统计范围，非严格等价或已确认小收益。
- 历史上下文：{"workload": "W4"}
- 正确性前提：{"cost_and_evidence_scope_defined": true}
- 收益条件：{}
- 失败分类：numerical_uncertainty
- 重试信号：已有对称 baseline noise floor 和状态解释
- 最小验证：重复 baseline/control 并比较 Critic Adam sketch，再判断约 0.154 秒收益
- 不应重复：未建立噪声地板前不把 REVIEW 改成默认启用
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case helios-small-gain-review-r14`
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r14/conclusion.json#L4)；定位：观察中的数值文本锚点 L4；同时阅读同节合同、状态与限制。原定位：status; numerical; saved_seconds

### 被停止的队列必须保持 NOT_RUN，压缩上下文不能重置预算

- ID：h100-stopped-campaign-ledger
- 状态：source_contract / HISTORICAL_STOPPED_NOT_RUN
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：r48 的停止状态覆盖 r47 的待执行计划；取消的 canary/queue 没有完成结果。r47 也保留了失败尝试计数与 GPU owner 未确认状态。
- 经验：恢复经验时读取最新状态而非执行旧 handoff 命令；技术修复、换会话或分支不能擦除失败成本和用户停止决定。
- 范围：H100 这次 campaign 的操作经验；当前用户的新任务需按新授权单独判断。
- 历史上下文：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "campaign", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"cost_and_evidence_scope_defined": true}
- 收益条件：{}
- 失败分类：not_established
- 重试信号：用户重新授权且资源所有权、失败账本已确认
- 最小验证：先读取最新停止状态与已有费用，重建明确 bounded run
- 不应重复：旧 r47 队列不能覆盖 r48 停止决定或视为已完成
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case h100-stopped-campaign-ledger`
- 来源：[sliortega295-ops/DMD-System / h100_campaign_handoff_20261016_r47.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/campaign/h100_campaign_handoff_20261016_r47.md#L1)；定位：报告范围 L1–L33；L1: R47 — SSH recovered; W2 completed; W1 canary import repaired; foreign GPU use blocks launch；L5: New measured W2 result；L17: W1 actual cheap failure, not new model/OOM/numeric failure；L24: Actual resource blocker after reconnect；L28: Next authorized serial work；原定位：该结果及 qualification / NOT_RUN 边界
- 来源：[sliortega295-ops/DMD-System / h100_campaign_handoff_20261016_r48.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/campaign/h100_campaign_handoff_20261016_r48.md#L1)；定位：报告范围 L1–L27；L1: R48 — Explicit user stop before machine reboot；L6: Actual verified stop；L18: Unfinished work is cancelled, not completed；L23: Evidence；原定位：该结果及 qualification / NOT_RUN 边界

### ResearchBoost：可改变的合同与内存目标决定搜索上限

- ID：rb-constraints-and-memory-objectives
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：其 lessons 中 LLaVA 的较大收益依赖用户后来允许的 padding/recompute 变更；另有 balancing 主要省内存而未加速。DMD2 后期小改动受噪声和 OOM 约束。
- 经验：把允许空间、容量改善、时间改善与停止理由分开；不拿改变任务后的最佳值当作原固定合同收益。
- 范围：外部跨任务经验，外部轮数、OOM 次数、噪声阈值不是本系统默认预算。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"cost_and_evidence_scope_defined": true}
- 收益条件：{}
- 失败分类：not_established
- 重试信号：用户明确授权语义范围变化或省内存目标
- 最小验证：重新冻结任务合同与 baseline，分开报告内存和时间收益
- 不应重复：不能自行放宽 padding/recompute 约束或把省内存叫加速
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case rb-constraints-and-memory-objectives`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / lessons.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/lessons.md#L1)；定位：报告范围 L1–L33；L1: Results and lessons across tasks；L3: Four tasks, four ceilings；L12: General lessons；原定位：Four tasks, four ceilings; Memory is an account
