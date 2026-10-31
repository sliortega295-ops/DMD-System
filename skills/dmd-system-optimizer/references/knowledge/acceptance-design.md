# 验收器的独立性、灵敏度与覆盖范围

固定并独立保存验收合同，用已知错误检查是否真的会拒绝；通过的结论只覆盖实际测试路径。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- acceptance_contract_frozen: true
- candidate_cannot_redefine_success: true

收益条件（不满足会降低收益预期，不等于语义错误）：

由当前 workload 的目标和测量确定。

最小验证：按当前风险挑选负对照和生命周期边界；核对判定器身份、同一路径执行与失败出口，不照搬外部阈值。

## 已整理案例

### 两组 offload scope 被旧验收器误判，复核原始证据即可修正

- ID：helios-scope-aware-validator
- 状态：source_contract / HISTORICAL_VALIDATOR_REPAIR
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：Score4 首版 summarizer 强求一个 final event 而报失败；同一不可变 trace 实际有两个已安装且均归零的 scope，修复 multiset 检查后通过，没有重跑模型。
- 经验：先区分执行失败与观测格式错误；修复验收器要保持同一语义要求，保留旧判定与同证据重放。
- 范围：该 activation-offload installed/final scope 校验。
- 条件：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / ra16_f5_glp_score4_20260809.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/ra16_f5_glp_score4_20260809.md)；定位：结果、合同、限制与失败记录

### ResearchBoost：让已知错误真正触发最终失败出口

- ID：rb-judge-negative-controls
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：公开 judge/numerics 文档要求用已知扰动和 mutation 检查判定路径；计算了中间 verdict 却未传到最终输出可能让全部候选通过。
- 经验：冻结外部验收器并校准其灵敏度；hash 检查证明身份，不等于独立权限隔离或完整测试覆盖。
- 范围：外部 harness 方法；不复制其统计阈值、N0/N1/N2 标准或固定任务权限。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-numerics-validation/SKILL.md)；定位：The evaluator itself can have bugs

### ResearchBoost：同一路径与原始 baseline 比较，任务变更需新合同

- ID：rb-judge-scope-and-contract
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：judge.md 区分 exact items、数值与 benefit，同时列出短轨迹、save 缺失和固定 forward-count 的局限。
- 经验：可借鉴独立验收设计，但应按 DMD 的合法调用合并与风险选择检查项；不能把外部规则整套当成本地优化禁令。
- 范围：ResearchBoost frozen-judge 设计的原创概括；本工具的检索/字段校验不是该外部 judge。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / judge.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/judge.md)；定位：Four gates; When the judge may change; Known limits
