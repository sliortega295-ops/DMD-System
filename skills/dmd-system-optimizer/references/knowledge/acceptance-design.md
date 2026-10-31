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

### ResearchBoost：让已知错误真正触发最终失败出口

- ID：rb-judge-negative-controls
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：公开 judge/numerics 文档要求用已知扰动和 mutation 检查判定路径；计算了中间 verdict 却未传到最终输出可能让全部候选通过。
- 经验：冻结外部验收器并校准其灵敏度；hash 检查证明身份，不等于独立权限隔离或完整测试覆盖。
- 范围：外部 harness 方法；不复制其统计阈值、N0/N1/N2 标准或固定任务权限。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"acceptance_contract_frozen": true, "candidate_cannot_redefine_success": true}
- 收益条件：{}
- 失败分类：not_established
- 重试信号：known-bad 修改能正确传播到最终 verdict
- 最小验证：注入错误输出、状态或计时回执并检查最终拒绝码
- 不应重复：计算中间 FAIL 但最终 PASS 的 judge 不能用于晋级
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case rb-judge-negative-controls`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-numerics-validation/SKILL.md#L1)；定位：报告范围 L1–L98；L6: Numerics tiers and validation；L14: N0–N5；L35: How thresholds are set；L57: Cheap investigation order (T0 → T3)；L68: When only one component cannot be bitwise: isolate it behind a stand-in；L81: Common pitfalls；原定位：The evaluator itself can have bugs

### ResearchBoost：同一路径与原始 baseline 比较，任务变更需新合同

- ID：rb-judge-scope-and-contract
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：judge.md 区分 exact items、数值与 benefit，同时列出短轨迹、save 缺失和固定 forward-count 的局限。
- 经验：可借鉴独立验收设计，但应按 DMD 的合法调用合并与风险选择检查项；不能把外部规则整套当成本地优化禁令。
- 范围：ResearchBoost frozen-judge 设计的原创概括；本工具的检索/字段校验不是该外部 judge。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"acceptance_contract_frozen": true, "candidate_cannot_redefine_success": true}
- 收益条件：{}
- 失败分类：not_established
- 重试信号：验收范围覆盖当前候选触及的边界
- 最小验证：核对 exact/numerics/benefit 与 save/长轨迹/逻辑工作范围
- 不应重复：外部固定 forward-count 或短测合同不能未经判断照搬
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case rb-judge-scope-and-contract`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / judge.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/judge.md#L1)；定位：报告范围 L1–L43；L1: The judge；L5: Four gates；L19: When the judge may change, and when it may not；L26: Fake gains the judge caught；L38: Known limits of the judge；原定位：Four gates; When the judge may change; Known limits

### ResearchBoost：隔离非bitwise组件，再验真实最终组合

- ID：rb-nonbitwise-isolation-final-combination
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：原方法用reference组件作为接口一致的替身，检查其余改动及组合；最终切回真实组件对原始baseline完成适当数值验收。
- 经验：替身通过只说明该配置下其他改动；真实组件、依赖布局/梯度或组合发生变化仍要检验。验证层级按当前合同选择，不照抄固定100步或外部收益阈值。
- 范围：外部训练验证方法，本系统未由此获得新的GPU数值结论。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"standin_interface_and_gradient_contract_match": true, "original_baseline_preserved": true}
- 收益条件：{}
- 失败分类：not_established
- 重试信号：一个已知非bitwise组件遮蔽其他改动差异
- 最小验证：先隔离组件核对其他路径，最后验证真实组合对original baseline的结果，并记录比较对象与范围
- 不应重复：不把该证据范围外的结果补成已测或通过，不因模型名称变化而重复相同已否决实验
- 代码状态：method_only；本条提供来源明确的方法或诊断，不提供未定位的runtime实现。
- 定位命令：`dmd_knowledge.py locate --case rb-nonbitwise-isolation-final-combination`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-numerics-validation/SKILL.md#L68)；定位：## When only one component cannot be bitwise
