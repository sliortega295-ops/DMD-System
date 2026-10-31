# 数据、冷启动、checkpoint 与持续运行成本

稳态训练窗口可能遗漏数据准备、首次编译、保存与恢复；交付加速时要说明用户实际支付哪些成本。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- data_and_progress_identity_preserved: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- reported_window_matches_user_objective: true

最小验证：在小预算下触发真实数据加载、phase 切换、save/resume；同时报告冷启动、稳态与必要的后续清理。

## 已整理案例

### ResearchBoost：judge 没测保存，真实训练第一次 save 才失败

- ID：rb-first-checkpoint-save-crash
- 状态：implementation_failure / EXTERNAL_REPORTED_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：其 lessons 报告真实运行在首次 checkpoint save 暴露故障，前面的稳态 judge 未执行该路径。
- 经验：长跑前用小预算触发真实 save/resume 和 phase 边界；具体 step 数按本地生命周期选择。
- 范围：外部训练生命周期缺口，不说明本地 checkpoint 已有此故障。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"data_and_progress_identity_preserved": true}
- 收益条件：{"reported_window_matches_user_objective": true}
- 失败分类：untested_save_path
- 重试信号：首次 save 路径与状态序列化已修复
- 最小验证：真实保存并恢复，检查 optimizer/RNG/keys 及峰值
- 不应重复：稳定 step 的 judge 不替代第一次 save/resume
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case rb-first-checkpoint-save-crash`
- 关联实现：[项目 checkpoint 保存与恢复入口](implementations.md#checkpoint-lifecycle)；adaptation_example；机制参考或旧版本示例；需核对实现差异并适配，未证明是历史计时所用源码。
- 函数入口：[src/distillschedule/training/checkpoint.py::save_checkpoint](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/training/checkpoint.py#L319-L456)
- 函数入口：[src/distillschedule/training/checkpoint.py::restore_checkpoint](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/training/checkpoint.py#L541-L559)
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / lessons.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/lessons.md#L1)；定位：报告范围 L1–L33；L1: Results and lessons across tasks；L3: Four tasks, four ceilings；L12: General lessons；原定位：From the judge to a real run

### ResearchBoost：计时窗口与优化贡献分母分别记录

- ID：rb-end-to-end-denominator
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：YOCO 区分 step/full run，LLM2Vec 区分阶段；R3 的 2.08× 包括 reward 服务优化贡献，1.88× 则两臂均使用优化后的 reward 服务。R3 RL step 仍含 rollout、reward service 和 optimizer。
- 经验：服务是否在计时窗口内，与两臂服务实现是否相同是两回事；R3 不能解释为计时排除 reward 服务。不同阶段和优化栈的倍数不能拼接。
- 范围：外部 LLM 各自实验合同，不能与 DMD 数值横向排榜。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"data_and_progress_identity_preserved": true}
- 收益条件：{"reported_window_matches_user_objective": true}
- 失败分类：not_established
- 重试信号：计时窗口和每臂服务实现身份已分别明确
- 最小验证：分别记录 step/full-run 范围、各臂 reward 版本及优化贡献；不要从标题倍数倒推计时范围
- 不应重复：不能把 R3 的 1.88× 写成排除了 reward service 的计时
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case rb-end-to-end-denominator`
- 结构化事实：{"r3_reward_service_in_timed_step": true, "r3_with_reward_optimization_speedup": 2.08, "r3_optimized_reward_both_arms_speedup": 1.88, "r3_comparison_axis": "optimization_contribution"}
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / llm.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/results/llm.md#L1)；定位：报告范围 L1–L56；L1: LLM；L7: Faster-NanoChat；L24: Faster-YOCO；L39: Faster-LLM2Vec；原定位：YOCO; R3; LLM2Vec
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / multimodal.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/results/multimodal.md#L56)；定位：With the optimized reward service on both sides
