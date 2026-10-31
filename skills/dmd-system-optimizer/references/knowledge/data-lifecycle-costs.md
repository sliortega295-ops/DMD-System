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
- 关联实现：[项目 checkpoint 保存与恢复入口](implementations.md#checkpoint-lifecycle)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/training/checkpoint.py::save_checkpoint](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/training/checkpoint.py#L319-L456)
- 函数入口：[src/distillschedule/training/checkpoint.py::restore_checkpoint](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/training/checkpoint.py#L541-L559)
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / lessons.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/lessons.md#L1)；定位：报告范围 L1–L33；L1: Results and lessons across tasks；L3: Four tasks, four ceilings；L12: General lessons；原定位：From the judge to a real run

### ResearchBoost：stage、step 与含外部服务/启动的收益不能混为一个倍数

- ID：rb-end-to-end-denominator
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：公开 LLM results 分别报告 YOCO step 与 full run、R3 是否包含 reward service、LLM2Vec 不同阶段的时间。
- 经验：按用户目标明确端到端窗口；多阶段倍数不能相乘，省下稳态时间还要覆盖必要启动、数据和外部服务。
- 范围：外部 LLM 各自实验合同，不能与 DMD 数值横向排榜。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"data_and_progress_identity_preserved": true}
- 收益条件：{"reported_window_matches_user_objective": true}
- 失败分类：not_established
- 重试信号：服务、数据阶段和启动成本的计时范围已明确
- 最小验证：同时记录 step、各阶段、外部 reward service 与 full run
- 不应重复：不同阶段和排除服务的比值不能拼成总体收益
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case rb-end-to-end-denominator`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / llm.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/results/llm.md#L1)；定位：报告范围 L1–L56；L1: LLM；L7: Faster-NanoChat；L24: Faster-YOCO；L39: Faster-LLM2Vec；原定位：YOCO; R3; LLM2Vec
