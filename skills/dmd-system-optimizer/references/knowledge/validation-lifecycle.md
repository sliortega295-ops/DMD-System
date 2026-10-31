# 按风险验证数值、状态与训练生命周期

不同改动需要不同深度的验证；训练 loss 走势、数值一致性和输出质量是不同判断。

检查相对 matched baseline 的额外偏离，不以 loss 上升或下降独立判正确。
finite、调用计数和版本相同不能替代梯度、参数与 optimizer 状态的检查。
已有 numerical FAIL 未解决时不做正式速度晋级；原因未知保留 REVIEW。
保存/恢复/导出/生成会触发训练窗口没有的分支与内存峰值；按交付范围测试端点。
根据风险冻结 V0/V1/V2/V3 验证深度；不因非 bitwise 自动要求固定 100 步。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- current_validation_contract_defined: true

收益条件（不满足会降低收益预期，不等于语义错误）：

由当前 workload 的目标和测量确定。

最小验证：先用能够暴露该风险的短测试；状态变化补 continuation/resume；长期近似或残余漂移问题再做轨迹和质量检查。

## 已整理案例

### W5：后向诊断并未消除原数值失败

- ID：helios-w5-numerical-failure
- 状态：correctness_failure / HISTORICAL_FAIL_UNRESOLVED
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：r1 记录两次真实 local backward diagnostics，但 prior numerical FAIL unresolved；没有 speed pair。
- 经验：运行诊断、finite 和完成调用不能替代原验收门；没有速度配对应记 NOT_RUN。
- 范围：W5 full-sequence Grid1000 近似合同；不能推广成 W6 continuous AR 的失败。
- 历史上下文：{"workload": "W5", "sigma_contract": "Grid1000"}
- 正确性前提：{"current_validation_contract_defined": true}
- 收益条件：{}
- 失败分类：numerical_cause_unresolved
- 重试信号：Grid1000 合同下已定位并修复原数值 FAIL
- 最小验证：先复现不变 baseline 与首次 backward 差异，再跑冻结 gate
- 不应重复：未清除原 FAIL 前不开始正式 speed pair
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-w5-numerical-failure`
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json)；定位：rows[workload=W5]; numerical_policy

### loss 上升应与匹配 baseline 比较

- ID：helios-loss-trend-lead
- 状态：unverified / UNVERIFIED_CONVERSATION_LEAD
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：本地会话纠正过把 loss 上升直接当作数值异常的判断；该会话没有补齐匹配 baseline 的一致性证据。
- 经验：关注优化版相对 baseline 的额外偏离和梯度/参数/optimizer 状态；单臂趋势不能裁决一致性。
- 范围：会话提炼的方法线索；不作为 H3 AR 正确或错误的实验结论。
- 历史上下文：{"workload": "W6"}
- 正确性前提：{"current_validation_contract_defined": true}
- 收益条件：{}
- 失败分类：missing_matched_baseline
- 重试信号：有匹配 baseline 与状态证据可判断 loss 变化
- 最小验证：比较同输入/RNG 的 loss、梯度和更新；先确定基线波动
- 不应重复：不能仅因 loss 上升断言错误或放宽阈值
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-loss-trend-lead`
- 来源：仅本地会话线索，未公开原始会话；缺少可核实的公开实验回执。

### 恢复时重复分配 Adam 状态可能出现额外峰值

- ID：livewan-checkpoint-memory-lead
- 状态：unverified / UNVERIFIED_CONVERSATION_LEAD
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：本地会话提出 LiveWan 恢复时避免第二份完整 Adam 分配；当时只有 CPU 检查线索，GPU 结果未在本库核实。
- 经验：训练稳态能运行，不说明恢复峰值可承受；检查载入过程是否复用现有 optimizer state。
- 范围：私有会话线索，未公开原始执行回执，不能称已解决所有恢复 OOM。
- 历史上下文：{"model_family": "livewan"}
- 正确性前提：{"current_validation_contract_defined": true}
- 收益条件：{}
- 失败分类：suspected_restore_allocation
- 重试信号：找到恢复路径代码并确认重复 Adam 分配
- 最小验证：先做状态 roundtrip 与恢复峰值记录，再验证真实设备容量
- 不应重复：CPU 检查线索不能证明 GPU 恢复内存已解决
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case livewan-checkpoint-memory-lead`
- 关联实现：[项目 checkpoint 保存与恢复入口](implementations.md#checkpoint-lifecycle)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/training/checkpoint.py::save_checkpoint](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/training/checkpoint.py#L319-L456)
- 函数入口：[src/distillschedule/training/checkpoint.py::restore_checkpoint](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/training/checkpoint.py#L541-L559)
- 来源：仅本地会话线索，未公开原始会话；缺少可核实的公开实验回执。

### LoRA 合并的 nearest 与 stochastic 两种舍入都未通过轨迹

- ID：wan-merged-lora-trajectory-rejected
- 状态：correctness_failure / HISTORICAL_TRAJECTORY_FAIL
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：30 cycles/16 ranks 的四臂研究中，两种 merged candidate 的 Critic 偏差约 27.62%/24.76%，baseline repeat 约 0.0748%；调用数、RNG、finite 和局部 G probe 不能替代状态验收。
- 经验：权重误差无偏不保证非线性训练无偏；修复一个局部指标后仍需检查 optimizer、Critic 与持续轨迹。
- 范围：Wan 该 merged LoRA 实现及原固定数值合同，正式 native paired timing 未完成。
- 历史上下文：{"workload": "W1", "model_family": "Wan1.3B", "training_mode": "training", "global_batch": 16, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"current_validation_contract_defined": true}
- 收益条件：{}
- 失败分类：training_state_drift
- 重试信号：merged 更新与精度差异有针对性修复
- 最小验证：先对齐 Critic 梯度/Adam 与 baseline noise floor，再跑冻结轨迹
- 不应重复：finite、调用数和局部 G probe 不能覆盖 24–28% Critic 偏差
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case wan-merged-lora-trajectory-rejected`
- 来源：[sliortega295-ops/DMD-System / summary.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/merged_lora_rejected/summary.md#L18)；定位：观察中的数值文本锚点 L18；同时阅读同节合同、状态与限制。原定位：结果与证据边界

### 候选和 unchanged repeat 都失败时，先定位 baseline 可重复性

- ID：planner-baseline-self-parity-fails
- 状态：inconclusive / HISTORICAL_PARITY_UNRESOLVED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：H3 depth1/depth2 与不变重复比较都未通过冻结数值 gate；首次 Fake backward 提供定位点，但 metadata/RNG 身份相同不证明全部消费值一致。
- 经验：不能简单把所有差异归于新 knob，也不能以 baseline 不稳定为由免验候选；先缩小并固定可判定合同。
- 范围：live H3 native parity audit，不改变失败状态或 tolerance。
- 历史上下文：{"training_mode": "training", "workload": "W5"}
- 正确性前提：{"current_validation_contract_defined": true}
- 收益条件：{}
- 失败分类：not_established
- 重试信号：baseline 自身不一致已解释并固定合同
- 最小验证：先比较不变重复的首次 Fake backward 与实际消费值
- 不应重复：不能用 metadata/RNG 名义相同跳过 baseline 失败
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case planner-baseline-self-parity-fails`
- 关联实现：[Planner 的已测成本及结果资格](implementations.md#planner-calibration)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/search/cost_model.py::CostModel](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/cost_model.py#L77-L195)
- 函数入口：[src/distillschedule/search/measurement.py::normalize_native](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/measurement.py#L18-L98)
- 来源：[sliortega295-ops/DMD-System / DMD_JOINT_PLANNER_LIVEWAN_H3_20260907.md](https://github.com/sliortega295-ops/DMD-System/blob/68c3c0b1bf8c3ad350b27b49344e9bb9402df3fa/docs/experiments/DMD_JOINT_PLANNER_LIVEWAN_H3_20260907.md#L1)；定位：报告范围 L1–L213；L1: Native planner continuation: LiveWan complete, H3 numerical gate blocked；L19: Environment and denominator；L41: LiveWan: completed loop；L74: Memory caveat；L81: H3: native integration works, numerical admission fails；L107: Actual blocker and repeat experiment；L146: Integration bugs fixed and preserved failures；L185: Evidence, replay and stopping point；原定位：结果、合同、限制与失败记录

### checkpoint replay 不能包住 optimizer 更新等副作用

- ID：dmd-pure-call-checkpoint-contract
- 状态：source_contract / SOURCE_CONTRACT_ONLY
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：能力契约将可重算的纯调用与包含 optimizer 的宏调用区分；Self-Forcing Generator checkpoint ON/OFF 还可能改变梯度/cache 语义。
- 经验：先证明 replay 的无副作用范围；checkpoint 开关并非在每个模型上都是纯物理优化。
- 范围：DMD capabilities，需读取目标 runtime 的实际 guards。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"current_validation_contract_defined": true, "pure_replay": true, "checkpoint_semantics_preserved": true}
- 收益条件：{}
- 失败分类：not_established
- 重试信号：待重算范围已证明是纯调用
- 最小验证：比较 forward/replay 的状态、RNG、cache 与梯度，覆盖真实调用边界
- 不应重复：不能 checkpoint 含 optimizer 或改变 Self-Forcing cache 语义的宏调用
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case dmd-pure-call-checkpoint-contract`
- 关联实现：[Helios FA2 selective checkpoint 保留](implementations.md#helios-selective-checkpoint)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/fsdp2_static.py::_selective_flash_checkpoint_contexts](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L84-L111)
- 函数入口：[src/distillschedule/runtime/fsdp2_static.py::configure_fsdp2_checkpointing](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L171-L290)
- 来源：[sliortega295-ops/DMD-System / optimization_capabilities.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/docs/guides/optimization_capabilities.md#L1)；定位：报告范围 L1–L239；L1: Physical capabilities and executable plans；L20: Reuse the existing runtime；L37: Four uses of one executable policy；L59: Model adaptation, not optimization-script copying；L76: Scoped native owner example: DMD2 UNet checkpointing；L113: Official Self-Forcing: Fake-only checkpoint policy；L135: Modern native flow planning is a separately qualified CPU slice；L171: Direct single-CUDA immutable sharing；原定位：结果与证据边界

### tiny trainable-state parity 不能覆盖全模型 AR rollout

- ID：h100-w6-tiny-parity-boundary
- 状态：source_contract / HISTORICAL_TINY_ONLY
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：对既有 tiny 产物做 CPU 后处理，FSDP 与 candidate 各 192 项 state comparisons 通过；当前全模型 C2 gate 仍 NOT_RUN，旧 FA3 full-model failures 没有被该结果取消。
- 经验：明确 fixture 能验证的映射/状态逻辑；shape、phase、attention、长 rollout 与容量需由目标测试补齐。
- 范围：W6 既有八 rank tiny 产物的 CPU 状态比对；没有新增 GPU 运行，也没有导出 rollout output parity。
- 历史上下文：{"workload": "W6", "training_mode": "training", "validation_scope": "tiny_state_cpu_postprocess", "target_hardware": "8x H100-80GB", "physical_gpus": null}
- 正确性前提：{"current_validation_contract_defined": true}
- 收益条件：{}
- 失败分类：not_established
- 重试信号：具备全模型相同数值合同与验证资源
- 最小验证：保留 tiny 192 项结果并补缺失的 full-model C2 gate
- 不应重复：CPU 后处理 PASS 不抹去旧 FA3 全模型失败
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case h100-w6-tiny-parity-boundary`
- 来源：[sliortega295-ops/DMD-System / w6_tiny_trainable_state_parity_r1.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W6/reports/w6_tiny_trainable_state_parity_r1.json)；定位：该结果及 qualification / NOT_RUN 边界

### ResearchBoost：文件缓存与依赖库可能消费全局随机流

- ID：rb-rng-dependency-side-effects
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：其 judge/lessons 报告 Triton file cache 与 xformers 消费 Python random，RNG fingerprint 检查因此拒绝候选。
- 经验：把非训练用途的随机源与训练 RNG 分开验证；只修正已确认的库级作用域，不能重置全局 RNG 掩盖差异。
- 范围：外部特定依赖版本中的 RNG side effect。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"current_validation_contract_defined": true}
- 收益条件：{}
- 失败分类：not_established
- 重试信号：库初始化/缓存对 RNG 的消费已识别
- 最小验证：在匹配位置检查 Python/设备 RNG 与实际样本，再修复状态管理
- 不应重复：不为绕过 gate 删除 RNG fingerprint
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case rb-rng-dependency-side-effects`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / lessons.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/lessons.md#L1)；定位：报告范围 L1–L33；L1: Results and lessons across tasks；L3: Four tasks, four ceilings；L12: General lessons；原定位：Third-party libraries changing state behind your back

### ResearchBoost：no-grad 覆盖 cache 仍可能留下旧梯度连接

- ID：rb-ghost-gradient-inplace-cache
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：numerics card 描述 grad forward 写入 cache 后，被 no-grad 就地覆盖，数值读取已变但旧 CopySlices 梯度边仍存活的情况。
- 经验：同时追踪缓存值和 autograd owner；若发现 baseline 语义缺陷，应建立修正后的新 baseline，不能暗改并宣称等价加速。
- 范围：外部 autoregressive in-place cache 风险，不说明本地每种 cache 都有问题。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"current_validation_contract_defined": true, "cache_autograd_lifetime_validated": true}
- 收益条件：{}
- 失败分类：not_established
- 重试信号：cache 不再跨 grad/no-grad 保留旧图或覆盖活跃存储
- 最小验证：构造最小 CopySlices/in-place 用例，比较值、grad_fn 与梯度
- 不应重复：数值内容更新不代表旧梯度边消失
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case rb-ghost-gradient-inplace-cache`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-numerics-validation/SKILL.md#L1)；定位：报告范围 L1–L98；L6: Numerics tiers and validation；L14: N0–N5；L35: How thresholds are set；L57: Cheap investigation order (T0 → T3)；L68: When only one component cannot be bitwise: isolate it behind a stand-in；L81: Common pitfalls；原定位：Common pitfalls / ghost gradient

### ResearchBoost：接近零的梯度相对误差需要诊断而非事后放宽门槛

- ID：rb-zero-gradient-error-localization
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：numerics card 区分局部数值定位、原生计算精度、baseline 自身差异与训练轨迹判定。
- 经验：使用更高精度参考或预定绝对误差辅助定位；绝不能因为 relative error 难看而在看到候选后改验收规则。
- 范围：外部数值诊断方法；容差由本 workload 冻结合同决定。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"current_validation_contract_defined": true}
- 收益条件：{}
- 失败分类：not_established
- 重试信号：baseline 自差与首个局部偏差已可复现
- 最小验证：先定位原生精度下误差，再验证真实 loss 的梯度和轨迹
- 不应重复：人工零梯度仅能帮助定位，不能代替训练验收
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case rb-zero-gradient-error-localization`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-numerics-validation/SKILL.md#L1)；定位：报告范围 L1–L98；L6: Numerics tiers and validation；L14: N0–N5；L35: How thresholds are set；L57: Cheap investigation order (T0 → T3)；L68: When only one component cannot be bitwise: isolate it behind a stand-in；L81: Common pitfalls；原定位：Common pitfalls / numerical debugging

### ResearchBoost H3：短测很快，较长轨迹暴露 checkpoint replay 数值问题

- ID：rb-h3-checkpoint-qknorm-regression
- 状态：correctness_failure / EXTERNAL_REPORTED_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：外部 video results 报告早期 H3 方案短测约 1.869×，25-step LoRA 梯度出现有限值尖峰并失败；定位到 QK RMSNorm 在 checkpoint forward/replay 选择不同 variant。
- 经验：测试必须跨实际 replay 与更新生命周期；finite、loss 正常和短测加速不能替代状态/轨迹验收。
- 范围：外部 H20 BF16 LoRA ZeRO-3、约 33750 tokens，非本仓库 W5/W6。
- 历史上下文：{"training_mode": "training", "model_family": "MiniMax-H3", "source_project": "ResearchBoost", "gpu_family": "H20"}
- 正确性前提：{"current_validation_contract_defined": true}
- 收益条件：{}
- 失败分类：checkpoint_variant_numerics
- 重试信号：checkpoint forward/replay QK RMSNorm variant 已统一
- 最小验证：先定位首次 LoRA 梯度尖峰，比较 replay 与导数再做冻结轨迹
- 不应重复：finite 或早期约 1.869 倍短测不能覆盖后期失败
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case rb-h3-checkpoint-qknorm-regression`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / video-generation.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/results/video-generation.md#L1)；定位：报告范围 L1–L35；L1: Video Generation；L7: Faster-SelfForcing；L22: Faster-MiniMax-H3；原定位：MiniMax-H3 trajectory failure and repair

### ResearchBoost H3：统一 forward/导数后重新通过其短轨迹 gate

- ID：rb-h3-qknorm-repair
- 状态：positive_observation / EXTERNAL_REPORTED_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：同一外部 H3 报告后续修复统一 QK RMSNorm forward 与导数，约 1.857×，三 seeds/50 steps 通过其验收。
- 经验：失败可以由明确实现修复重新进入验证；该外部短轨迹通过不自动成为本地 H100 全模型证明或长期质量结论。
- 范围：ResearchBoost H3 round50 报告范围，未在本任务复现。
- 历史上下文：{"training_mode": "training", "model_family": "MiniMax-H3", "source_project": "ResearchBoost", "gpu_family": "H20"}
- 正确性前提：{"current_validation_contract_defined": true}
- 收益条件：{}
- 失败分类：not_established
- 重试信号：当前实现具有同样 forward/derivative 一致性
- 最小验证：在目标源码重验 forward/replay 与多 seed 轨迹；保留外部报告身份
- 不应重复：外部三 seed/50-step 结果不自动验收本系统
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case rb-h3-qknorm-repair`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / video-generation.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/results/video-generation.md#L1)；定位：报告范围 L1–L35；L1: Video Generation；L7: Faster-SelfForcing；L22: Faster-MiniMax-H3；原定位：MiniMax-H3 later repaired candidate

### Resident/FSDP/DDP 都完成更新，跨布局数值仍可能失败

- ID：wan-placement-numerics-failure
- 状态：correctness_failure / HISTORICAL_STRICT_PARITY_FAIL
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：四个固定 source arm 均通过 runtime dtype/更新检查，候选相对 resident 的 80/80 rollout rows 均未过冻结 parity；正式配对计时未启动。
- 经验：dtype 修复和 hook 兼容修复不能取消剩余梯度/Adam 差异；先在通信前、SUM/divide 后、clipping 后定位，不能从相似偏差直接宣布根因。
- 范围：Wan PR22、BF16 frozen base/FP32 LoRA、world16 B16、原 P0 合同。
- 历史上下文：{"workload": "W1", "model_family": "Wan1.3B", "training_mode": "training", "global_batch": 16, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"current_validation_contract_defined": true}
- 收益条件：{}
- 失败分类：cross_placement_numerics_unresolved
- 重试信号：placement 后 rollout 差异有定位修复
- 最小验证：先复测 resident control 自差和 80 条 rollout，过冻结 gate 后才计时
- 不应重复：dtype/更新检查通过不能覆盖 80/80 parity 失败
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case wan-placement-numerics-failure`
- 来源：[sliortega295-ops/DMD-System / wan13b_fair_pr22_iteration328_20260822.md](https://github.com/sliortega295-ops/DMD-System/blob/524048c94c6374ae5071bc23e93daf19fc1e0bb9/docs/experiments/wan13b_fair_pr22_iteration328_20260822.md#L1)；定位：报告范围 L1–L101；L1: Iteration 328 — Wan1.3B fair Resident/FSDP2 result；L3: Status；L10: Source and compatibility patch；L28: Frozen workload；L46: Per-placement execution；L57: Strict cross-placement correctness；L82: Conclusion boundary；L90: Resource and cleanup evidence；原定位：结果、合同、限制与失败记录
