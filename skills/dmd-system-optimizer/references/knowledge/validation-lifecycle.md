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
- 条件：{"workload": "W5", "sigma_contract": "Grid1000"}
- 失败分类：numerical_cause_unresolved
- 重试条件：定位并修复该失败，保持原验收标准；重新验证后才比较速度。
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json)；定位：rows[workload=W5]; numerical_policy

### loss 上升应与匹配 baseline 比较

- ID：helios-loss-trend-lead
- 状态：unverified / UNVERIFIED_CONVERSATION_LEAD
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：本地会话纠正过把 loss 上升直接当作数值异常的判断；该会话没有补齐匹配 baseline 的一致性证据。
- 经验：关注优化版相对 baseline 的额外偏离和梯度/参数/optimizer 状态；单臂趋势不能裁决一致性。
- 范围：会话提炼的方法线索；不作为 H3 AR 正确或错误的实验结论。
- 条件：{"workload": "W6"}
- 失败分类：missing_matched_baseline
- 重试条件：取得匹配 baseline 和状态/轨迹证据后再判定。
- 来源：仅本地会话线索，未公开原始会话；缺少可核实的公开实验回执。

### 恢复时重复分配 Adam 状态可能出现额外峰值

- ID：livewan-checkpoint-memory-lead
- 状态：unverified / UNVERIFIED_CONVERSATION_LEAD
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：本地会话提出 LiveWan 恢复时避免第二份完整 Adam 分配；当时只有 CPU 检查线索，GPU 结果未在本库核实。
- 经验：训练稳态能运行，不说明恢复峰值可承受；检查载入过程是否复用现有 optimizer state。
- 范围：私有会话线索，未公开原始执行回执，不能称已解决所有恢复 OOM。
- 条件：{"model_family": "livewan"}
- 失败分类：suspected_restore_allocation
- 重试条件：检查实际 checkpoint restore 实现，测量恢复峰值并核对恢复后 optimizer 更新。
- 来源：仅本地会话线索，未公开原始会话；缺少可核实的公开实验回执。

### LoRA 合并的 nearest 与 stochastic 两种舍入都未通过轨迹

- ID：wan-merged-lora-trajectory-rejected
- 状态：correctness_failure / HISTORICAL_TRAJECTORY_FAIL
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：30 cycles/16 ranks 的四臂研究中，两种 merged candidate 的 Critic 偏差约 27.62%/24.76%，baseline repeat 约 0.0748%；调用数、RNG、finite 和局部 G probe 不能替代状态验收。
- 经验：权重误差无偏不保证非线性训练无偏；修复一个局部指标后仍需检查 optimizer、Critic 与持续轨迹。
- 范围：Wan 该 merged LoRA 实现及原固定数值合同，正式 native paired timing 未完成。
- 条件：{"workload": "W1", "model_family": "Wan1.3B", "training_mode": "training", "global_batch": 16, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：training_state_drift
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / summary.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/merged_lora_rejected/summary.md)；定位：结果与证据边界

### 候选和 unchanged repeat 都失败时，先定位 baseline 可重复性

- ID：planner-baseline-self-parity-fails
- 状态：inconclusive / HISTORICAL_PARITY_UNRESOLVED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：H3 depth1/depth2 与不变重复比较都未通过冻结数值 gate；首次 Fake backward 提供定位点，但 metadata/RNG 身份相同不证明全部消费值一致。
- 经验：不能简单把所有差异归于新 knob，也不能以 baseline 不稳定为由免验候选；先缩小并固定可判定合同。
- 范围：live H3 native parity audit，不改变失败状态或 tolerance。
- 条件：{"training_mode": "training", "workload": "W5"}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / DMD_JOINT_PLANNER_LIVEWAN_H3_20260907.md](https://github.com/sliortega295-ops/DMD-System/blob/68c3c0b1bf8c3ad350b27b49344e9bb9402df3fa/docs/experiments/DMD_JOINT_PLANNER_LIVEWAN_H3_20260907.md)；定位：结果、合同、限制与失败记录

### checkpoint replay 不能包住 optimizer 更新等副作用

- ID：dmd-pure-call-checkpoint-contract
- 状态：source_contract / SOURCE_CONTRACT_ONLY
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：能力契约将可重算的纯调用与包含 optimizer 的宏调用区分；Self-Forcing Generator checkpoint ON/OFF 还可能改变梯度/cache 语义。
- 经验：先证明 replay 的无副作用范围；checkpoint 开关并非在每个模型上都是纯物理优化。
- 范围：DMD capabilities，需读取目标 runtime 的实际 guards。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / optimization_capabilities.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/docs/guides/optimization_capabilities.md)；定位：结果与证据边界

### tiny trainable-state parity 不能覆盖全模型 AR rollout

- ID：h100-w6-tiny-parity-boundary
- 状态：source_contract / HISTORICAL_TINY_ONLY
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：对既有 tiny 产物做 CPU 后处理，FSDP 与 candidate 各 192 项 state comparisons 通过；当前全模型 C2 gate 仍 NOT_RUN，旧 FA3 full-model failures 没有被该结果取消。
- 经验：明确 fixture 能验证的映射/状态逻辑；shape、phase、attention、长 rollout 与容量需由目标测试补齐。
- 范围：W6 既有八 rank tiny 产物的 CPU 状态比对；没有新增 GPU 运行，也没有导出 rollout output parity。
- 条件：{"workload": "W6", "training_mode": "training", "validation_scope": "tiny_state_cpu_postprocess", "target_hardware": "8x H100-80GB", "physical_gpus": null}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / w6_tiny_trainable_state_parity_r1.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W6/reports/w6_tiny_trainable_state_parity_r1.json)；定位：该结果及 qualification / NOT_RUN 边界

### ResearchBoost：文件缓存与依赖库可能消费全局随机流

- ID：rb-rng-dependency-side-effects
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：其 judge/lessons 报告 Triton file cache 与 xformers 消费 Python random，RNG fingerprint 检查因此拒绝候选。
- 经验：把非训练用途的随机源与训练 RNG 分开验证；只修正已确认的库级作用域，不能重置全局 RNG 掩盖差异。
- 范围：外部特定依赖版本中的 RNG side effect。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / lessons.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/lessons.md)；定位：Third-party libraries changing state behind your back

### ResearchBoost：no-grad 覆盖 cache 仍可能留下旧梯度连接

- ID：rb-ghost-gradient-inplace-cache
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：numerics card 描述 grad forward 写入 cache 后，被 no-grad 就地覆盖，数值读取已变但旧 CopySlices 梯度边仍存活的情况。
- 经验：同时追踪缓存值和 autograd owner；若发现 baseline 语义缺陷，应建立修正后的新 baseline，不能暗改并宣称等价加速。
- 范围：外部 autoregressive in-place cache 风险，不说明本地每种 cache 都有问题。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-numerics-validation/SKILL.md)；定位：Common pitfalls / ghost gradient

### ResearchBoost：接近零的梯度相对误差需要诊断而非事后放宽门槛

- ID：rb-zero-gradient-error-localization
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：numerics card 区分局部数值定位、原生计算精度、baseline 自身差异与训练轨迹判定。
- 经验：使用更高精度参考或预定绝对误差辅助定位；绝不能因为 relative error 难看而在看到候选后改验收规则。
- 范围：外部数值诊断方法；容差由本 workload 冻结合同决定。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-numerics-validation/SKILL.md)；定位：Common pitfalls / numerical debugging

### ResearchBoost H3：短测很快，较长轨迹暴露 checkpoint replay 数值问题

- ID：rb-h3-checkpoint-qknorm-regression
- 状态：correctness_failure / EXTERNAL_REPORTED_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：外部 video results 报告早期 H3 方案短测约 1.869×，25-step LoRA 梯度出现有限值尖峰并失败；定位到 QK RMSNorm 在 checkpoint forward/replay 选择不同 variant。
- 经验：测试必须跨实际 replay 与更新生命周期；finite、loss 正常和短测加速不能替代状态/轨迹验收。
- 范围：外部 H20 BF16 LoRA ZeRO-3、约 33750 tokens，非本仓库 W5/W6。
- 条件：{"training_mode": "training", "model_family": "MiniMax-H3", "source_project": "ResearchBoost", "gpu_family": "H20"}
- 失败分类：checkpoint_variant_numerics
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / video-generation.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/results/video-generation.md)；定位：MiniMax-H3 trajectory failure and repair

### ResearchBoost H3：统一 forward/导数后重新通过其短轨迹 gate

- ID：rb-h3-qknorm-repair
- 状态：positive_observation / EXTERNAL_REPORTED_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：同一外部 H3 报告后续修复统一 QK RMSNorm forward 与导数，约 1.857×，三 seeds/50 steps 通过其验收。
- 经验：失败可以由明确实现修复重新进入验证；该外部短轨迹通过不自动成为本地 H100 全模型证明或长期质量结论。
- 范围：ResearchBoost H3 round50 报告范围，未在本任务复现。
- 条件：{"training_mode": "training", "model_family": "MiniMax-H3", "source_project": "ResearchBoost", "gpu_family": "H20"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / video-generation.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/results/video-generation.md)；定位：MiniMax-H3 later repaired candidate

### Resident/FSDP/DDP 都完成更新，跨布局数值仍可能失败

- ID：wan-placement-numerics-failure
- 状态：correctness_failure / HISTORICAL_STRICT_PARITY_FAIL
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：四个固定 source arm 均通过 runtime dtype/更新检查，候选相对 resident 的 80/80 rollout rows 均未过冻结 parity；正式配对计时未启动。
- 经验：dtype 修复和 hook 兼容修复不能取消剩余梯度/Adam 差异；先在通信前、SUM/divide 后、clipping 后定位，不能从相似偏差直接宣布根因。
- 范围：Wan PR22、BF16 frozen base/FP32 LoRA、world16 B16、原 P0 合同。
- 条件：{"workload": "W1", "model_family": "Wan1.3B", "training_mode": "training", "global_batch": 16, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：cross_placement_numerics_unresolved
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / wan13b_fair_pr22_iteration328_20260822.md](https://github.com/sliortega295-ops/DMD-System/blob/524048c94c6374ae5071bc23e93daf19fc1e0bb9/docs/experiments/wan13b_fair_pr22_iteration328_20260822.md)；定位：结果、合同、限制与失败记录
