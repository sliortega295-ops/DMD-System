# Attention backend 与真实调用覆盖

可用且更快的 backend 依赖真实 shape、dtype、梯度和缓存路径；局部 benchmark 只支持对应调用。

先检查现有 backend 和 guards，再决定新增 kernel。
实际被路由的 shape 与 primitive 探针覆盖分开；测过若干长度不能证明训练实际出现了它们。
快路径确实启用也可能整体变慢；检查布局拷贝、同步、workspace、replay 与 backward。
新 backend 和同步消除要分别归因；不得借旧 backend 的结果宣称新组合增量。
推理路径支持不等于可微调用已验证。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- attention_semantics_supported: true
- backward_and_cache_validated: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- target_shapes_hit_fast_path: true
- backend_e2e_advantage: true

最小验证：枚举实际 shape/dtype/grad/mask/cache 调用，记录 fast-path 与 fallback 次数，再测完整周期。

## 已整理案例

### W6：FA3 已启用仍可能整体变慢

- ID：helios-w6-fa3-r43
- 状态：negative_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：实际 FA3 每周期每 rank 350 次调用；C2 255.528302602→257.709695683 秒。
- 经验：快路径命中证据与收益证据分开；后续去同步的收益不把这次 backend 试验改写为成功。
- 范围：W6 r43 的同源单配对，grad/selected attention 未改，numerical REVIEW。
- 历史上下文：{"workload": "W6", "ranks": 16}
- 正确性前提：{"attention_semantics_supported": true, "backward_and_cache_validated": true}
- 收益条件：{"target_shapes_hit_fast_path": true, "backend_e2e_advantage": true}
- 失败分类：net_slowdown_cause_not_isolated
- 重试信号：当前 shape/backend 版本或关键路径比例变化
- 最小验证：确认实际 FA3 调用覆盖，再比较完整周期和 forward/backward 数值
- 不应重复：不能从其他模型的 FA3 快值推定 W6 必快
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-w6-fa3-r43`
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r43/conclusion.json#L8)；定位：观察中的数值文本锚点 L8；同时阅读同节合同、状态与限制。原定位：actual_FA3_calls_per_cycle_per_rank; control_C2_s; candidate_C2_s

### W1：primitive 覆盖与实际路由范围必须分开

- ID：helios-w1-fa3-r25
- 状态：positive_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：原生 C2 单配对节省 0.324606359 秒；实际只路由 11040-token 调用；1880/4308 strict tensor comparisons not allclose。
- 经验：不能用探针覆盖其他长度来宣称它们在整模型内生效；正向时间不消除 numerical REVIEW。
- 范围：W1 r25 手工 FA3 路由与当时 retain4 对照，非默认或长期晋级。
- 历史上下文：{"workload": "W1", "ranks": 16}
- 正确性前提：{"attention_semantics_supported": true, "backward_and_cache_validated": true}
- 收益条件：{"target_shapes_hit_fast_path": true, "backend_e2e_advantage": true}
- 失败分类：not_established
- 重试信号：实际路由覆盖改善且数值失败已有修复证据
- 最小验证：重查 11040-token 路由与严格 tensor comparisons，过门后才计时
- 不应重复：未解释 1880 项不一致前不以小时间差晋级
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-w1-fa3-r25`
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r25/conclusion.json)；定位：saved_s; coverage_caveat; review_note; long_trajectory

### 两次 SAC 目标计数为零，真正原因是运行时落到 SDPA

- ID：helios-fa2-backend-missing
- 状态：implementation_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：预期 varlen/fixed FA2 target 都未命中；审计发现 launcher 未导出 LOCAL_KERNELS。固定本地 FA2 身份后 96.529020→95.725157 秒。
- 经验：先证明实际 backend 与目标 op 命中，再判断选择性保留是否无效；安装成功不等于运行路径生效。
- 范围：No-GAN closure 的 pinned local FA2 与该 launcher。
- 历史上下文：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": false, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"attention_semantics_supported": true, "backward_and_cache_validated": true, "attention_backend": "FA2"}
- 收益条件：{"target_shapes_hit_fast_path": true, "backend_e2e_advantage": true}
- 失败分类：runtime_backend_fallback
- 重试信号：运行环境确实加载所需 local FA2 实现
- 最小验证：核对 LOCAL_KERNELS、实际函数身份及 varlen/fixed 命中计数
- 不应重复：开关声明不等于 runtime 使用该 backend
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-fa2-backend-missing`
- 关联实现：[Helios FA2 selective checkpoint 保留](implementations.md#helios-selective-checkpoint)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/fsdp2_static.py::_selective_flash_checkpoint_contexts](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L84-L111)
- 函数入口：[src/distillschedule/runtime/fsdp2_static.py::configure_fsdp2_checkpointing](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L171-L290)
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/111e293dc79978132af91ab5515a096d941b94fe/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md#L1)；定位：报告范围 L1–L146；L1: Helios14B node-HSDP optimization closure (Iterations 368-382)；L3: Status and claim boundary；L27: New experiment ladder；L52: Effective methods；L54: Explicit pinned local FlashAttention-2；L70: Critic selective FA2 checkpoint-output retention；L87: Rejected and bounded-negative methods；L105: Post-winner Nsight result；原定位：结果、合同、限制与失败记录

### ResearchBoost/SGLang：稀疏/缓存/量化快路径需单独质量合同

- ID：rb-inference-approximation-boundary
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：外部 diffusion card 将近似缓存、稀疏与量化等方法列为可用方向，但这些路径并不天然提供逐位相同输出。
- 经验：只把它们作为另一个获准的近似问题候选；不能把 inference 视觉可接受当作训练梯度/轨迹等价。
- 范围：外部 diffusion inference 工具目录，具体方法各自质量范围。
- 历史上下文：{"training_mode": "inference", "approximation_permitted": true, "source_project": "SGLang"}
- 正确性前提：{"attention_semantics_supported": true, "backward_and_cache_validated": true, "training_mode": "inference", "approximation_authorized": true, "quality_contract_frozen": true}
- 收益条件：{"target_shapes_hit_fast_path": true, "backend_e2e_advantage": true}
- 失败分类：not_established
- 重试信号：用户授权近似并冻结质量/误差门
- 最小验证：明确缓存/稀疏/量化的质量评测与真实服务收益
- 不应重复：不能把近似方向自动当作严格等价优化
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case rb-inference-approximation-boundary`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/sglang-diffusion-performance/SKILL.md#L1)；定位：报告范围 L1–L565；L6: SGLang Diffusion Performance Tuning；L10: Start from the model's cookbook page；L38: Native Backend Gate；L51: Section 1: Lossless Optimizations；L73: Choosing what goes in `--layerwise-offload-components`；L105: Prefetch depth has a knee；L118: Host memory is part of the placement decision；L135: When the transfer knobs do nothing；原定位：Approximate inference paths
