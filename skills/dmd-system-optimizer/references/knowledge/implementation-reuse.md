# 先核对已有实现，再决定移植或重写

已有源码提供候选实现；模型名、接口相同或库存存在都不证明当前调用可用。

机制级经验可以跨模型检索；具体代码必须满足实现级限制。
记录精确源码版本和依赖，但不要把某一 GPU 名当作整个机制的永久适用边界。
未生效的快路径、库版本问题和真正无收益分开；失败类型决定下一步。
组合后重新检查资源和状态，不把多个独立 microbenchmark 收益直接相加。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- implementation_guards_match: true
- license_and_dependencies_checked: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- fast_path_active: true

最小验证：读实际入口、guard、fallback、精度和 backward；用当前调用的最小测试确认命中，再做组合校准。

## 已整理案例

### 14 个已有实现入口可检索，仍需检查 guards

- ID：dmd-implementation-inventory
- 状态：source_contract / SOURCE_ONLY_GPU_NOT_RUN
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：该快照 implementation_library 的 ENTRIES 有14个入口，涵盖 modulation、LoRA、RMSNorm、splitKV、SwiGLU 等。
- 经验：库存为复用提供起点；其自身也要求 workload adapter、成本校准和组合验证。
- 范围：静态库存，不执行模型导入，不宣称14项都适配新 workload。
- 历史上下文：{}
- 正确性前提：{"implementation_guards_match": true, "license_and_dependencies_checked": true}
- 收益条件：{"fast_path_active": true}
- 失败分类：not_established
- 重试信号：实际目标 guard 与清单描述存在差异
- 最小验证：沿清单打开实现与调用者，记录 dtype/grad/shape/fallback 和测试缺口
- 不应重复：不能把 inventory 中出现名字当成实现可用
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case dmd-implementation-inventory`
- 来源：[sliortega295-ops/DMD-System / implementation_library.py](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/implementation_library.py)；定位：ENTRIES; implementation_inventory.search_admission

### H100 W1 的已测组合与后续未测 norm 代码分开

- ID：h100-w1-source-qualified-pair
- 状态：positive_observation / HISTORICAL_H100_SINGLE_TRIAL_REVIEW
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：7a668bcb 源码上的 matched 观察为 52.066607→37.669795 秒；之后的 norm/import 修复不是这个计时的来源。
- 经验：移植历史优化包时，分清已测 source/config 与后续代码；A100 源码测试不能自动成为 H100 性能证据。
- 范围：W1 Wan H100、单次 matched pair，numerical policy REVIEW，严格 full-state qualification 未闭合。
- 历史上下文：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W1", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"implementation_guards_match": true, "license_and_dependencies_checked": true}
- 收益条件：{"fast_path_active": true}
- 失败分类：not_established
- 重试信号：需要在当前源码重建 matched pair
- 最小验证：核对 7a668bcb 与后续 norm/import 修复的差异，再跑同源数值/时间
- 不应重复：不得把旧 52.07→37.67 秒归给之后的修复
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case h100-w1-source-qualified-pair`
- 来源：[sliortega295-ops/DMD-System / wan_matched_r1_result.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W1/artifacts/wan_matched_r1_result.json#L4)；定位：观察中的数值文本锚点 L4；同时阅读同节合同、状态与限制。原定位：该结果及 qualification / NOT_RUN 边界

### 离线 import 故障发生在模型之前，修复代码不等于新 GPU 结果

- ID：h100-offline-import-and-wrapper-identity
- 状态：implementation_failure / HISTORICAL_ENVIRONMENT_FAILURE
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：W1 canary 被 package init 自动选择的 FA3 Hub 请求触发 OfflineModeIsEnabled；生产 offline binder 修复后，包装器身份还需 unwrap 到实际函数。新 GPU canary NOT_RUN。
- 经验：验证依赖解析、真正导入/执行的函数与 pinned identity；保留旧有效测试，但不要把 import 修复宣称成已复测性能。
- 范围：H100 campaign r47 的 W1 离线 canary 恢复。
- 历史上下文：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W1", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"implementation_guards_match": true, "license_and_dependencies_checked": true}
- 收益条件：{"fast_path_active": true}
- 失败分类：offline_import_resolution
- 重试信号：offline binder 与实际 wrapper 身份已核实
- 最小验证：最小 import 检查无 Hub 请求、unwrap 指向预期函数，再做缺失 canary
- 不应重复：修复 import 不等于新增 GPU PASS
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case h100-offline-import-and-wrapper-identity`
- 来源：[sliortega295-ops/DMD-System / h100_campaign_handoff_20261016_r47.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/campaign/h100_campaign_handoff_20261016_r47.md#L1)；定位：报告范围 L1–L33；L1: R47 — SSH recovered; W2 completed; W1 canary import repaired; foreign GPU use blocks launch；L5: New measured W2 result；L17: W1 actual cheap failure, not new model/OOM/numeric failure；L24: Actual resource blocker after reconnect；L28: Next authorized serial work；原定位：该结果及 qualification / NOT_RUN 边界

### ResearchBoost：冻结计算权重可预转换，状态与输入梯度需单独核对

- ID：rb-frozen-dtype-not-whole-model
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：多 forward card 将重复 cast、冻结权重计算 dtype 和其他 buffer/训练状态分开处理。
- 经验：候选应只作用于可证明兼容的对象；冻结参数仍可能需要输入梯度，整模型强转 dtype 可能改变未允许的状态。
- 范围：外部 frozen-module 优化思路，目标 dtype/AMP 路径需本地实证。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"implementation_guards_match": true, "license_and_dependencies_checked": true}
- 收益条件：{"fast_path_active": true}
- 失败分类：not_established
- 重试信号：cast 热点来自真正冻结且语义兼容的权重
- 最小验证：分别检查冻结权重、buffer、LoRA、Adam dtype 与梯度
- 不应重复：不能对整个模型粗暴 cast 以消除重复转换
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case rb-frozen-dtype-not-whole-model`
- 关联实现：[Wan 冻结 LoRA 路径的 cast cache wrapper](implementations.md#wan-cast-cache)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/wan13b_lora_cast.py::install_generator_nograd_lora_cast_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_cast.py#L265-L427)
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-fsdp-multi-forward/SKILL.md#L1)；定位：报告范围 L1–L88；L6: FSDP: several forwards in one iteration；L24: 1. All-gather once per iteration；L33: 2. Cast frozen models to the compute dtype before wrapping in FSDP；L45: 3. Keep the frozen encoder's parameters resident；L54: 4. Cross-attention K/V cache；L63: 5. Remove host syncs；L72: 6. Spend the freed memory on speed；L83: What ships with it；原定位：frozen weights / compute dtype

### 配置写着 GPU residual，实际旧 runtime 根本没有实现这个参数

- ID：wan-runtime-capability-noop
- 状态：positive_observation / HISTORICAL_MECHANISM_DIAGNOSTIC
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：旧 source 不接受 residual_storage 也不读相应 block 字段，CPU staging 继续发生；换到支持该能力的 pinned runtime 后，16 GPU 上旧 residual copy signatures 归零。
- 经验：能力声明要绑定真实 helper/signature/call sites 与所有 rank receipt；消除 memcpy 的机制证据不等于其余 NCCL/空隙变化都已因果归因。
- 范围：Wan resident GPU-residual correction；单次 unprofiled diagnostic，strict cross-arm parity 未作该诊断阻断，formal_eligibility=false。
- 历史上下文：{"workload": "W1", "model_family": "Wan1.3B", "training_mode": "training", "global_batch": 16, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"implementation_guards_match": true, "license_and_dependencies_checked": true}
- 收益条件：{"fast_path_active": true}
- 失败分类：not_established
- 重试信号：目标 runtime 真正支持并消费 residual_storage
- 最小验证：检查实际函数签名/字段消费与全部 rank copy signatures
- 不应重复：配置项存在但未被读取不能视为 offload 关闭
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case wan-runtime-capability-noop`
- 来源：[sliortega295-ops/DMD-System / wan13b_resident_gpu_residual_iteration336_20260823.md](https://github.com/sliortega295-ops/DMD-System/blob/af240f5e53f9a43eefe523a6473aa6a9fba0164e/docs/experiments/wan13b_resident_gpu_residual_iteration336_20260823.md#L1)；定位：报告范围 L1–L105；L1: Iteration 336 — Wan1.3B resident GPU-residual correction；L3: Status；L11: Root cause and fix；L28: Frozen workload；L37: World16 execution；L47: Exact post-fix transfer evidence；L74: Diagnostic timeline comparison；L90: Unprofiled timing boundary；原定位：结果、合同、限制与失败记录
