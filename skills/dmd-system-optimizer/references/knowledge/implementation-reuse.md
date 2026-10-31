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
- 条件：{}
- 失败分类：not_established
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[sliortega295-ops/DMD-System / implementation_library.py](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/implementation_library.py)；定位：ENTRIES; implementation_inventory.search_admission

### H100 W1 的已测组合与后续未测 norm 代码分开

- ID：h100-w1-source-qualified-pair
- 状态：positive_observation / HISTORICAL_H100_SINGLE_TRIAL_REVIEW
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：7a668bcb 源码上的 matched 观察为 52.066607→37.669795 秒；之后的 norm/import 修复不是这个计时的来源。
- 经验：移植历史优化包时，分清已测 source/config 与后续代码；A100 源码测试不能自动成为 H100 性能证据。
- 范围：W1 Wan H100、单次 matched pair，numerical policy REVIEW，严格 full-state qualification 未闭合。
- 条件：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W1", "gpu_family": "H100", "gpu_memory_gb": 80}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / wan_matched_r1_result.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W1/artifacts/wan_matched_r1_result.json)；定位：该结果及 qualification / NOT_RUN 边界

### 离线 import 故障发生在模型之前，修复代码不等于新 GPU 结果

- ID：h100-offline-import-and-wrapper-identity
- 状态：implementation_failure / HISTORICAL_ENVIRONMENT_FAILURE
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：W1 canary 被 package init 自动选择的 FA3 Hub 请求触发 OfflineModeIsEnabled；生产 offline binder 修复后，包装器身份还需 unwrap 到实际函数。新 GPU canary NOT_RUN。
- 经验：验证依赖解析、真正导入/执行的函数与 pinned identity；保留旧有效测试，但不要把 import 修复宣称成已复测性能。
- 范围：H100 campaign r47 的 W1 离线 canary 恢复。
- 条件：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W1", "gpu_family": "H100", "gpu_memory_gb": 80}
- 失败分类：offline_import_resolution
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / h100_campaign_handoff_20261016_r47.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/campaign/h100_campaign_handoff_20261016_r47.md)；定位：该结果及 qualification / NOT_RUN 边界

### ResearchBoost：冻结计算权重可预转换，状态与输入梯度需单独核对

- ID：rb-frozen-dtype-not-whole-model
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：多 forward card 将重复 cast、冻结权重计算 dtype 和其他 buffer/训练状态分开处理。
- 经验：候选应只作用于可证明兼容的对象；冻结参数仍可能需要输入梯度，整模型强转 dtype 可能改变未允许的状态。
- 范围：外部 frozen-module 优化思路，目标 dtype/AMP 路径需本地实证。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-fsdp-multi-forward/SKILL.md)；定位：frozen weights / compute dtype

### 配置写着 GPU residual，实际旧 runtime 根本没有实现这个参数

- ID：wan-runtime-capability-noop
- 状态：positive_observation / HISTORICAL_MECHANISM_DIAGNOSTIC
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：旧 source 不接受 residual_storage 也不读相应 block 字段，CPU staging 继续发生；换到支持该能力的 pinned runtime 后，16 GPU 上旧 residual copy signatures 归零。
- 经验：能力声明要绑定真实 helper/signature/call sites 与所有 rank receipt；消除 memcpy 的机制证据不等于其余 NCCL/空隙变化都已因果归因。
- 范围：Wan resident GPU-residual correction；单次 unprofiled diagnostic，strict cross-arm parity 未作该诊断阻断，formal_eligibility=false。
- 条件：{"workload": "W1", "model_family": "Wan1.3B", "training_mode": "training", "global_batch": 16, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / wan13b_resident_gpu_residual_iteration336_20260823.md](https://github.com/sliortega295-ops/DMD-System/blob/af240f5e53f9a43eefe523a6473aa6a9fba0164e/docs/experiments/wan13b_resident_gpu_residual_iteration336_20260823.md)；定位：结果、合同、限制与失败记录
