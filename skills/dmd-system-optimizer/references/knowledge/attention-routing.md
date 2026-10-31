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
- 条件：{"workload": "W6", "ranks": 16}
- 失败分类：net_slowdown_cause_not_isolated
- 重试条件：后端调用边界、同步、layout 或真实几何改变后，重新测该组合。
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r43/conclusion.json)；定位：actual_FA3_calls_per_cycle_per_rank; control_C2_s; candidate_C2_s

### W1：primitive 覆盖与实际路由范围必须分开

- ID：helios-w1-fa3-r25
- 状态：positive_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：原生 C2 单配对节省 0.324606359 秒；实际只路由 11040-token 调用；1880/4308 strict tensor comparisons not allclose。
- 经验：不能用探针覆盖其他长度来宣称它们在整模型内生效；正向时间不消除 numerical REVIEW。
- 范围：W1 r25 手工 FA3 路由与当时 retain4 对照，非默认或长期晋级。
- 条件：{"workload": "W1", "ranks": 16}
- 失败分类：not_established
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r25/conclusion.json)；定位：saved_s; coverage_caveat; review_note; long_trajectory

### 两次 SAC 目标计数为零，真正原因是运行时落到 SDPA

- ID：helios-fa2-backend-missing
- 状态：implementation_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：预期 varlen/fixed FA2 target 都未命中；审计发现 launcher 未导出 LOCAL_KERNELS。固定本地 FA2 身份后 96.529020→95.725157 秒。
- 经验：先证明实际 backend 与目标 op 命中，再判断选择性保留是否无效；安装成功不等于运行路径生效。
- 范围：No-GAN closure 的 pinned local FA2 与该 launcher。
- 条件：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": false, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：runtime_backend_fallback
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/111e293dc79978132af91ab5515a096d941b94fe/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md)；定位：结果、合同、限制与失败记录

### ResearchBoost/SGLang：稀疏/缓存/量化快路径需单独质量合同

- ID：rb-inference-approximation-boundary
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：外部 diffusion card 将近似缓存、稀疏与量化等方法列为可用方向，但这些路径并不天然提供逐位相同输出。
- 经验：只把它们作为另一个获准的近似问题候选；不能把 inference 视觉可接受当作训练梯度/轨迹等价。
- 范围：外部 diffusion inference 工具目录，具体方法各自质量范围。
- 条件：{"training_mode": "inference", "approximation_permitted": true, "source_project": "SGLang"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/sglang-diffusion-performance/SKILL.md)；定位：Approximate inference paths
