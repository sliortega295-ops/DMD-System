# 缓存、预计算与状态失效

重复计算可以被缓存替换，但 key、版本、随机性和维护成本决定复用是否正确且合算。

冻结参数不等于输出恒定：输入、噪声、条件和 timestep 都可能变化。
缓存优化器更新前的结果供更新后使用会改变训练；需要证明生命周期。
缓存可能减少计算却增加带宽、内存、同步或索引成本，完整周期可能变慢。
模式相同但 reuse 次数不同可以翻转收益；因缓存维护成本失败的案例可在复用次数显著增加后重试。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- cache_key_complete: true
- invalidation_validated: true
- cached_gradients_preserved: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- cache_reuse_sufficient: true
- saved_compute_exceeds_cache_cost: true

最小验证：先列 value 依赖的输入、shape、dtype、conditioning、timestep、权重、adapter、模式与随机性；验证命中和失效。

## 已整理案例

### W6：AdaLN bank 尝试没有带来时间收益

- ID：helios-w6-adaln-r41
- 状态：negative_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：C2 255.602136769→256.312003678 秒；exact cache input gate 通过，参数未删除、权重未合并。
- 经验：缓存输入条件通过不保证净收益；需要区分计算节省和维护/访问成本。
- 范围：W6 r41 单周期配对，完整状态或轨迹 REVIEW。
- 条件：{"workload": "W6", "ranks": 16}
- 失败分类：net_slowdown_cause_not_isolated
- 重试条件：reuse 数量、缓存布局或访问成本改变后，用可判别的小测量重新估算。
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r41/conclusion.json)；定位：exact_cache_input_gate; control_C2_s; candidate_C2_s; numerical_status

### W1：RETAIN4 与 frozen_down 的条件边际不可相加

- ID：helios-w1-frozen-down-r1
- 状态：negative_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：20.028682893→20.169802765 秒，MEASURED_NO_C2_GAIN；范围标为 RETAIN4_VS_FROZEN_DOWN_ONLY。
- 经验：两个替代策略的比较不支持叠加它们的收益。
- 范围：r1 followup 对应源和该对照，不是所有 frozen_down 路径都无效。
- 条件：{"workload": "W1"}
- 失败分类：net_slowdown_comparison_scope
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json)；定位：followups.wan_frozen_down

### 缓存 cast 不一定值得其维护成本

- ID：wan-cast-cache-negative
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：Wan resident cast-cache 单配对 27.050073→27.166219 秒，未晋级。
- 经验：先证实重复 cast 的暴露成本与版本失效成本；缓存本身不是收益证据。
- 范围：该 Wan frozen cast-cache 实现及当时 substrate。
- 条件：{"workload": "W1", "model_family": "Wan1.3B", "training_mode": "training", "global_batch": 16, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：cache_net_regression
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / WAN13B_RESIDENT_OPTIMIZATION_LADDER_20260824.md](https://github.com/sliortega295-ops/DMD-System/blob/9186aee022e7d8933ca84429bde7f54868f203de/docs/experiments/WAN13B_RESIDENT_OPTIMIZATION_LADDER_20260824.md)；定位：结果、合同、限制与失败记录

### history self-attention K/V cache 依赖的 attention 模式不成立

- ID：helios-history-kv-contract-rejected
- 状态：correctness_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：live workload 为 restrict_self_attn=false；已有 history KV 实现要求 restricted self-attention，因此未启动全模型计时。
- 经验：缓存正确性由 attention 依赖决定；不能为了复用缓存而偷偷改变 mask/attention contract。
- 范围：Helios No-GAN 的该 history KV 路径；其他合法跨步缓存仍可评估。
- 条件：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": false, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：attention_dependency_mismatch
- 重试条件：提供不改变 live attention 依赖的新 cache 实现，或由用户明确建立另一个 workload 合同。
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/111e293dc79978132af91ab5515a096d941b94fe/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md)；定位：结果、合同、限制与失败记录

### No-GAN cross-KV 命中 880 次仍几乎没有 E2E 收益

- ID：helios-crosskv-nogan-negligible
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：每 rank 200 fills/880 reuses 且零 mismatch，Full5 96.529020→96.497111 秒，只约 0.033%。
- 经验：高命中率不说明省下关键路径大头；保留机制正确性证据，避免将微小单次差值称为稳定加速。
- 范围：Helios No-GAN node-HSDP 的 per-rollout cross-attention cache。
- 条件：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": false, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：insufficient_exposed_saved_compute
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/111e293dc79978132af91ab5515a096d941b94fe/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md)；定位：结果、合同、限制与失败记录

### GAN-on 的 cross-KV 单配对约有 0.75% 改善

- ID：helios-crosskv-gan-positive
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：同样 200 fills/880 reuses，限定 no-grad B1 且 key 绑定 Generator 版本与 encoder storage/version；142.295342→141.235767 秒。
- 经验：同类 cache 的收益随调用图改变；额外 uncached-reference 校验运行很慢，不能拿它作正式性能分母。
- 范围：GAN-on C8 per-rollout cache，不能直接复用到不同 conditioning 或可微路径。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration359_c8_crosskv.md](https://github.com/sliortega295-ops/DMD-System/blob/7f7a5c28db628a22ad494d4533590dc71b3d9afa/docs/experiments/gan_cuda_baseline_opt_20260829/iteration359_c8_crosskv.md)；定位：结果、合同、限制与失败记录

### 静态 context 缓存零误差，完整周期仍略慢

- ID：helios-static-context-cache-negative
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：history patch、RoPE、text 缓存每 rollout 5 fills/22 reuses；零误差但 138.176083→138.243935 秒，未晋级。
- 经验：完整 key 和命中证明正确工作；是否值得保留还要看维护与节省的净成本，不能只展示 cache hit。
- 范围：GAN-on、完整 pyramid context key、同权重 rollout。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：cache_net_cost
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration366_c13_static_context.md](https://github.com/sliortega295-ops/DMD-System/blob/2d70afbbc3f28d828304b4f08cd6440c4e3e7a71/docs/experiments/gan_cuda_baseline_opt_20260829/iteration366_c13_static_context.md)；定位：结果、合同、限制与失败记录

### ResearchBoost：wrapper 重建容器会让 host metadata 更新丢失

- ID：rb-fsdp-container-host-state
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：其 FSDP card 提醒 kwargs 中 dict/list 可能经 wrapper 重建，内部写入的 metadata 不一定回到调用者的容器。
- 经验：以实际 wrapper 的 alias/owner 行为验证 host cache；换成持有状态的对象也需要测试，不能假定所有版本相同。
- 范围：外部 FSDP 容器语义经验，限定具体 wrapper 版本。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-fsdp-multi-forward/SKILL.md)；定位：kwargs container reconstruction / host state
