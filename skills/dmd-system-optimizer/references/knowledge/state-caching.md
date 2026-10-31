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
- 历史上下文：{"workload": "W6", "ranks": 16}
- 正确性前提：{"cache_key_complete": true, "invalidation_validated": true, "cached_gradients_preserved": true}
- 收益条件：{"cache_reuse_sufficient": true, "saved_compute_exceeds_cache_cost": true}
- 失败分类：net_slowdown_cause_not_isolated
- 重试信号：cache 命中、搬运或建表成本相对原点有变化
- 最小验证：检查 exact inputs、失效、保留参数，再分开建表成本和完整周期
- 不应重复：零误差不构成重复同一慢 cache 方案的理由
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-w6-adaln-r41`
- 关联实现：[H3 冻结 AdaLN projection bank](implementations.md#h3-adaln-bank)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/backends/training/minimax_h3/adaln_cache.py::install_adaln_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/adaln_cache.py#L233-L268)
- 函数入口：[src/distillschedule/backends/training/minimax_h3/adaln_cache.py::AdaLNCacheController](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/adaln_cache.py#L51-L175)
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r41/conclusion.json#L8)；定位：观察中的数值文本锚点 L8；同时阅读同节合同、状态与限制。原定位：exact_cache_input_gate; control_C2_s; candidate_C2_s; numerical_status

### W1：RETAIN4 与 frozen_down 的条件边际不可相加

- ID：helios-w1-frozen-down-r1
- 状态：negative_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：20.028682893→20.169802765 秒，MEASURED_NO_C2_GAIN；范围标为 RETAIN4_VS_FROZEN_DOWN_ONLY。
- 经验：两个替代策略的比较不支持叠加它们的收益。
- 范围：r1 followup 对应源和该对照，不是所有 frozen_down 路径都无效。
- 历史上下文：{"workload": "W1"}
- 正确性前提：{"cache_key_complete": true, "invalidation_validated": true, "cached_gradients_preserved": true}
- 收益条件：{"cache_reuse_sufficient": true, "saved_compute_exceeds_cache_cost": true}
- 失败分类：net_slowdown_comparison_scope
- 重试信号：当前 frozen-down 对关键路径占比增加
- 最小验证：以同一 retain4 control 做隔离配对，验证冻结值和更新独立
- 不应重复：不把不同 retention substrate 的时间直接相减
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-w1-frozen-down-r1`
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json#L104)；定位：观察中的数值文本锚点 L104；同时阅读同节合同、状态与限制。原定位：followups.wan_frozen_down

### 缓存 cast 不一定值得其维护成本

- ID：wan-cast-cache-negative
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：Wan resident cast-cache 单配对 27.050073→27.166219 秒，未晋级。
- 经验：先证实重复 cast 的暴露成本与版本失效成本；缓存本身不是收益证据。
- 范围：该 Wan frozen cast-cache 实现及当时 substrate。
- 历史上下文：{"workload": "W1", "model_family": "Wan1.3B", "training_mode": "training", "global_batch": 16, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"cache_key_complete": true, "invalidation_validated": true, "cached_gradients_preserved": true, "grad_mode": "no_grad"}
- 收益条件：{"cache_reuse_sufficient": true, "saved_compute_exceeds_cache_cost": true}
- 失败分类：cache_net_regression
- 重试信号：重复 cast 暴露增加且版本失效成本更低
- 最小验证：先计 cast 次数和 key/版本失效，再测 no-grad 完整周期
- 不应重复：相同 shape 的单配对无收益不宜反复消耗预算
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case wan-cast-cache-negative`
- 关联实现：[Wan 冻结 LoRA 路径的 cast cache wrapper](implementations.md#wan-cast-cache)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/wan13b_lora_cast.py::install_generator_nograd_lora_cast_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_cast.py#L265-L427)
- 来源：[sliortega295-ops/DMD-System / WAN13B_RESIDENT_OPTIMIZATION_LADDER_20260824.md](https://github.com/sliortega295-ops/DMD-System/blob/9186aee022e7d8933ca84429bde7f54868f203de/docs/experiments/WAN13B_RESIDENT_OPTIMIZATION_LADDER_20260824.md#L47)；定位：观察中的数值文本锚点 L47；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### history self-attention K/V cache 依赖的 attention 模式不成立

- ID：helios-history-kv-contract-rejected
- 状态：correctness_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：live workload 为 restrict_self_attn=false；已有 history KV 实现要求 restricted self-attention，因此未启动全模型计时。
- 经验：缓存正确性由 attention 依赖决定；不能为了复用缓存而偷偷改变 mask/attention contract。
- 范围：Helios No-GAN 的该 history KV 路径；其他合法跨步缓存仍可评估。
- 历史上下文：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": false, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"cache_key_complete": true, "invalidation_validated": true, "cached_gradients_preserved": true, "restrict_self_attn": true}
- 收益条件：{"cache_reuse_sufficient": true, "saved_compute_exceeds_cache_cost": true}
- 失败分类：attention_dependency_mismatch
- 重试信号：已有适配 unrestricted attention 的正确实现或任务本来要求 restricted
- 最小验证：先对齐 attention mask/history 语义与 self-KV cache 合同
- 不应重复：不能为了命中已有实现擅自改 restrict_self_attn
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-history-kv-contract-rejected`
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/111e293dc79978132af91ab5515a096d941b94fe/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md#L1)；定位：报告范围 L1–L146；L1: Helios14B node-HSDP optimization closure (Iterations 368-382)；L3: Status and claim boundary；L27: New experiment ladder；L52: Effective methods；L54: Explicit pinned local FlashAttention-2；L70: Critic selective FA2 checkpoint-output retention；L87: Rejected and bounded-negative methods；L105: Post-winner Nsight result；原定位：结果、合同、限制与失败记录

### No-GAN cross-KV 命中 880 次仍几乎没有 E2E 收益

- ID：helios-crosskv-nogan-negligible
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：每 rank 200 fills/880 reuses 且零 mismatch，Full5 96.529020→96.497111 秒，只约 0.033%。
- 经验：高命中率不说明省下关键路径大头；保留机制正确性证据，避免将微小单次差值称为稳定加速。
- 范围：Helios No-GAN node-HSDP 的 per-rollout cross-attention cache。
- 历史上下文：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": false, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"cache_key_complete": true, "invalidation_validated": true, "cached_gradients_preserved": true, "grad_mode": "no_grad", "microbatch": 1, "weights_stable_during_reuse": true}
- 收益条件：{"cache_reuse_sufficient": true, "saved_compute_exceeds_cache_cost": true}
- 失败分类：insufficient_exposed_saved_compute
- 重试信号：目标 KV 投影占比或复用收益明显高于 No-GAN 旧点
- 最小验证：验证 no-grad B1、版本/key 与 uncached 输出，分开计填充/复用/整周期
- 不应重复：200 fills/880 reuses 不能单独证明加速
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-crosskv-nogan-negligible`
- 关联实现：[Helios 单 rollout cross-attention KV cache](implementations.md#helios-cross-kv)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/helios_cross_kv_cache.py::install_generator_nograd_cross_kv_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_cross_kv_cache.py#L101-L246)
- 函数入口：[src/distillschedule/runtime/helios_cross_kv_cache.py::generator_nograd_cross_kv_cache_scope](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_cross_kv_cache.py#L269-L473)
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/111e293dc79978132af91ab5515a096d941b94fe/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md#L1)；定位：报告范围 L1–L146；L1: Helios14B node-HSDP optimization closure (Iterations 368-382)；L3: Status and claim boundary；L27: New experiment ladder；L52: Effective methods；L54: Explicit pinned local FlashAttention-2；L70: Critic selective FA2 checkpoint-output retention；L87: Rejected and bounded-negative methods；L105: Post-winner Nsight result；原定位：结果、合同、限制与失败记录

### GAN-on 的 cross-KV 单配对约有 0.75% 改善

- ID：helios-crosskv-gan-positive
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：同样 200 fills/880 reuses，限定 no-grad B1 且 key 绑定 Generator 版本与 encoder storage/version；142.295342→141.235767 秒。
- 经验：同类 cache 的收益随调用图改变；额外 uncached-reference 校验运行很慢，不能拿它作正式性能分母。
- 范围：GAN-on C8 per-rollout cache，不能直接复用到不同 conditioning 或可微路径。
- 历史上下文：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"cache_key_complete": true, "invalidation_validated": true, "cached_gradients_preserved": true, "grad_mode": "no_grad", "microbatch": 1, "weights_stable_during_reuse": true}
- 收益条件：{"cache_reuse_sufficient": true, "saved_compute_exceeds_cache_cost": true}
- 失败分类：not_established
- 重试信号：当前 GAN 调用图仍支持稳定 no-grad B1 conditioning
- 最小验证：核对权重和 encoder storage/version 失效，独立做正确性与正式配对
- 不应重复：不能把很慢的 uncached-reference 校验路径作性能分母
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-crosskv-gan-positive`
- 关联实现：[Helios 单 rollout cross-attention KV cache](implementations.md#helios-cross-kv)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/helios_cross_kv_cache.py::install_generator_nograd_cross_kv_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_cross_kv_cache.py#L101-L246)
- 函数入口：[src/distillschedule/runtime/helios_cross_kv_cache.py::generator_nograd_cross_kv_cache_scope](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_cross_kv_cache.py#L269-L473)
- 来源：[sliortega295-ops/DMD-System / iteration359_c8_crosskv.md](https://github.com/sliortega295-ops/DMD-System/blob/7f7a5c28db628a22ad494d4533590dc71b3d9afa/docs/experiments/gan_cuda_baseline_opt_20260829/iteration359_c8_crosskv.md#L48)；定位：观察中的数值文本锚点 L48；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### 静态 context 缓存零误差，完整周期仍略慢

- ID：helios-static-context-cache-negative
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：history patch、RoPE、text 缓存每 rollout 5 fills/22 reuses；零误差但 138.176083→138.243935 秒，未晋级。
- 经验：完整 key 和命中证明正确工作；是否值得保留还要看维护与节省的净成本，不能只展示 cache hit。
- 范围：GAN-on、完整 pyramid context key、同权重 rollout。
- 历史上下文：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"cache_key_complete": true, "invalidation_validated": true, "cached_gradients_preserved": true, "grad_mode": "no_grad"}
- 收益条件：{"cache_reuse_sufficient": true, "saved_compute_exceeds_cache_cost": true}
- 失败分类：cache_net_cost
- 重试信号：静态 patch/RoPE/text 计算占比或缓存开销发生变化
- 最小验证：先测 5 fills/22 reuses 对应计算净节省，再比完整周期
- 不应重复：零误差与 cache 命中不构成收益证明
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-static-context-cache-negative`
- 关联实现：[Helios rollout 静态 context cache](implementations.md#helios-static-context)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/helios_rollout_context_cache.py::install_generator_nograd_static_context_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_rollout_context_cache.py#L153-L404)
- 函数入口：[src/distillschedule/runtime/helios_rollout_context_cache.py::generator_nograd_static_context_cache_scope](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_rollout_context_cache.py#L408-L596)
- 来源：[sliortega295-ops/DMD-System / iteration366_c13_static_context.md](https://github.com/sliortega295-ops/DMD-System/blob/2d70afbbc3f28d828304b4f08cd6440c4e3e7a71/docs/experiments/gan_cuda_baseline_opt_20260829/iteration366_c13_static_context.md#L18)；定位：观察中的数值文本锚点 L18；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### ResearchBoost：wrapper 重建容器会让 host metadata 更新丢失

- ID：rb-fsdp-container-host-state
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：其 FSDP card 提醒 kwargs 中 dict/list 可能经 wrapper 重建，内部写入的 metadata 不一定回到调用者的容器。
- 经验：以实际 wrapper 的 alias/owner 行为验证 host cache；换成持有状态的对象也需要测试，不能假定所有版本相同。
- 范围：外部 FSDP 容器语义经验，限定具体 wrapper 版本。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"cache_key_complete": true, "invalidation_validated": true, "cached_gradients_preserved": true}
- 收益条件：{"cache_reuse_sufficient": true, "saved_compute_exceeds_cache_cost": true}
- 失败分类：not_established
- 重试信号：当前 wrapper 会重建 dict/list 等容器
- 最小验证：最小 FSDP 调用比较容器 identity 与 metadata 写回
- 不应重复：不假设函数内修改 kwargs 一定反馈到调用者
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case rb-fsdp-container-host-state`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-fsdp-multi-forward/SKILL.md#L1)；定位：报告范围 L1–L88；L6: FSDP: several forwards in one iteration；L24: 1. All-gather once per iteration；L33: 2. Cast frozen models to the compute dtype before wrapping in FSDP；L45: 3. Keep the frozen encoder's parameters resident；L54: 4. Cross-attention K/V cache；L63: 5. Remove host syncs；L72: 6. Spend the freed memory on speed；L83: What ships with it；原定位：kwargs container reconstruction / host state
