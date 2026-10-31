# 实现与代码定位

此表来自 implementation-map.json。所有链接固定到源码提交与函数行号；`locate --repo` 额外核对文件 hash 和 AST 符号。相关当前代码不等于历史计时所用实现。

测试入口仅说明能在哪里检查覆盖；未列出测试时明确记录缺口。此轮没有运行这里列出的 GPU/训练验收，helper 的 CPU 检查也不能替代它们。

<a id="fsdp-phase-retention"></a>

## FSDP2 完整单元的受控参数驻留

- ID：`fsdp-phase-retention`；用途：runtime_component
- 范围：现有 FSDP2 primitive，要求全 rank 同意物化顺序；不保证旧 r36/任何新模型容量。
- 正确性前提：{"weights_stable_during_reuse": true, "whole_checkpoint_units": true, "state_release_validated": true}
- 收益条件：{"repeated_forwards": true, "exposed_gather": true, "memory_headroom": true}
- 接入：通过 fsdp2_variable_forward_phase(enabled=..., with_grad=..., expected_units=...) 包住合法 phase；有梯度时绑定真实 checkpoint block 类型，保留原 engine 的 backward/step。
- 回退：新进程使用 enabled=False 的原 control；已进入的 scope 在异常或 step 前必须完成 release，不能把该 engine view 带出 scope。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：所列测试不构成当前 workload 的实机完整周期/组合/恢复验收；这些检查仍需按改动风险选择。
- 入口：[src/distillschedule/runtime/fsdp2_variable_forward.py::fsdp2_variable_forward_phase](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_variable_forward.py#L98-L142)
- 接入代码：[src/distillschedule/runtime/fsdp2_variable_forward.py::_PhaseEngine.step](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_variable_forward.py#L53-L62)
- 接入代码：[src/distillschedule/runtime/fsdp2_variable_forward.py::_PhaseEngine.release](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_variable_forward.py#L73-L94)
- 测试：[tests/test_runtime_fsdp2_variable_forward.py::test_materializes_before_calls_releases_before_original_step_and_restores_policy](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_runtime_fsdp2_variable_forward.py#L68-L85)
- 测试：[tests/test_runtime_fsdp2_variable_forward.py::test_failure_restores_every_unit_without_committing](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_runtime_fsdp2_variable_forward.py#L102-L119)
- 核对命令：`dmd_knowledge.py locate --implementation fsdp-phase-retention --repo /project/DMD-System --show-code`

<a id="helios-rollout-residency"></a>

## Helios 单次 no-grad rollout 的参数驻留

- ID：`helios-rollout-residency`；用途：runtime_component
- 范围：固定 Helios FSDP 单元 census；外部 ResearchBoost 仅是同机制建议。
- 正确性前提：{"grad_mode": "no_grad", "weights_stable_during_reuse": true, "state_release_validated": true}
- 收益条件：{"repeated_forwards": true, "exposed_gather": true, "memory_headroom": true}
- 接入：核对 policy.generator_nograd_persistent_unshard 与 flush_allocator_cache；只包 no-grad rollout，进入下一个更新阶段前退出 scope。
- 回退：用新 control 进程关闭 generator_nograd_persistent_unshard；不得仅删除 flush 后沿用旧容量验收。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：所列测试不构成当前 workload 的实机完整周期/组合/恢复验收；这些检查仍需按改动风险选择。
- 入口：[src/distillschedule/runtime/fsdp2_nograd_persistence.py::generator_nograd_persistent_unshard_scope](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_nograd_persistence.py#L43-L162)
- 接入代码：[src/distillschedule/runtime/fsdp2_p0.py::StaticFSDP2P0Policy](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_p0.py#L47-L231)
- 测试：[tests/test_training_helios_gan_native_history.py::test_original_detached_persistence_preserves_all_rank_local_calls_without_nested_phase](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_training_helios_gan_native_history.py#L466-L494)
- 核对命令：`dmd_knowledge.py locate --implementation helios-rollout-residency --repo /project/DMD-System --show-code`

<a id="helios-cross-kv"></a>

## Helios 单 rollout cross-attention KV cache

- ID：`helios-cross-kv`；用途：runtime_component
- 范围：只缓存 pre-norm encoder KV；不能替代 history self-attention cache，也不能进入可微调用。
- 正确性前提：{"grad_mode": "no_grad", "microbatch": 1, "cache_key_complete": true, "weights_stable_during_reuse": true, "invalidation_validated": true, "upstream_source_hash_matches": true}
- 收益条件：{"cache_reuse_sufficient": true, "saved_compute_exceeds_cache_cost": true}
- 接入：先 source-pin 安装 generator wrapper，再在 generator_nograd_cross_kv_cache_scope 内传 source_encoder_hidden_states；policy.generator_nograd_cross_kv_cache 仅支持 B1。
- 回退：在干净进程关闭 generator_nograd_cross_kv_cache，使用原始投影路径；已安装 wrapper 的进程不能以再次调用 disabled 安装器代替真正回退。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：未在该源码快照中定位到专门验证此组件前后向/完整周期的测试；不得把配置检查当作该项验收。
- 入口：[src/distillschedule/runtime/helios_cross_kv_cache.py::install_generator_nograd_cross_kv_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_cross_kv_cache.py#L101-L246)
- 入口：[src/distillschedule/runtime/helios_cross_kv_cache.py::generator_nograd_cross_kv_cache_scope](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_cross_kv_cache.py#L269-L473)
- 接入代码：[src/distillschedule/runtime/fsdp2_p0.py::StaticFSDP2P0Policy](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_p0.py#L47-L231)
- 接入代码：[scripts/distillschedule/probe_static_fsdp2_p0_world16.py::_build_static_model](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/scripts/distillschedule/probe_static_fsdp2_p0_world16.py#L314-L317)
- 接入代码：[runtime/static_fsdp2_p0_helios.py::run_generator_trace_rollout_batch](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/static_fsdp2_p0_helios.py#L146-L151)
- 核对命令：`dmd_knowledge.py locate --implementation helios-cross-kv --repo /project/DMD-System --show-code`

<a id="helios-static-context"></a>

## Helios rollout 静态 context cache

- ID：`helios-static-context`；用途：runtime_component
- 范围：绑定源码和 tensor identity/version/layout；零误差不意味着有性能收益。
- 正确性前提：{"grad_mode": "no_grad", "cache_key_complete": true, "invalidation_validated": true, "upstream_source_hash_matches": true}
- 收益条件：{"cache_reuse_sufficient": true, "saved_compute_exceeds_cache_cost": true}
- 接入：核对 generator_nograd_static_context_cache；仅复用 history patch/RoPE/text 等静态项，当前 latent 和 timestep 仍进入原始路径。
- 回退：用原 control 在新进程关闭 static context cache；退出 scope 后检查所有 payload 释放。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：未在该源码快照中定位到专门验证此组件前后向/完整周期的测试；不得把配置检查当作该项验收。
- 入口：[src/distillschedule/runtime/helios_rollout_context_cache.py::install_generator_nograd_static_context_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_rollout_context_cache.py#L153-L404)
- 入口：[src/distillschedule/runtime/helios_rollout_context_cache.py::generator_nograd_static_context_cache_scope](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_rollout_context_cache.py#L408-L596)
- 接入代码：[src/distillschedule/runtime/fsdp2_p0.py::StaticFSDP2P0Policy](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_p0.py#L47-L231)
- 接入代码：[scripts/distillschedule/probe_static_fsdp2_p0_world16.py::_build_static_model](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/scripts/distillschedule/probe_static_fsdp2_p0_world16.py#L318-L322)
- 接入代码：[runtime/static_fsdp2_p0_helios.py::run_generator_trace_rollout_batch](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/static_fsdp2_p0_helios.py#L151-L159)
- 核对命令：`dmd_knowledge.py locate --implementation helios-static-context --repo /project/DMD-System --show-code`

<a id="wan-lora-epilogue"></a>

## Wan no-grad LoRA 融合及实际安装 wrapper

- ID：`wan-lora-epilogue`；用途：runtime_component
- 范围：BF16 base + FP32 LoRA update；保留两次 FP32 round-to-nearest 再转 BF16。
- 正确性前提：{"grad_mode": "no_grad", "base_dtype": "bfloat16", "lora_update_dtype": "float32", "contiguous_inputs": true, "numerical_contract_preserved": true}
- 收益条件：{"epilogue_on_critical_path": true}
- 接入：经 install_generator_nograd_lora_fused_epilogue 的角色、源码和 no-grad guard 接入；不要把底层 Triton 函数直接塞进可微 Linear。
- 回退：恢复原 PEFT forward 的干净进程，关闭融合开关；底层函数本身没有 autograd fallback。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：所列测试将 fused kernel 替换成 CPU eager mock，验证 wrapper/cache 组合与 grad passthrough；不覆盖真实 CUDA kernel、当前 workload 或完整训练周期。
- 独立备选：[Wan 可微 LoRA epilogue 的独立备选](#wan-lora-grad-epilogue)；需要梯度时检查这个独立的 autograd 备选；它不继承 no-grad 案例的历史收益或验收。
- 入口：[src/distillschedule/runtime/wan13b_lora_epilogue.py::fused_lora_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_epilogue.py#L55-L92)
- 接入代码：[src/distillschedule/runtime/wan13b_lora_cast.py::install_generator_nograd_lora_fused_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_cast.py#L430-L610)
- 接入代码：[scripts/distillschedule/probe_wan13b_fair_world16.py::_build_model_fair](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/scripts/distillschedule/probe_wan13b_fair_world16.py#L257-L260)
- 测试：[tests/test_refactor_wan13b_cast_composition.py::test_fused_cast_composition_output_gradient_input_version_and_scope](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_wan13b_cast_composition.py#L20-L84)
- 核对命令：`dmd_knowledge.py locate --implementation wan-lora-epilogue --repo /project/DMD-System --show-code`

<a id="wan-lora-grad-epilogue"></a>

## Wan 可微 LoRA epilogue 的独立备选

- ID：`wan-lora-grad-epilogue`；用途：runtime_component
- 范围：可微前向/checkpoint replay 的独立候选，底层 BF16/FP32 内核与 no-grad 路径共享但导数由单独 autograd 实现；没有从 no-grad 历史结果继承收益。
- 正确性前提：{"grad_mode": "grad_enabled", "base_dtype": "bfloat16", "lora_update_dtype": "float32", "single_vanilla_active_adapter": true, "upstream_source_hash_matches": true, "checkpoint_replay_validated": true}
- 收益条件：{"epilogue_on_critical_path": true}
- 接入：通过 DISTILLSCHEDULE_WAN13B_GRAD_LORA_FUSED_EPILOGUE 显式 opt-in；分别读取 autograd forward/backward 与 installer 的 PEFT source/adapter guard，检查实际 receipt。
- 回退：新进程设置该变量为 0 并恢复原 PEFT 路径；不要从失败后残留的 wrapper/optimizer 状态继续作为 control。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：CPU eager forward 注入测试覆盖 native input gradients，receipt 测试检查命中声明；真实 Triton CUDA 路径、checkpoint replay 与完整周期仍需当前验证。
- 入口：[src/distillschedule/runtime/wan13b_lora_grad_epilogue.py::fused_lora_epilogue_with_grad](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_grad_epilogue.py#L79-L85)
- 入口：[src/distillschedule/runtime/wan13b_lora_grad_epilogue.py::make_epilogue_autograd_function](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_grad_epilogue.py#L51-L76)
- 接入代码：[src/distillschedule/runtime/wan13b_lora_grad_epilogue.py::install_grad_lora_fused_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_grad_epilogue.py#L88-L154)
- 接入代码：[scripts/distillschedule/probe_wan13b_fair_world16.py::_build_model_fair](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/scripts/distillschedule/probe_wan13b_fair_world16.py#L266-L266)
- 测试：[tests/test_refactor_wan13b_grad_epilogue.py::test_real_cpu_autograd_matches_native_output_and_both_gradients](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_wan13b_grad_epilogue.py#L42-L60)
- 测试：[tests/test_refactor_wan13b_grad_epilogue.py::test_runtime_receipts_require_actual_execution](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_wan13b_grad_epilogue.py#L82-L90)
- 核对命令：`dmd_knowledge.py locate --implementation wan-lora-grad-epilogue --repo /project/DMD-System --show-code`

<a id="wan-cast-cache"></a>

## Wan 冻结 LoRA 路径的 cast cache wrapper

- ID：`wan-cast-cache`；用途：runtime_component
- 范围：当前相关 wrapper，不证明历史 cache 的时间可转移；外部 dtype 方法范围更广。
- 正确性前提：{"grad_mode": "no_grad", "weights_stable_during_reuse": true, "invalidation_validated": true}
- 收益条件：{"cache_reuse_sufficient": true, "saved_compute_exceeds_cache_cost": true}
- 接入：读取 wrapper 的 cast 对象与源 hash gate，只处理被允许的冻结计算路径，不转换训练 master/optimizer state。
- 回退：新进程禁用 cast cache 并恢复源 forward；重新核对实际 cast 次数和参数 dtype。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：所列测试将 fused kernel 替换成 CPU eager mock，验证 wrapper/cache 组合与 grad passthrough；不覆盖真实 CUDA kernel、当前 workload 或完整训练周期。
- 入口：[src/distillschedule/runtime/wan13b_lora_cast.py::install_generator_nograd_lora_cast_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_cast.py#L265-L427)
- 接入代码：[src/distillschedule/runtime/wan13b_lora_cast.py::install_generator_nograd_lora_cast_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_cast.py#L265-L427)
- 测试：[tests/test_refactor_wan13b_cast_composition.py::test_fused_cast_composition_output_gradient_input_version_and_scope](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_wan13b_cast_composition.py#L20-L84)
- 核对命令：`dmd_knowledge.py locate --implementation wan-cast-cache --repo /project/DMD-System --show-code`

<a id="helios-modulation-chunks"></a>

## Helios modulation chunk 参数入口

- ID：`helios-modulation-chunks`；用途：runtime_component
- 范围：只改变 token 独立操作的分块 launch；不是修改 attention/MLP 逻辑形状。
- 正确性前提：{"model_family": "Helios14B", "helios_block_count": 40, "role": ["generator", "critic"]}
- 收益条件：{"pointwise_launch_cost_exposed": true}
- 接入：chunk_tokens 仅接受 512/1024/2048，逐个核对 40 个 HeliosTransformerBlock；不要将该入口用于 Wan 的同名概念。
- 回退：从干净 control 进程设置 512；实现拒绝已经被另一候选修改过的 chunk 策略。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：未在该源码快照中定位到专门验证此组件前后向/完整周期的测试；不得把配置检查当作该项验收。
- 入口：[src/distillschedule/runtime/helios_modulation_chunks.py::configure_helios_modulation_chunk_tokens](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_modulation_chunks.py#L19-L85)
- 接入代码：[src/distillschedule/runtime/fsdp2_p0.py::StaticFSDP2P0Policy](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_p0.py#L47-L231)
- 核对命令：`dmd_knowledge.py locate --implementation helios-modulation-chunks --repo /project/DMD-System --show-code`

<a id="helios-modnorm"></a>

## Helios modulation norm 的安装与融合实现

- ID：`helios-modnorm`；用途：runtime_component
- 范围：当前源码中存在多条互斥融合路径；按所选分支验证。
- 正确性前提：{"model_family": "Helios14B", "upstream_source_hash_matches": true, "gradient_path_supported": true, "numerical_contract_preserved": true}
- 收益条件：{"norm_on_critical_path": true}
- 接入：先读 installer 的 distinct forward/backward 选项和 source guards；仅调用 epilogue kernel 不等于安装完整 LayerNorm/backward 融合。
- 回退：在新进程恢复原 source norm 与 backward；按输出及梯度分别验证，不能只用某个 canary 代替整体。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：未在该源码快照中定位到专门验证此组件前后向/完整周期的测试；不得把配置检查当作该项验收。
- 入口：[src/distillschedule/runtime/helios_modulated_norm_fusion.py::install_exact_modulated_norm_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_modulated_norm_fusion.py#L24-L774)
- 入口：[src/distillschedule/runtime/helios_modulated_norm_kernel.py::fused_modulated_norm_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_modulated_norm_kernel.py#L207-L263)
- 接入代码：[src/distillschedule/runtime/helios_modulated_norm_fusion.py::install_exact_modulated_norm_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_modulated_norm_fusion.py#L24-L774)
- 核对命令：`dmd_knowledge.py locate --implementation helios-modnorm --repo /project/DMD-System --show-code`

<a id="helios-crossnorm-canary"></a>

## Helios LayerNorm/modulation canary

- ID：`helios-crossnorm-canary`；用途：diagnostic_only
- 范围：诊断参考；不把当前模块与历史 cross-norm 实验视为同一个实现。
- 正确性前提：{"numerical_contract_preserved": true, "dtype_layout_supported": true}
- 收益条件：{"norm_on_critical_path": true}
- 接入：此入口是相关 LayerNorm canary，不是历史 cross-norm 候选的已验证直接替换；先核对 affine、dtype、reduction 与 backward。
- 回退：保持原 norm 默认路径；独立 canary 未通过不能接入全模型。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：未在该源码快照中定位到专门验证此组件前后向/完整周期的测试；不得把配置检查当作该项验收。
- 入口：[src/distillschedule/runtime/helios_fused_layernorm_kernel.py::fused_bf16_layernorm_modulation](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_fused_layernorm_kernel.py#L135-L222)
- 接入代码：[src/distillschedule/runtime/helios_modulated_norm_fusion.py::install_exact_modulated_norm_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_modulated_norm_fusion.py#L24-L774)
- 核对命令：`dmd_knowledge.py locate --implementation helios-crossnorm-canary --repo /project/DMD-System --show-code`

<a id="livewan-generator-modulation"></a>

## LiveWan Generator modulation canary 与 runtime scope

- ID：`livewan-generator-modulation`；用途：runtime_component
- 范围：C32 指定 Generator canary/运行域；没有跨角色通用支持。
- 正确性前提：{"grad_mode": "no_grad", "microbatch": 1, "width": 1536, "scale_shift_dtype": "float32", "role": "generator"}
- 收益条件：{"pointwise_launch_cost_exposed": true}
- 接入：明确通过 canary/runtime scope 启用；检查 Generator 的 FP32 scale/shift，不能套到 Score 的不同 dtype。
- 回退：退出 scope 并使用 default-off 安装路径；runtime receipt 不满足该方案时回到原实现。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：所列测试不构成当前 workload 的实机完整周期/组合/恢复验收；这些检查仍需按改动风险选择。
- 入口：[src/distillschedule/runtime/livewan_generator_modulation.py::build](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/livewan_generator_modulation.py#L18-L65)
- 入口：[src/distillschedule/runtime/livewan_generator_modulation.py::model_runtime_scope](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/livewan_generator_modulation.py#L140-L146)
- 接入代码：[src/distillschedule/runtime/livewan_generator_modulation.py::install](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/livewan_generator_modulation.py#L131-L136)
- 测试：[tests/test_refactor_livewan_generator_modulation.py::test_no_grad_required_before_import_or_device_access](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_livewan_generator_modulation.py#L16-L21)
- 测试：[tests/test_refactor_livewan_generator_modulation.py::test_only_installed_on_explicit_validated_g_scope](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_livewan_generator_modulation.py#L35-L39)
- 核对命令：`dmd_knowledge.py locate --implementation livewan-generator-modulation --repo /project/DMD-System --show-code`

<a id="helios-history-fusion"></a>

## Helios history/current add+cat 融合

- ID：`helios-history-fusion`；用途：runtime_component
- 范围：与 C25 同机制的 source-pinned helper，当前组合仍需完整周期验证。
- 正确性前提：{"upstream_source_hash_matches": true, "gradient_path_supported": true, "numerical_contract_preserved": true}
- 收益条件：{"pointwise_launch_cost_exposed": true}
- 接入：仅在源 hash 和 rewrite 模板完全匹配时安装；核对 actual calls/fallback 与训练梯度。
- 回退：新进程不安装该 rewrite，保留原 forward；不能在未知源版本上模糊替换文本。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：未在该源码快照中定位到专门验证此组件前后向/完整周期的测试；不得把配置检查当作该项验收。
- 入口：[src/distillschedule/runtime/helios_history_current_fusion.py::install_exact_history_current_fusion](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_history_current_fusion.py#L77-L179)
- 接入代码：[src/distillschedule/runtime/helios_history_current_fusion.py::_rewrite_forward_source](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_history_current_fusion.py#L43-L74)
- 核对命令：`dmd_knowledge.py locate --implementation helios-history-fusion --repo /project/DMD-System --show-code`

<a id="helios-fsdp-layout"></a>

## Helios world16 FSDP/node-HSDP 配置

- ID：`helios-fsdp-layout`；用途：runtime_component
- 范围：Helios 固定 profile 的并行 helper，不是任意节点数的通用 HSDP 接口。
- 正确性前提：{"logical_batch_preserved": true, "gradient_reduction_preserved": true, "mesh_contract_validated": true}
- 收益条件：{"topology_calibrated": true, "memory_headroom": true}
- 接入：读取 mesh policy 的 replicate/shard 约束、完整模型构造峰值和原训练时钟；GAS/逻辑 batch 不随 mesh 改动。
- 回退：新进程恢复 world16-fsdp control；并行布局不能在已有 optimizer state 上随意就地切换。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：未在该源码快照中定位到专门验证此组件前后向/完整周期的测试；不得把配置检查当作该项验收。
- 入口：[src/distillschedule/runtime/fsdp2_p0.py::create_device_mesh](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_p0.py#L273-L288)
- 入口：[src/distillschedule/runtime/fsdp2_p0.py::apply_blockwise_fsdp2_policy](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_p0.py#L291-L398)
- 接入代码：[src/distillschedule/runtime/fsdp2_p0.py::StaticFSDP2P0Policy](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_p0.py#L47-L231)
- 核对命令：`dmd_knowledge.py locate --implementation helios-fsdp-layout --repo /project/DMD-System --show-code`

<a id="optimizer-dispatch-warmup"></a>

## FSDP2 optimizer dispatch 预热与状态恢复

- ID：`optimizer-dispatch-warmup`；用途：runtime_component
- 范围：不能将真实训练更新挪出计时窗口；原 warmup 对 state 的恢复逻辑是必读入口。
- 正确性前提：{"optimizer_state_restored": true, "weights_restored": true, "rng_restored": true, "update_clocks_preserved": true}
- 收益条件：{"cold_dispatch_is_relevant": true}
- 接入：沿 build_fsdp2_engine 的原 warmup 参数接入；比较预热前后全部参数/梯度/Adam/scheduler/RNG，单列 setup 时间。
- 回退：使用关闭 warmup 的新 control 进程；恢复不完整时丢弃该初始化状态。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：未在该源码快照中定位到专门验证此组件前后向/完整周期的测试；不得把配置检查当作该项验收。
- 入口：[src/distillschedule/runtime/fsdp2_static.py::_optimizer_dispatch_warmup](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L545-L775)
- 接入代码：[src/distillschedule/runtime/fsdp2_static.py::build_fsdp2_engine](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L779-L828)
- 核对命令：`dmd_knowledge.py locate --implementation optimizer-dispatch-warmup --repo /project/DMD-System --show-code`

<a id="helios-selective-checkpoint"></a>

## Helios FA2 selective checkpoint 保留

- ID：`helios-selective-checkpoint`；用途：runtime_component
- 范围：FA2 算子级保留策略可供最小适配；外层 FSDP2/P0 训练接入固定 world16/B16 和40个block。不是无条件保存全部 attention；真实 backend 和最坏更新阶段容量都需验收。
- 正确性前提：{"attention_backend": "FA2", "checkpoint_replay_pure": true, "gradient_path_supported": true}
- 收益条件：{"recompute_cost_exposed": true, "memory_headroom": true}
- 接入：读取 selective Flash checkpoint policy；先证明真实 FA2 op 被命中，再选择保留密度。保持原 checkpoint 的 RNG 与反向路径。 区分策略函数与外层训练接入：后者的 world16/B16 不能直接用于8卡；适配策略须保持目标逻辑工作，重新验证真实算子、block身份及梯度路径。
- 回退：恢复原 checkpoint dispatcher/原 stride；不删除 optimizer/update 的语义步骤。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：未在该源码快照中定位到专门验证此组件前后向/完整周期的测试；不得把配置检查当作该项验收。
- 入口：[src/distillschedule/runtime/fsdp2_static.py::_selective_flash_checkpoint_contexts](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L84-L111)
- 入口：[src/distillschedule/runtime/fsdp2_static.py::configure_fsdp2_checkpointing](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_static.py#L171-L290)
- 接入代码：[src/distillschedule/runtime/fsdp2_p0.py::StaticFSDP2P0Policy](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/fsdp2_p0.py#L47-L231)
- 核对命令：`dmd_knowledge.py locate --implementation helios-selective-checkpoint --repo /project/DMD-System --show-code`

<a id="block-activation-retention"></a>

## 选择完整 block 保留 autograd 激活

- ID：`block-activation-retention`；用途：runtime_component
- 范围：当前相关 block retention，不是历史 tail8 patch 的身份保证。
- 正确性前提：{"checkpoint_replay_pure": true, "gradient_path_supported": true, "whole_checkpoint_units": true}
- 收益条件：{"recompute_cost_exposed": true, "memory_headroom": true}
- 接入：按实际模块选择 opt-in block retention，未选择模块保留原 dispatcher；先校准候选内存成本。
- 回退：不安装/选择空集合，恢复原 checkpoint；不能从 tail8 OOM 推导当前允许的 block 数。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：所列测试不构成当前 workload 的实机完整周期/组合/恢复验收；这些检查仍需按改动风险选择。
- 入口：[src/distillschedule/runtime/checkpoint_block_retention.py::install](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/checkpoint_block_retention.py#L11-L62)
- 接入代码：[src/distillschedule/runtime/checkpoint_block_retention.py::install](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/checkpoint_block_retention.py#L11-L62)
- 测试：[tests/test_checkpoint_block_retention.py::test_original_forward_backward_adam_rng_and_selected_recompute](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_checkpoint_block_retention.py#L33-L47)
- 测试：[tests/test_checkpoint_block_retention.py::test_uncalibrated_retention_is_rejected_by_calibration_before_cuda](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_checkpoint_block_retention.py#L87-L97)
- 核对命令：`dmd_knowledge.py locate --implementation block-activation-retention --repo /project/DMD-System --show-code`

<a id="h3-boundary-offload"></a>

## H3 checkpoint 边界 CPU offload

- ID：`h3-boundary-offload`；用途：runtime_component
- 范围：CPU 边界搬运与 full model GPU residency 可同时存在；配置测试不证明完整状态等价。
- 正确性前提：{"checkpoint_replay_pure": true, "storage_aliases_preserved": true, "offload_completion_order_validated": true}
- 收益条件：{"host_memory_headroom": true, "device_capacity_bottleneck": true}
- 接入：通过 H3 checkpoint_boundary_cpu policy 选择精确边界；检查 registration、pack/unpack、实际传输字节和最晚 backward。
- 回退：保持原 checkpoint policy 的新进程；边界对象仍存活时不可拆掉 hook 或提前释放 CPU storage。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：所列测试检查配置字段确实生效；不是 boundary pack/unpack 的真实 GPU 数值、峰值与完整 backward 验收。
- 入口：[src/distillschedule/backends/training/minimax_h3/checkpoint_boundary_offload.py::install_checkpoint_boundary_cpu_offload](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/checkpoint_boundary_offload.py#L187-L336)
- 接入代码：[src/distillschedule/backends/training/minimax_h3/checkpoint_boundary_offload.py::CheckpointBoundaryOffloadRegistration](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/checkpoint_boundary_offload.py#L156-L184)
- 测试：[tests/test_refactor_h3_cache_workflow.py::test_selected_fields_cannot_be_ignored_by_actual_config](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_h3_cache_workflow.py#L84-L98)
- 核对命令：`dmd_knowledge.py locate --implementation h3-boundary-offload --repo /project/DMD-System --show-code`

<a id="h3-attention-replay"></a>

## H3 FA3 replay cache 与 staged 路径

- ID：`h3-attention-replay`；用途：runtime_component
- 范围：FA3 专用当前参考；W6 的 FA2 对照不能直接启用此模块。
- 正确性前提：{"attention_backend": "FA3", "checkpoint_replay_pure": true, "cache_key_complete": true, "offload_completion_order_validated": true}
- 收益条件：{"recompute_cost_exposed": true, "host_memory_headroom": true, "memory_headroom": true}
- 接入：核对 block_indices、storage 与 max_d2h_inflight；staged 路径进入 backward 前准备，并用 validate_cycle 检查实际执行。
- 回退：用原 control 新进程禁用 replay cache；CPU-stage 卡住时先用有限超时采集等待，不改名为 OOM 或默认重试。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：所列为 workflow/receipt/配置验证；没有覆盖 CPU-stage futex 停滞的根因、真实 CUDA kernel 数值或完整训练周期。
- 入口：[src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py::install_fa3_replay_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py#L880-L1131)
- 入口：[src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py::prepare_staged_cache_before_backward](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py#L854-L872)
- 接入代码：[src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py::parse_storage](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py#L819-L825)
- 接入代码：[src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py::validate_cycle](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py#L1134-L1224)
- 测试：[tests/test_refactor_h3_cache_workflow.py::test_receipt_selected_cache_execution_and_all_ranks_fail_closed](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_h3_cache_workflow.py#L147-L161)
- 测试：[tests/test_refactor_h3_cache_workflow.py::test_inactive_cuda_prefetch_is_not_an_invented_candidate](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_h3_cache_workflow.py#L70-L79)
- 核对命令：`dmd_knowledge.py locate --implementation h3-attention-replay --repo /project/DMD-System --show-code`

<a id="h3-adaln-bank"></a>

## H3 冻结 AdaLN projection bank

- ID：`h3-adaln-bank`；用途：runtime_component
- 范围：当前 H3 bank 参考；不把其文件 hash 当作 r41 性能结论的新验证。
- 正确性前提：{"adaln_parameters_frozen": true, "cache_key_complete": true, "invalidation_validated": true}
- 收益条件：{"cache_reuse_sufficient": true, "saved_compute_exceeds_cache_cost": true}
- 接入：从 install_adaln_cache 检查冻结参数断言与 cache scope；不删除注册参数、不合并训练权重。
- 回退：新进程保留原 AdaLN 模块，关闭该 cache/bank；比较完整 cycle，不能仅展示命中率。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：未在该源码快照中定位到专门验证此组件前后向/完整周期的测试；不得把配置检查当作该项验收。
- 入口：[src/distillschedule/backends/training/minimax_h3/adaln_cache.py::install_adaln_cache](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/adaln_cache.py#L233-L268)
- 入口：[src/distillschedule/backends/training/minimax_h3/adaln_cache.py::AdaLNCacheController](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/adaln_cache.py#L51-L175)
- 接入代码：[src/distillschedule/backends/training/minimax_h3/adaln_cache.py::adaln_bank](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/adaln_cache.py#L271-L273)
- 接入代码：[src/distillschedule/backends/training/minimax_h3/adaln_cache.py::adaln_controller](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/backends/training/minimax_h3/adaln_cache.py#L276-L278)
- 核对命令：`dmd_knowledge.py locate --implementation h3-adaln-bank --repo /project/DMD-System --show-code`

<a id="legacy-saved-tensor-offload"></a>

## 旧 Route-A 精确 saved-tensor offload

- ID：`legacy-saved-tensor-offload`；用途：historical_component
- 范围：保留的历史 Route-A 代码，依赖其运行器；不是当前 native trainer 的即插即用组件。
- 正确性前提：{"grad_mode": "grad_enabled", "storage_aliases_preserved": true, "offload_completion_order_validated": true}
- 收益条件：{"host_memory_headroom": true, "transfer_cost_measured": true}
- 接入：旧运行器入口：只在已确认的顶层 grad-enabled 调用安装 allowlist hook，区分 original/recompute/final scope。
- 回退：干净 control 中不安装 hook；所有 outstanding unpack/alias 生命周期结束后才可卸载。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：所列测试不构成当前 workload 的实机完整周期/组合/恢复验收；这些检查仍需按改动风险选择。
- 入口：[runtime/activation_offload.py::install_activation_offload_hooks](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/activation_offload.py#L362-L407)
- 入口：[runtime/activation_offload.py::AuditedSavedTensorOffloader](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/activation_offload.py#L61-L347)
- 接入代码：[runtime/activation_offload.py::ActivationOffloadRegistration](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/activation_offload.py#L351-L359)
- 测试：[tests/test_activation_offload.py::test_offload_signature_allowlist_is_exact](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_activation_offload.py#L140-L167)
- 测试：[tests/test_activation_offload.py::test_pinned_signature_allowlist_cuda_parity](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_activation_offload.py#L171-L203)
- 核对命令：`dmd_knowledge.py locate --implementation legacy-saved-tensor-offload --repo /project/DMD-System --show-code`

<a id="legacy-pinned-materialization"></a>

## 旧 Route-A pinned target 与有界 staging

- ID：`legacy-pinned-materialization`；用途：historical_component
- 范围：历史角色转换参考，不保证节点并发吞吐；冷准备和最终清理都需计入目标窗口。
- 正确性前提：{"storage_aliases_preserved": true, "offload_completion_order_validated": true, "registered_metadata_preserved": true}
- 收益条件：{"transfer_cost_measured": true, "host_memory_headroom": true}
- 接入：选择 persistent target 或 bounded staging 中的一条；分别核对 pin、DMA、等待、清理的实际生命周期，读对象/alias 限制。
- 回退：在干净运行中恢复原 module_to transition；先等待 pending DMA 和显式 drain cleanup，不能只去掉 join。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：所列测试不构成当前 workload 的实机完整周期/组合/恢复验收；这些检查仍需按改动风险选择。
- 入口：[runtime/bounded_materialization.py::pin_module_cpu_registered_state](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L189-L238)
- 入口：[runtime/bounded_materialization.py::materialize_module_from_persistent_pinned_cpu](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L242-L331)
- 入口：[runtime/bounded_materialization.py::materialize_module_from_cpu_with_bounded_pinned_staging](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L335-L480)
- 接入代码：[runtime/bounded_materialization.py::start_deferred_host_allocator_cleanup](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L37-L72)
- 接入代码：[runtime/bounded_materialization.py::drain_deferred_host_allocator_cleanups](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/bounded_materialization.py#L75-L93)
- 测试：[tests/test_bounded_materialization.py::test_bounded_pinned_materialization_is_value_and_metadata_exact](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_bounded_materialization.py#L45-L72)
- 测试：[tests/test_bounded_materialization.py::test_post_full5_cleanup_requires_explicit_start_then_drain](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_bounded_materialization.py#L176-L191)
- 核对命令：`dmd_knowledge.py locate --implementation legacy-pinned-materialization --repo /project/DMD-System --show-code`

<a id="legacy-frozen-streaming"></a>

## 旧 Route-A 不可变 frozen-block streaming

- ID：`legacy-frozen-streaming`；用途：historical_component
- 范围：旧 DeepSpeed/Route-A streaming 参考；Full5 容量和最小集合需要重新测量。
- 正确性前提：{"streamed_parameters_frozen": true, "optimizer_ownership_preserved": true, "storage_aliases_preserved": true}
- 收益条件：{"device_capacity_bottleneck": true, "host_memory_headroom": true}
- 接入：只选择 model.blocks 中 immutable 且不属于 optimizer 的参数；记录 Generator 和 Critic 各自 peak，不迁移 LoRA/Adam。
- 回退：使用原 placement 的新进程；不能把失败阶段不相关的更多 Generator block 当作 Critic 容量修复。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：未在该源码快照中定位到专门验证此组件前后向/完整周期的测试；不得把配置检查当作该项验收。
- 入口：[runtime/frozen_block_streaming.py::install_frozen_block_streamer](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/frozen_block_streaming.py#L1095-L1099)
- 接入代码：[runtime/frozen_block_streaming.py::selected_frozen_block_parameter_ids](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/frozen_block_streaming.py#L90-L138)
- 核对命令：`dmd_knowledge.py locate --implementation legacy-frozen-streaming --repo /project/DMD-System --show-code`

<a id="legacy-saved-tensor-census"></a>

## 保存张量的无策略改变 census

- ID：`legacy-saved-tensor-census`；用途：diagnostic_only
- 范围：诊断工具；带观测的耗时不能作正式加速。
- 正确性前提：{"census_preserves_tensors": true}
- 收益条件：{}
- 接入：用 pass-through census 标注 storage/alias/shape 与调用 scope；配合 runtime correlation，不能仅凭相同尺寸认定 memcpy 来源。
- 回退：在正式 unprofiled control 中不安装 census，保留原始训练输入与计时边界。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：未在该源码快照中定位到专门验证此组件前后向/完整周期的测试；不得把配置检查当作该项验收。
- 入口：[runtime/saved_tensor_census.py::install_saved_tensor_census_hooks](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/saved_tensor_census.py#L164-L193)
- 接入代码：[runtime/saved_tensor_census.py::PassThroughSavedTensorCensus](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/saved_tensor_census.py#L29-L143)
- 核对命令：`dmd_knowledge.py locate --implementation legacy-saved-tensor-census --repo /project/DMD-System --show-code`

<a id="legacy-zero3-leaf"></a>

## ZeRO-3 模块粒度与 parent-first leaf 选择

- ID：`legacy-zero3-leaf`；用途：historical_component
- 范围：历史 ZeRO-3 调参所需 inventory，不是任意阈值的通用最优值。
- 正确性前提：{"leaf_dtype_compatible": true, "collective_order_preserved": true}
- 收益条件：{"topology_calibrated": true}
- 接入：先枚举实际模块和 dtype，再显式选择真实 block；该 census 不会替用户开启 overlap_comm 或改 bucket。
- 回退：保留原 leaf/persistence/prefetch 设置；更改通信后重新测 buffers 和关键路径。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：所列测试不构成当前 workload 的实机完整周期/组合/恢复验收；这些检查仍需按改动风险选择。
- 入口：[runtime/zero3_module_granularity.py::census_zero3_module_granularity](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/zero3_module_granularity.py#L25-L86)
- 入口：[runtime/zero3_module_granularity.py::selected_preorder_leaf_records](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/zero3_module_granularity.py#L89-L111)
- 接入代码：[runtime/zero3_module_granularity.py::ModuleGranularityRecord](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/runtime/zero3_module_granularity.py#L12-L22)
- 测试：[tests/test_zero3_module_granularity.py::test_parent_first_selection_does_not_duplicate_descendants](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_zero3_module_granularity.py#L31-L38)
- 核对命令：`dmd_knowledge.py locate --implementation legacy-zero3-leaf --repo /project/DMD-System --show-code`

<a id="rolephase-transport"></a>

## 角色流水的 FIFO P2P 与转换合同

- ID：`rolephase-transport`；用途：runtime_component
- 范围：Gloo FIFO 测试只检查传输协议；不证明 GPU 重叠或全周期速度。
- 正确性前提：{"collective_order_preserved": true, "logical_batch_preserved": true, "state_release_validated": true}
- 收益条件：{"role_service_costs_measured": true, "transfer_cost_measured": true}
- 接入：将角色布局、header/version、FIFO 和 transition 合同一起接入；原证明只支持实际测过的布局/阶段。
- 回退：恢复顺序 control，并 flush 所有在途 P2P；不能比较不同物理 batch 来声明同工作量收益。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：Gloo/FIFO 与 transition 协议测试不证明 GPU P2P 数值、目标拓扑死锁安全或流水端到端收益。
- 入口：[src/distillschedule/runtime/gan_rolephase_transport.py::RolePhaseCriticP2PTransport](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/gan_rolephase_transport.py#L105-L258)
- 接入代码：[src/distillschedule/runtime/gan_rolephase_transition.py::RolePhaseTransitionContract](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/gan_rolephase_transition.py#L16-L50)
- 测试：[tests/test_refactor_gan_rolephase_transport_torch.py::test_two_rank_gloo_transport_is_fifo_and_identity_bound](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_gan_rolephase_transport_torch.py#L94-L98)
- 测试：[tests/test_refactor_gan_rolephase_transition.py::test_transition_contract_rejects_hidden_or_repeated_costs](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_gan_rolephase_transition.py#L23-L29)
- 核对命令：`dmd_knowledge.py locate --implementation rolephase-transport --repo /project/DMD-System --show-code`

<a id="profile-attribution"></a>

## Nsight SQLite 区间与 rank 覆盖分析

- ID：`profile-attribution`；用途：analysis_tool
- 范围：分析工具，不自动采集 profile，也不把非重叠搬运算成可消除关键路径。 当前 search/profile.py 固定2节点×8卡、rank=node*8+device；不能直接接受H100单节点8卡数据。
- 正确性前提：{"profile_source_identity_matches": true, "rank_coverage_known": true, "timer_scope_known": true, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 收益条件：{}
- 接入：输入实际 trace manifest，先通过 validate_profile，再读 union/overlap、rank 覆盖与源身份；对 Nsight Compute 的 occupancy 还需其专用指标。
- 回退：覆盖不足保留 partial/unknown，回到原 trace/未观测计时；不填补缺失 rank 或把空洞归因为 CPU。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：所列解析/区间测试不证明输入 trace 的完整性、真实因果关系或 profiler 无扰动。
- 入口：[src/distillschedule/search/profile.py::parse_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L140-L211)
- 入口：[src/distillschedule/search/profile.py::validate_profile](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L214-L232)
- 接入代码：[src/distillschedule/search/profile.py::feature_weights](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/profile.py#L235-L251)
- 测试：[tests/test_refactor_profile_guided_search.py::test_sqlite_per_device_union_pid_and_reconciliation](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_profile_guided_search.py#L213-L225)
- 测试：[tests/test_refactor_profile_guided_search.py::test_phase_partition_matches_full_scan_with_cross_boundary_copies](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_profile_guided_search.py#L246-L257)
- 核对命令：`dmd_knowledge.py locate --implementation profile-attribution --repo /project/DMD-System --show-code`

<a id="planner-calibration"></a>

## Planner 的已测成本及结果资格

- ID：`planner-calibration`；用途：analysis_tool
- 范围：解析与资格检查入口；source/config/shape/state policy 变化后成本必须重新核对。
- 正确性前提：{"cost_source_identity_matches": true, "logical_work_preserved": true, "state_policy_bound": true}
- 收益条件：{}
- 接入：使用真实 component contract/calibration 与 native receipt；未测成本保持 missing，比较当前 manual control 和候选。
- 回退：缺 calibration 或验收失败则保持原 control；不把 tie-break 结果报告成实测收益。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：所列合同/解析测试不证明成本在新 shape/storage/topology 下准确，也不证明实际计划更快。
- 入口：[src/distillschedule/search/cost_model.py::CostModel](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/cost_model.py#L77-L195)
- 入口：[src/distillschedule/search/measurement.py::normalize_native](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/measurement.py#L18-L98)
- 接入代码：[src/distillschedule/search/helios_action_costs.py::component_contract](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/helios_action_costs.py#L68-L91)
- 测试：[tests/test_refactor_profile_guided_search.py::test_native_measurement_fail_closed](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_profile_guided_search.py#L95-L100)
- 测试：[tests/test_refactor_h3_cache_workflow.py::test_shape_and_storage_costs_and_comparisons_are_not_reused](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_h3_cache_workflow.py#L164-L183)
- 核对命令：`dmd_knowledge.py locate --implementation planner-calibration --repo /project/DMD-System --show-code`

<a id="checkpoint-lifecycle"></a>

## 项目 checkpoint 保存与恢复入口

- ID：`checkpoint-lifecycle`；用途：runtime_component
- 范围：当前框架入口；不证明历史 LiveWan 恢复峰值已修复，外部保存失败也非本地复现。
- 正确性前提：{"parameter_mapping_preserved": true, "optimizer_mapping_preserved": true, "rng_restored": true}
- 收益条件：{"host_memory_headroom": true}
- 接入：沿目标 trainer 的 checkpoint 接口保留 canonical 参数/Adam/EMA/RNG/进度映射；_stage_state 是检查临时分配与重复状态的入口。
- 回退：保留旧可恢复 checkpoint；新布局/映射的保存恢复未通过前不得覆盖唯一 control checkpoint。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：未在该源码快照中定位到专门验证此组件前后向/完整周期的测试；不得把配置检查当作该项验收。
- 入口：[src/distillschedule/training/checkpoint.py::save_checkpoint](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/training/checkpoint.py#L319-L456)
- 入口：[src/distillschedule/training/checkpoint.py::restore_checkpoint](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/training/checkpoint.py#L541-L559)
- 接入代码：[src/distillschedule/training/checkpoint.py::_stage_state](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/training/checkpoint.py#L177-L222)
- 接入代码：[src/distillschedule/training/checkpoint.py::validate_rng](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/training/checkpoint.py#L266-L279)
- 核对命令：`dmd_knowledge.py locate --implementation checkpoint-lifecycle --repo /project/DMD-System --show-code`

<a id="wan-rollout-graph"></a>

## Wan Resident B1 no-grad CUDA Graph 参考

- ID：`wan-rollout-graph`；用途：runtime_component
- 范围：本地训练中的 no-grad 子路径参考；外部 SGLang 推理案例不验证这个模块。
- 正确性前提：{"grad_mode": "no_grad", "microbatch": 1, "parameters_resident": true, "stable_tensor_addresses": true, "dropout_zero": true}
- 收益条件：{"host_launch_cost_exposed": true, "memory_headroom": true, "capture_amortized": true}
- 接入：通过 DISTILLSCHEDULE_WAN13B_GENERATOR_ROLLOUT_CUDA_GRAPH 显式 opt-in，核对 requested/install/validate 及每个 capture key。
- 回退：在新进程将该变量置 0，恢复原调用；graph output clone 与 pool 必须覆盖完整生命周期。
- 验证范围：本次只核对固定源码、定义和测试位置，未重新执行所列组件测试。所列测试覆盖以正文为准；helper 的 CPU 测试不等于组件训练验收。
- 测试缺口：CPU dispatch protocol 和 receipt 测试不执行真实 CUDA Graph；capture/replay 数值、峰值与完整周期仍需验证。
- 入口：[src/distillschedule/runtime/wan13b_rollout_cuda_graph.py::RolloutCudaGraph](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_rollout_cuda_graph.py#L77-L244)
- 接入代码：[src/distillschedule/runtime/wan13b_rollout_cuda_graph.py::requested](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_rollout_cuda_graph.py#L18-L22)
- 接入代码：[src/distillschedule/runtime/wan13b_rollout_cuda_graph.py::validate](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_rollout_cuda_graph.py#L303-L311)
- 测试：[tests/test_refactor_wan13b_rollout_cuda_graph.py::test_dispatch_protocol_live_inputs_weights_and_output_lifetime_cpu](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_wan13b_rollout_cuda_graph.py#L52-L109)
- 测试：[tests/test_refactor_wan13b_rollout_cuda_graph.py::test_reference_cannot_be_missing_or_in_native_timer](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/tests/test_refactor_wan13b_rollout_cuda_graph.py#L135-L144)
- 核对命令：`dmd_knowledge.py locate --implementation wan-rollout-graph --repo /project/DMD-System --show-code`

<a id="w6-native-metadata-barrier"></a>

## W6 r46 原生 metadata barrier 恢复 packet

- ID：`w6-native-metadata-barrier`；用途：historical_component
- 范围：历史 W6 AR、同 FA3 原始 varlen 路径，两周期/16 ranks；非通用去同步优化。
- 正确性前提：{"workload": "W6", "attention_backend": "FA3", "metadata_semantics_preserved": true, "physical_gpus": 16}
- 收益条件：{"same_backend_comparison": true, "metadata_barrier_policy_measured": true}
- 接入：检查 --restore-native-barrier、cu_q.tolist()、350 次读取 census 与原始 backend；只在匹配 packet 内显式开启。
- 回退：新进程关闭 restore-native-barrier，使用同 FA3 control；保留原生注意力与输入。
- 验证范围：本轮核对固定源码和符号，未执行该历史 GPU/训练实验；测试链接只提供导航。
- 测试缺口：当前 workload 的梯度、组合、容量和恢复仍需验证。
- 入口：[artifacts/publications/20261019/manual_packets/state_retention/r46/scripts/native_r2/native_metadata_barrier.py::NativeMetadataBarrier](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/manual_packets/state_retention/r46/scripts/native_r2/native_metadata_barrier.py#L7-L45)
- 接入代码：[artifacts/publications/20261019/manual_packets/state_retention/r46/scripts/native_r2/native_metadata_barrier.py::install_on_session](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/manual_packets/state_retention/r46/scripts/native_r2/native_metadata_barrier.py#L48-L55)
- 接入代码：[artifacts/publications/20261019/manual_packets/state_retention/r46/scripts/native_r2/metadata_barrier_worker.py::main](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/manual_packets/state_retention/r46/scripts/native_r2/metadata_barrier_worker.py#L6-L48)
- 测试：[artifacts/publications/20261019/manual_packets/state_retention/r46/tests/test_metadata_barrier.py::test_actual_tolist_before_original_call_and_two_cycle_read_census](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/manual_packets/state_retention/r46/tests/test_metadata_barrier.py#L24-L43)
- 核对命令：`dmd_knowledge.py locate --implementation w6-native-metadata-barrier --repo /project/DMD-System --show-code`

<a id="w3-frozen-ffn-up-replay"></a>

## W3 r1 冻结 FFN-up checkpoint 结果复用

- ID：`w3-frozen-ffn-up-replay`；用途：historical_component
- 范围：155ffbb5 历史 GAN Critic blocks38/39、两周期/10次C更新；接入硬编码 A100/SM80、world16 与 local_rank=rank%8、Torch 2.10.0+cu128/CUDA 12.8；旧 addmm ABI，不支持任意层或解析二阶 backward。
- 正确性前提：{"workload": "W3", "grad_mode": "with_grad", "frozen_base_only": true, "replay_scope_validated": true, "gpu_family": "A100", "physical_gpus": 16, "gpus_per_node": 8, "torch_version": "2.10.0+cu128", "cuda_version": "12.8"}
- 收益条件：{"replay_reuses_saved_output": true, "saved_compute_exceeds_cache_cost": true, "memory_headroom": true}
- 接入：历史 NativeDownSAC 的 up_blocks 只接受 () 或 (38,39)；由 candidate_bundle.up_blocks 传入；核对 frozen-up guard、版本/graph FIFO 与释放。manual_runtime_scope 明确拒绝非 A100/SM80、非 world16 或不同 Torch/CUDA；H100/8卡只能提取机制后重新适配，不应删除 guard 冒充原样兼容。硬件/拓扑限制属于此历史接入，不能据此否定结果缓存机制在其他环境的可能收益。
- 回退：使用相同历史 control 的 up_blocks=()，保留已存在 down replay 路径与原训练语义。
- 验证范围：本轮核对固定源码和符号，未执行该历史 GPU/训练实验；测试链接只提供导航。
- 测试缺口：当前 workload 的梯度、组合、容量和恢复仍需验证。
- 入口：[src/distillschedule/runtime/helios_gan_manual/down_replay.py::NativeDownSAC.contexts](https://github.com/sliortega295-ops/DMD-System/blob/155ffbb585356a29b544573a391eb456b790a15a/src/distillschedule/runtime/helios_gan_manual/down_replay.py#L66-L115)
- 接入代码：[src/distillschedule/runtime/helios_gan_manual/down_replay.py::NativeDownSAC.__init__](https://github.com/sliortega295-ops/DMD-System/blob/155ffbb585356a29b544573a391eb456b790a15a/src/distillschedule/runtime/helios_gan_manual/down_replay.py#L36-L54)
- 接入代码：[src/distillschedule/runtime/helios_gan_manual/runtime.py::manual_runtime_scope](https://github.com/sliortega295-ops/DMD-System/blob/155ffbb585356a29b544573a391eb456b790a15a/src/distillschedule/runtime/helios_gan_manual/runtime.py#L114-L227)
- 测试：[tests/test_gan_frozen_up_retention.py::test_frozen_up_operand_contract_and_no_trainable_cache](https://github.com/sliortega295-ops/DMD-System/blob/155ffbb585356a29b544573a391eb456b790a15a/tests/test_gan_frozen_up_retention.py#L22-L33)
- 测试：[tests/test_gan_frozen_up_retention.py::test_real_selective_addmm_replay_preserves_finite_difference_r1_gradients_rng](https://github.com/sliortega295-ops/DMD-System/blob/155ffbb585356a29b544573a391eb456b790a15a/tests/test_gan_frozen_up_retention.py#L36-L63)
- 核对命令：`dmd_knowledge.py locate --implementation w3-frozen-ffn-up-replay --repo /project/DMD-System --show-code`

<a id="role-producer-dispatch"></a>

## 按 producer_dispatch 执行真实 forward/backward

- ID：`role-producer-dispatch`；用途：runtime_component
- 范围：resident replicated role pools；CPU/Gloo 合同，不支持自动推导任意 FSDP/offload 调度。
- 正确性前提：{"resident_replicated_pools": true, "dispatch_identity_validated": true, "logical_state_mapping_preserved": true}
- 收益条件：{"cost_domain_matches_target": true}
- 接入：将已验证 producer_dispatch 交给 RolePoolExecutor；核对 rank一致性、实际forward/backward事件与版本，不能只修改 cost-model 输出。
- 回退：使用原已验收的 producer 顺序；不能只替换预测表而宣称执行恢复。
- 验证范围：本轮核对固定源码和符号，未执行该历史 GPU/训练实验；测试链接只提供导航。
- 测试缺口：未定位到覆盖该精确历史变体的独立测试；不能借邻近测试声明验收。
- 入口：[src/distillschedule/open_workload/role_transport.py::RolePoolExecutor._run_cycle](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/open_workload/role_transport.py#L276-L466)
- 接入代码：[src/distillschedule/open_workload/role_transport.py::RolePoolExecutor._prepare](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/open_workload/role_transport.py#L157-L225)
- 接入代码：[src/distillschedule/open_workload/role_transport.py::RolePoolExecutor.run](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/open_workload/role_transport.py#L227-L255)
- 核对命令：`dmd_knowledge.py locate --implementation role-producer-dispatch --repo /project/DMD-System --show-code`

<a id="wan-exact-ema-11-20"></a>

## W1 r34 已知11–20步的原生EMA图

- ID：`wan-exact-ema-11-20`；用途：historical_component
- 范围：只支持原reference FP32 resident EMA、优化步10起、11..20的十个已知decay和两个原生周期；不是无限训练图。
- 正确性前提：{"workload": "W1", "ema_coefficients": "known_steps_11_20", "resident_fp32_reference_ema": true}
- 收益条件：{"launch_bound": true, "capture_amortized": true}
- 接入：核对schedule/identity/stream guards；通过explicit_worker显式启用；超出已知区间需重新设计并验收，不能删掉 guard。
- 回退：新进程使用 enabled=False 的原 reference EMA。
- 验证范围：本轮核对固定源码和符号，未执行该历史 GPU/训练实验；测试链接只提供导航。
- 测试缺口：当前 workload 的梯度、组合、容量和恢复仍需验证。
- 入口：[artifacts/publications/20261019/manual_packets/state_retention/r34/scripts/exact_ema_graph.py::ExactEMAStep](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/manual_packets/state_retention/r34/scripts/exact_ema_graph.py#L13-L29)
- 入口：[artifacts/publications/20261019/manual_packets/state_retention/r34/scripts/exact_ema_graph.py::capture](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/manual_packets/state_retention/r34/scripts/exact_ema_graph.py#L31-L57)
- 接入代码：[artifacts/publications/20261019/manual_packets/state_retention/r34/scripts/exact_ema_graph.py::explicit_worker](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/manual_packets/state_retention/r34/scripts/exact_ema_graph.py#L60-L99)
- 测试：[artifacts/publications/20261019/manual_packets/state_retention/r34/tests/test_exact_ema_graph.py::test_ten_real_cpu_updates_bitwise_output_metadata_and_rng](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/manual_packets/state_retention/r34/tests/test_exact_ema_graph.py#L27-L38)
- 核对命令：`dmd_knowledge.py locate --implementation wan-exact-ema-11-20 --repo /project/DMD-System --show-code`

<a id="helios-trainable-timestep-residual"></a>

## Helios trainable timestep residual backward

- ID：`helios-trainable-timestep-residual`；用途：runtime_component
- 范围：当前Helios phase_local梯度实现；有具体shape、dtype和reduction约束，不是no-grad输出缓存。
- 正确性前提：{"grad_mode": "with_grad", "timestep_gradients_preserved": true, "bf16_before_reduction": true}
- 收益条件：{"pointwise_staging_on_critical_path": true}
- 接入：函数名含nograd的安装入口也包含训练态分支，逐项核对trainable_residual_backward及phase_local条件，确认真实调用。
- 回退：恢复原生residual backward路径；保留梯度和reduction顺序。
- 验证范围：本轮核对固定源码和符号，未执行该历史 GPU/训练实验；测试链接只提供导航。
- 测试缺口：未定位到覆盖该精确历史变体的独立测试；不能借邻近测试声明验收。
- 入口：[src/distillschedule/runtime/helios_residual_kernel.py::fused_gated_residual_trainable_temb_backward_chunk](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_residual_kernel.py#L327-L391)
- 接入代码：[src/distillschedule/runtime/helios_residual_fusion.py::install_generator_nograd_residual_fusion](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_residual_fusion.py#L58-L587)
- 核对命令：`dmd_knowledge.py locate --implementation helios-trainable-timestep-residual --repo /project/DMD-System --show-code`

<a id="h3-block49-hybrid-output-cache"></a>

## W6 r1 block49 GPU结果缓存

- ID：`h3-block49-hybrid-output-cache`；用途：historical_component
- 范围：45938af8历史原生selected-output pool；只有block49从pinned CPU改留GPU，严格绑定原consumer/version。
- 正确性前提：{"workload": "W6", "cache_key_complete": true, "cached_gradients_preserved": true}
- 收益条件：{"saved_cache_transfer_exceeds_overhead": true, "memory_headroom": true}
- 接入：只在对应历史strategy配置启用GPU_BLOCKS=(49,)；核对原父模块身份、GPU输出存储计数与consumer释放。
- 回退：恢复原SHA绑定parent pool及pinned CPU存储策略，不改变原始模型或attention。
- 验证范围：本轮核对固定源码和符号，未执行该历史 GPU/训练实验；测试链接只提供导航。
- 测试缺口：当前 workload 的梯度、组合、容量和恢复仍需验证。
- 入口：[src/distillschedule/strategies/h3_ar_hybrid_cache.py::pool_class](https://github.com/sliortega295-ops/DMD-System/blob/45938af88d65ddb15b96ec1a181b65aa4e92de70/src/distillschedule/strategies/h3_ar_hybrid_cache.py#L13-L124)
- 接入代码：[src/distillschedule/strategies/h3_ar_hybrid_cache.py::install](https://github.com/sliortega295-ops/DMD-System/blob/45938af88d65ddb15b96ec1a181b65aa4e92de70/src/distillschedule/strategies/h3_ar_hybrid_cache.py#L127-L137)
- 接入代码：[src/distillschedule/strategies/h3_ar_hybrid_cache.py::validate_receipt](https://github.com/sliortega295-ops/DMD-System/blob/45938af88d65ddb15b96ec1a181b65aa4e92de70/src/distillschedule/strategies/h3_ar_hybrid_cache.py#L140-L147)
- 测试：[tests/test_h3_ar_hybrid_cache.py::test_full_original_consumer_order_cpu_loss_grad_rng_adam_and_actual_retention](https://github.com/sliortega295-ops/DMD-System/blob/45938af88d65ddb15b96ec1a181b65aa4e92de70/tests/test_h3_ar_hybrid_cache.py#L57-L69)
- 核对命令：`dmd_knowledge.py locate --implementation h3-block49-hybrid-output-cache --repo /project/DMD-System --show-code`

