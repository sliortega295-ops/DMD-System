# 融合、精度边界与 no-grad 复用

融合可减少中间访存和 launch，但 dtype、舍入点、布局及 backward 决定实现是否可复用。

同一个代数式可以有不同 BF16/FP32 中间值。FMA、乘加顺序和 reduction 改动必须符合预先声明的数值合同。
保留 no-grad 专用实现的边界；调用函数本身没有 autograd 支持时，不能因为名字相似就用于训练。
Generator 与冻结 Score 可能使用不同 dtype；复用前读调用方和包装器，而非只看 kernel 签名。
先复用适用的已有实现或库，写新 kernel 需要说明缺口。
非 bitwise 本身不决定验证步数，按实际风险和项目标准选择。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- dtype_layout_supported: true
- gradient_path_supported: true
- numerical_contract_preserved: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- fusion_cost_on_critical_path: true
- saved_traffic_exceeds_overhead: true

最小验证：按实际调用比较 forward、输入/参数梯度和保存中间量；确认 fast path 命中，计算额外 copy 与完整周期成本。

## 已整理案例

### Wan LoRA epilogue：保留 FP32 乘加舍入与 BF16 输出

- ID：wan-lora-rounding-source
- 状态：source_contract / SOURCE_ONLY_GPU_NOT_RUN
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：源码用独立 mul.rn.f32/add.rn.f32 后 BF16 store；要求 BF16 base、FP32 update、同形且连续的 CUDA 输入。
- 经验：代数等价不足以复用实现；这条路径为 no-grad 设计，训练梯度需另行支持和验证。
- 范围：静态源码核对，未在本次执行 CUDA 或训练验收。
- 历史上下文：{"model_family": "wan", "grad_mode": "no_grad", "base_dtype": "bf16", "lora_dtype": "fp32"}
- 正确性前提：{"dtype_layout_supported": true, "gradient_path_supported": true, "numerical_contract_preserved": true, "grad_mode": "no_grad", "base_dtype": "bfloat16", "lora_update_dtype": "float32", "contiguous_inputs": true}
- 收益条件：{"fusion_cost_on_critical_path": true, "saved_traffic_exceeds_overhead": true}
- 失败分类：not_established
- 重试信号：当前仍有匹配 dtype/layout 的 no-grad LoRA epilogue
- 最小验证：经实际 wrapper 检查 BF16 base、FP32 update、连续形状与两次舍入
- 不应重复：不能把无 autograd 的底层函数移入可微路径
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case wan-lora-rounding-source`
- 关联实现：[Wan no-grad LoRA 融合及实际安装 wrapper](implementations.md#wan-lora-epilogue)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/wan13b_lora_epilogue.py::fused_lora_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_epilogue.py#L55-L92)
- 来源：[sliortega295-ops/DMD-System / wan13b_lora_epilogue.py](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_epilogue.py)；定位：module docstring; fused_lora_epilogue; inline PTX

### LiveWan：Generator 与 Score 的调制 dtype 不同

- ID：livewan-modulation-source
- 状态：source_contract / SOURCE_ONLY_GPU_NOT_RUN
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：源码注释区分 G 的 FP32 scale 与 frozen Score 的 BF16 scale；G 实现有 no-grad、B1、width1536 与源码 guard。
- 经验：不能只因数学形式相同就跨角色套 kernel；检查 dtype、broadcast、grad 与 wrapper scope。
- 范围：C32 canary 实现静态核对；不是全模型通用实现或本次 GPU 验证。
- 历史上下文：{"model_family": "livewan", "grad_mode": "no_grad", "batch": 1, "width": 1536}
- 正确性前提：{"dtype_layout_supported": true, "gradient_path_supported": true, "numerical_contract_preserved": true, "grad_mode": "no_grad", "microbatch": 1, "width": 1536, "scale_shift_dtype": "float32"}
- 收益条件：{"fusion_cost_on_critical_path": true, "saved_traffic_exceeds_overhead": true}
- 失败分类：not_established
- 重试信号：目标是匹配的 Generator no-grad B1 width1536 路径
- 最小验证：检查 FP32 scale/shift、源码 guard 和实际命中，再对照原 forward
- 不应重复：不能把 Score 的 BF16 modulation 当成同一精度合同
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case livewan-modulation-source`
- 关联实现：[LiveWan Generator modulation canary 与 runtime scope](implementations.md#livewan-generator-modulation)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/livewan_generator_modulation.py::build](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/livewan_generator_modulation.py#L18-L65)
- 函数入口：[src/distillschedule/runtime/livewan_generator_modulation.py::model_runtime_scope](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/livewan_generator_modulation.py#L140-L146)
- 来源：[sliortega295-ops/DMD-System / livewan_generator_modulation.py](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/livewan_generator_modulation.py)；定位：build.op guards; canary_scope; PLAN_CONTRACT

### Wan modulation chunk 扩大得到明显单配对收益

- ID：wan-modulation-chunk2048
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：固定 Tree 通信与 resident substrate，chunk512→2048：34.473506→27.233468 秒。另一个 async128→256 配对几乎无变化。
- 经验：chunk 参数对算子形状与调用频率敏感；不能把 Wan 的收益比例搬到 Helios 或与异源 ladder 项逐项相加。
- 范围：Wan1.3B resident、该 modulation 实现和 matched pair。
- 历史上下文：{"workload": "W1", "model_family": "Wan1.3B", "training_mode": "training", "global_batch": 16, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"dtype_layout_supported": true, "gradient_path_supported": true, "numerical_contract_preserved": true}
- 收益条件：{"fusion_cost_on_critical_path": true, "saved_traffic_exceeds_overhead": true}
- 失败分类：not_established
- 重试信号：当前 Wan shape 与 chunk 成本结构相近
- 最小验证：固定 Tree/resident substrate 单独扫描 chunk，检查数值和峰值
- 不应重复：不能套用 Helios 的同名 chunk helper 或混入 async 改动
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case wan-modulation-chunk2048`
- 来源：[sliortega295-ops/DMD-System / WAN13B_RESIDENT_OPTIMIZATION_LADDER_20260824.md](https://github.com/sliortega295-ops/DMD-System/blob/9186aee022e7d8933ca84429bde7f54868f203de/docs/experiments/WAN13B_RESIDENT_OPTIMIZATION_LADDER_20260824.md#L43)；定位：观察中的数值文本锚点 L43；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### Helios 的同类 chunk 扩大只有小收益或回退

- ID：helios-modulation-chunk-neutral
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：No-GAN ladder 的 chunk2048 观察为 116.008085 秒，未优于此前点；GAN-on 独立配对 143.379704→143.206713 秒，仅约 0.12%。
- 经验：模型家族、序列长度、角色比例和原有融合会改变 chunk 选择；一次跨 workload 成功不构成通用参数。
- 范围：Helios No-GAN 与 GAN-on 是两个独立合同，不能直接比绝对秒数。
- 历史上下文：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"dtype_layout_supported": true, "gradient_path_supported": true, "numerical_contract_preserved": true}
- 收益条件：{"fusion_cost_on_critical_path": true, "saved_traffic_exceeds_overhead": true}
- 失败分类：workload_dependent_small_or_negative_gain
- 重试信号：当前形状、batch 或 modulation 比例改变
- 最小验证：同一 GAN 模式做 chunk 对照并比较端到端噪声
- 不应重复：Wan 的大收益不是 Helios 重试的证据
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-modulation-chunk-neutral`
- 关联实现：[Helios modulation chunk 参数入口](implementations.md#helios-modulation-chunks)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/helios_modulation_chunks.py::configure_helios_modulation_chunk_tokens](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_modulation_chunks.py#L19-L85)
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_LADDER_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/c93c2736e0fc528e72e63229998d21d87181d346/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_LADDER_20260825.md#L43)；定位：观察中的数值文本锚点 L43；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / iteration354_modulation_chunk2048.md](https://github.com/sliortega295-ops/DMD-System/blob/92de9f07af81a9e7ff96de755aafdcf4dadadf2b/docs/experiments/gan_cuda_baseline_opt_20260829/iteration354_modulation_chunk2048.md#L34)；定位：观察中的数值文本锚点 L34；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### no-grad LoRA epilogue 在 Wan 有实测收益

- ID：wan-lora-epilogue-positive
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：保留 BF16 输出与 FP32 scale/add 舍入契约的 no-grad epilogue 配对：27.052164→26.227823 秒。
- 经验：收益依赖调用实际命中 no-grad 支持域；可微 Generator 路径仍需要独立 backward 支持与测试。
- 范围：Wan resident ladder、指定 no-grad 调用，不覆盖 trainable forward。
- 历史上下文：{"workload": "W1", "model_family": "Wan1.3B", "training_mode": "training", "global_batch": 16, "grad_mode": "no_grad", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"dtype_layout_supported": true, "gradient_path_supported": true, "numerical_contract_preserved": true, "grad_mode": "no_grad", "base_dtype": "bfloat16", "lora_update_dtype": "float32", "contiguous_inputs": true}
- 收益条件：{"fusion_cost_on_critical_path": true, "saved_traffic_exceeds_overhead": true}
- 失败分类：not_established
- 重试信号：当前 no-grad epilogue 仍在关键路径
- 最小验证：验证 BF16/FP32 两次舍入、wrapper 命中与完整周期
- 不应重复：不要把这份 no-grad 实现直接用于梯度路径
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case wan-lora-epilogue-positive`
- 关联实现：[Wan no-grad LoRA 融合及实际安装 wrapper](implementations.md#wan-lora-epilogue)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/wan13b_lora_epilogue.py::fused_lora_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_epilogue.py#L55-L92)
- 来源：[sliortega295-ops/DMD-System / WAN13B_RESIDENT_OPTIMIZATION_LADDER_20260824.md](https://github.com/sliortega295-ops/DMD-System/blob/9186aee022e7d8933ca84429bde7f54868f203de/docs/experiments/WAN13B_RESIDENT_OPTIMIZATION_LADDER_20260824.md#L7)；定位：观察中的数值文本锚点 L7；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### 显式 modulation backward 避免逐 chunk autograd 构图

- ID：helios-modnorm-explicit-backward
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：保持 native FP32 LN backward 与显式 temb 梯度，GAN-on 141.235767→138.462140 秒；B1/B3 的 hidden/output bitwise，temb 满足当时 P0 而非 bitwise。
- 经验：分别记录每个输出/梯度的精度等级；完整周期 pass 不等于长期质量或所有中间值逐位一致。
- 范围：Helios modulation norm、指定 B1/B3 和 dtype。
- 历史上下文：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"dtype_layout_supported": true, "gradient_path_supported": true, "numerical_contract_preserved": true}
- 收益条件：{"fusion_cost_on_critical_path": true, "saved_traffic_exceeds_overhead": true}
- 失败分类：not_established
- 重试信号：目标 shape 与梯度语义满足当前 modnorm 实现
- 最小验证：同时比较 hidden/output 与 temb 梯度，按冻结容差验收
- 不应重复：不能把 hidden bitwise 写成所有梯度 bitwise
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-modnorm-explicit-backward`
- 关联实现：[Helios modulation norm 的安装与融合实现](implementations.md#helios-modnorm)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/helios_modulated_norm_fusion.py::install_exact_modulated_norm_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_modulated_norm_fusion.py#L24-L774)
- 函数入口：[src/distillschedule/runtime/helios_modulated_norm_kernel.py::fused_modulated_norm_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_modulated_norm_kernel.py#L207-L263)
- 来源：[sliortega295-ops/DMD-System / iteration363_c11_modnorm_backward.md](https://github.com/sliortega295-ops/DMD-System/blob/06b53e85e6598bbc43d93c785a85520da7c3f716/docs/experiments/gan_cuda_baseline_opt_20260829/iteration363_c11_modnorm_backward.md#L5)；定位：观察中的数值文本锚点 L5；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### 局部融合的倍数必须换算为完整周期收益

- ID：helios-fused-modnorm-forward-backward
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：前向 LN/modulation canary 约 5.5–6.2×，Full5 138.176083→136.084682 秒；后续独立 backward 融合使 136.084682→131.562027 秒。
- 经验：保存中间量、FP32 reduction 与 BF16 写回都属契约；局部倍数不能直接写成系统倍数或独立相乘。
- 范围：width5120 的 GAN-on 两个相继 substrate，各有单次配对。
- 历史上下文：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"dtype_layout_supported": true, "gradient_path_supported": true, "numerical_contract_preserved": true}
- 收益条件：{"fusion_cost_on_critical_path": true, "saved_traffic_exceeds_overhead": true}
- 失败分类：not_established
- 重试信号：前后向各自有可测暴露成本
- 最小验证：分别核对 forward 与 backward variant，再测完整周期组合
- 不应重复：微测 5–6 倍不能当作训练周期加速
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-fused-modnorm-forward-backward`
- 关联实现：[Helios modulation norm 的安装与融合实现](implementations.md#helios-modnorm)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/helios_modulated_norm_fusion.py::install_exact_modulated_norm_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_modulated_norm_fusion.py#L24-L774)
- 函数入口：[src/distillschedule/runtime/helios_modulated_norm_kernel.py::fused_modulated_norm_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_modulated_norm_kernel.py#L207-L263)
- 来源：[sliortega295-ops/DMD-System / iteration367_c14_fused_layernorm.md](https://github.com/sliortega295-ops/DMD-System/blob/b6cdeed6211bcfb5433840da68d2339b427946f0/docs/experiments/gan_cuda_baseline_opt_20260829/iteration367_c14_fused_layernorm.md#L1)；定位：报告范围 L1–L40；L1: Iteration 367: C14 fused BF16 LayerNorm and modulation；L3: Frozen comparison；L14: Evidence；L31: Decision；原定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / iteration369_c15_fused_backward.md](https://github.com/sliortega295-ops/DMD-System/blob/1ef0e907aff7a64d4b459f0f41af64c499d3689b/docs/experiments/gan_cuda_baseline_opt_20260829/iteration369_c15_fused_backward.md#L20)；定位：观察中的数值文本锚点 L20；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### cross LayerNorm 微基准快 5.7–6.35×，Full5 却变慢

- ID：helios-bf16-crossnorm-reversal
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：No-GAN direct BF16 frozen cross norm 在孤立 canary bitwise，Full5 从 94.461700 变为 95.538187 秒，回退。
- 经验：kernel 单独更快仍需核对布局、组合开销与完整周期；该实现的回退不封禁所有 cross norm 融合。
- 范围：No-GAN Iteration382、当时组合与 B1 shapes。
- 历史上下文：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": false, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"dtype_layout_supported": true, "gradient_path_supported": true, "numerical_contract_preserved": true}
- 收益条件：{"fusion_cost_on_critical_path": true, "saved_traffic_exceeds_overhead": true}
- 失败分类：micro_to_e2e_reversal
- 重试信号：当前 crossnorm 关键路径或 kernel/shape 改变
- 最小验证：canary 过门后保留同源 Full5 对照
- 不应重复：孤立 bitwise 不构成忽略 Full5 回退的理由
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-bf16-crossnorm-reversal`
- 关联实现：[Helios LayerNorm/modulation canary](implementations.md#helios-crossnorm-canary)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/helios_fused_layernorm_kernel.py::fused_bf16_layernorm_modulation](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_fused_layernorm_kernel.py#L135-L222)
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/111e293dc79978132af91ab5515a096d941b94fe/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md#L1)；定位：报告范围 L1–L146；L1: Helios14B node-HSDP optimization closure (Iterations 368-382)；L3: Status and claim boundary；L27: New experiment ladder；L52: Effective methods；L54: Explicit pinned local FlashAttention-2；L70: Critic selective FA2 checkpoint-output retention；L87: Rejected and bounded-negative methods；L105: Post-winner Nsight result；原定位：结果、合同、限制与失败记录

### 后续 GAN frozen cross norm 只有 canary，未完成 world16

- ID：helios-frozen-crossnorm-unrun
- 状态：inconclusive / HISTORICAL_CANARY_ONLY_NOT_RUN
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：B1/B3 canary 约 2.98/6.88×，P0 output/hidden pass 但非 bitwise；world16 NOT_RUN。两次早期失败分别是 import 路径和 writer 目录问题。
- 经验：canary 数值/速度、有效结果文件、全模型验证是分开的完成状态；不要将未跑候选并入最佳 Full5。
- 范围：GAN-on C27、frozen affine width5120、仅 microbenchmark。
- 历史上下文：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"dtype_layout_supported": true, "gradient_path_supported": true, "numerical_contract_preserved": true}
- 收益条件：{"fusion_cost_on_critical_path": true, "saved_traffic_exceeds_overhead": true}
- 失败分类：not_established
- 重试信号：import/writer 问题修复且获准补齐全模型验证
- 最小验证：先复查 B1/B3 容差与梯度，再跑缺失的目标规模验证
- 不应重复：不能把 canary 时间写成 world16 已测收益
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-frozen-crossnorm-unrun`
- 关联实现：[Helios LayerNorm/modulation canary](implementations.md#helios-crossnorm-canary)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/helios_fused_layernorm_kernel.py::fused_bf16_layernorm_modulation](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_fused_layernorm_kernel.py#L135-L222)
- 来源：[sliortega295-ops/DMD-System / iteration386_c27_frozen_cross_norm.md](https://github.com/sliortega295-ops/DMD-System/blob/a78827099c89795416f9ac6c77284dd5bacb0383/docs/experiments/gan_cuda_baseline_opt_20260829/iteration386_c27_frozen_cross_norm.md#L1)；定位：报告范围 L1–L83；L1: Iteration 386: C27 frozen cross-attention LayerNorm canary；L3: Pause-point summary；L16: Source-qualified hypothesis；L38: Canary evidence；L73: Decision and stop boundary；原定位：结果、合同、限制与失败记录

### 小型 add+cat 融合有完整周期小收益

- ID：helios-history-addcat-fusion
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：保持原 prefix 与当前 BF16 sum，backward 仅正确 views；micro 约 1.59/2.09×，Full5 125.814004→125.258736 秒，约 0.44%。
- 经验：小收益可以保留，但不能因此继续无边界微调；确认 fast-path count、fallback 与局部梯度。
- 范围：GAN-on C25，2,360 calls/rank、零 fallback。
- 历史上下文：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"dtype_layout_supported": true, "gradient_path_supported": true, "numerical_contract_preserved": true}
- 收益条件：{"fusion_cost_on_critical_path": true, "saved_traffic_exceeds_overhead": true}
- 失败分类：not_established
- 重试信号：当前 history prefix 与 add/cat 流量占比值得融合
- 最小验证：验证 BF16 sum、prefix 与 backward views，再比完整周期
- 不应重复：micro 的 1.59/2.09 倍不直接等于约 0.44% 之外的新收益
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-history-addcat-fusion`
- 关联实现：[Helios history/current add+cat 融合](implementations.md#helios-history-fusion)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/helios_history_current_fusion.py::install_exact_history_current_fusion](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_history_current_fusion.py#L77-L179)
- 来源：[sliortega295-ops/DMD-System / iteration380_c25_history_current_fusion.md](https://github.com/sliortega295-ops/DMD-System/blob/160f19543cfdefd71ba9f8dc73bd79cad7b6d207/docs/experiments/gan_cuda_baseline_opt_20260829/iteration380_c25_history_current_fusion.md#L5)；定位：观察中的数值文本锚点 L5；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### ResearchBoost：融合进 GEMM 可能输给库 GEMM 加独立融合

- ID：rb-vendor-gemm-fusion-tradeoff
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：fusion card 将优化库 GEMM 与其前后内存操作分开比较，指出硬件 lowering 和激活近似式都会影响融合代价。
- 经验：同时测 vendor GEMM+独立 kernel 与更大融合，保留准确激活数学；不能只因 launch 更少就选后者。
- 范围：外部 Transformer/DiT fusion 经验，架构与 dtype 相关。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"dtype_layout_supported": true, "gradient_path_supported": true, "numerical_contract_preserved": true}
- 收益条件：{"fusion_cost_on_critical_path": true, "saved_traffic_exceeds_overhead": true}
- 失败分类：not_established
- 重试信号：融合额外访存的节省大于丢失 vendor GEMM 优势
- 最小验证：保留高效 GEMM control，比较 epilogue 与全融合的真实 shape/数值
- 不应重复：不因减少 kernel 数就重写高性能 GEMM
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case rb-vendor-gemm-fusion-tradeoff`
- 关联实现：[Wan no-grad LoRA 融合及实际安装 wrapper](implementations.md#wan-lora-epilogue)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/wan13b_lora_epilogue.py::fused_lora_epilogue](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_epilogue.py#L55-L92)
- 关联实现：[Wan 可微 LoRA epilogue 的独立备选](implementations.md#wan-lora-grad-epilogue)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/wan13b_lora_grad_epilogue.py::fused_lora_epilogue_with_grad](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_grad_epilogue.py#L79-L85)
- 函数入口：[src/distillschedule/runtime/wan13b_lora_grad_epilogue.py::make_epilogue_autograd_function](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_grad_epilogue.py#L51-L76)
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-fusion-patterns/SKILL.md#L1)；定位：报告范围 L1–L179；L6: Common fusion paths and examples；L10: Read these first；L33: 1. Residual add + norm；L38: reference；L41: fused: read x and y once, write h (still needed by the residual stream) and out；L49: 2. Norm + scale and shift (+ gate): adaLN in DiT；L63: 3. Bias + activation (+ gated activation)；L68: reference；原定位：Read these first / GEMM prologue and epilogue

### ResearchBoost：QKV packing 需覆盖 optimizer、checkpoint 与 reduction 顺序

- ID：rb-qkv-layout-state-mapping
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：packing 改变参数 keys/shapes；布局 copy 消失还可能改变后继 reduction 的顺序。
- 经验：建立参数、梯度、Adam/EMA、save/resume 的 canonical 映射；forward shape 匹配不足以证明训练等价。
- 范围：外部 QKV packing 与布局融合建议；本地已有 guard 优先。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"dtype_layout_supported": true, "gradient_path_supported": true, "numerical_contract_preserved": true, "checkpoint_state_mapping_validated": true}
- 收益条件：{"fusion_cost_on_critical_path": true, "saved_traffic_exceeds_overhead": true}
- 失败分类：not_established
- 重试信号：packing 方案有完整参数/优化器/checkpoint 映射
- 最小验证：做 save/resume roundtrip 与梯度/reduction 对照
- 不应重复：keys/shapes 变化不能只验当前 forward
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case rb-qkv-layout-state-mapping`
- 关联实现：[项目 checkpoint 保存与恢复入口](implementations.md#checkpoint-lifecycle)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/training/checkpoint.py::save_checkpoint](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/training/checkpoint.py#L319-L456)
- 函数入口：[src/distillschedule/training/checkpoint.py::restore_checkpoint](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/training/checkpoint.py#L541-L559)
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-fusion-patterns/SKILL.md#L1)；定位：报告范围 L1–L179；L6: Common fusion paths and examples；L10: Read these first；L33: 1. Residual add + norm；L38: reference；L41: fused: read x and y once, write h (still needed by the residual stream) and out；L49: 2. Norm + scale and shift (+ gate): adaLN in DiT；L63: 3. Bias + activation (+ gated activation)；L68: reference；原定位：QKV packing; Removing layout copies

### ResearchBoost：复现库 reduction 需要核对分区、舍入和算术顺序

- ID：rb-reduction-architecture-numerics
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：fusion card 讨论线程分区、分阶段 reduction、epsilon、rsqrt 与 FMA 造成的逐位差异。
- 经验：A100 上满足的 numerical contract 到 H100 仍需核对实际库 kernel；“同数学表达式”不能代替同执行精度。
- 范围：外部 reduction 复现方法，不提供跨架构 bitwise 保证。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"dtype_layout_supported": true, "gradient_path_supported": true, "numerical_contract_preserved": true}
- 收益条件：{"fusion_cost_on_critical_path": true, "saved_traffic_exceeds_overhead": true}
- 失败分类：not_established
- 重试信号：目标设备/variant 的 reduction 数值差异已定位
- 最小验证：检查 epsilon、rsqrt、FMA、累加精度与反向，在目标架构跑 canary
- 不应重复：另一架构上的逐位结果不能担保当前硬件
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case rb-reduction-architecture-numerics`
- 关联实现：[Helios LayerNorm/modulation canary](implementations.md#helios-crossnorm-canary)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/runtime/helios_fused_layernorm_kernel.py::fused_bf16_layernorm_modulation](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/helios_fused_layernorm_kernel.py#L135-L222)
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-fusion-patterns/SKILL.md#L1)；定位：报告范围 L1–L179；L6: Common fusion paths and examples；L10: Read these first；L33: 1. Residual add + norm；L38: reference；L41: fused: read x and y once, write h (still needed by the residual stream) and out；L49: 2. Norm + scale and shift (+ gate): adaLN in DiT；L63: 3. Bias + activation (+ gated activation)；L68: reference；原定位：Reproducing library reductions bit for bit
