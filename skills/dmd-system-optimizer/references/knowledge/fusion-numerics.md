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
- 条件：{"model_family": "wan", "grad_mode": "no_grad", "base_dtype": "bf16", "lora_dtype": "fp32"}
- 失败分类：not_established
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[sliortega295-ops/DMD-System / wan13b_lora_epilogue.py](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_epilogue.py)；定位：module docstring; fused_lora_epilogue; inline PTX

### LiveWan：Generator 与 Score 的调制 dtype 不同

- ID：livewan-modulation-source
- 状态：source_contract / SOURCE_ONLY_GPU_NOT_RUN
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：源码注释区分 G 的 FP32 scale 与 frozen Score 的 BF16 scale；G 实现有 no-grad、B1、width1536 与源码 guard。
- 经验：不能只因数学形式相同就跨角色套 kernel；检查 dtype、broadcast、grad 与 wrapper scope。
- 范围：C32 canary 实现静态核对；不是全模型通用实现或本次 GPU 验证。
- 条件：{"model_family": "livewan", "grad_mode": "no_grad", "batch": 1, "width": 1536}
- 失败分类：not_established
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[sliortega295-ops/DMD-System / livewan_generator_modulation.py](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/livewan_generator_modulation.py)；定位：build.op guards; canary_scope; PLAN_CONTRACT

### Wan modulation chunk 扩大得到明显单配对收益

- ID：wan-modulation-chunk2048
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：固定 Tree 通信与 resident substrate，chunk512→2048：34.473506→27.233468 秒。另一个 async128→256 配对几乎无变化。
- 经验：chunk 参数对算子形状与调用频率敏感；不能把 Wan 的收益比例搬到 Helios 或与异源 ladder 项逐项相加。
- 范围：Wan1.3B resident、该 modulation 实现和 matched pair。
- 条件：{"workload": "W1", "model_family": "Wan1.3B", "training_mode": "training", "global_batch": 16, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / WAN13B_RESIDENT_OPTIMIZATION_LADDER_20260824.md](https://github.com/sliortega295-ops/DMD-System/blob/9186aee022e7d8933ca84429bde7f54868f203de/docs/experiments/WAN13B_RESIDENT_OPTIMIZATION_LADDER_20260824.md)；定位：结果、合同、限制与失败记录

### Helios 的同类 chunk 扩大只有小收益或回退

- ID：helios-modulation-chunk-neutral
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：No-GAN ladder 的 chunk2048 观察为 116.008085 秒，未优于此前点；GAN-on 独立配对 143.379704→143.206713 秒，仅约 0.12%。
- 经验：模型家族、序列长度、角色比例和原有融合会改变 chunk 选择；一次跨 workload 成功不构成通用参数。
- 范围：Helios No-GAN 与 GAN-on 是两个独立合同，不能直接比绝对秒数。
- 条件：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：workload_dependent_small_or_negative_gain
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_LADDER_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/c93c2736e0fc528e72e63229998d21d87181d346/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_LADDER_20260825.md)；定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / iteration354_modulation_chunk2048.md](https://github.com/sliortega295-ops/DMD-System/blob/92de9f07af81a9e7ff96de755aafdcf4dadadf2b/docs/experiments/gan_cuda_baseline_opt_20260829/iteration354_modulation_chunk2048.md)；定位：结果、合同、限制与失败记录

### no-grad LoRA epilogue 在 Wan 有实测收益

- ID：wan-lora-epilogue-positive
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：保留 BF16 输出与 FP32 scale/add 舍入契约的 no-grad epilogue 配对：27.052164→26.227823 秒。
- 经验：收益依赖调用实际命中 no-grad 支持域；可微 Generator 路径仍需要独立 backward 支持与测试。
- 范围：Wan resident ladder、指定 no-grad 调用，不覆盖 trainable forward。
- 条件：{"workload": "W1", "model_family": "Wan1.3B", "training_mode": "training", "global_batch": 16, "grad_mode": "no_grad", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / WAN13B_RESIDENT_OPTIMIZATION_LADDER_20260824.md](https://github.com/sliortega295-ops/DMD-System/blob/9186aee022e7d8933ca84429bde7f54868f203de/docs/experiments/WAN13B_RESIDENT_OPTIMIZATION_LADDER_20260824.md)；定位：结果、合同、限制与失败记录

### 显式 modulation backward 避免逐 chunk autograd 构图

- ID：helios-modnorm-explicit-backward
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：保持 native FP32 LN backward 与显式 temb 梯度，GAN-on 141.235767→138.462140 秒；B1/B3 的 hidden/output bitwise，temb 满足当时 P0 而非 bitwise。
- 经验：分别记录每个输出/梯度的精度等级；完整周期 pass 不等于长期质量或所有中间值逐位一致。
- 范围：Helios modulation norm、指定 B1/B3 和 dtype。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration363_c11_modnorm_backward.md](https://github.com/sliortega295-ops/DMD-System/blob/06b53e85e6598bbc43d93c785a85520da7c3f716/docs/experiments/gan_cuda_baseline_opt_20260829/iteration363_c11_modnorm_backward.md)；定位：结果、合同、限制与失败记录

### 局部融合的倍数必须换算为完整周期收益

- ID：helios-fused-modnorm-forward-backward
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：前向 LN/modulation canary 约 5.5–6.2×，Full5 138.176083→136.084682 秒；后续独立 backward 融合使 136.084682→131.562027 秒。
- 经验：保存中间量、FP32 reduction 与 BF16 写回都属契约；局部倍数不能直接写成系统倍数或独立相乘。
- 范围：width5120 的 GAN-on 两个相继 substrate，各有单次配对。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration367_c14_fused_layernorm.md](https://github.com/sliortega295-ops/DMD-System/blob/b6cdeed6211bcfb5433840da68d2339b427946f0/docs/experiments/gan_cuda_baseline_opt_20260829/iteration367_c14_fused_layernorm.md)；定位：结果、合同、限制与失败记录
- 来源：[sliortega295-ops/DMD-System / iteration369_c15_fused_backward.md](https://github.com/sliortega295-ops/DMD-System/blob/1ef0e907aff7a64d4b459f0f41af64c499d3689b/docs/experiments/gan_cuda_baseline_opt_20260829/iteration369_c15_fused_backward.md)；定位：结果、合同、限制与失败记录

### cross LayerNorm 微基准快 5.7–6.35×，Full5 却变慢

- ID：helios-bf16-crossnorm-reversal
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：No-GAN direct BF16 frozen cross norm 在孤立 canary bitwise，Full5 从 94.461700 变为 95.538187 秒，回退。
- 经验：kernel 单独更快仍需核对布局、组合开销与完整周期；该实现的回退不封禁所有 cross norm 融合。
- 范围：No-GAN Iteration382、当时组合与 B1 shapes。
- 条件：{"workload": "W2", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": false, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：micro_to_e2e_reversal
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md](https://github.com/sliortega295-ops/DMD-System/blob/111e293dc79978132af91ab5515a096d941b94fe/docs/experiments/HELIOS14B_HSDP_OPTIMIZATION_CLOSURE_20260825.md)；定位：结果、合同、限制与失败记录

### 后续 GAN frozen cross norm 只有 canary，未完成 world16

- ID：helios-frozen-crossnorm-unrun
- 状态：inconclusive / HISTORICAL_CANARY_ONLY_NOT_RUN
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：B1/B3 canary 约 2.98/6.88×，P0 output/hidden pass 但非 bitwise；world16 NOT_RUN。两次早期失败分别是 import 路径和 writer 目录问题。
- 经验：canary 数值/速度、有效结果文件、全模型验证是分开的完成状态；不要将未跑候选并入最佳 Full5。
- 范围：GAN-on C27、frozen affine width5120、仅 microbenchmark。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration386_c27_frozen_cross_norm.md](https://github.com/sliortega295-ops/DMD-System/blob/a78827099c89795416f9ac6c77284dd5bacb0383/docs/experiments/gan_cuda_baseline_opt_20260829/iteration386_c27_frozen_cross_norm.md)；定位：结果、合同、限制与失败记录

### 小型 add+cat 融合有完整周期小收益

- ID：helios-history-addcat-fusion
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：保持原 prefix 与当前 BF16 sum，backward 仅正确 views；micro 约 1.59/2.09×，Full5 125.814004→125.258736 秒，约 0.44%。
- 经验：小收益可以保留，但不能因此继续无边界微调；确认 fast-path count、fallback 与局部梯度。
- 范围：GAN-on C25，2,360 calls/rank、零 fallback。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration380_c25_history_current_fusion.md](https://github.com/sliortega295-ops/DMD-System/blob/160f19543cfdefd71ba9f8dc73bd79cad7b6d207/docs/experiments/gan_cuda_baseline_opt_20260829/iteration380_c25_history_current_fusion.md)；定位：结果、合同、限制与失败记录

### ResearchBoost：融合进 GEMM 可能输给库 GEMM 加独立融合

- ID：rb-vendor-gemm-fusion-tradeoff
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：fusion card 将优化库 GEMM 与其前后内存操作分开比较，指出硬件 lowering 和激活近似式都会影响融合代价。
- 经验：同时测 vendor GEMM+独立 kernel 与更大融合，保留准确激活数学；不能只因 launch 更少就选后者。
- 范围：外部 Transformer/DiT fusion 经验，架构与 dtype 相关。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-fusion-patterns/SKILL.md)；定位：Read these first / GEMM prologue and epilogue

### ResearchBoost：QKV packing 需覆盖 optimizer、checkpoint 与 reduction 顺序

- ID：rb-qkv-layout-state-mapping
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：packing 改变参数 keys/shapes；布局 copy 消失还可能改变后继 reduction 的顺序。
- 经验：建立参数、梯度、Adam/EMA、save/resume 的 canonical 映射；forward shape 匹配不足以证明训练等价。
- 范围：外部 QKV packing 与布局融合建议；本地已有 guard 优先。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-fusion-patterns/SKILL.md)；定位：QKV packing; Removing layout copies

### ResearchBoost：复现库 reduction 需要核对分区、舍入和算术顺序

- ID：rb-reduction-architecture-numerics
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：fusion card 讨论线程分区、分阶段 reduction、epsilon、rsqrt 与 FMA 造成的逐位差异。
- 经验：A100 上满足的 numerical contract 到 H100 仍需核对实际库 kernel；“同数学表达式”不能代替同执行精度。
- 范围：外部 reduction 复现方法，不提供跨架构 bitwise 保证。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-fusion-patterns/SKILL.md)；定位：Reproducing library reductions bit for bit
