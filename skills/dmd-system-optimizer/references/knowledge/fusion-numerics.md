# 融合、精度边界与 no-grad 复用

融合可减少中间访存和 launch，但 dtype、舍入点、布局及 backward 决定实现是否可复用。

同一个代数式可以有不同 BF16/FP32 中间值。FMA、乘加顺序和 reduction 改动必须符合预先声明的数值合同。
保留 no-grad 专用实现的边界；调用函数本身没有 autograd 支持时，不能因为名字相似就用于训练。
Generator 与冻结 Score 可能使用不同 dtype；复用前读调用方和包装器，而非只看 kernel 签名。
先复用适用的已有实现或库，写新 kernel 需要说明缺口。
非 bitwise 本身不决定验证步数，按实际风险和项目标准选择。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- dtype_layout_supported: True
- gradient_path_supported: True
- numerical_contract_preserved: True

收益条件（不满足会降低收益预期，不等于语义错误）：

- fusion_cost_on_critical_path: True
- saved_traffic_exceeds_overhead: True

最小验证：按实际调用比较 forward、输入/参数梯度和保存中间量；确认 fast path 命中，计算额外 copy 与完整周期成本。

## 已整理案例

### Wan LoRA epilogue：保留 FP32 乘加舍入与 BF16 输出

- ID：wan-lora-rounding-source
- 状态：source_contract / SOURCE_ONLY_GPU_NOT_RUN
- 观察：源码用独立 mul.rn.f32/add.rn.f32 后 BF16 store；要求 BF16 base、FP32 update、同形且连续的 CUDA 输入。
- 经验：代数等价不足以复用实现；这条路径为 no-grad 设计，训练梯度需另行支持和验证。
- 范围：静态源码核对，未在本次执行 CUDA 或训练验收。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[wan13b_lora_epilogue.py](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/wan13b_lora_epilogue.py)

### LiveWan：Generator 与 Score 的调制 dtype 不同

- ID：livewan-modulation-source
- 状态：source_contract / SOURCE_ONLY_GPU_NOT_RUN
- 观察：源码注释区分 G 的 FP32 scale 与 frozen Score 的 BF16 scale；G 实现有 no-grad、B1、width1536 与源码 guard。
- 经验：不能只因数学形式相同就跨角色套 kernel；检查 dtype、broadcast、grad 与 wrapper scope。
- 范围：C32 canary 实现静态核对；不是全模型通用实现或本次 GPU 验证。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[livewan_generator_modulation.py](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/runtime/livewan_generator_modulation.py)
