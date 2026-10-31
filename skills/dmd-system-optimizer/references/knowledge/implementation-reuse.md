# 先核对已有实现，再决定移植或重写

已有源码提供候选实现；模型名、接口相同或库存存在都不证明当前调用可用。

机制级经验可以跨模型检索；具体代码必须满足实现级限制。
记录精确源码版本和依赖，但不要把某一 GPU 名当作整个机制的永久适用边界。
未生效的快路径、库版本问题和真正无收益分开；失败类型决定下一步。
组合后重新检查资源和状态，不把多个独立 microbenchmark 收益直接相加。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- implementation_guards_match: True
- license_and_dependencies_checked: True

收益条件（不满足会降低收益预期，不等于语义错误）：

- fast_path_active: True

最小验证：读实际入口、guard、fallback、精度和 backward；用当前调用的最小测试确认命中，再做组合校准。

## 已整理案例

### 14 个已有实现入口可检索，仍需检查 guards

- ID：dmd-implementation-inventory
- 状态：source_contract / SOURCE_ONLY_GPU_NOT_RUN
- 观察：该快照 implementation_library 的 ENTRIES 有14个入口，涵盖 modulation、LoRA、RMSNorm、splitKV、SwiGLU 等。
- 经验：库存为复用提供起点；其自身也要求 workload adapter、成本校准和组合验证。
- 范围：静态库存，不执行模型导入，不宣称14项都适配新 workload。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[implementation_library.py](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/implementation_library.py)
