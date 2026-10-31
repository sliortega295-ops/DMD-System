# 按风险验证数值、状态与训练生命周期

不同改动需要不同深度的验证；训练 loss 走势、数值一致性和输出质量是不同判断。

检查相对 matched baseline 的额外偏离，不以 loss 上升或下降独立判正确。
finite、调用计数和版本相同不能替代梯度、参数与 optimizer 状态的检查。
已有 numerical FAIL 未解决时不做正式速度晋级；原因未知保留 REVIEW。
保存/恢复/导出/生成会触发训练窗口没有的分支与内存峰值；按交付范围测试端点。
根据风险冻结 V0/V1/V2/V3 验证深度；不因非 bitwise 自动要求固定 100 步。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- current_validation_contract_defined: True

收益条件（不满足会降低收益预期，不等于语义错误）：

- 无统一性能门槛；根据当前目标判断。

最小验证：先用能够暴露该风险的短测试；状态变化补 continuation/resume；长期近似或残余漂移问题再做轨迹和质量检查。

## 已整理案例

### W5：后向诊断并未消除原数值失败

- ID：helios-w5-numerical-failure
- 状态：correctness_failure / HISTORICAL_FAIL_UNRESOLVED
- 观察：r1 记录两次真实 local backward diagnostics，但 prior numerical FAIL unresolved；没有 speed pair。
- 经验：运行诊断、finite 和完成调用不能替代原验收门；没有速度配对应记 NOT_RUN。
- 范围：W5 full-sequence Grid1000 近似合同；不能推广成 W6 continuous AR 的失败。
- 重试条件：定位并修复该失败，保持原验收标准；重新验证后才比较速度。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json)

### loss 上升应与匹配 baseline 比较

- ID：helios-loss-trend-lead
- 状态：unverified / UNVERIFIED_CONVERSATION_LEAD
- 观察：本地会话纠正过把 loss 上升直接当作数值异常的判断；该会话没有补齐匹配 baseline 的一致性证据。
- 经验：关注优化版相对 baseline 的额外偏离和梯度/参数/optimizer 状态；单臂趋势不能裁决一致性。
- 范围：会话提炼的方法线索；不作为 H3 AR 正确或错误的实验结论。
- 重试条件：取得匹配 baseline 和状态/轨迹证据后再判定。
- 来源：已脱敏的本地会话线索；公开可复核回执缺失，不能作为实验验证。

### 恢复时重复分配 Adam 状态可能出现额外峰值

- ID：livewan-checkpoint-memory-lead
- 状态：unverified / UNVERIFIED_CONVERSATION_LEAD
- 观察：本地会话提出 LiveWan 恢复时避免第二份完整 Adam 分配；当时只有 CPU 检查线索，GPU 结果未在本库核实。
- 经验：训练稳态能运行，不说明恢复峰值可承受；检查载入过程是否复用现有 optimizer state。
- 范围：私有会话线索，未公开原始执行回执，不能称已解决所有恢复 OOM。
- 重试条件：检查实际 checkpoint restore 实现，测量恢复峰值并核对恢复后 optimizer 更新。
- 来源：已脱敏的本地会话线索；公开可复核回执缺失，不能作为实验验证。
