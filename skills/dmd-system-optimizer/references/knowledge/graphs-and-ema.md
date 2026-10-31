# 图捕获、EMA 与固定状态假设

图捕获与批量提交可减少 host 开销；形状、系数、状态地址和阶段变化会影响正确性与命中率。

把固定十个 EMA 系数的试验限定在对应 steps，不能直接推广到无限训练。
训练、EMA、no-grad 和 checkpoint 路径可能触发不同图；单层 fixture 不覆盖全流程。
编译成本和 warmup 单列；短任务的最优选择可能与长任务不同。
改动后重新看提交开销是否仍在关键路径；不能从总耗时减少直接推断新并发发生。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- dynamic_state_and_coefficients_preserved: True
- capture_paths_validated: True

收益条件（不满足会降低收益预期，不等于语义错误）：

- launch_bound: True
- capture_amortized: True

最小验证：核对 capture/replay 与 fallback 计数，跨阶段和系数变化测试；比较短工作和稳态完整周期。

## 已整理案例

### W1：固定 EMA 系数图的有效期只覆盖已测 steps

- ID：helios-w1-ema-r34
- 状态：positive_observation / HISTORICAL_REVIEW
- 观察：C2 19.720700→19.549642 秒，约0.171秒；仅已知 EMA decays11–20 两周期；REVIEW。
- 经验：小正值可保留为组合候选；有限系数范围不能包装成生产无限训练支持。
- 范围：r34 exact EMA submission graph，同 streams/order；长期未跑，生产未晋级。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r34/conclusion.json)
