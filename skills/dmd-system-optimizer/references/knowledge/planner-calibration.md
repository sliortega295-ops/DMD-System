# Planner 成本转移、搜索空间与实际执行

计划只在已校准的动作、资源和状态策略内有预测意义；还要证明运行器实际执行了所选动作。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- action_semantics_and_dispatch_validated: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- cost_domain_matches_target: true

最小验证：列出有测量和缺测的动作，按资源争用建依赖；比较手工 control、预测与执行，而不是拼接各处最优。

## 已整理案例

### Planner 复现接近手工最优，不等于独立新增加速

- ID：planner-manual-rediscovery
- 状态：source_contract / HISTORICAL_REVIEW
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：ret4/window3 的 planner 点 20.103775 秒，手工 ret4/window5 为 20.040550 秒；仍有 numerical REVIEW/失败记录。
- 经验：说明自动找到合适计划的能力，同时把系统 substrate 收益与 planner 的额外增量分开；成本表必须绑定 state policy。
- 范围：state retention r18 已校准动作域。
- 历史上下文：{"training_mode": "training", "workload": "W1"}
- 正确性前提：{"action_semantics_and_dispatch_validated": true}
- 收益条件：{"cost_domain_matches_target": true}
- 失败分类：not_established
- 重试信号：新成本或搜索空间能够区分当前手工最好点
- 最小验证：同源比较 planner ret4/window3 与手工 control 并完成 numerical gate
- 不应重复：重新找到已有点不能作为 planner 独立提升
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case planner-manual-rediscovery`
- 关联实现：[Planner 的已测成本及结果资格](implementations.md#planner-calibration)；diagnostic_tool；用于分析、诊断或 canary；不是可接入的 runtime 优化，拓扑和输入格式仍需核对。
- 函数入口：[src/distillschedule/search/cost_model.py::CostModel](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/cost_model.py#L77-L195)
- 函数入口：[src/distillschedule/search/measurement.py::normalize_native](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/measurement.py#L18-L98)
- 来源：[sliortega295-ops/DMD-System / conclusion.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r18/conclusion.md#L7)；定位：观察中的数值文本锚点 L7；同时阅读同节合同、状态与限制。原定位：结果与证据边界

### 缺测动作不应虚构成本，相同预测只说明 tie-break

- ID：planner-missing-costs-tie
- 状态：source_contract / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：H3 action closure 的 20 assignments 中 5 feasible、15 uncalibrated；多个 window 的预测相同，确定性选中 window5 不是它更快的证据。
- 经验：搜索空间受已测成本与共同 world16 资源约束；不能凭动作名假设免费 G/C overlap。
- 范围：H3 prepared-action cost closure 的既定资源模型。
- 历史上下文：{"training_mode": "training", "workload": "W5"}
- 正确性前提：{"action_semantics_and_dispatch_validated": true}
- 收益条件：{"cost_domain_matches_target": true}
- 失败分类：not_established
- 重试信号：缺失 action/window 成本已有实测校准
- 最小验证：检查各 assignment feasible/calibrated 状态和预测差异
- 不应重复：确定性 tie-break 选 window5 不证明它最快
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case planner-missing-costs-tie`
- 关联实现：[Planner 的已测成本及结果资格](implementations.md#planner-calibration)；diagnostic_tool；用于分析、诊断或 canary；不是可接入的 runtime 优化，拓扑和输入格式仍需核对。
- 函数入口：[src/distillschedule/search/cost_model.py::CostModel](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/cost_model.py#L77-L195)
- 函数入口：[src/distillschedule/search/measurement.py::normalize_native](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/measurement.py#L18-L98)
- 来源：[sliortega295-ops/DMD-System / DMD_H3_ACTION_PERFORMANCE_CLOSURE_20260907.md](https://github.com/sliortega295-ops/DMD-System/blob/68c3c0b1bf8c3ad350b27b49344e9bb9402df3fa/docs/experiments/DMD_H3_ACTION_PERFORMANCE_CLOSURE_20260907.md#L1)；定位：报告范围 L1–L140；L1: H3 measured action costs → common search → native performance closure；L8: Result；L30: What was actually searched and executed；L59: Hard gates and numerical boundary；L82: Memory and transfers；L102: Reproduction, failures and source fix；L134: Stop boundary；原定位：结果、合同、限制与失败记录

### 计划中的 producer 顺序必须落到真实 forward/backward

- ID：planner-dispatch-realized
- 状态：source_contract / SOURCE_CONTRACT_ONLY
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：producer dispatch 合同要求各 rank 对计划一致并执行选中的实际调用；仅改变表中顺序、运行器仍固定 FIFO 不能证明调度生效。
- 经验：提前 backward 释放激活的预测需用执行事件和峰值验证；CPU/Gloo probe 不自动证明 CUDA 或共享 NIC 成本。
- 范围：DMD role producer dispatcher 的源码/CPU 支持范围。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"action_semantics_and_dispatch_validated": true}
- 收益条件：{"cost_domain_matches_target": true}
- 失败分类：not_established
- 重试信号：运行器能够执行而非只打印选定计划
- 最小验证：跨 rank 核对 plan identity、调用顺序和实际调度事件
- 不应重复：planner 表里换序但 runtime FIFO 不算调度生效
- 代码状态：linked；关系按具体机制和代码职责标注；源码匹配不等于当前 workload 已验证。
- 定位命令：`dmd_knowledge.py locate --case planner-dispatch-realized`
- 关联实现：[按 producer_dispatch 执行真实 forward/backward](implementations.md#role-producer-dispatch)；reusable_candidate；此处是实际执行序列的代码；成本分析另行阅读，CPU测试不证明目标GPU吞吐。
- 函数入口：[src/distillschedule/open_workload/role_transport.py::RolePoolExecutor._run_cycle](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/open_workload/role_transport.py#L276-L466)
- 来源：[sliortega295-ops/DMD-System / ROLE_PRODUCER_DISPATCH.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/docs/status/ROLE_PRODUCER_DISPATCH.md#L1)；定位：报告范围 L1–L69；L1: Component-derived producer dispatch on the existing role runtime；L20: Evidence；L32: Boundaries；L49: Dispatch safety and a non-preset scheduling example；原定位：结果与证据边界
