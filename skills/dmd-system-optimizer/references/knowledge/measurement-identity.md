# 对照身份、计时范围与因果归因

只有逻辑工作、资源和窗口可比较，时间差才支持所宣称的优化增量。

固定原 baseline 和当前已验证 control。修正错误 workload 需要新 baseline，不给修正计加速贡献。
profile 用来解释机制，正式收益用 unprofiled 对照；NVTX envelope 和累计 kernel 时间不等于关键路径可节省时间。
单次诊断、用户汇总表、跨 revision 比值和匹配实验分别标注。
计时后 observer 失败不抹去已有计时，也不把整次运行改成 PASS。
同一优化在不同窗口收益可能相反：启动、steady state、Full5、checkpoint 分开描述。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- matched_comparison_contract: True

收益条件（不满足会降低收益预期，不等于语义错误）：

- representative_measurement: True

最小验证：记录源码含 dirty 改动、配置、输入/状态、硬件、timer、rank 聚合和独立重复；拆开主表、消融与当前 control。

## 已整理案例

### W3：计时完成与 observer 失败分别保留

- ID：helios-w3-observer-failure
- 状态：inconclusive / HISTORICAL_REVIEW
- 观察：N70-OFF C2 116.253523015 秒，16 ranks merge=0；原回执计时后 observer KeyError 失败，strict numerical REVIEW。
- 经验：保留可支持的计时事实，同时保留原失败；CPU 离线核对不等于端到端 PASS。
- 范围：指定 no-merge retime；formal_speedup_eligible=false，不是旧113.59秒记录。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md)；[gan_nomerge_retime.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/results_correction/evidence/gan_nomerge/gan_nomerge_retime.json)

### W4：新两臂与旧 baseline 的来源不同

- ID：helios-w4-ablation-sources
- 状态：inconclusive / HISTORICAL_REVIEW
- 观察：R3 新 only-A 和 complete-B 同源；旧 baseline 与 joint-full 来自父源，没有在 R3 重跑全四格。
- 经验：可以引用各自观察；不能拼成同源完整因果消融，也不能因有四个数字就相加边际贡献。
- 范围：W4 R3 对照来源审计。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/livewan_r3/reports/conclusion.json)

### 保存计划接近手工结果与全系统加速分别归因

- ID：helios-planner-increment
- 状态：source_contract / HISTORICAL_EVIDENCE_SCOPE
- 观察：公开纠偏文档列出手工/保存计划逐项配对，并保留 W5 NOT_RUN。
- 经验：planner 复现手工选择的增量用手工作对照；不能把相对普通 FSDP 的系统收益全算成搜索收益。
- 范围：20261019 六工作量历史索引；不代表全局最优或当前 head GPU 验收。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md)
