# 对照身份、计时范围与因果归因

只有逻辑工作、资源和窗口可比较，时间差才支持所宣称的优化增量。

固定原 baseline 和当前已验证 control。修正错误 workload 需要新 baseline，不给修正计加速贡献。
profile 用来解释机制，正式收益用 unprofiled 对照；NVTX envelope 和累计 kernel 时间不等于关键路径可节省时间。
单次诊断、用户汇总表、跨 revision 比值和匹配实验分别标注。
计时后 observer 失败不抹去已有计时，也不把整次运行改成 PASS。
同一优化在不同窗口收益可能相反：启动、steady state、Full5、checkpoint 分开描述。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- matched_comparison_contract: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- representative_measurement: true

最小验证：记录源码含 dirty 改动、配置、输入/状态、硬件、timer、rank 聚合和独立重复；拆开主表、消融与当前 control。

## 已整理案例

### W3：计时完成与 observer 失败分别保留

- ID：helios-w3-observer-failure
- 状态：inconclusive / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：N70-OFF C2 116.253523015 秒，16 ranks merge=0；原回执计时后 observer KeyError 失败，strict numerical REVIEW。
- 经验：保留可支持的计时事实，同时保留原失败；CPU 离线核对不等于端到端 PASS。
- 范围：指定 no-merge retime；formal_speedup_eligible=false，不是旧113.59秒记录。
- 历史上下文：{"workload": "W3", "merged_lora": false}
- 正确性前提：{"matched_comparison_contract": true}
- 收益条件：{"representative_measurement": true}
- 失败分类：post_timer_instrumentation_failure
- 重试信号：observer 错误已修复且保留不可变原回执
- 最小验证：先离线修复解析并确认可用证据范围，需要时再跑缺失数值检查
- 不应重复：不能把 timer 完成等同于 observer 和严格验证通过
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case helios-w3-observer-failure`
- 来源：[sliortega295-ops/DMD-System / SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md#L39)；定位：观察中的数值文本锚点 L39；同时阅读同节合同、状态与限制。原定位：section 2: N70-OFF retime and preserved failure
- 来源：[sliortega295-ops/DMD-System / gan_nomerge_retime.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/results_correction/evidence/gan_nomerge/gan_nomerge_retime.json#L32)；定位：观察中的数值文本锚点 L32；同时阅读同节合同、状态与限制。原定位：status; strict_numerical_status; rank records

### W4：新两臂与旧 baseline 的来源不同

- ID：helios-w4-ablation-sources
- 状态：inconclusive / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：R3 新 only-A 和 complete-B 同源；旧 baseline 与 joint-full 来自父源，没有在 R3 重跑全四格。
- 经验：可以引用各自观察；不能拼成同源完整因果消融，也不能因有四个数字就相加边际贡献。
- 范围：W4 R3 对照来源审计。
- 历史上下文：{"workload": "W4"}
- 正确性前提：{"matched_comparison_contract": true}
- 收益条件：{"representative_measurement": true}
- 失败分类：comparison_identity
- 重试信号：有资源补齐同源四格或只能发布受限比较
- 最小验证：先核对每格 source/config；仅对同源格计算因果增量
- 不应重复：不把父源 baseline/joint-full 与 R3 混成完整消融
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case helios-w4-ablation-sources`
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/livewan_r3/reports/conclusion.json)；定位：source_boundary; baseline; joint_reference

### 保存计划接近手工结果与全系统加速分别归因

- ID：helios-planner-increment
- 状态：source_contract / HISTORICAL_EVIDENCE_SCOPE
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：公开纠偏文档列出手工/保存计划逐项配对，并保留 W5 NOT_RUN。
- 经验：planner 复现手工选择的增量用手工作对照；不能把相对普通 FSDP 的系统收益全算成搜索收益。
- 范围：20261019 六工作量历史索引；不代表全局最优或当前 head GPU 验收。
- 历史上下文：{}
- 正确性前提：{"matched_comparison_contract": true}
- 收益条件：{"representative_measurement": true}
- 失败分类：not_established
- 重试信号：当前 planner 能超过已知手工 control
- 最小验证：按 workload 对齐 saved plan、实际 dispatch 和完整周期；保留 W5 NOT_RUN
- 不应重复：不能用原 baseline 与 planner 的总差代替 planner 独立贡献
- 代码状态：linked；已定位相关源码；关系标注区分当前参考与旧实现，不能据此认定它就是历史计时源码。
- 定位命令：`dmd_knowledge.py locate --case helios-planner-increment`
- 关联实现：[Planner 的已测成本及结果资格](implementations.md#planner-calibration)；current_related；固定源码中的相关实现，供当前任务检查或适配；不是该历史计时所用源码的身份声明。
- 函数入口：[src/distillschedule/search/cost_model.py::CostModel](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/cost_model.py#L77-L195)
- 函数入口：[src/distillschedule/search/measurement.py::normalize_native](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/src/distillschedule/search/measurement.py#L18-L98)
- 来源：[sliortega295-ops/DMD-System / SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md#L1)；定位：报告范围 L1–L122；L1: 六个 workload：当前结果口径、旧审计勘误与源码阅读指南；L7: 1. 范围必须是六项，不是五项；L20: 2. 用户当前论文主表快照：A100 六项；L37: 必须更正：W3 的 113.59s 不是当前无 merged-LoRA 数字；L47: 不能再沿用旧审计的三种替代；L53: 3. 主表之后还有新结果：不要隐去，也不要直接替换旧分母；L67: 4. H100 六项均已有结果，不能只写 W1/W2 找到了；L86: 5. 自动规划器：最新证据是“保存计划实际执行并接近手工”，不是把全部系统收益算成搜索增量；原定位：section 5: manual vs saved plan

### 两波 GAS 必须重放同一逻辑操作的 selected-call 语义

- ID：helios-gas-trace-baseline-repair
- 状态：correctness_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：旧 48.551620 秒 anchor 第二波只连接最后一个 selected call；官方 strict-v4 两波均需三个 selected call，旧时间不能作为新合同分母。
- 经验：修正训练图要重建 baseline；因少做 backward 得到的旧快值不能成为优化目标或算作加速。
- 范围：官方同步退出的 Step10-A DP8 MB1/GAS2。
- 历史上下文：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"matched_comparison_contract": true}
- 收益条件：{"representative_measurement": true}
- 失败分类：different_backward_work
- 重试信号：两波 selected calls 已符合 strict-v4 语义
- 最小验证：核对每波三次调用和梯度连接，然后重建同合同 baseline
- 不应重复：不继续用少算逻辑工作的 48.55 秒旧分母
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case helios-gas-trace-baseline-repair`
- 来源：[sliortega295-ops/DMD-System / rb16_full5_2g6c_continuous_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_full5_2g6c_continuous_20260810.md#L87)；定位：观察中的数值文本锚点 L87；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### B32 的 B16 等效时间不是实际 B16 训练记录

- ID：livewan-b32-normalization
- 状态：source_contract / HISTORICAL_DIAGNOSTIC_NORMALIZATION
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：LiveWan B32 108.583634 秒可除二得到 B16-equivalent 54.291817 秒，但 slots16–31 复用同一批 feature rows；两个 B16 overlap/RNG-elision 试验还略慢。
- 经验：报告真实 batch、样本身份与资源；算术归一化可用于描述，不能替代相同 batch 和数据契约实测。
- 范围：LiveWan continuous B16/B32 historical probe。
- 历史上下文：{"workload": "W4", "model_family": "LiveWan", "training_mode": "training"}
- 正确性前提：{"matched_comparison_contract": true}
- 收益条件：{"representative_measurement": true}
- 失败分类：not_established
- 重试信号：新比较具有真实等量且不同样本的逻辑工作
- 最小验证：核对 slots、feature rows、RNG 与 batch denominator
- 不应重复：B32 除二不能掩盖重复 feature rows
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case livewan-b32-normalization`
- 来源：[sliortega295-ops/DMD-System / LIVEWAN_CONTINUOUS_B16_B32_20260901.md](https://github.com/sliortega295-ops/DMD-System/blob/54351707288c8d8a8d2efd789ece708852a07033/docs/experiments/LIVEWAN_CONTINUOUS_B16_B32_20260901.md#L8)；定位：观察中的数值文本锚点 L8；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### W6 C1 与 C2 相差很大，不能混用周期分母

- ID：h100-w6-warmup-cycle-denominator
- 状态：source_contract / HISTORICAL_H100_SINGLE_TRIAL_REVIEW
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：candidate C1 359.380 秒而 C2 278.886 秒；startup/warmup 与持续周期分开。
- 经验：所有 arm 必须比较同一个窗口与周期序号；将热候选对冷 baseline 会夸大收益。
- 范围：W6 same0419 matrix 的计时字段，不构成新重复试验。
- 历史上下文：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W6", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"matched_comparison_contract": true}
- 收益条件：{"representative_measurement": true}
- 失败分类：not_established
- 重试信号：运行时长改变使 warmup 摊销发生变化
- 最小验证：分别报告 startup、C1、C2 和 full-run 时间
- 不应重复：359 秒 C1 与 279 秒 C2 不能互作等价分母
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case h100-w6-warmup-cycle-denominator`
- 来源：[sliortega295-ops/DMD-System / report.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W6/artifacts/w6_resident_adaln16_staging_20261015_r1/reports/completed_same0419_matrix_r1/report.md#L7)；定位：观察中的数值文本锚点 L7；同时阅读同节合同、状态与限制。原定位：该结果及 qualification / NOT_RUN 边界

### ResearchBoost：共享机器的慢 baseline 会夸大候选收益

- ID：rb-timing-contaminated-baselines
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：公开 lessons 记录受其他作业影响的 baseline 窗口产生虚高加速；小差值与 run-to-run 噪声难分离。
- 经验：保留每个窗口、交错对照与资源竞争证据，既防候选受扰也防 baseline 受扰；阈值按当前合同预先定义。
- 范围：外部测量经验，不照搬其固定百分比或重复次数。
- 历史上下文：{"training_mode": "training"}
- 正确性前提：{"matched_comparison_contract": true}
- 收益条件：{"representative_measurement": true}
- 失败分类：not_established
- 重试信号：资源隔离与交错 baseline 的波动可控
- 最小验证：记录并发作业/资源状态，重测 matched pair 与噪声
- 不应重复：受污染慢 baseline 不能作为加速分母
- 代码状态：method_only；这是验收、计量或决策方法；本条不交付可直接接入的 runtime 优化实现。请按最小验证检查当前项目。
- 定位命令：`dmd_knowledge.py locate --case rb-timing-contaminated-baselines`
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / lessons.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/lessons.md#L1)；定位：报告范围 L1–L33；L1: Results and lessons across tasks；L3: Four tasks, four ceilings；L12: General lessons；原定位：Measurement
