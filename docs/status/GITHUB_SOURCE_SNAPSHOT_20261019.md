# DMD-System 源码与实测证据索引（2026-10-19）

**这是源码发布和证据归档，不是把所有研究分支合并到 main。**

**H100 六 workload 后续补充发布：**请同时读 [H100 源码／历史实测索引](H100_SOURCE_SNAPSHOT_20261019.md)。它补齐 W1 后续 CPU 候选、W3 独立来源分支、W4 Teacher 常驻、W5 GPU Attention 缓存和 W6 原生 FA2／状态管理源码，并归档 Git 外执行 packet；不是重新启动 H100。下面原发布中的“仅固定依赖 8fc”范围不代表这份后续补充的全部范围。

核对时 main 为 `e69f4bf8b8bb64f54542fea74c452b53e090b526`，仍是早期五 workload 集成入口。
下面的优化、搜索、计划执行器在独立分支；原实验提交身份保持不变。
本次只补 GitHub 源码与小型证据，没有新 GPU 实验，没有更改 main、模型数学、canonical 默认值或验收阈值。

## 1. 应从哪里读代码

| 内容 | 固定实验/源码提交 | 分支 | 范围 |
|---|---|---|---|
| 搜索器、成本绑定、计划执行器 | [ea291293](https://github.com/sliortega295-ops/DMD-System/tree/ea29129342741b54e59c4773e07454f541fadcc7) | `codex/wan-state-policy-planner-20261017` | 实测成本→事件前沿搜索→保存计划→原生执行；含 C51、H3 AR、A100/H100 适配以及 Wan 状态策略成本键 |
| W1 Wan1.3B 四格与 FSDP 兼容 B | [7fa27ff8](https://github.com/sliortega295-ops/DMD-System/tree/7fa27ff8cab0a483a65fd857e0df762d46194d6d) | `codex/w1-r2-fsdp-compatible-b-20261019` | 保留原布局的 B 是兼容子集，不能称完整 B；C1 元数据通信修复不更改训练 Tree 通信 |
| W1 手工 large-shape attention 候选 | [a5b550af](https://github.com/sliortega295-ops/DMD-System/tree/a5b550af56b3eb7e8b5f570be94540134fc49ad5) | `codex/wan-large-attention-backend-20261018` | 手工 opt-in；r34 EMA 外部补充源码另见下面归档；不是已进入自动规划器的组合 |
| W2 Helios No-GAN 状态保留 | [e2374f08](https://github.com/sliortega295-ops/DMD-System/tree/e2374f084977a31ef1fd06a2e81a5e9582af818c) | `codex/nogan-partial-parameter-retention-20261018` | G 阶段内前24 block 参数保留与 Critic 有界保留，单对诊断收益 |
| W3 Helios GAN 状态保留 | [155ffbb5](https://github.com/sliortega295-ops/DMD-System/tree/155ffbb585356a29b544573a391eb456b790a15a) | `codex/state-retention-six-20261017` | 冻结 FFN-up replay 保留；不是已否决的 merged-LoRA |
| W4 Self-Forcing 完整 B／重新选择 A | [053bdcad](https://github.com/sliortega295-ops/DMD-System/tree/053bdcad2cf0e5388738156599f96a6ec5905518) | `codex/livewan-r2-capacity-offload-20261019` | 完整 C51 B 接到原 FSDP16；独立 G8-FSDP8 仅 A，原始 OOM/offload/子集记录保留 |
| W4 tail4 独立候选 | [0dfce74b](https://github.com/sliortega295-ops/DMD-System/tree/0dfce74b66499419fdf38ec40b5e621544d8b904) | `codex/livewan-c51-tail4-census-20261017` | 实际重放计数修复；不是完整 B 的同义词 |
| W5 H3 Full-Sequence/Grid1000 诊断 | [285b31e3](https://github.com/sliortega295-ops/DMD-System/tree/285b31e3ec4789d7f98816265f19f7853827b3e2) | `codex/h3-full-local-state-diagnostic-20261018` | 共享底座、Grid/cache 与数值差异定位；原 FAIL 保留，不发布新的 passing speed pair |
| W6 H3 AR/DiffSynth 候选 | [69a94649](https://github.com/sliortega295-ops/DMD-System/tree/69a9464997de16deafd5786338f538ba7930d9a5) | `codex/h3-ar-uncached-rollout-fa3-20261018` | ZeRO-3 adapter、exact frozen bank／uncached FA3 实验；r41/r43 慢于对照，r46 独立源码见下 |
| PP＋FSDP＋DP 原型 | [7cde0ff8](https://github.com/sliortega295-ops/DMD-System/tree/7cde0ff8101725d8bab2843c52dcde127b09525a) | `codex/wan-grid-static-shared-20261014` | PP2×shard4×replica2、TP1 的 CPU 原型；不是旧 TP4 实测，不宣称六任务 GPU 完成 |
| 已否决 merged-LoRA 研究源码 | [c6987619](https://github.com/sliortega295-ops/DMD-System/tree/c69876197aeaccbab19b515b54d1022f07634422) | `codex/wan13b-shadow-noise30-20261012` | 30-cycle 轨迹/舍入研究，保留负结果，不能当作有效加速或默认候选 |

机器可读索引：[source_manifest.json](../../artifacts/publications/20261019/source_manifest.json)。
这些是**可追溯的不同源**，不是一个已经合并并重新验收的最终二进制。源码被上传也不表示某候选获准晋级默认值。

H100 adapter 还固定依赖独立原生源码 [8fc70e8](https://github.com/sliortega295-ops/DMD-System/tree/8fc70e8ce9f6e0243604812120119c9f4c078e55)，一并上传到 `agent/h100-helios-nograd-retain-20261014`。仅在本地读取与做 CPU 协议验证，没有连接或恢复 H100 队列。

## 2. 搜索保存计划是否真的执行过

以下是各行自己的同源、原参数、无 profiler 的单对原生 C2；C1 预热，不平均两周期。
这是“自动选出的计划与手工方案接近”的证据，不是相对 FSDP2 baseline 的加速表。

| Workload | 手工 C2 (s) | 保存计划实跑 C2 (s) | 相对手工差距 | 状态 |
|---|---:|---:|---:|---|
| W1 Wan1.3B（retain4） | 20.040550 | 20.103775 | +0.3155% | 16/16，REVIEW |
| W2 Helios No-GAN | 84.867318 | 84.958650 | +0.1076% | 16/16，REVIEW |
| W3 Helios GAN-on | 116.805525 | 118.764660 | +1.6773% | 16/16，REVIEW，显式1GiB容量政策 |
| W4 Self-Forcing C51 | 42.558984 | 42.496301 | −0.1473% | 16/16，REVIEW |
| W5 H3 Full-Sequence/Grid1000 | NOT_RUN | NOT_RUN | — | sampled 数值 FAIL 未解决，未绕过门做速度配对 |
| W6 H3 AR/DiffSynth | 255.501369 | 255.442535 | −0.0230% | 16/16，REVIEW，原容量政策 |

[逐行 source 与证据](../../artifacts/publications/20261019/evidence/planner/six_workload_native_match.json)，[CSV](../../artifacts/publications/20261019/planner_native_matches.csv)。
C34 的 46.8943／47.1225s 另存，只是历史注册入口配对，不能替代 C51。
H100 仅接入代码和 CPU 验证，本次没有接触或补跑 H100 GPU。

搜索是已支持候选/实测成本范围内的受约束事件前沿搜索，而非 unrestricted global optimum。
已有小规模 oracle 对照不证明所有分辨率、布局、kernel 组合全局最优。
新输入必须重新校准、满足资产/源/容量/语义准入；不能用历史总耗时充当新组件成本。

### 规划器阅读与复现入口

请在 **ea291293** 对应源码读取：
- `docs/runbooks/native_planner_workflow.md`
- `docs/runbooks/wan_compute_state_planning_v1.md`
- `docs/runbooks/livewan_c51_planner_v1.md`
- `docs/runbooks/original_h3_ar_native.md`
- `src/distillschedule/search/frontier_schedule.py`
- `scripts/distillctl.py` 的 `workloads native` 命令

原源码 runbook 的 NOT_RUN 表示编写时状态；本次归档的后续实际回执为更新后的测量依据。
[Wan r18 保存计划与资格资料](../../artifacts/publications/20261019/evidence/planner/wan_retain4/)。
参数、任务、计划 SHA 与两节点 effective contract 必须匹配，不允许降级成固定手工入口冒充计划执行。

## 3. 最近 W4 仅 A／完整仅 B 的结果

原合同：368×640×81、实际 global B16、continuous sigma、seed20260830，G1/C5/EMA0；2×8 A100。
一次初始化连续4周期，每格 n=1；C1预热，主数字只取 C2，无 profiler。

| 配置 | 训练 source | C2 (s) | allocated / reserved peak (GiB) |
|---|---|---:|---:|
| 既有 FSDP2 baseline | a62cafee | 83.853875 | 13.70 / 38.37 |
| 重新选择仅 A：G8-FSDP8、C8常驻、T8-FSDP8；B全关 | 053bdcad | 82.330023 | 18.96 / 37.66 |
| 完整仅 B：原 FSDP16 顺序，18项 C51 B 作用域适配 | 053bdcad | 42.123914 | 15.48 / 38.38 |
| 既有联合完整系统（未重跑） | a62cafee | 42.407027 | 35.16 / 38.07 |

[完整 R3 报告](../../artifacts/publications/20261019/evidence/livewan_r3/reports/conclusion.md)、[原生 summary 与实际计数](../../artifacts/publications/20261019/evidence/livewan_r3/)。
新两格同源；旧 baseline/full 不是在新源重跑，因此跨源比率只是描述性诊断。
完整 B 终态 driver-free 最低采样4.375MiB，不是整个周期的最小余量，也不是有宽裕容量的证明。
调用、版本、有限值、缓存生命周期检查通过，严格数值仍 REVIEW；没有用 finite 冒充全状态/长期轨迹等价。

论文指定主 baseline 仍为 **FSDP2 84.26s**，历史完整系统 **42.45s**；本次不会用 hybrid sequential 62.45s 替换它。
原先仅 B 的79.15s是兼容子集，不能称完整 B；仅 A 的72.12s包含 CPU optimizer offload，不能称纯布局最优。
W1 FSDP兼容 B 为28.477869s，是明确的子集，数值原状态 NOT_RUN 不升级。

## 4. Git 之外的真实实验源码也补齐

[manual_packets](../../artifacts/publications/20261019/manual_packets/) 保存逐字节原始脚本、SHA 与配置：
- Wan r30 流提交／r34 原始 EMA recurrence 的 CUDA Graph 手工组合；r34 C2 19.720700→19.549642s，**仅已知 EMA steps11–20 两周期**，不宣称生产通用支持。
- H3 AR r46 metadata barrier：同 FA3 路径262.384847→253.740367s，数值 REVIEW；历史 SDPA 255.528303s只是另一批 provenance。

这些脚本当时位于远端实验目录，不在训练 Git HEAD 中。只上传训练分支不足以复原它们，因此同时保存外部源码及原配置。
归档保留当时绝对路径与 SHA；不要直接在本发布分支上启动它们。按各 manifest 指定的训练 HEAD、依赖、资产与原有锁/容量 gate 部署。
主机搬迁需要显式重新绑定路径与派生 packet SHA，不能编辑冻结旧回执或用此绕过源检查。

全部测量保留自己的 REVIEW/FAIL/NOT_RUN、n、timer 和 source；组件时间不能作为整周期收益，各项条件边际不能相加当总加速。
18GiB AdaLN 表、模型、checkpoint、大型 trace、SQLite、nsys/ncu 原始文件未上传，只保留远端路径及小型证据。

## 5. 校验归档完整性

```bash
python3 scripts/distillschedule/check_source_publication.py
```

该检查只验证发布文件与已归档 source/SHA/六任务范围，不触碰 GPU，也不替代训练或集群验收。
历史报告中的 `push=false` / `no_push=true` 是**实验结束当时**的冻结记录；本次后续上传不追改这些原始回执。

本次发布前的聚焦 CPU 校验：LiveWan151通过；规划器182通过（另12项因缺显式native root先skip，随后绑定8fc原生源码的专门回归13/13通过）；外部脚本35通过；归档检查3通过。[机器回执](../../artifacts/publications/20261019/publication_validation.json)。这些是 CPU 检查，不替代 GPU 或数值验收。
