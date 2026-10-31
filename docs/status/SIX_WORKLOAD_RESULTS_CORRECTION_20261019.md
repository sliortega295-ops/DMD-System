# 六个 workload：当前结果口径、旧审计勘误与源码阅读指南

**请先读本页，再审计性能。** 本页针对旧版 `DMD_six_workload_performance_audit.md` 只读早期 registry／少量历史报告，进而将历史中间结果当成项目最新进展的问题。旧记录不删除，但不能据此说项目只有五项、H100 W3–W6 没结果或 W6 没有保存计划实跑。

本页汇总的是**已存在的历史实测、用户确认的论文汇总快照以及各自限制**。本次仅补文档和小型证据，没有新 GPU 实验、改训练数学、放宽数值门或合并 main。不同对照的分母不能互换；最新发布不等于六项已完成严格数值／长期收敛验收。

## 1. 范围必须是六项，不是五项

| ID | Workload | 不可合并或混用的合同 |
|---|---|---|
| W1 | Wan1.3B／Wan2.1-1.3B-AR | 1.3B→1.3B，Helios 分层自回归 No-GAN DMD |
| W2 | Helios14B No-GAN | 分层自回归，无 GAN，G1/C5/EMA5 |
| W3 | Helios14B GAN-on | GAN-G/GAN-D 分支，原 GAN 更新／EMA 合同，不是 No-GAN EMA5 |
| W4 | Self-Forcing DMD／LiveWan | 14B Teacher→1.3B Student，流式自回归；A100 论文 full 为 C51，不是 C34 |
| W5 | Audio–Video DMD (Full-Sequence)／H3 Grid1000 | 全序列 AV，4-step、G1/F5、LoRA128；Grid1000 是近似 sigma 分布 |
| W6 | Autoregressive Audio–Video DMD／Original DiffSynth H3 | 原生自回归 AV，8-step、G1/F1、LoRA64、previous+current chunk；不能套 W5 Grid |

旧五个 canonical profile 与新 W6 opt-in adapter 可以同时存在。只数旧 registry 文件不能判断整个项目只有五个 workload。

## 2. 用户当前论文主表快照：A100 六项

硬件：2×8 A100-40GB。**下面逐字保留用户当前截图列值，不伪装成本次新测或同一 HEAD 的统一配对。** 未在本页重新绑定的原始 run ID／主表 training HEAD 留空，见机器可读数据；特别是 RealHF-like 列不能仅凭名字推断 TP/PP/FSDP 拓扑或已完成六项新 PP 基线。

| ID | FSDP2 baseline／reference (s) | Resident／offload reference (s) | RealHF-like reference (s) | DMD-System，表中值 (s) | 相除的参考比值 |
|---|---:|---:|---:|---:|---:|
| W1 | 36.175 | 35.05 | 39.43 | 20.588 | 1.757× |
| W2 | 109.17 | 688.73 | 156.25 | 82.99 | 1.315× |
| W3 | 164.30 | 762.44 | 203.74 | **113.59，N70-ON 历史值，见更正** | **1.446×，不能标作最新 N70-OFF 收益** |
| W4 | **84.26** | 69.25 | 62.38 | **42.45** | **1.985×** |
| W5 | **911.77** | — | — | **767.02** | **1.189×** |
| W6 | 320.00 | — | — | 255.671 | 1.252×参考比值 |

W2/W3 的 688.73/762.44s 是旧角色切换／offload 路径，不能称“全部模型完整 GPU 常驻”。表中空项是未提供，不是零秒。W6 旧基线与优化臂的计时／精度合同仍需独立核对；此处相除不是新增 matched speedup。用户当前主表也不是自动 Planner 相对手工版的增量收益表。

来源：本轮用户确认的截图，SHA 与结构化列值已保存到 [results.json](../../artifacts/publications/20261019/results_correction/results.json)；[可读 CSV](../../artifacts/publications/20261019/results_correction/A100_user_reported_main_table.csv)。截图本身未上传。

### 必须更正：W3 的 113.59s 不是当前无 merged-LoRA 数字

已实际完成 N70-OFF retime：`gan_nomerge_down40_native_c2_world16_20261013_r1`，训练源 `ba52a2446b7f090cd31af9504679e68770d32603`，C1=117.716096268s，**原生 C2=116.253523015s**。16/16 rank 的实际 merge 次数全部为 **0**，保留 down40 和原 GAN 工作量。

- 相对纸面 164.30s 参考的时间比为 **1.413291×**；相对本轮 163.90s 参考为 **1.409850×**。这不把两个来源变成新配对。
- 原运行在计时结束后的附加 observer 上触发 schema KeyError。原 FAIL 回执保留，CPU 离线复核确认 timer／调用／版本及 merge=0 已完成；**不是把失败回执改成端到端 PASS**。
- 严格全状态数值仍 REVIEW，`formal_speedup_eligible=false`。更正论文数字时应附该边界，不能把113.59s继续写成“当前正确性已通过的无 N70 最佳值”。

原始小报告现已一并发布：[完整说明](../../artifacts/publications/20261019/results_correction/evidence/gan_nomerge/gan_nomerge_retime.md)、[机器回执](../../artifacts/publications/20261019/results_correction/evidence/gan_nomerge/gan_nomerge_retime.json)、[计时外 schema 修复 CPU 证据](../../artifacts/publications/20261019/results_correction/evidence/gan_nomerge/post_timer_schema_bug_fix_CPU.json)。没有为本页重跑，也没有追改旧文件。

### 不能再沿用旧审计的三种替代

1. **W4 主表是 FSDP2 84.26→42.45s。** 62.4535078→42.3895898s 属于另一组 hybrid-sequential 辅助对照；它约1.47×，不能替换主表分母，也不能将62.4535改名61.99。旧 padding／物理调用审计意见单列，不据此擅自删除用户指定的 FSDP 主对照，更不能宣称严格审计已全部解决。
2. **W5 812.029→772.332s 只是一段缓存增量。** 不能用它代替用户主表911.77→767.02s的系统汇总，也不能把二者混成同源新实验。Grid1000 近似必须明确；不是 exact-equivalent continuous DMD。
3. **W6 255s 结果和保存计划实跑均存在。** 不能因为早期“五配置”文档没写就标不存在；但旧316s首次 update、320s参考与新C2要分别标计时范围，不能拼成未做过的公平配对。

## 3. 主表之后还有新结果：不要隐去，也不要直接替换旧分母

这些是另有来源的增量／消融，不是把各次最快值拼成“最终统一成绩”。

| 范围 | 本行实际比较 (s) | 来源及边界 |
|---|---:|---|
| W1 手工组合候选 r34 | 19.720700→19.549642 | [r34](../../artifacts/publications/20261019/evidence/state_retention/r34/conclusion.json)，`a5b550af` 加 Git 外 packet；仅已知 EMA steps11–20 两周期，REVIEW、非生产通用晋级，不直接替代20.588s主表 |
| W2 阶段内部分参数保留 r36 | 83.970328→83.023065 | [r36](../../artifacts/publications/20261019/evidence/state_retention/r36/conclusion.json)，`e2374f08`；本行同源增量约0.947s，非109.17s新基线配对 |
| W3 冻结 FFN-up replay 保留 | 117.074207→116.673757 | [六项 r1](../../artifacts/publications/20261019/evidence/state_retention/r1/conclusion.json)，`155ffbb5`；约0.400s增量，REVIEW，不与113.59s无条件叠加 |
| W4 重新选择仅A／完整仅B | 仅A **82.330023**；完整仅B **42.123914** | [R3](../../artifacts/publications/20261019/evidence/livewan_r3/reports/conclusion.json)，新两格同源 `053bdcad`；旧83.853875 baseline与42.407027联合full来自父源，不能伪称新源重跑全四格 |
| W6 同 FA3 的 metadata barrier 消除 r46 | 262.384847→253.740367 | [r46](../../artifacts/publications/20261019/evidence/state_retention/r46/conclusion.json)，`69a94649`加外部 packet；同FA3下约8.644s增量，REVIEW，非把旧SDPA255.67当新配对分母 |

W4 旧79.15s仅B是兼容子集；72.12s仅A包含 CPU optimizer offload；都不能代替后续完整B／重选A结果。完整B的4.375MiB是末边界采样余量，不是证明全程容量宽裕。[总源码索引](GITHUB_SOURCE_SNAPSHOT_20261019.md)列明这些版本的代码与外部 packet。

## 4. H100 六项均已有结果，不能只写 W1/W2 找到了

8张物理 H100-80GB，16个独立逻辑样本槽位。不是16张物理H100，也不与A100秒数直接做硬件比。

| ID | 最新归档对照定义 | 对照 (s) | 优化 (s) | 观察比值 | 精确计时源 |
|---|---|---:|---:|---:|---|
| W1 | Resident reference | 52.066607 | 37.669795 | 1.382× | `7a668bcb`；后续97b8 norm端到端NOT_RUN |
| W2 | 同源 retained-G，内部同步开／关 | 95.534631 | 94.311433 | 1.013× | `8fc70e8c` |
| W3 | 官方角色切换／优化实现 | 192.986046 | 117.343507 | 1.645× | `624515e7` |
| W4 | Resident／Teacher常驻 | 45.361398 | 45.257227 | 1.002× | `4d7c589a`；后续分析源1b8236c2 |
| W5 | 同源Grid1000 FSDP／GPU Attention cache | 638.522138 | 539.718383 | 1.183× | H3仓库 `9bfd3eda` |
| W6 | 同源原生FSDP／FA2 candidate | 320.666055 | 278.886442 | 1.150× | `0419b96b` |

W2 较早138.371948→98.213873s保留为旧同批对照；138.371948→最新94.311433s约1.467×只能标跨revision描述性比值。W1更早56.798→37.511s三组配对也保留，但不是最新Resident单对的分母。不能为了更新一列把不同批次拼接。

W5 Grid近似与W6 continuous AR严格分开；W6当前成功路径是FA2，FA3完整模型门仍FAIL；不能声称测到了60% MFU。以上主要是n=1诊断，不升级全状态／长轨迹数值结论。

**代码与小证据均已补发：**[H100六项来源索引](H100_SOURCE_SNAPSHOT_20261019.md)、[精确结果CSV](../../artifacts/publications/20261019/h100/measured_results.csv)、[manifest与原始小报告](../../artifacts/publications/20261019/h100/source_manifest.json)。不要只查9月30日的旧报告后写“H100 W3/W4/W5/W6未找到”。

## 5. 自动规划器：最新证据是“保存计划实际执行并接近手工”，不是把全部系统收益算成搜索增量

以下是各行自己的同源原参数、无profiler原生C2。不是相对普通FSDP的主表，也不是所有组合全局最优或优于手工的证明。

| ID | 手工 C2 (s) | 保存计划实跑 C2 (s) | 相对手工差距 | 状态 |
|---|---:|---:|---:|---|
| W1 | 20.040550 | 20.103775 | +0.3155% | 16/16，REVIEW |
| W2 | 84.867318 | 84.958650 | +0.1076% | 16/16，REVIEW |
| W3 | 116.805525 | 118.764660 | +1.6773% | 16/16，REVIEW，显式1GiB容量政策 |
| W4 C51 | 42.558984 | 42.496301 | −0.1473% | 16/16，REVIEW |
| W5 | NOT_RUN | NOT_RUN | — | 后向采样数值FAIL未解决，没有绕门做速度配对 |
| W6 | 255.501369 | 255.442535 | −0.0230% | 16/16，REVIEW，原容量政策 |

[逐行source、原始证据与失败边界](../../artifacts/publications/20261019/evidence/planner/six_workload_native_match.json)。这比旧审计的“Wan/LiveWan初次重选、Helios93/91s、H3只有769s独立C2”更新；不能继续拿旧描述否定后来的保存计划实跑。也不能反过来声称W5失败已解决、H100自动实测闭环全部完成或main已合并。

## 6. 截图中的消融列有算术未同步，且不能用算术替代实测

截图中仅布局耗时更新后，后两项仍沿用了旧分解。**只按现显示的三列做差**，应为下表；这不是新增或认可同源因果消融。

| ID | 基线→仅布局→完整，截图三列(s) | 布局差值(s) | 状态追加差值(s) | 总差值(s) |
|---|---|---:|---:|---:|
| W2 | 109.17→100.48→82.99 | 8.69 | 17.49 | 26.18 |
| W3 | 164.30→150.93→113.59 | 13.37 | 37.34 | 50.71 |
| W5 | 911.77→877.02→767.02 | 34.75 | 110.00 | 144.75 |

不能继续同时展示W2的1.69/24.49、W3的0.37/50.34、W5的6.75/138.00而又保留上面新的中间列。若中间值来源尚未绑定，则先标“用户表中值／待来源核对”，不要为使贡献均衡而反推实测或把条件边际相加。
W1截图37.017→25.159→20.364是一组不同于主表36.175→20.588的嵌套消融；应保留自己的分母。W4后续已新增R3真实仅A/完整B结果，不能仍写“仅B入口完全没有”。

## 7. 网页版应按这个顺序重新整理

1. 以本页固定 **六项范围与当前用户主表快照**，但显式保留W3已知更正、W5近似、W6参考比值和缺失source绑定。
2. 看[总源码索引](GITHUB_SOURCE_SNAPSHOT_20261019.md)和[H100索引](H100_SOURCE_SNAPSHOT_20261019.md)，进入对应**固定训练提交**，不是只停留在main、registry或最新分支名。
3. 从每行manifest里的原生计时／run ID／配置hash追证据；区分系统主表、同源增量、跨revision参考、手工/保存计划配对和profile时间。不要将NOT_RUN改0，不把finite/census替代数值等价。
4. 已发源码在独立研究分支；main仍是早期入口。**“main没集成”不等于“优化代码未上传”，“代码上传”也不等于“统一运行时全部验收完成”。** 不要用旧negative/pending报告覆盖较晚真实结果，也不要删除真实FAIL。
5. 模型、输入大文件、checkpoint、18GiB Grid表、大型profile留远端。另一个100-cycle loss任务不属于此次性能源码发布；本页不声称已完成六项真实长轨迹或质量证明。

无需为整理这份纠偏重跑GPU或估算缺失成绩。没有可靠绑定的项准确列缺口；已存在的后续实测必须先读完再下“缺结果”的结论。
