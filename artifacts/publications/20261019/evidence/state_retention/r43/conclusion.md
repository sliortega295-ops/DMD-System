# H3 AR：H56 未缓存 rollout FA3 原生对照

Source: `69a9464997de16deafd5786338f538ba7930d9a5`。手工 opt-in，不接自动规划器；仅替换未缓存的7次Student no-grad主Attention，保留独立padding段、原grad与checkpoint缓存；未组合失败的AdaLN bank。

| 配置 | C1(s) | C2(s) | Peak allocated(GiB) | Peak reserved(GiB) | Ranks |
|---|---:|---:|---:|---:|---:|
| manual-control | 261.255514 | 255.528303 | 33.728 | 38.486 | 16/16 |
| manual-child | 258.457772 | 257.709696 | 33.728 | 38.490 | 16/16 |

单对原生C2节省 -2.181393s；比值 0.991535x。

两节点原始 owner/capacity/OOM 检查保持。原checkpoint Attention缓存、调用/更新/输入/RNG与每rank每cycle350次实际FA3计数通过；这不是全部参数/Adam逐元素或长期轨迹等价证明。REVIEW 保留，不冒充严格数值或统计结论。

原始回执保留远端 `/share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/state_retention_six_workloads_20261017/r43/native`；本地 `actual-native-pair.json` 为小型逐rank派生证据。

## 晋级结论
不晋级：完整原生C2慢2.181393s；R42局部约10%收益未兑现。下一步仅一次有界当前候选C2诊断，核实真实Attention、NCCL重叠与输入stride，不把微基准收益冒充训练加速。
