# H3 AR：相同FA3只恢复原生metadata同步

Source: `69a9464997de16deafd5786338f538ba7930d9a5`。手工 opt-in，不接自动规划器。两个arm使用相同FA3、grad/checkpoint缓存和原生timer；只在候选恢复原cu_seqlens.tolist同步，实际350次/rank/cycle；没有AdaLN bank。

| 配置 | C1(s) | C2(s) | Peak allocated(GiB) | Peak reserved(GiB) | Ranks |
|---|---:|---:|---:|---:|---:|
| manual-control | 256.572890 | 262.384847 | 33.728 | 38.512 | 16/16 |
| manual-child | 256.167687 | 253.740367 | 33.728 | 38.445 | 16/16 |

单对原生C2节省 8.644480s；比值 1.034068x。

两节点原始 owner/capacity/OOM 检查保持。原checkpoint Attention缓存、调用/更新/输入/RNG、每rank每cycle350次FA3与0/350次实际metadata同步计数通过；这不是全部参数/Adam逐元素或长期轨迹等价证明。REVIEW 保留，不冒充严格数值或统计结论。

原始回执保留远端 `/share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/state_retention_six_workloads_20261017/r46/native`；本地 `actual-native-pair.json` 为小型逐rank派生证据。

## 对原最优版的边界
本对照只隔离相同FA3路径里的同步恢复：262.384847→253.740367s，少8.644480s。历史原始SDPA同训练HEAD记录255.528303s，仅作provenance；与候选差1.787935s，不是另一组新鲜同源完整A/B。数值REVIEW、单对描述性，不宣称长期等价或显著性。未晋级默认值/调度器。
