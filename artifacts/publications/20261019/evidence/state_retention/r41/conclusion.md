# H3 AR：精确冻结 AdaLN bank 原生对照

Source: `ffdea1695fc7c2da7444eca2f27062356e6f4f40`。手工 opt-in，不接自动规划器；没有删除13B参数、没有Grid、merged-LoRA或改训练调用。

| 配置 | C1(s) | C2(s) | Peak allocated(GiB) | Peak reserved(GiB) | Ranks |
|---|---:|---:|---:|---:|---:|
| manual-control | 257.450137 | 255.602137 | 33.728 | 38.484 | 16/16 |
| manual-child | 258.752205 | 256.312004 | 33.428 | 38.422 | 16/16 |

单对原生C2节省 -0.709867s；比值 0.997230x。

两节点原始 owner/capacity/OOM 检查保持。缓存逐输入精确相等与冻结分片版本、调用/更新/输入/RNG通过；这不是全部参数/Adam逐元素或长期轨迹等价证明。REVIEW 保留，不冒充严格数值或统计结论。

原始回执保留远端 `/share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/state_retention_six_workloads_20261017/r41/native`；本地 `actual-native-pair.json` 为小型逐rank派生证据。

## 晋级决定

**不晋级**：C2反而慢0.710s，且DMD loss最大相对偏离约29.72%、Critic loss约6.90%；精确缓存输入检查不能替代全状态数值验收。没有宣称训练等价，原控制组仍保留。两节点已实时复核GPU为空、节点锁可获取。
