# No-GAN：阶段内部分权重驻留（R36）

同源 `e2374f084977a31ef1fd06a2e81a5e9582af818c`，不接自动搜索或调度器。默认路径和 canonical profile 不变。两臂都使用相同五次 B1 rollout 的连续 scope，仅候选保留前24个 FSDP block 的完整参数；不是 activation cache，也不是 merged-LoRA。

| 原生两-cycle | C1(s，仅预热) | C2(s) | 峰值 allocated(GiB) | reserved(GiB) |
|---|---:|---:|---:|---:|
| 原逐block reshard | 91.612056 | 83.970328 | 26.418 | 37.336 |
| 前24 block 阶段驻留 | 90.375074 | 83.023065 | 30.911 | 37.400 |

描述性节省 **0.947264s / 1.128%**，单对比率1.011410×。不作统计显著性、长期等价或生产默认晋级声明。资格运行单列，未冒充原生 C2。

16/16、32 rank-cycle，G4/C20/EMA20；每scope27次真实 G forward；24单元在scope末仍展开，恢复后0个，版本/grad-enabled违规0。原样本/RNG/精度/调用/更新与容量检查通过。数值 REVIEW，原比较器 coarse REVIEW，最差采样 nL2 0.000109120779；不是全状态/长轨迹证明。

两节点无新增CUDA/host OOM，原owner cleanup及新鲜GPU空闲/节点锁复核通过。原2GiB政策约束allocated highwater，不是driver-free2GiB；本次最小采样driver空闲control50MiB、candidate116MiB，不能声称始终有2GiB物理空闲。采样host/cgroup峰值分别82.984/82.485GiB，不是连续RSS峰值。

结论：确实减少了重复物化机会，但仅获得约1%整周期收益；不能把预测减少的all-gather数当作已测NCCL开销。保留为可组合小收益，不继续扫24→32等参数。下一步用当前timeline确认剩余compute/launch大头，而不是继续以更多显存换同类通信。

证据：`nogan-pair-summary.json`、4阶段原始回执、`artifacts/fresh-cleanup.json`、source.patch及源码清单。累计92个owner phase，含失败/组件/诊断；旧账不清零。未push、未改main/Downloads/H100。
