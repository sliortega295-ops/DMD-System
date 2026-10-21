# 六个 workload：状态保留与重算优化实测

## 结论先行

六项均已实际尝试，不是仅 CPU/source 接通。四项在原有优化版上观察到小幅 C2 收益；H3-AR 的 GPU 缓存候选变慢，未采用；H3 全序列完成两项数值诊断，原数值 FAIL 尚未关闭，未绕过门做速度配对。

本表是**当前优化控制组 → 本轮同源候选**，不是原始 FSDP2 baseline → DMD-System 的论文总加速。每臂单次初始化连续两 cycle，C1 预热，C2 原生无 profiler；不同源码的增量不相加。

| Workload | 本轮机制 | 控制 C2 (s) | 候选 C2 (s) | 节省 (s) | 候选 allocated 峰值 (GiB) | 状态 |
|---|---|---:|---:|---:|---:|---|
| Wan1.3B | Critic 最后4 block 保留原 autograd 激活，省 checkpoint 重算 | 20.324208 | 20.002455 | 0.321753 | 26.692 | 单对正向观察；REVIEW |
| Helios No-GAN | Critic 最后2 block 保留原 autograd 激活 | 84.840473 | 84.345194 | 0.495279 | 22.733 | 单对正向观察；REVIEW |
| Helios GAN-on | 在现有 down40/FA3/N70-OFF 上保留两个 frozen FFN-up 结果 | 117.074207 | 116.673757 | 0.400450 | 37.179 | 单对正向观察；REVIEW |
| LiveWan | C51 上增加 Critic frozen FFN-down checkpoint 结果复用 | 42.486977 | 42.243719 | 0.243258 | 35.152 | 单对正向观察；REVIEW |
| H3 Full Sequence Grid1000 | 真实相同输入 raw FA3 backward：block49 / block10 | — | — | — | — | 数值定位；没有速度准入 |
| H3 AR DiffSynth | 最后1个 block 的 Attention 结果从 CPU 改留 GPU | 255.363758 | 260.582584 | -5.218826 | 34.178 | 负收益，不采用；REVIEW |

所有已完成计时臂各16/16 ranks，调用、样本/输入、版本、finite 与相应容量合同通过；这不等于数值严格等价。REVIEW 原样保留，不改阈值，没有将本轮结果自动晋级 main 或规划器。未采集的新 kernel/NCCL 时间、MFU、完整生命周期和长轨迹不补估计值。

## Wan 的进一步边界：不是显存越满越快

1. **Whole-block tail8：CUDA OOM，未得到候选 C2。** 原生报错申请378MiB，driver 仅374.38MiB；报文分别记录 allocated30.09GiB、private pool2.77GiB、reserved-unused7.07GiB。它们不是三项可直接相加的独立内存。保留完整失败与清理；未重试/改余量。原错误处理未提供精确 op/cycle 栈，该位置标未观测，而不猜 FFN/某个 block。
2. **更细的 frozen FFN-down 复用真实跑通，但更慢。** 同源 `fa249cc03447ed8263f9b16b1bf048ab7a41111e`：retain4 C2=20.028683s，projection-only C2=20.169803s，慢0.141120s。allocated 26.692→23.420GiB，reserved 33.943→30.867GiB。只说明本硬件上的时间/内存折中，不外推低显存卡可 fit。
3. 每 rank 真实 Python eager/capture 共120次 frozen GEMM 原算、120次缓存读、0 failure，覆盖30 block。**不是人为填每周期/每次 CUDA Graph replay 的计数。** 原 full4308个 trainable/Adam/EMA tensor 比较仍 REVIEW，max_abs=1.18473854e-05，不声称长期等价。

## 为什么有些全局峰值不变

LiveWan 的 Generator/Score 必须分开看：本轮 Score allocated17.867→21.073GiB，全局峰值仍35.152GiB。保留更多 Score 状态确实执行了，但整个流水 C2 仅省0.243s。Host receive wait 是 CPU 包围时间，不是已测的 exposed GPU bubble，不能直接加到关键路径。

H3-AR 的软件记账确实显示 CPU copy 请求字节减少，但 C2 变慢5.219s。它不是新 CUPTI memcpy 时间，也没有证据把回退归为 allocator cache-flush；不能把“少搬字节”直接当收益，更没有据此盲目扩大 GPU cache。

## H3 全序列数值问题仍未闭合

两项实际 world16 两周期诊断均16/16。相同完整 dout/Q/K/V/out/LSE 下：block49 DQ nL2约3e-6至1e-5；block10约1.67e-5。DK/DV 在所测重复中 bitwise 一致。deterministic=True 两次 DQ 仍不同，不能把此 flag 当作本构建的确定性保证。

这只证实局部 backward 存在很小的非确定性；**尚未说明此前6–11% loss 波动无害或预取完全等价**。旧16点 sketch 与全张量范数不可按倍数比较。W5 没有新 C2 速度、cost 校准或数值 PASS。后续优先查真实 upstream dout/全张量缓存 round-trip，以及 BF16 梯度/Adam 的误差放大；原用户“不做 H3 速度配对”保持。

## 源码、证据与边界

- 每个 JSON 详报绑定 source HEAD、原生 rank refs、配置/输入 hash、计数和资源回执；`six-progress.csv/json` 可机器读取，`conclusion.json` 汇总来源与最终清理。新增源码均为默认关闭的实验路径，旧分支/失败未覆盖。
- 部分派生 run_id 沿用旧模板的 `tail8` 名称前缀；**不要据名字推断机制**。真实唯一证据为 source HEAD + 独立 artifact_root + config SHA + actual strategy/census。所有旧 run 原样保留，未重写为新源。
- 24次启动包含资格、诊断及2次失败，不是24次统计重复。W1/W2/W3/W4/W5/W6 分别6/4/4/6/2/2次，旧账保留，未换名清零。没有 profiler 速度、两周期平均或挑选多次最快值。
- 最终两个节点 GPU compute apps 为空，节点锁及 root scheduler 锁可取得并释放；本轮 host/cgroup oom、oom_kill 增量0。**有一项明确 CUDA OOM 失败**（tail8），不能称本轮全部容量 PASS。最终空闲只是收尾观测，不代表未来永久占用权。
- 大型 raw/state 留远端持久目录 `/share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/state_retention_six_workloads_20261017/r1/`；本地只小报告/索引/源补丁。未 push、未改 main 默认、未动 Downloads/飞书/论文，未联系 H100。

## 未完成不隐去

本轮六项“逐一尝试”完成，不代表六项都已新提速、严格数值验收或整个论文任务完成。W5 数值定位仍开放；所有 REVIEW 的噪声底/长期验证尚缺；新实验候选的自动 planner 搜索/成本准入未完成。原主线的阶段/storage 生命周期、纯布局消融、PP/FSDP/DP、69帧 timer 等不因本轮而自动标完成。
