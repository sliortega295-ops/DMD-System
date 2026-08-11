# Iteration 308：Step10-A 与参数重分配联合优化正式结果

日期：2026-08-10
状态：**Measured / PASS_FORMAL_TARGETS_MET**

## 1. 冻结合同

- 官方 384×640，strict-v4 trace `654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3`。
- Step10-A：DP8 ZeRO-3，MB1/GAS2/global batch 16，两次 selected graph、两次 backward、一次 G2→G3。
- 目标阶段：4G+12C（每节点 2G+6C），Critic DP12 ZeRO-1 CPU optimizer，C10→C15，EMA10→EMA15。
- 未改变模型数学、selected calls、loss、batch/update 语义、optimizer/EMA 语义或 role placement。

## 2. 实现的物理调度优化

1. **Step10-A block-integral fetch**：只把 40 个 `HeliosTransformerBlock` 显式注册为 ZeRO-3 leaf，使每个 block 内部的参数获取按 block 聚合；embedding 等混合 dtype 模块仍走原已验证细粒度路径。该改动只改变参数 materialization 粒度，不改变计算图。
2. **Transition persistent pinned cache**：在排除的 cold preparation 中把 switch target 从 pageable CPU state 转为 persistent pinned state；计时区间内通过独立 H2D stream 直接搬运，并把 pinned allocator trim 延后到 Full5 语义边界。Step15-A timer 后在主线程同步清理，清理时间单独计量。没有后台 CUDA 清理，也没有把 cleanup 隐藏成训练时间收益。

Scalar `module_granularity_threshold=8M` 路径因连带选中 Timestep/PixArt 等模块导致 Float/BF16 dtype 错误，已作为失败保留并关闭；正式候选没有放宽 dtype 或 correctness。

## 3. 正确性

绑定 exact run `rb16_f5_iter307_explicit_block_leaf_correctness_20260810_001`，summary SHA-256 `1ff4b4a11d81053f3149d0a7f24bf9b7d6b51bc0bfc7f18ad8b22ea969f12377`：

- 16/16 ranks PASS；
- 每个 Step10 Generator rank 恰好 40 个 block leaves；
- G3/C15/EMA15；60/60 payloads；
- exact trace/version/queue wrapper 无错误；
- 无 GPU/cgroup OOM。

## 4. 三次独立正式低开销计时

| Round | Step10-A world max (s) | Transition (s) | Critic phase (s) | Full5 timer (s) | Post-boundary cleanup (s) | Inclusive (s) | samples/s | samples/s/GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 104.775 | 21.683 | 167.179 | 290.584 | 13.643 | 304.226 | 0.206481 | 0.012905 |
| 2 | 100.689 | 17.776 | 150.816 | 265.316 | 12.034 | 277.350 | 0.226146 | 0.014134 |
| 3 | 104.687 | 17.393 | 158.494 | 279.934 | 13.082 | 293.016 | 0.214336 | 0.013396 |

中位数：

- Step10-A：**104.687 s**（range 100.689–104.775，CV 1.84%）；
- Transition：**17.776 s**（range 17.393–21.683，CV 10.23%）；
- Full5 timer：**279.934 s**（CV 3.72%）；
- Full5 + cleanup：**293.016 s**；
- reported throughput：**0.214336 samples/s**，**0.013396 samples/s/GPU**。

## 5. 目标与可比性

- Step10-A 三轮均在 85–105 s，目标通过；相对同为 low-overhead strict-v4 Step10-only 的 Iter287 world-max 129.004 s，中位数减少 **18.85%**（1.232×）。
- Transition 三轮均在 10–25 s，目标通过。相对 Iter293 的 49.839 s historical exact/profile anchor，原始比值为 2.804×；但 Iter293 不是 paired formal low-overhead run，因此不把该比值写成最终因果 speedup。
- Full5 是 batch12 resource-throughput Pareto，不是 matched-batch16 Route-A speedup。Iter293 341.819 s 也只作为历史 profile anchor。

## 6. 资源与限制

- node0 cgroup peak：664.75–666.23 GiB；node1：239.39–240.02 GiB。
- target Critic max allocated 约 36.24 GiB；正式监控 GPU 峰值约 39.4–39.5 GiB，GPU headroom 很小。
- post-boundary cleanup 仍需 12.03–13.64 s；它不属于 Step10/transition/Full5 语义 timer，但若下一周期紧接执行，必须纳入 inclusive makespan。
- Transition CV 10.23%，正式论文因果 speedup仍需要同模式 paired baseline；当前证据足以证明两个预注册工程目标稳定达到。
- 最终代码验证：使用冻结 Python 环境和正确 repository/runtime `PYTHONPATH`，**245 tests passed**。一次遗漏 `PYTHONPATH` 的命令产生 7 个 import collection errors，已保留为 invocation error，不属于实验失败。

## 7. 结论

在不改变 Helios 训练语义的前提下，**block-only ZeRO-3 leaf grouping** 将 Step10-A 稳定压到 100.69–104.77 s；**persistent pinned transition cache** 将参数重分配稳定压到 17.39–21.68 s。两个目标均通过，且 exact Full5 correctness 已闭环。
