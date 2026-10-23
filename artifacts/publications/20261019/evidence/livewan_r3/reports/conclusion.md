# W4 Self-Forcing：仅A与完整仅B重新实测

原合同：368×640×81、global B16、continuous sigma、seed20260830，G1/C5/EMA0。两台8×A100，4个连续周期；C1预热，以下只取C2；每格n=1，无profiler。

| 新配置 | C2 (s) | 相对既有FSDP2 83.853875s | 节省(s / %) | 峰值allocated / reserved (GiB) | 完成 |
|---|---:|---:|---:|---:|---|
| 仅A：G8 FSDP8、C8常驻、T8 FSDP8；B全关，无offload | 82.330023 | 1.0185× | 1.524 / 1.82% | 18.96 / 37.66 | 16/16 ×4cycles |
| 完整仅B：原FSDP16顺序，接齐18项既有B机制 | 42.123914 | 1.9906× | 41.730 / 49.77% | 15.48 / 38.38 | 16/16 ×4cycles |

若按论文FSDP2主表84.26s引用，两格分别为1.0234× / 2.0003×；不把它冒充本轮新跑基线。既有联合方案42.407027s未重跑。

## 这次修正了什么

- 仅A不再把固定resident布局加CPU optimizer offload的72.12s当最佳纯A。本轮重选可行G分片、保留完整流水和梯度通信调度，所有B关闭；实测收益只有约1.8%。这是一个有界可行候选，不是整个布局空间全局最优证明。
- 完整仅B不再用79.15s的四算子兼容子集。原FSDP16 groups/units及顺序不变：将原C51的G/Score算子按角色适配，G单次no-grad rollout内保留同组gathered权重/autocast与lazy text K/V；退出先清KV/autocast，再reshard，commit前恢复；Critic沿原FA3 SAC缓存，原梯度/RNG链不变。没有merged-LoRA。
- **结果：完整B在原FSDP布局下已达到42.12s，与既有联合方案42.41s接近。** 这说明旧“仅B”子集严重低估B的独立作用。不要据这一对n=1跨源结果声称联合调度普遍无效、或仅B在统计上优于联合。

## 实际机制验收与边界

每个rank四周期：960checkpoint创建、1920函数执行、960replay；其中Student360、Critic600、Teacher0，Critic FA3实际缓存读600、失败0。每rank G版本4/C20/EMA0；20次G rollout scope、700次原35个unit保留/reshard/恢复；text KV构建600 blocks，结束缓存0 bytes。13个真实kernel/backend计数族全部执行，连同KV/cast/SAC/optimizer/full-context组成18项B清单。逐rank计数见CSV。

16/16 ranks、64 rank-cycle、调用与版本/finite/lifetime检查通过；严格数值仍REVIEW，不把finite升级成全状态或长轨迹等价。没有新CUDA/host OOM；node1旧oom_kill=3在前后未增。

仅B终态driver-free最低4.375MiB，allocated峰值15.48GiB但reserved38.38GiB；reserved含可复用allocator缓存。终态采样不是整个周期最小余量，不能宣称容量裕量充足；未修改容量政策。直接硬件autocast命中率未独立测量。

源：两格均053bdcad2cf0e5388738156599f96a6ec5905518，parent42f4f855。基线/联合a62ca是保留的旧源速度诊断；新两格同源，不能写成基线也在新HEAD重跑。两节点真实Torch2.10聚焦CPU各151/151、零skip；旧CPU失败及修复均保留。

总账12→14，新增仅2次native。基线/联合未重跑，余下2次未使用，不自动开启其他队列。两节点owned GPU进程已无、锁最终可取。即时B清理回读曾遇node0锁释放竞态（GPU已空）；后续新鲜回读两节点GO，未强解锁或杀进程。

没有push、改main、H100、飞书或Downloads。大raw留远端持久目录；此处保存小摘要与hash索引。
