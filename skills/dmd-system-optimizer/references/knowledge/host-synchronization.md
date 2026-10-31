# 消除 metadata 同步与 CPU 提交空隙

频繁读取 GPU 上的 metadata 可能阻断 CPU 提交；可推导的 host 状态可减少这些同步。

CPU 时间段很长不等于 GPU 空闲；先关联实际 GPU activity 和依赖。
只有已知且不依赖异步结果的状态可以放在 host 管理；数据决定的分支不能直接猜测。
跨 wrapper 调用要核对 host 状态实际持有者，避免副本更新后调用方仍读旧值。
同一 attention backend 的单因素对照可回答 metadata 的增量；更换 backend 的旧计时不能当其分母。
收益可能随 batch、角色数和图捕获改变。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- metadata_semantics_preserved: true
- host_state_lifetime_validated: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- host_sync_on_critical_path: true

最小验证：定位 item/DtoH/同步调用及其依赖；用开关只改变 metadata 路径，确认调用、版本和缓存输入一致。

## 已整理案例

### W6：同 FA3 对照下 metadata barrier 消除有正向观察

- ID：helios-w6-metadata-r46
- 状态：positive_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：两臂同 FA3：C2 262.384847224→253.740367298 秒，节省 8.644479926 秒；单配对 REVIEW。
- 经验：这是 metadata 变化的该次增量；不能把更早 SDPA 的 255.528 秒代入分母。
- 范围：W6 r46 相同训练源加外部 packet；非长期数值或最终全系统收益。
- 条件：{"workload": "W6", "attention_backend": "FA3", "ranks": 16}
- 失败分类：not_established
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r46/conclusion.json)；定位：control_C2_s; candidate_C2_s; same_FA3_backend_both_arms; interpretation

### 去掉若干内部同步后只获得很小完整周期改善

- ID：helios-internal-fences-small
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：保留外部 timer fences，移除一次 app empty_cache 与六次内部 rollout synchronize；138.462140→138.176083 秒，约 0.21%。
- 经验：CPU API wall 中长时间等待可能只是等 GPU；不可把整段同步时间全算作可消除收益。
- 范围：GAN-on C12 的这些 fence，其他依赖仍保持。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration365_c12_rollout_nosync.md](https://github.com/sliortega295-ops/DMD-System/blob/2883f88a14654d3e6369c4f14de01a1f932edf35/docs/experiments/gan_cuda_baseline_opt_20260829/iteration365_c12_rollout_nosync.md)；定位：结果、合同、限制与失败记录

### 只跳过计时模式不用的 scalar logging

- ID：helios-scalar-log-elision
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：每次 Critic update 四次无消费者的 item 转换，在 correctness 模式保留，正式计时共省 20 次；131.562027→130.630471 秒。
- 经验：先证明值不参与训练控制、输出承诺或验收；不要删除必需的观测来制造收益。
- 范围：GAN-on C17、固定 timing-mode logging 契约。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration372_c17_scalar_logs.md](https://github.com/sliortega295-ops/DMD-System/blob/6137c4cfa762f7b73aaa08614bd4ac6b19466070/docs/experiments/gan_cuda_baseline_opt_20260829/iteration372_c17_scalar_logs.md)；定位：结果、合同、限制与失败记录

### helper 的全局 free_memory/GC 也可能干扰训练提交

- ID：helios-free-memory-gc
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：限定 timing 模式移除四次 GAN helper free_memory/GC，correctness 保留；130.630471→129.441379 秒。最初 SSH 断开在构造前发生。
- 经验：核对对象生命周期后再改变 cleanup；连接故障不算模型失败，也不能由此重置实验次数。
- 范围：GAN-on C18，作用域局限于已验证 helper。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration373_c18_generator_free_memory.md](https://github.com/sliortega295-ops/DMD-System/blob/d91c3727cb22642dc9f54061315f15703dfe623a/docs/experiments/gan_cuda_baseline_opt_20260829/iteration373_c18_generator_free_memory.md)；定位：结果、合同、限制与失败记录

### H100 W2 只有同源 fence-only 差值能归因给 fence

- ID：h100-w2-fence-only-pair
- 状态：positive_observation / HISTORICAL_H100_SINGLE_TRIAL_REVIEW
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：同 8fc70e8 上 95.534631493→94.311432633 秒；41 个单元在阶段结束前 reshard，保留前后 allocator flush 与外 timer fences。
- 经验：更早 official-swap 的 138.371948 秒跨了 source/策略，不能将整个差距算成 fence 消除收益。
- 范围：W2 Helios No-GAN、同源 retained-G 内部 fence 实验，n=1。
- 条件：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W2", "gpu_family": "H100", "gpu_memory_gb": 80}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / result.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W2/artifacts/helios_h100_rollout_fence_probe_20261015_r1/reports/completed_fence_pair_r1/result.json)；定位：该结果及 qualification / NOT_RUN 边界
