# 分配器、构造峰值与运行模式

存活张量、缓存、碎片与驱动空闲是不同容量项；能完成运行与运行更快需要分别验证。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- live_storage_ownership_known: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- allocator_cost_or_capacity_is_limiting: true

最小验证：分构造、稳态、checkpoint replay 与恢复记录最坏 rank；正确性模式和计时模式都要核对峰值。

## 已整理案例

### to_empty 或 swap 无法保证旧分布式状态真正释放

- ID：helios-role-transition-alias-owners
- 状态：implementation_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：连续切换时旧 Generator 的 Engine、hooks 和 partition aliases 仍持有 CUDA 存储；to_empty 未释放，swap_tensors 被 live use count 拒绝。最终按所有权释放过时分片才通过。
- 经验：必须证明没有未来消费者并清点所有存储 owner；模块表面换设备不等于 Engine/optimizer 引用已解除。
- 范围：8G4S4T→4G12C 的旧 DeepSpeed 表示释放；不授权删除当前仍有效的状态。
- 条件：{"model_family": "Helios14B", "training_mode": "training", "resolution": "384x640", "trace": null, "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：obsolete_owner_retains_storage
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / rb16_full5_2g6c_continuous_20260810.md](https://github.com/sliortega295-ops/DMD-System/blob/15a0511803ba97d860f80b6ecd5b88121cbccdf7/docs/experiments/rb16_full5_2g6c_continuous_20260810.md)；定位：结果、合同、限制与失败记录

### 正确性模式能跑，去掉 flush 后的正式计时仍可能 OOM

- ID：helios-no-flush-timing-oom
- 状态：implementation_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：去掉 G residency 前后两次 empty_cache，correctness 通过；正式 timing 在 292 MiB 分配上 OOM，存活 allocated 约 36.43 GiB，同时有约 1.91 GiB reserved-unallocated。
- 经验：计时路径也需要容量验收；reserved slack 并不保证能满足特定分配，不能把 OOM 归为数值错误。
- 范围：GAN-on C23，无 flush 的 allocator 布局。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：timing_path_allocator_capacity
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration378_c23_allocator_noflush.md](https://github.com/sliortega295-ops/DMD-System/blob/7f64327754a31a409f581470f588c5e94a2787b4/docs/experiments/gan_cuda_baseline_opt_20260829/iteration378_c23_allocator_noflush.md)；定位：结果、合同、限制与失败记录

### expandable_segments 修复容量却比有 flush 的 control 慢

- ID：helios-expandable-segments-cost
- 状态：negative_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：修复 no-flush OOM 后计时 127.619804 秒，对比有 flush 的 125.814004 秒慢约 1.44%。
- 经验：容量可行性修复值得记录，但只有完整时间比较才能判定性能；不同碎片或显存条件可重新评估。
- 范围：GAN-on C24 的 allocator 组合，单次观察。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：capacity_fix_runtime_cost
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration379_c24_expandable_noflush.md](https://github.com/sliortega295-ops/DMD-System/blob/ae6f2b28b97739143b02f21c6496d3afbb5765d0/docs/experiments/gan_cuda_baseline_opt_20260829/iteration379_c24_expandable_noflush.md)；定位：结果、合同、限制与失败记录

### 80GB H100 上纯 resident H3 仍在第二次 Fake backward OOM

- ID：h100-w5-pure-resident-oom
- 状态：implementation_failure / HISTORICAL_H100_CUDA_OOM
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：Student1/Fake1 后 checkpoint replay 请求 2.02 GiB 时 GPU allocated 约 76.17 GiB、free 约 770 MiB；这是 CUDA OOM，不是 host OOM。
- 经验：最大阶段与多更新生命周期决定容量，不能用模型权重大小或第一次更新成功代替验证。
- 范围：W5 Grid1000 approximate training contract 的纯 GPU resident arm。
- 条件：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W5", "gpu_family": "H100", "gpu_memory_gb": 80}
- 失败分类：late_fake_backward_capacity
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / w5_resident_actual_capacity_failure_r4.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W5/reports/w5_resident_actual_capacity_failure_r4.json)；定位：该结果及 qualification / NOT_RUN 边界

### GPU 容量改善可能转移为很大的 host 内存压力

- ID：h100-w6-host-budget
- 状态：source_contract / HISTORICAL_H100_SINGLE_TRIAL_REVIEW
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：W6 candidate sampled cgroup peak 约 1256 GiB；该值含 page cache/kernel/anonymous，不等于进程 RSS 之和。Resident 最低 sampled free 约 0.458 GiB。
- 经验：容量模型同时包含 host 与 device，记录采样范围；完成一次不证明长期余量或更小主机可运行。
- 范围：W6 该 H100 主机与同源矩阵，不能把 GiB 数值作为别的 workload 的固定需求。
- 条件：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W6", "gpu_family": "H100", "gpu_memory_gb": 80}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / report.md](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W6/artifacts/w6_resident_adaln16_staging_20261015_r1/reports/completed_same0419_matrix_r1/report.md)；定位：该结果及 qualification / NOT_RUN 边界
