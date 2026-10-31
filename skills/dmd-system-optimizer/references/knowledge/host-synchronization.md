# metadata 同步策略与 CPU 提交次序

metadata 读取可能阻断 CPU 提交，也会改变提交次序与资源竞争。先分清恢复、增添或删除同步；收益方向必须由当前证据确定。

CPU 时间段很长不等于 GPU 空闲；先关联实际 GPU activity 和依赖。
只有已知且不依赖异步结果的状态可以放在 host 管理；数据决定的分支不能直接猜测。
跨 wrapper 调用要核对 host 状态实际持有者，避免副本更新后调用方仍读旧值。
同一 attention backend 的配对可以记录 metadata 策略变化的观察差值；一次差值不独立解释其原因，更换 backend 的旧计时不能当其分母。
收益可能随 batch、角色数和图捕获改变。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- metadata_semantics_preserved: true
- host_state_lifetime_validated: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- host_policy_e2e_effect_measured: true

最小验证：先核对动作是增添、恢复还是删除同步；检查实际消费者、提交次序与同源端到端证据，不预设删除一定更快。

## 已整理案例

### W6 r46：恢复原生 metadata barrier 后的单次正向观察

- ID：helios-w6-metadata-r46
- 状态：positive_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：同 FA3 的单次配对：每 rank 每周期 metadata 读取 control 0→candidate 350 次；C2 262.384847224→253.740367298 秒，差值 8.644479926 秒，数值 REVIEW。
- 经验：启用 restore-native-cu-metadata-barrier.v1 会恢复 cu_q.tolist() 读取/同步。该次时间改善的底层原因未隔离，不能推出删除同步有益或恢复 barrier 普遍有益。
- 范围：W6 r46 相同训练源加外部 packet；非长期数值或最终全系统收益。
- 历史上下文：{"workload": "W6", "attention_backend": "FA3", "ranks": 16}
- 正确性前提：{"metadata_semantics_preserved": true, "host_state_lifetime_validated": true}
- 收益条件：{"same_backend_comparison": true, "metadata_barrier_policy_measured": true}
- 失败分类：not_established
- 重试信号：同 FA3、相同读取值与原生 barrier 路径可独立比较
- 最小验证：先核对 0/350 读取计数和真实 tolist 路径，再检查梯度/状态及同源交错配对
- 不应重复：不把恢复 barrier 写成删除同步，不以单配对推断通用因果规律，也不混入 backend 更换
- 代码状态：linked；关系按具体机制和代码职责标注；源码匹配不等于当前 workload 已验证。
- 定位命令：`dmd_knowledge.py locate --case helios-w6-metadata-r46`
- 结构化事实：{"intervention": "restore_native_metadata_barrier", "metadata_reads_per_rank_cycle": {"control": 0, "candidate": 350}, "control_C2_s": 262.384847224, "candidate_C2_s": 253.740367298, "numerical_status": "REVIEW", "paired_runs": 1}
- 归因边界：{"status": "MECHANISM_UNRESOLVED", "claim": "观察到恢复 barrier 后的单次改善；改善原因未完成隔离。"}
- 关联实现：[W6 r46 原生 metadata barrier 恢复 packet](implementations.md#w6-native-metadata-barrier)；historical_implementation；r46 归档 packet，实际恢复原生 metadata 读取；训练源与外部 packet 身份分别核对。
- 函数入口：[artifacts/publications/20261019/manual_packets/state_retention/r46/scripts/native_r2/native_metadata_barrier.py::NativeMetadataBarrier](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/manual_packets/state_retention/r46/scripts/native_r2/native_metadata_barrier.py#L7-L45)
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r46/conclusion.json#L12)；定位：观察中的数值文本锚点 L12；同时阅读同节合同、状态与限制。原定位：control_C2_s; candidate_C2_s; same_FA3_backend_both_arms; interpretation
- 来源：[sliortega295-ops/DMD-System / native_metadata_barrier.py](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/manual_packets/state_retention/r46/scripts/native_r2/native_metadata_barrier.py#L4)；定位：POLICY='restore-native-cu-metadata-barrier.v1'

### 去掉若干内部同步后只获得很小完整周期改善

- ID：helios-internal-fences-small
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：保留外部 timer fences，移除一次 app empty_cache 与六次内部 rollout synchronize；138.462140→138.176083 秒，约 0.21%。
- 经验：CPU API wall 中长时间等待可能只是等 GPU；不可把整段同步时间全算作可消除收益。
- 范围：GAN-on C12 的这些 fence，其他依赖仍保持。
- 历史上下文：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"metadata_semantics_preserved": true, "host_state_lifetime_validated": true}
- 收益条件：{"host_sync_on_critical_path": true}
- 失败分类：not_established
- 重试信号：内部 fence 不承载语义依赖且 trace 显示等待
- 最小验证：保持外 timer fences，验证删除处的消费者与异步完成
- 不应重复：不通过移除计时边界同步制造快值
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-internal-fences-small`
- 来源：[sliortega295-ops/DMD-System / iteration365_c12_rollout_nosync.md](https://github.com/sliortega295-ops/DMD-System/blob/2883f88a14654d3e6369c4f14de01a1f932edf35/docs/experiments/gan_cuda_baseline_opt_20260829/iteration365_c12_rollout_nosync.md#L1)；定位：报告范围 L1–L42；L1: Iteration 365: C12 explicit rollout-fence cleanup；L3: Contract；L12: Evidence；L36: Result boundary；原定位：结果、合同、限制与失败记录

### 只跳过计时模式不用的 scalar logging

- ID：helios-scalar-log-elision
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：每次 Critic update 四次无消费者的 item 转换，在 correctness 模式保留，正式计时共省 20 次；131.562027→130.630471 秒。
- 经验：先证明值不参与训练控制、输出承诺或验收；不要删除必需的观测来制造收益。 旧 correctness 分支保留调用，它的 PASS 不能证明省略调用的 timing 分支获得相同检查；迁移需验实际加速分支。
- 范围：GAN-on C17、固定 timing-mode logging 契约。 两模式路径不同，timing 分支的等同检查覆盖尚未建立。
- 历史上下文：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"metadata_semantics_preserved": true, "host_state_lifetime_validated": true, "timing_path_consumers_checked": true, "timing_path_state_lifetime_validated": true}
- 收益条件：{"host_sync_on_critical_path": true}
- 失败分类：not_established
- 重试信号：目标 scalar 转换确实无当前消费者
- 最小验证：核对日志/检查路径消费者，比较 timing 与 correctness 模式的有效行为；在实际省略调用的分支检查数值、日志消费者、对象释放与多周期状态
- 不应重复：不删除用户需要的指标或验收值
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-scalar-log-elision`
- 结构化事实：{"correctness_branch": "original_calls_retained", "timing_branch": "calls_elided", "same_path_validation": false}
- 来源：[sliortega295-ops/DMD-System / iteration372_c17_scalar_logs.md](https://github.com/sliortega295-ops/DMD-System/blob/6137c4cfa762f7b73aaa08614bd4ac6b19466070/docs/experiments/gan_cuda_baseline_opt_20260829/iteration372_c17_scalar_logs.md#L31)；定位：观察中的数值文本锚点 L31；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### helper 的全局 free_memory/GC 也可能干扰训练提交

- ID：helios-free-memory-gc
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：限定 timing 模式移除四次 GAN helper free_memory/GC，correctness 保留；130.630471→129.441379 秒。最初 SSH 断开在构造前发生。
- 经验：核对对象生命周期后再改变 cleanup；连接故障不算模型失败，也不能由此重置实验次数。 旧 correctness 分支保留调用，它的 PASS 不能证明省略调用的 timing 分支获得相同检查；迁移需验实际加速分支。
- 范围：GAN-on C18，作用域局限于已验证 helper。 两模式路径不同，timing 分支的等同检查覆盖尚未建立。
- 历史上下文：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 正确性前提：{"metadata_semantics_preserved": true, "host_state_lifetime_validated": true, "timing_path_consumers_checked": true, "timing_path_state_lifetime_validated": true}
- 收益条件：{"host_sync_on_critical_path": true}
- 失败分类：not_established
- 重试信号：GC/free_memory 不再是容量或生命周期必要条件
- 最小验证：在目标 timing 模式测多周期峰值与完成释放；在实际省略调用的分支检查数值、日志消费者、对象释放与多周期状态
- 不应重复：不能把构造前 SSH 中断归因于候选或忽略模式差别
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case helios-free-memory-gc`
- 结构化事实：{"correctness_branch": "original_calls_retained", "timing_branch": "calls_elided", "same_path_validation": false}
- 来源：[sliortega295-ops/DMD-System / iteration373_c18_generator_free_memory.md](https://github.com/sliortega295-ops/DMD-System/blob/d91c3727cb22642dc9f54061315f15703dfe623a/docs/experiments/gan_cuda_baseline_opt_20260829/iteration373_c18_generator_free_memory.md#L6)；定位：观察中的数值文本锚点 L6；同时阅读同节合同、状态与限制。原定位：结果、合同、限制与失败记录

### H100 W2 只有同源 fence-only 差值能归因给 fence

- ID：h100-w2-fence-only-pair
- 状态：positive_observation / HISTORICAL_H100_SINGLE_TRIAL_REVIEW
- 来源类别：Helios H100 history；本次未复现实验。
- 观察：同 8fc70e8 上 95.534631493→94.311432633 秒；41 个单元在阶段结束前 reshard，保留前后 allocator flush 与外 timer fences。
- 经验：更早 official-swap 的 138.371948 秒跨了 source/策略，不能将整个差距算成 fence 消除收益。
- 范围：W2 Helios No-GAN、同源 retained-G 内部 fence 实验，n=1。
- 历史上下文：{"physical_gpus": 8, "logical_sample_slots": 16, "training_mode": "training", "workload": "W2", "gpu_family": "H100", "gpu_memory_gb": 80}
- 正确性前提：{"metadata_semantics_preserved": true, "host_state_lifetime_validated": true}
- 收益条件：{"host_sync_on_critical_path": true}
- 失败分类：not_established
- 重试信号：当前同源 fence-only 改动仍合法
- 最小验证：保留 41 单元 reshard、allocator flush、外 timer fence 后配对
- 不应重复：不能把去内部 fence 写成无限驻留或删除全部同步
- 代码状态：unresolved；已保留报告/线索与适用边界，但未定位到可核对符号和接入方式的对应实现；不能把同名或邻近模块当作该案例代码。
- 定位命令：`dmd_knowledge.py locate --case h100-w2-fence-only-pair`
- 来源：[sliortega295-ops/DMD-System / result.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/h100/evidence/W2/artifacts/helios_h100_rollout_fence_probe_20261015_r1/reports/completed_fence_pair_r1/result.json)；定位：该结果及 qualification / NOT_RUN 边界
