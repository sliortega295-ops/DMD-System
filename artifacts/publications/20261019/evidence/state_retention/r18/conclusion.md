# Wan：状态保留策略的真实搜索与保存计划执行

同源 `ea29129342741b54e59c4773e07454f541fadcc7`，384×640×33、world16、B1/global16、G1/C5/EMA5，FP32 LoRA/BF16 base、原 bootstrap/trace/RNG不变。不开 profiler；每臂同一进程连续两周期，C1只作预热。原始4阶段 owner 均完整结束，原子节点锁覆盖审计至清理。

| 原生计时臂 | C1 (s) | C2 (s) | allocated峰值 (GiB) | reserved峰值 (GiB) | ranks |
|---|---:|---:|---:|---:|---:|
| 手工 retain4/window5 | 31.975596 | 20.040550 | 26.6918 | 33.9434 | 16/16 |
| 搜索保存计划 retain4/window3 | 31.966593 | 20.103775 | 26.6249 | 33.5469 | 16/16 |

保存计划比手工慢 **0.063226s / 0.3155%**，描述性单对达到≤3%匹配目标。不是新加速纪录，也不是统计等价/全局最优声明。

## 这次新增的系统能力

- 实际分别校准“不保留”和“Critic后4block正常autograd保留”，成本键显式包含状态策略；不能混借另一策略的成本。
- 原事件前沿搜索使用实测 action成本、DAG依赖与内存上限；实际选中了保留block26–29、rollout窗口3，而非手工窗口5。
- 保存计划SHA `25585cd17e342ba33d3d4a6b167582d84b115d0853b03b2af50ac6547268fee0` 真正交给原始运行器；16份worker回执均绑定该计划。
- 实际runtime回执证明Generator未保留、Critic仅后4block保留；每rank记录retained forward16、checkpoint forward120。**此Python计数只涵盖eager/graph构建，不冒充CUDA Graph每次replay的Python计数**。原有图/调用/参数版本验收完整保留。
- 无 merged-LoRA、无参数精度变化、无新offload。原2GiB allocated容量余量及64GiB host政策不变。

## 数值、显存与证据边界

原完整trainable/Adam/EMA共4308个tensor比较：**REVIEW**，1524项未达原严格allclose，最大绝对误差1.1738608e-05；阈值未改，不能称严格数值等价或长期收敛通过。

两节点cgroup oom/oom_kill增量均0；新鲜清理复核GPU无进程、节点锁可重新获取。driver free仅有原计时端点样本（手工1.9750GiB、计划2.3715GiB），不是连续最小余量。host peak未在此pair采集。

本批4次native，累计本连续campaign62次（旧失败/组件/诊断均留账，不清零）。源码在独立分支，未push、未改main默认/H100/Downloads。

完整机器结果：`reports/conclusion.json`；资格/计划：`campaign/manifests/qualification_and_plan.json`；计时CSV：`campaign/reports/wan_native_pair.csv`；实际效果及清理：`artifacts/fresh-cleanup-and-effects.json`。大型状态和成本留远端。
