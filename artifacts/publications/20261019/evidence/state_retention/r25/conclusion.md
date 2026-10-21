# Wan 大形状Attention后端：完整原生C2正收益

同源 `a5b550af56b3eb7e8b5f570be94540134fc49ad5`，384×640×33，world16，B1/global16，G1/C5/EMA5；private frozen bases，retainC26-29，LoRA FP32，原有其余优化不变。不开profiler，每臂只跑一次同进程两周期，C1预热、C2取16rank最大。

| 手工配置 | C1(s) | C2(s) | peak allocated(GiB) | peak reserved(GiB) | 完成 |
|---|---:|---:|---:|---:|---|
| Native attention control |35.640501042|20.005457799|26.691761|33.943359|16/16×2cycle|
| 大形状FA3＋其他保持native |31.395128444|19.680851440|26.691757|33.943359|16/16×2cycle|

C2减少 **0.324606359s（1.623%）**，描述性 **1.016494×**。这是单配对，不是统计显著性或正式长期数值验收。C1不用于加速计算。

- 原应用调用、输入trace、初末版本、finite、dtype、原2GiBallocated-capacity/64GiBhost准入等检查通过。两臂最终G4/C20/EMA20。实际LoRA参数/Adam/EMA全状态4308tensors比较保留 **REVIEW**（1880未满足原1e-5/1e-7严格allclose，maxabs1.305e-5），不把它叫严格等价。
- 实际每rank FA3 Python eager/capture调用644（grad344/nograd300），均为 **11040-token**；其余1964次源代码调用保留native。CUDA Graph重放不增加Python计数，不能把644当整个周期kernel数。
- 组件2160/8640形状有正收益，但本轮实际未命中；这些M尺寸不能未经验证当作真实attention-sequence长度。下一步最高信息价值是收齐真实fallback几何和FP32LoRA GEMM形状，再选择路径，而不是按猜测扩大白名单。
- 两节点两臂owned清理及新的GPU空闲/锁释放均通过，host/cgroup OOM增量0。边界采样最小driver free≈1.975GiB，不是全程最低值。Host峰值未连续测量，不补估计。
- 无融合权重、无共享底座/额外保留块，无batch/schedule/optimizer修改，独立opt-in **未接调度器、未push、未改main或Downloads**。

源码patch与bundle在 `artifacts/`；两节点目标2.10 CPU77及实际loader检查、此前两个metadata-only CPU报告错误均保留。本方向2/2启动，累计campaign74个owner phase，包括所有旧失败/组件；大状态全部留远端。
