# Attention backend 与真实调用覆盖

可用且更快的 backend 依赖真实 shape、dtype、梯度和缓存路径；局部 benchmark 只支持对应调用。

先检查现有 backend 和 guards，再决定新增 kernel。
实际被路由的 shape 与 primitive 探针覆盖分开；测过若干长度不能证明训练实际出现了它们。
快路径确实启用也可能整体变慢；检查布局拷贝、同步、workspace、replay 与 backward。
新 backend 和同步消除要分别归因；不得借旧 backend 的结果宣称新组合增量。
推理路径支持不等于可微调用已验证。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- attention_semantics_supported: True
- backward_and_cache_validated: True

收益条件（不满足会降低收益预期，不等于语义错误）：

- target_shapes_hit_fast_path: True
- backend_e2e_advantage: True

最小验证：枚举实际 shape/dtype/grad/mask/cache 调用，记录 fast-path 与 fallback 次数，再测完整周期。

## 已整理案例

### W6：FA3 已启用仍可能整体变慢

- ID：helios-w6-fa3-r43
- 状态：negative_observation / HISTORICAL_REVIEW
- 观察：实际 FA3 每周期每 rank 350 次调用；C2 255.528302602→257.709695683 秒。
- 经验：快路径命中证据与收益证据分开；后续去同步的收益不把这次 backend 试验改写为成功。
- 范围：W6 r43 的同源单配对，grad/selected attention 未改，numerical REVIEW。
- 重试条件：后端调用边界、同步、layout 或真实几何改变后，重新测该组合。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r43/conclusion.json)

### W1：primitive 覆盖与实际路由范围必须分开

- ID：helios-w1-fa3-r25
- 状态：positive_observation / HISTORICAL_REVIEW
- 观察：原生 C2 单配对节省 0.324606359 秒；实际只路由 11040-token 调用；1880/4308 strict tensor comparisons not allclose。
- 经验：不能用探针覆盖其他长度来宣称它们在整模型内生效；正向时间不消除 numerical REVIEW。
- 范围：W1 r25 手工 FA3 路由与当时 retain4 对照，非默认或长期晋级。
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r25/conclusion.json)
