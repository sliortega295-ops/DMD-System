# 图捕获、EMA 与固定状态假设

图捕获与批量提交可减少 host 开销；形状、系数、状态地址和阶段变化会影响正确性与命中率。

把固定十个 EMA 系数的试验限定在对应 steps，不能直接推广到无限训练。
训练、EMA、no-grad 和 checkpoint 路径可能触发不同图；单层 fixture 不覆盖全流程。
编译成本和 warmup 单列；短任务的最优选择可能与长任务不同。
改动后重新看提交开销是否仍在关键路径；不能从总耗时减少直接推断新并发发生。

## 如何判断当前适用范围

正确性前提（未知需要核实，冲突不可直接套用）：

- dynamic_state_and_coefficients_preserved: true
- capture_paths_validated: true

收益条件（不满足会降低收益预期，不等于语义错误）：

- launch_bound: true
- capture_amortized: true

最小验证：核对 capture/replay 与 fallback 计数，跨阶段和系数变化测试；比较短工作和稳态完整周期。

## 已整理案例

### W1：固定 EMA 系数图的有效期只覆盖已测 steps

- ID：helios-w1-ema-r34
- 状态：positive_observation / HISTORICAL_REVIEW
- 来源类别：Helios optimization history；本次未复现实验。
- 观察：C2 19.720700→19.549642 秒，约0.171秒；仅已知 EMA decays11–20 两周期；REVIEW。
- 经验：小正值可保留为组合候选；有限系数范围不能包装成生产无限训练支持。
- 范围：r34 exact EMA submission graph，同 streams/order；长期未跑，生产未晋级。
- 条件：{"workload": "W1", "ema_coefficients": "known_steps_11_20"}
- 失败分类：not_established
- 重试条件：在目标 workload 上重新确认条件、数值和完整周期收益。
- 来源：[sliortega295-ops/DMD-System / conclusion.json](https://github.com/sliortega295-ops/DMD-System/blob/fee3e29e25ba481bb87d42270bb3af57a3676ccd/artifacts/publications/20261019/evidence/state_retention/r34/conclusion.json)；定位：scope; coefficient_scope; claim; production_ready

### Wan block 级 torch.compile 变慢且严格 rollout 不通过

- ID：wan-block-compile-fails
- 状态：correctness_failure / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：30 blocks 编译路径有实际命中；diagnostic 27.181843→33.730888 秒，80/80 rollout strict checks 失败。另有冷编译成本，不能减掉估算成本伪造快值。
- 经验：编译需要同时验证命中、冷启动、稳态与数值；这是该分块编译范围的失败，不是整个 compile 机制的黑名单。
- 范围：Wan1.3B no-grad Generator block canary、该版本和 shapes。
- 条件：{"workload": "W1", "model_family": "Wan1.3B", "training_mode": "training", "global_batch": 16, "grad_mode": "no_grad", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：compile_numerics_and_runtime
- 重试条件：只有编译范围、kernel lowering 或数值实现等发生可验证变化后重试，并重新测冷/热成本。
- 来源：[sliortega295-ops/DMD-System / WAN13B_TORCH_COMPILE_CANARY_20260824.md](https://github.com/sliortega295-ops/DMD-System/blob/b9dd25e36dfa45e6768428eb08f549afa7d3895e/docs/experiments/WAN13B_TORCH_COMPILE_CANARY_20260824.md)；定位：结果、合同、限制与失败记录

### 预热 optimizer dispatch 的前提是恢复所有训练状态

- ID：helios-optimizer-warmup-restored
- 状态：positive_observation / HISTORICAL_SOURCE_REPORTED
- 来源类别：Helios A100 history；本次未复现实验。
- 观察：正式 timer 前零梯度/零学习率/零 weight-decay dispatch，并恢复参数、Adam、step、清空梯度；16 ranks 验证后 151.851949→144.621745 秒。
- 经验：避免首次调度开销可有用，但不能把真实更新移到 timer 外；mesh 应比较逻辑签名而非 Python 对象身份。
- 范围：GAN-on CUDA optimizer 首次 dispatch，cold setup 单独计算。
- 条件：{"workload": "W3", "model_family": "Helios14B", "training_mode": "training", "global_batch": 16, "gan_enabled": true, "update_ratio": "G1/C5/EMA1", "gpu_family": "A100", "gpu_memory_gb": 40, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 16}
- 失败分类：not_established
- 重试条件：当前 workload 的相关前提、容量或实现改变后，先做最小判别测试，再以匹配对照验证完整周期。
- 来源：[sliortega295-ops/DMD-System / iteration352_optimizer_warmup.md](https://github.com/sliortega295-ops/DMD-System/blob/dcc4a740a8dbf17969a7618f1514c7d6438e9931/docs/experiments/gan_cuda_baseline_opt_20260829/iteration352_optimizer_warmup.md)；定位：结果、合同、限制与失败记录

### ResearchBoost：SAC/dispatch mode 可让 compile 路径静默回到 eager

- ID：rb-compile-dispatch-fallback
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：action ladder 报告 Torch2.7 下编译函数被某些 dispatch mode 调用后，后续调用未再走预期编译 kernel。
- 经验：用实际 kernel/counter 区分“编译无收益”和“编译未生效”；升级版本后重新验证，不把外部 workaround 固定为通用禁令。
- 范围：外部版本限定的 compile/SAC 经验，不保证当前 Torch 行为相同。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-action-ladder/SKILL.md)；定位：Compile falls back / proving compile effective

### ResearchBoost：单层 compile 测试漏掉全模型的重编译与 fallback

- ID：rb-compile-full-flow-specializations
- 状态：source_contract / EXTERNAL_GUIDANCE_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：同一 forward code object 被多层、grad/no-grad、EMA 与不同 shapes 共用，可能触及版本相关的 recompile limit。
- 经验：先找为何重编译，再选择形状/作用域；验证完整训练流程，不把提高 limit 当默认解决办法。
- 范围：外部 regional compilation 方法建议；具体阈值不继承。
- 条件：{"training_mode": "training"}
- 失败分类：not_established
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-action-ladder/SKILL.md)；定位：Recompile limit

### ResearchBoost/SGLang：推理 graph capture shape 与服务请求不符会失效

- ID：rb-inference-graph-shape-miss
- 状态：negative_observation / EXTERNAL_REPORTED_NOT_REPRODUCED
- 来源类别：ResearchBoost external；本次未复现实验。
- 观察：其 inference card 报告 SANA-Video 捕获 81 frames 而实际服务 17 frames 时没命中并变慢；改为目标形状后有改善。
- 经验：核对 capture key、实际形状和命中；这是推理范围的经验，迁移训练还需 backward、参数地址、RNG 与更新验证。
- 范围：外部 SGLang diffusion inference 案例，不是 DMD training graph 验收。
- 条件：{"training_mode": "inference", "model_family": "SANA-Video", "source_project": "SGLang"}
- 失败分类：capture_shape_mismatch
- 重试条件：在目标 workload、库版本与冻结验收合同下先验证触发条件；外部成功/失败均不直接授权采用或永久否决。
- 来源：[AI4AI-for-Anything/ResearchBoost-Agent / SKILL.md](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/sglang-diffusion-performance/SKILL.md)；定位：SANA-Video graph capture shape
