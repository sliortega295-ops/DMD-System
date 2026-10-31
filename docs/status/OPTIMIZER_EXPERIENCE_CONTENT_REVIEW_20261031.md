# Optimizer 经验内容修正与适用范围

本次修正针对公开 Skill 在 `5e2af222` 中的经验内容和代码导航。修改首先落到
`helios-experience-pack.json`、`implementation-map.json` 和来源索引，再生成 Markdown。
这是历史来源核验，不是重新做 GPU 实验或为当前 workload 证明加速。

## 纠正的机制与事实

| 案例 | 正确记录与复用边界 |
| --- | --- |
| `helios-w6-metadata-r46` | 同 FA3 下**恢复**原生 metadata barrier；每 rank 每周期读取 **0→350**。C2 262.384847224→253.740367298 秒为单配对，数值 REVIEW，改善原因未隔离。 |
| `helios-w3-retention-r1` | 缓存 Critic blocks38/39 冻结 FFN-up 的计算结果，减少 checkpoint 重算；不是参数驻留。定位到 `155ffbb5` 的 `NativeDownSAC` up 分支和真实接入。 |
| `helios-full5-pr8-oom` | g8 在 Step10 **Generator** checkpoint backward 的216MiB临时量分配失败；g12之后的Critic失败单独保留。没有有效完整计时。 |
| `helios-w1-tail8-oom` | whole-block激活保留增加峰值，收益条件是减少重算；不能继承减少gather的解释。 |
| `helios-profile-teardown-incomplete` | 卡住的是rank0、rank8两个控制rank；其他14个完成。低开销计时与profile收尾失败分开。 |
| `rb-end-to-end-denominator` | R3 2.08×计入reward服务优化贡献；1.88×两臂均用优化后的服务。RL step仍包含服务，不能解释成排除服务计时。补上固定版本`docs/results/multimodal.md`。 |
| `rb-inference-offload-vram-dependent` | 关闭offload需完整模型及workspace能驻留；保持inference/B300历史范围，不能继承容量不足时开启offload的条件。 |

同类检查还纠正了 `helios-w6-retention-r1`：它把block49的原生计算结果从
pinned CPU缓存改留GPU，其他块保持父路径；不是FSDP参数驻留。固定历史实现为
`45938af8:src/distillschedule/strategies/h3_ar_hybrid_cache.py`，当次回退原因未隔离。
该提交只在本地，原GitHub链接不可访问；引用的两份源码已逐字节保存到
[公开归档](../../artifacts/publications/20261031/optimizer_reference_sources/h3_gpu_tail1/README.md)，
机器导航改用公开归档提交`48134cde`。归档保留原hash和来源，依赖原生helper，不是独立可运行组件。

原回执保留。上游[六项结果汇总](SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md)已加入r46勘误；
检索仍指向旧固定版本汇总时，来源元数据和案例引用也携带内容勘误，防止再次学习错误方向。

## 条件、数值和代码关系

- 参数驻留、整块激活保留、checkpoint算子结果复用分开归组和建模。
- modulation chunk与FA2选择性保留的W2/W3结果保留独立子观察、来源和更新合同；
  完整和简短检索都返回子观察对当前上下文的比较，不能借父条目的W2标签覆盖W3。
- No-GAN cross-KV约0.033%差值、static-context约0.049%回退保留原数值，状态改为未决；
  缓存维护开销等解释保留为未隔离假设，不认定稳定的失败规律。
- scalar logging和free_memory明确标注correctness保留调用、timing省略调用；
  旧分支PASS不资格化实际加速分支。新环境需核对消费者及状态生命周期。
- 代码关系分为 `historical_implementation`、`reusable_candidate`、`adaptation_example`、
  `diagnostic_tool`、`conceptual_reference`，任何关系都不自动给出直接复用许可。
  旧私人记录的宽泛标签仍可读，但必须检查具体关系。
- QKV布局条目关联的checkpoint代码只属概念参考；producer dispatch改连真实
  `RolePoolExecutor._run_cycle`；profile parser声明固定2×8/world16约束，H100单机8卡会显示冲突。
- r34 EMA和r46 barrier补上归档源码。EMA仅覆盖原生reference FP32路径的11..20步，
  不支持无限训练；历史W3 up replay与当前同名down-only文件也不混用。

## 两次独立使用检查

新上下文Codex分别使用分发副本处理W6/FA3/r46和单机8张H100/W3重算问题，
完成检索、源码定位和有边界的复用准备；没有GPU、依赖安装、训练代码修改或追加预算。
两次均正常收尾，没有循环索取条件或把未知状态改写成PASS。

- W6试用识别出恢复metadata读取的真实方向、单配对REVIEW，以及当前checkout缺少旧worker接口，
  将方案保留为待最小适配的候选。试用同时发现相邻r43的“后续去同步”文字仍可能误导，已改正。
- W3试用从源码识别出历史接入强制A100/SM80、world16和固定Torch/CUDA，
  但初版locator未显式显示冲突；这些约束已加入实现JSON。相同H100/8卡输入复查后返回
  `DO_NOT_APPLY_AS_IS`，同时保留结果缓存机制可重新适配的空间。
- 固定2×8的profile包装器正确拒绝原样用于8卡；FA2策略函数与外层world16/B16训练接入的范围也已说明。

W3首次使用`locate --show-code`时自动展开关联历史组件测试摘录，虽未运行测试，
仍不能称为完全未接触测试材料的盲测。两次试用不是任意用户、Agent或环境都不会卡住的证明。
试用发现项的后续复查为确定性检索/回归检查，没有增加GPU或数值实验覆盖。

## 新增的六条有来源经验

| ID | 证据边界 |
| --- | --- |
| `rb-nonbitwise-isolation-final-combination` | reference替身隔离其他改动；真实最终组合仍需对原始baseline验收，不照搬固定100步。 |
| `helios-b2-rollout-numerical-rejection` | 兼容日程合批仍未通过数值门；保留15份失败rank回执，不能补全第16份，速度NOT_RUN。 |
| `helios-trainable-phase-local-timestep` | 保留BF16-before-reduction与timestep梯度的训练态路线，不用no-grad缓存替代。 |
| `helios-frozen-qkv-canary-no-benefit` | M6144局部canary无改善，world16正式计时NOT_RUN；不是已测出的全模型回退。 |
| `helios-capacity-policy-rejection-not-oom` | C25完成correctness和校准但不满足预测容量余量，CUDA OOM=false；校准时间不是native性能。 |
| `helios-logical-native-clock-mapping` | systems-only bootstrap的逻辑G200/C1000与真实Adam/scheduler G2/C10分别保存；只映射标签，不改训练时钟。 |

各条完整引用、commit、hash、条件和重试探针见[经验库](../../skills/dmd-system-optimizer/references/knowledge/README.md)。
现有134条记录包含方法和未核实线索，不能计为134个独立已验证优化。

## 验证与未覆盖范围

内容回归覆盖已知的方向、角色、rank、分母、结果对象、验证路径、子观察匹配和关系降级。
它能阻止这些已知错配被重新引入，不是自动证明任意新增经验真实的Judge。
原有36个自然语言检索夹具保留；激活保留夹具改用纠正后的机制分类。

本地203项Skill CPU测试通过，包括22项内容回归；Skill结构、Markdown链接和差异检查通过。
训练源码Python编译检查与17份shell脚本语法检查通过；完整仓库refactor测试未运行，
当前本地Python没有pytest。本次没有改动训练runtime，也没有安装依赖或运行GPU实验。

首次CI暴露浅克隆缺少历史源码；工作流现按实现清单抓取固定提交，保留源码和符号校验。
从本地浅克隆复现并修正后，导航测试通过。公开访问检查确认当前案例及实现引用涉及的
36个固定提交可读取；这证明资料可取得，不代替历史结论的真伪或新workload验收。

本轮核对103份去重案例证据文件的固定hash及125个固定源码符号，未运行引用中的GPU测试。
源码存在、测试存在与当前训练验证通过是三个不同结论。未声明完成所有历史材料的全面审计，
也未把缺证据线索升级；更多覆盖应以原始来源和可分辨结论为依据。
