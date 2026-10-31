# 用 Codex 与条件化经验库优化自己的 workload

DMD-System 提供可独立安装的
[dmd-system-optimizer](../../skills/dmd-system-optimizer/SKILL.md) Skill。
Codex 读取当前工作量，检索机制及历史正反案例，核对适用条件，
然后在用户给定的资源范围内优化。可直接处理现有训练仓库，
也可使用 DMD-System recipe，不要求先迁移训练代码。

## 安装与使用

在本仓库运行：

    python3 tools/install_optimizer_skill.py

安装器把完整 Skill 放到 Codex 的 skills 目录。已有同名但不同内容的版本会被保留；
可用 --destination 指定其他安装位置，或显式管理旧版本。
仅需要 Python 3.10+ 标准库，不导入 Torch、下载模型或启动 GPU。
之后在能发现该 Skill 的新 Codex 会话中，可以直接说：

> 用 $dmd-system-optimizer 优化这个训练 workload。先读取代码和运行方式，
> 检索相关经验及失败案例，说明哪些条件符合、哪些还未知，再在已有预算内推进。

提供已有运行命令、资源和目标即可；Codex 先读取项目能找到的信息。
只要求分析时会保持分析范围。缺失条件不能用历史数字填补。

## 经验范围

当前有 [19 个机制主题与 128 条案例/方法条目](../../skills/dmd-system-optimizer/references/knowledge/README.md)，
包括 Helios/Wan 连续执行与调度、H100 W1–W6、pinned staging、allocator、
融合、编译、缓存、profiling 和 ResearchBoost 的外部经验。
其中 122 条关联公开来源，6 条会话线索保持未核实；实测与方法建议分开。
这不是 128 次独立成功实验，也不声称找全全部历史。
[覆盖报告](../../skills/dmd-system-optimizer/references/knowledge/coverage.md)
列出历史筛选方法、来源统计和未完成项。

每条模式区分：

- 正确性前提：冲突时不能直接应用；未知时先核实。
- 收益条件：决定是否值得尝试；与语义错误分开。
- 历史案例：保留 workload、阶段、状态、对照、结果和失败范围。
- 重试条件：容量、复用次数、实现、拓扑或证据改变后如何重新判断。

同一模式同时返回正向、负向、未决与源码案例。模型名只是上下文；
收益不能跨 workload 自动外推，失败不会形成全局黑名单。
候选仍需在当前代码、瓶颈和验收合同下验证。

## 无 embedding 检索

    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py recall --query "参数保留 residency" --limit 2
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py recall --query "FA3 attention" --repo .
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py sources --query "H100" --limit 10
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py recall --brief --query "同样的东西算了好多遍"
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py case --id helios-crosskv-nogan-negligible
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py locate --case helios-crosskv-gan-positive --repo . --show-code

支持中文关键词、常见口语症状与英文术语；含糊的等待会提示先诊断原因。
查询文字只用于排序假设，不推定硬件、梯度或其他 workload 事实。
--context 接收当前条件 JSON；未知写 null。
--repo 核对 DMD 本地证据，历史文件可从本地已有的固定 Git 提交读取；不会联网拉取。
外部来源可用 --source-repo OWNER/REPO=/absolute/checkout 指定独立 checkout，
不会把 ResearchBoost 路径误当成本仓库文件。没有 checkout 时仍可读摘要和固定链接。
sources 可搜索 136 份来源元数据，包括尚未提炼完的报告；索引命中不是验证通过。
检索不联网、调用模型或创建向量数据库。Codex 按需读取相关 Markdown 卡片，
判断适用性；检索命中本身不代表批准应用。

每条案例有独立的正确性/收益比较与具体重试计划。`locate` 可从经验定位到
[29 个实现条目](../../skills/dmd-system-optimizer/references/knowledge/implementations.md)，
提供固定 commit 的函数行号、接入开关、guard、测试和回退。
71 条经验有相关代码链接，19 条是纯方法，38 条的对应实现尚未定位。
其中 13 个实现尚未找到专门测试入口；所有条目保留测试尚未覆盖的范围。“相关当前实现”不冒充历史计时源码。
`--repo` 会核对源码 hash 与符号，`--show-code` 只读取有限片段，不导入或执行代码。
具体实现可能比通用经验更窄，必须同时检查两层条件，例如 no-grad/B1 的 cross-KV
不能直接用于需要梯度的调用。没有本地 checkout 时仍返回固定 GitHub 链接并保留未核对状态。

新观察保存到项目本地经验目录；`recall --store /project/memory` 可统一检索公共与私人经验，
使用相同的中文检索、条件归一化和正负案例分组，`case --id ... --store ...` 展开私人记录。
纠正关系在结果截断前解析，返回有效后继与适用范围；H100 下的新结论不会全局抹掉旧硬件经验。
`finish` 将实验结算、决策和经验写回接为幂等操作，中断后可用 `flush-memory` 恢复。
导入原始会话需要显式指定文件，始终作为未验证材料。
原始会话、私有路径和大实验文件不进入公开包。
公共经验更新通过审核后的提交完成，不自动上传用户数据。

## 首版工具与边界

另附可选的性能估算、机会排序和持久预算工具；已有 runner/ledger 可继续复用。
探索优先级与收益保留分开，稳定正确的小收益可以保留；
默认启用还需适用性、组合与回退验证。非 bitwise 改动按风险选择验证深度。

`REVIEW` 允许继续补验证；候选源码/配置、计时样本与旧判断保留不可变历史。
失败尝试都记成本，只有事先约定且有 runner 正文证明的无效计时可在总次数限额内补测；
有效慢样本不能删除，候选 OOM/数值错误不能伪装成环境故障。
只有容量收益时，可在正确性和容量验证后保留为默认关闭的 `KEEP_OPTIONAL`，同时注明时延代价是否已知。
完整速度确认只用于少量最终候选；六个 A–B–A block 是 18 次独立进程运行。
只做正确性或容量验证时可设 `timing: null`，baseline 时间也可为空，不必填虚构计时。
CPU 条件可明确填写 0 张 GPU；实际的 CPU 验收保持 observed 身份。

[checker 薄适配](../../skills/dmd-system-optimizer/references/checker-receipts.md)
解析现有 JUnit checker 和 runner 正文，绑定源码、配置、输入、run、比较对象与验证范围；
手填 PASS 或计时与正文冲突时拒绝。默认启用还要求最终组合对原始 baseline 的验收，
不能只逐轮对着上一个 control 放行。`ReuseDecision` 记录已有实现、最小接入、实际路径与回退证据。
它们仍不能证明进程身份真实性、验收器覆盖充分性或阻止绕过工具启动进程。

CPU 集成测试已经覆盖独立小项目检索实现、最小接入、实际数值检查、保留/回退、
Memory 写回与新进程检索；模型是 scalar protocol stand-in，计时明确 synthetic。
GPU/FSDP/真实梯度与长轨迹验证未运行，不能从这条流程测试推导大模型加速。
详见 [辅助工具](../../skills/dmd-system-optimizer/references/campaign-tools.md)。

## 维护与来源

新增经验先说明机制、正确性前提、收益条件与失败类型，再关联案例。
未知原因保留未知；纠正旧结论使用新案例和 supersedes，不删除负结果。
机器包位于 Skill 的 assets/helios-experience-pack.json，
可读卡片位于 references/knowledge/。修改结构化条目后生成卡片并验证：

    python3 skills/dmd-system-optimizer/scripts/render_knowledge.py
    python3 -m unittest discover -s skills/dmd-system-optimizer/tests -v
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py validate-pack

设计参考 ResearchBoost 的经验先验、当前任务重新验证和条件化快速路径清单：
[methodology](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/docs/methodology.md)、
[FSDP multi-forward](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-fsdp-multi-forward/SKILL.md)、
[action ladder](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/908c6a13c41fcb6da6d0237aba2574d368ffbec8/skills/train-opt-action-ladder/SKILL.md)。
本包未复制 ResearchBoost 代码或技能卡；24 条外部条目是带固定引用的原创概括，
方法建议、外部实测与推理范围均有标注，不能直接当作本地训练结果。
