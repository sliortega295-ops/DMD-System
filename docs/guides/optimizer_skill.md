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

首版有 [12 个机制主题与 24 个案例](../../skills/dmd-system-optimizer/references/knowledge/README.md)：
参数驻留、容量交换、角色调度、host 同步、attention 路由、融合与精度、
图与 EMA、状态缓存、计时身份、数值与生命周期验证、实现复用、实验决策。
21 个案例关联公开源码或报告；3 个会话线索保留未验证状态。
这不是 24 次本次新实验，也不意味着已覆盖全部 Helios 历史。

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

支持中文关键词与英文术语。--context 接收当前条件 JSON；未知写 null。
--repo 核对本地公开证据 hash；没有 checkout 时仍可读随 Skill 分发的摘要和固定来源链接。
检索不联网、调用模型或创建向量数据库。Codex 按需读取相关 Markdown 卡片，
判断适用性；检索命中本身不代表批准应用。

新观察可保存到项目本地经验目录，用 record/search 查询；
导入原始会话需要显式指定文件，始终作为未验证材料。
原始会话、私有路径和大实验文件不进入公开包。
公共经验更新通过审核后的提交完成，不自动上传用户数据。

## 首版工具与边界

另附可选的性能估算、机会排序和持久预算工具；已有 runner/ledger 可继续复用。
探索优先级与收益保留分开，稳定正确的小收益可以保留；
默认启用还需适用性、组合与回退验证。非 bitwise 改动按风险选择验证深度。

辅助工具检查声明、hash、预算与决策条件。它们不能阻止绕过工具启动进程，
也不能通过一个 PASS 字段证明训练真实正确。
本次交付验证检索、条件处理、记账、拒绝路径和安装可移植性；
GPU 加速与长轨迹验证未运行。
详见 [辅助工具](../../skills/dmd-system-optimizer/references/campaign-tools.md)。

## 维护与来源

新增经验先说明机制、正确性前提、收益条件与失败类型，再关联案例。
未知原因保留未知；纠正旧结论使用新案例和 supersedes，不删除负结果。
机器包位于 Skill 的 assets/helios-experience-pack.json，
可读卡片位于 references/knowledge/。修改时同步两者并运行：

    python3 -m unittest discover -s skills/dmd-system-optimizer/tests -v
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py validate-pack

设计参考 ResearchBoost 的经验先验、当前任务重新验证和条件化快速路径清单：
[methodology](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/main/docs/methodology.md)、
[FSDP multi-forward](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/main/skills/train-opt-fsdp-multi-forward/SKILL.md)、
[action ladder](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/main/skills/train-opt-action-ladder/SKILL.md)。
本包未复制 ResearchBoost 代码或技能卡；案例来自本项目公开证据和单独标注的会话提炼。
