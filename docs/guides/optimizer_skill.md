# 用 Codex 直接优化 workload

[dmd-system-optimizer](https://github.com/sliortega295-ops/DMD-System/blob/main/skills/dmd-system-optimizer/SKILL.md)
是优化方法、公共经验库和代码导航，不是实验审批系统。使用项目现有训练入口、checker 和 WORK_LOG。

## 安装与开始

Python 训练包与 Codex Skill 是两个独立入口。`pip install distillschedule` 不会把
Skill 安装到 Codex 发现目录。本页可以随 wheel 阅读；下面的安装、更新和维护命令
在 [DMD-System 源码 checkout](https://github.com/sliortega295-ops/DMD-System) 根目录执行。
引用的 Skill 文档使用在线源码链接，不假设 wheel 带有整个经验库和测试目录。

    python3 tools/install_optimizer_skill.py

只需 Python 3.10+ 标准库，不加载模型或启动 GPU。默认安装到
`~/.agents/skills/dmd-system-optimizer`。在本仓库使用时，`.agents/skills/`
已链接到同一份源码，不必再复制。其他项目使用个人安装副本即可。

    python3 tools/install_optimizer_skill.py --update

更新会将旧目录（包括本地修改）完整移到返回的 backup 路径，再安装新版；
不会自动合并旧修改或上传私人记录。备份位于 Skill 发现目录之外。
自定义位置显式传 `--destination /path/to/skill`；不会扫描或删除旧安装。
没有 `--update` 时拒绝覆盖不同内容。

新会话中可直接说：

> 用 $dmd-system-optimizer 优化当前 workload。先找可复用实现和历史经验，
> 解决端到端瓶颈。每轮快速检查，最后验证组合，用现有 runner 和 WORK_LOG。

提供已有命令、目标和授权资源即可。项目里已有的信息不重复询问；没有 GPU 仍可分析源码。

## 默认流程

**查实现与经验 → 找瓶颈 → 修改 → 快速检查/测时 → 保留或回退 → 记结果。**

说明为何值得尝试即可，不要求完整 MFU、DAG、机会表或精确 GPU-hours。
做不起、用不了的大优化不能一直挡住可行的小改动。正确且可重复的 1% 值得保留；
噪声内的单次差值标为待确认，显存改善可作为可选能力并注明时延代价。

每轮检查真正改到的前后向、更新或状态路径。高风险缓存、精度与分布式改动尽早验证；
不因非 bitwise 自动跑 100 steps。最终组合与原始 baseline 比较，按风险和用途补连续训练、
多 seed 或质量测试。保留全部有效测量，不靠挑最快的一次制造收益。

预算用于防止盲试，不设固定“两次封禁”或小 Kernel 配额。
**明确的资源上限、设备授权和停止要求仍然有效。** 预算不足时保留最佳已验证版本、
候选补丁和缺失检查，不为了凑次数继续运行。

## 查当前实现，再看历史经验

源码 checkout 中：

    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py navigate --repo . --query "官方 Self-Forcing"
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py recall --brief --query "参数反复上卡"
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py case --id helios-crosskv-gan-positive
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py locate --case helios-crosskv-gan-positive --repo . --show-code

独立安装时将脚本前缀换成安装路径。显式 `--repo` 读取当前注册表和 recipes；
未指定时使用标明版本的发布快照。某个未完成 recipe 会被跳过并给出警告，
不会让其他有效入口消失。定位代码不代表它已在当前环境验证或一定更快。

简版按结果类别展示少量相关经验，同时给出总数和翻页入口；
`--case-offset`、`--all-cases` 或 `case --id` 可展开，完整历史不删除。
只读影响当前决策的几个正反案例。检查形状、精度、梯度和生命周期；
相关诊断工具不等于可直接套用的优化。

个人连续工作优先查当前可用且已授权的 Codex 原生记忆/历史；实验事实写入 WORK_LOG。
不默认创建另一套私人 Memory。已有明确授权的索引可按需继续使用；索引失败保留日志和错误、
正常交接，不额外跑实验、不删除失败历史。私人记录不会自动上传。
详见 [Memory 与复用](https://github.com/sliortega295-ops/DMD-System/blob/main/skills/dmd-system-optimizer/references/reuse-memory.md)。

## 当前保留与删除的内容

公共库保留 [21 个机制主题、109 条经验/方法](https://github.com/sliortega295-ops/DMD-System/blob/main/skills/dmd-system-optimizer/references/knowledge/README.md)
和 [29 个实现/工具条目](https://github.com/sliortega295-ops/DMD-System/blob/main/skills/dmd-system-optimizer/references/knowledge/implementations.md)。
106 条有公开来源，3 条仍是未核实线索；不是 109 次独立成功实验。
当前能力导航另行从现有注册表和 recipe 推导，不将历史库限制误当作模型支持范围。

早期 Route-A/Route-B 专属入口及其 25 条案例、6 个实现、25 个来源入口已退役；
见 [退役说明](../architecture/legacy_route_retirement.md)。当前训练依赖的公共代码不按日期删除。
旧 SQLite campaign、回执适配、ReuseDecision 审批、rank/gate 及专属表单已删除。
只保留检索、导航和只读计算；见 [辅助工具](https://github.com/sliortega295-ops/DMD-System/blob/main/skills/dmd-system-optimizer/references/helpers.md)。
历史源码仍按固定版本追溯，旧失败不因工具退役而变成通过。

## 维护

在源码 checkout 中：

    python3 -m unittest discover -s skills/dmd-system-optimizer/tests -v
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py validate-pack

公共经验修改先改机器包，再用 render_knowledge.py 生成卡片。
CPU 流程和检索测试不证明 GPU 加速或训练质量；模型继续使用自己的验收器。
