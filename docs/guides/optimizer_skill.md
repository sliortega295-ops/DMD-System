# 用 Codex 直接优化 workload

[dmd-system-optimizer](../../skills/dmd-system-optimizer/SKILL.md) 是优化方法、
经验库和代码导航，不是实验审批系统。直接使用项目现有训练入口、checker 和 WORK_LOG。

## 安装与开始

    python3 tools/install_optimizer_skill.py

只需 Python 3.10+ 标准库，不加载模型或启动 GPU。默认安装到
`~/.agents/skills/dmd-system-optimizer`。在本仓库中使用时，`.agents/skills/`
已链接到同一份源码，不必再复制。其他项目使用个人安装副本即可。

更新个人安装副本：

    python3 tools/install_optimizer_skill.py --update

更新会把旧目录（包括本地修改）完整移到返回的 backup 路径，再安装新版；
不会自动合并旧修改或上传私人记录。备份位于 Skill 发现目录之外，避免旧版重复加载。
使用旧客户端或自定义位置时显式传 `--destination /path/to/skill`；不会扫描或删除
`~/.codex` 中的旧安装。没有 `--update` 时仍拒绝覆盖不同内容。

新会话中可直接说：

> 用 $dmd-system-optimizer 优化当前 workload。先找可复用代码和历史经验，
> 解决端到端瓶颈。每轮快速检查，最后验证组合，用现有 runner 和 WORK_LOG。

提供已有命令、目标和授权资源即可。项目里已有的信息不重复询问；没有 GPU 时仍可分析源码。

## 默认流程

**查实现与经验 → 找瓶颈 → 修改 → 快速检查/测时 → 保留或回退 → 记经验。**

说明为什么值得尝试即可，不要求完整 MFU、DAG、机会表或精确 GPU-hours。
做不起、用不了的大优化不能一直挡住可行的小改动。正确且可重复的 1% 也值得保留；
一次噪声内差值只能标待确认，省显存能力可以保留为可选项并注明时延代价。

每轮检查真正改到的前后向、更新或状态路径。高风险缓存、精度与分布式改动尽早验证；
不因非 bitwise 自动跑 100 steps。最终组合与原始 baseline 比较，按风险和用途补连续训练、
多 seed 或质量测试。保留全部有效测量，不靠挑最快的一次制造收益。

预算用于防止盲试：记录大致消耗，无进展时重新判断瓶颈，不设固定“两次封禁”或小 Kernel 配额。
**明确的资源上限、设备授权和停止要求仍然有效。** 不够完成验收时保留最佳已验证版本、
候选补丁和缺失检查，不为了凑次数继续运行。

## 查经验和代码

    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py recall --brief --query "参数反复上卡"
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py case --id helios-crosskv-gan-positive
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py locate --case helios-crosskv-gan-positive --repo . --show-code

只读影响当前决策的几个正反案例。context 和私有 store 按需提供，不是开工表单。
检查形状、精度、梯度和生命周期；相关诊断工具不等于可直接套用的优化。

新结果先写 WORK_LOG，必要时用 record 加入跨会话索引。索引失败时保留日志和错误、正常交接，
不额外跑实验、不删除失败历史。私人记录不会自动上传。
详见 [Memory 与复用](../../skills/dmd-system-optimizer/references/reuse-memory.md)。

## 当前保留与删除的内容

保留 [21 个机制主题、109 条经验/方法条目](../../skills/dmd-system-optimizer/references/knowledge/README.md)
和 [29 个实现/工具条目](../../skills/dmd-system-optimizer/references/knowledge/implementations.md)。
106 条有公开来源，3 条仍是未核实线索；不是 109 次独立成功实验。

早期 Route-A/Route-B、固定角色流水、旧 offload/staging 实现及配套入口已退役，
相应 25 条案例、6 个实现入口和 25 个来源入口也已撤下。
见 [退役说明](../architecture/legacy_route_retirement.md)。当前训练依赖的公共代码不是按日期一刀切删除。

Skill 中旧 SQLite campaign、回执适配、ReuseDecision 审批、rank/gate 及其表单和专属测试
也已删除，不再作为“高级可选模式”交付。只保留检索、代码导航与只读计算工具。
命令见 [辅助工具](../../skills/dmd-system-optimizer/references/helpers.md)。
历史源码可从 Git 追溯；用户已有数据库、实验结果、失败记录及资源消耗不会被清理脚本删除。
旧验收失败不会因为工具退役而变成通过。

## 维护

    python3 -m unittest discover -s skills/dmd-system-optimizer/tests -v
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py validate-pack

经验修改先改机器包，再用 render_knowledge.py 生成卡片。
测试保留原生 CPU 检查、错误梯度、复用/回退和跨进程 Memory；这些不证明 GPU 加速或真实训练质量。
当前模型仍使用自己的验收。旧实验报告只作来源，不是当前默认操作指南。
