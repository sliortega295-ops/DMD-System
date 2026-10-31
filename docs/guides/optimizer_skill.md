# 用 Codex 直接优化 workload：轻量流程优先

[dmd-system-optimizer](../../skills/dmd-system-optimizer/SKILL.md) 的默认定位是：
**一个有经验的性能工程师，而不是实验审批系统。**
Codex 使用现有训练仓库和测试命令，先找可复用实现与历史正反经验，再优化实际瓶颈。
不要求迁移 Trainer、创建数据库、填写完整 JSON 或先给出精确 GPU-hours。

## 安装后直接使用

在 DMD-System 仓库运行：

    python3 tools/install_optimizer_skill.py

安装器仅需 Python 3.10+ 标准库，不导入模型或启动 GPU。
已有同名不同内容版本不会被覆盖；按安装器提示显式管理旧副本或使用 `--destination`。
在能发现 Skill 的新 Codex 会话中说：

> 用 $dmd-system-optimizer 优化当前 DMD workload。先复用已有代码和经验，
> 解决当前端到端瓶颈。每轮做相关快速检查，最终验证组合。
> 用项目现有 runner 和 WORK_LOG，别为了填表或凑测试次数停住。

给出已有命令、目标与可用资源即可；信息已在项目里就不重复询问。
仅要求分析时不启动实验。没有 GPU 时仍可完成源码检查和适用的 CPU 测试。

## 默认只做这一圈

**查现成实现 → 看瓶颈 → 改一项或一组相关代码 → 快速检查/测时 → 保留或回退 → 记经验。**

每轮说明为什么改、可能省在哪里、主要风险是什么；粗略估计或一个便宜探针就够了。
不用为了开始工作先补齐 MFU、完整 DAG、四层机会表或所有资源字段。
有大收益机会时优先考虑，但做不起、用不了的大方案不能一直挡住可行的小改动。

正确且可重复的 1% 也能保留。一次差值分不清噪声就标“有希望/待确认”，不冒充已验证收益；
省显存的功能可以默认关闭保留，注明时延代价。是否继续深挖与是否保留是两回事。

## 验证分两段

**探索时便宜地筛选。** 检查真正改到的前后向、更新或状态路径。
没有必要每改一处就启动完整集群，更不因非 bitwise 自动跑 100 steps。
缓存、分布式、精度与状态生命周期等高风险改动，要尽早检查对应风险。

**最终组合认真验证。** 对原始 baseline 检查完整逻辑更新和匹配的无 profiler 计时；
需要时补连续训练、多 seed 或生成质量。不能只看 loss 曲线，也不能只验证旧分支。
不设默认六组 A–B–A；按波动、结论用途和资源决定复测力度，保留全部有效样本。
未完成时交付最佳已验证版本、候选补丁和缺失检查，不强行制造 PASS。

## 预算主要用于防止盲试

记录大致花了多久、试过什么。连续几轮无收益就重新判断瓶颈，
有新证据再继续，否则换可行方向或交付负结果。没有默认“两次失败封禁”或“小 Kernel 10% 配额”。

**用户明确的总量上限、资源分配和停止指令仍是硬边界。**
没填数字不代表可以无限使用资源；也不能因为账本字段为空就阻止已授权的普通工作。
给最终检查留时间，预算不足就保存现有成果，不为了凑次数继续跑。

## 查经验、找代码、沉淀新经验

    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py recall --brief --query "参数反复上卡"
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py case --id helios-crosskv-gan-positive
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py locate --case helios-crosskv-gan-positive --repo . --show-code

只展开影响当前决策的几个正反案例；`--context` 和私有 `--store` 都是按需提供，
不是开始优化前必须填的表。检查实际实现的形状、精度、梯度和生命周期；
历史失败不等于永远无效，相关诊断工具也不等于可直接套用的优化。

新结果先记项目 WORK_LOG：机制、代码、命令、真实输出、结论和再尝试条件。
需要跨会话检索时再用 `dmd_knowledge.py record` 写入明确指定的私有目录。
索引失败就保留日志和错误、正常交接，不额外跑实验；不能把坏库说成空库或删除失败历史。
公开/私人记录可用同一个 `recall --store` 检索，但不会自动上传私人会话或数据。
详见 [Memory 与复用](../../skills/dmd-system-optimizer/references/reuse-memory.md)。

现有 [21 个机制主题、109 条经验/方法条目](../../skills/dmd-system-optimizer/references/knowledge/README.md)
和 [29 个实现/工具条目](../../skills/dmd-system-optimizer/references/knowledge/implementations.md) 保留。
其中 106 条有公开来源，3 条仍是未核实线索；不是 109 次独立成功实验。
早期 Route-A/Route-B、固定角色流水和旧 offload/staging 路线已退出当前分发包：
撤下25条案例、6个实现入口和25个来源入口。旧内容仅从Git历史追溯；
当前框架/FSDP2阶段、H100和ResearchBoost经验继续按适用范围保留。
[覆盖与缺口](../../skills/dmd-system-optimizer/references/knowledge/coverage.md) 继续单独维护。

## 高级工具不再是默认控制器

`dmd_perf.py gate`、SQLite campaign、ReuseDecision、checker 回执适配和固定样本确认
保留给显式选用的正式实验或项目要求。普通优化不必调用它们。
项目已有 runner/checker/日志就继续用，不制造假 JUnit、假时间或第二份账本。
已采用的冻结协议、失败记录与已花预算不能通过切换模式抹掉。
详见 [高级工具](../../skills/dmd-system-optimizer/references/campaign-tools.md)。

## 维护与测试范围

    python3 -m unittest discover -s skills/dmd-system-optimizer/tests -v
    python3 skills/dmd-system-optimizer/scripts/dmd_knowledge.py validate-pack

经验内容改动先修改机器包，再运行 `render_knowledge.py` 生成卡片。
轻量工作流测试使用真实 CPU 标量检查与原始日志，不依赖账本或回执适配；
合成计时只验证工具行为，不证明 GPU 加速、真实训练梯度或 Codex 的所有行为。
正式部署仍需要当前 workload 自己的验证。
