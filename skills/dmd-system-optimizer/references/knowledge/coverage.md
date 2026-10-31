# 经验覆盖范围与证据边界

截至 2026-10-31：**21 个机制主题、134 条案例与方法条目、141 份固定版本来源**。
条目不等于独立实验或验证成功的方案；本轮是来源核验和内容修正，没有新 GPU 实验。

| 来源类别 | 条目数 |
| --- | ---: |
| Helios optimization history | 24 |
| Helios A100 history | 69 |
| Helios H100 history | 16 |
| ResearchBoost external | 25 |

| 记录状态 | 条目数 |
| --- | ---: |
| positive_observation | 35 |
| negative_observation | 17 |
| implementation_failure | 12 |
| inconclusive | 12 |
| source_contract | 44 |
| unverified | 6 |
| correctness_failure | 8 |

128 条有公开引用；6 条未核实会话线索保持线索等级。历史 PASS、REVIEW、FAIL、NOT_RUN 和外部报告均按各自范围保留。

## 历史筛选与本轮修正

前期按确切会话身份筛选主会话1,108条与H100会话568条assistant最终消息；1,676条经空白归一化后有1,243条不同正文、433条重复。只用摘要寻找线索，再核对公开报告；不是逐句审完全部会话。私人原始会话没有上传。

本轮纠正七条明确错误，并检查同类条件与代码关系；新增六条有来源的数值隔离、B2失败、训练态timestep、QKV canary、容量政策和时钟映射经验。相邻W6 r1也查出计算结果缓存误归为参数驻留的问题，已纠正。

核验范围是已列明的内容问题与同类错配；不声明完成全部历史材料的逐项内容审计。

## 代码导航覆盖

35 个实现/工具条目，62 份固定版本Python文件；这是代码身份统计，不是新增实测报告。

- `linked`：74 条经验。
- `method_only`：24 条经验。
- `unresolved`：36 条经验。

关系明确区分历史实际实现、当前复用候选、需适配示例、诊断工具、概念相关。所有导航结果的 `direct_reuse_verified=false`；hash/符号吻合不能证明当前训练验收或可直接迁移。

15 个实现没有定位到专门测试入口。其余测试也仅供检查覆盖，未在本轮执行组件GPU/训练验收。见 [实现索引](implementations.md)。

## 未完成范围

- `DISCOVERED_NOT_CURATED`：38 份来源。
- `CURATED_EXCERPTS`：103 份来源。

六条会话线索涉及早期流水、loss趋势、恢复Adam峰值、NUMA、staging扩展、H3 block-major；缺公开回执，不能当成根因或速度结论。

来源索引只表示发现或抽取了有关片段，行锚点只辅助阅读。没有穷尽远端大文件、重开停止的campaign或重跑全部历史验收。独立子观察分别保留workload/更新合同比较；微小单次差值和未隔离解释不升级成稳定规律。

使用 `dmd_knowledge.py sources --query ...` 搜索尚未提炼的来源；见 [source index](../../assets/experience-source-index.json)。

## 维护

统计由 [coverage JSON](../../assets/experience-coverage.json) 渲染；经验与代码关系由JSON生成Markdown。ResearchBoost固定为 [908c6a13](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/tree/908c6a13c41fcb6da6d0237aba2574d368ffbec8)，只做有归属的原创概括，不复制其代码或Skill卡。复用前仍须检查对应许可和当前适用性。
