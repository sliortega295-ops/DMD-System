# DMD optimization decision — <candidate-id>

1. **对象与目标差距**：workload / control-id / timer-scope / baseline → current → target。
2. **机制**：M/L/Q/K；具体改变什么工作、依赖或资源，而不是“优化一下”。
3. **证据**：artifact+hash+locator；原始总时长 / interval union / 非重叠 / 关键路径归因明确区分。
4. **收益**：物理/条件上限；低/中/高净 E2E 节省（含开销）；下一处瓶颈；未知及公式。
5. **语义与容量**：版本/随机性/梯度/LoRA/EMA/数值；HBM、host、pinned、workspace 和通信竞争。
6. **判别探针**：用什么最小试验区分哪个假设；验收/否决标准；不能借机无限调参。
7. **成本与预算**：工程、编译、排队、GPU-hours、full runs、正确性验证；已有消耗一起扣。
8. **决定**：ADMIT / PROBE / DEFER / INVALID；为何不是其他机会；切换/停止/重试条件。

这是工作记录，不是额外的人类审批。已授权且条件满足就继续。未测数据写 unknown。
