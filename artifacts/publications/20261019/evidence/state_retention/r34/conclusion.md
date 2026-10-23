# Exact EMA Graph：小收益，不是几秒级突破

| 同源原生配置 | C1(s) | C2(s) | peak allocated(GiB) | reserved(GiB) |
|---|---:|---:|---:|---:|
| R30双流＋提前1rollout，原EMA |31.527182641|19.720700253|26.691757|34.025391|
| 同样路径，只把原EMA序列捕获成Graph |31.597990563|19.549641884|26.691757|34.048828|

同源描述差值0.171058369s、减少0.8674%。保留为可组合小收益，但不当作统计确认或几秒收益。未接入调度器/main/default，未push。

16/16×2 cycles，G2/C10/EMA10→G4/C20/EMA20；原trace/调用/版本/finite/精度/容量通过。每rank control真实10reference EMA；child真实10Graph replay，原sub/mul/sub FP32运算与10个decay完全不变，GPU canary618实际形状/权重逐次bitwise一致。不是foreach/fused Adam/merged LoRA。原生4308全state比较REVIEW：1588 tensor不满足原allclose门，maxabs1.12862854e-05。不放宽门、不称长期等价。

该原型为两cycle已知decay11..20的显式缓存，捕获在初始化且成本单列（component建图约0.223s），不宣称无限步训练已经可交付。不是删除EMA工作，不能把profile中的8.5s EMA host envelope算成可省时间。修复后的G/C真实重叠尚未重采，不能由原生墙钟倒推。

原2GiB allocated highwater/64GiB host和节点FD锁未改，无新CUDA/hostOOM，当前双节点GPU空闲、锁重新获取释放通过。基础HEAD`a5b550af56b3eb7e8b5f570be94540134fc49ad5`＋SHA-bound显式源`6a7f0cc83725b884a92ed07ff70150f89da20d65f29a1ed420a28b7eca02235d`。R32预CUDA metadata KeyError及额外CPU harness SyntaxError均保留；R33 GPU机制gate和R34双节点25CPU gate均通过后才跑本原生配对。全部旧费用保留，累计86owner phases。

下一大机会：首次C0仍必须等同步r1 host handler返回（EMA尚未运行，故此依赖未由本补丁改变）。独立G主机producer可在r0完成后让主线程提交C0，同时后台G线程准备r1；须保持私有G/C参数、独立输入/RNG、原5次C/EMA顺序、逐样本事件及allocator消费者记录。先CPU依赖与禁用生产者collective/RNG测试，再有界同源native，不继续扫流窗口。
