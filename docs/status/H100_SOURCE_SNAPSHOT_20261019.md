# H100 六 workload 源码与历史实测索引

这是对 [A100／规划器源码索引](GITHUB_SOURCE_SNAPSHOT_20261019.md) 的补充。**发布代码和小型证据，不是重新运行 H100，也不是合并 main。**

六项分别是 Wan1.3B、Helios No-GAN、Helios GAN-on、Self-Forcing／LiveWan、H3 全序列 AV 和 H3 自回归 AV；不能把两项 H3 合成一项。
H100 历史实验使用 8 张物理 H100-80GB、16 个独立逻辑样本槽位。这里不是 16 张物理 H100，也不与 A100 表直接拼成同批实验。

## 1. 代码从哪里读

| Workload | 实际计时源码 | 已发布分支／补充源码 | 解释 |
|---|---|---|---|
| W1 Wan1.3B | [7a668bcb](https://github.com/sliortega295-ops/DMD-System/tree/7a668bcb12ad37d831dced3aca1be4ec7be0a2fe) | [97b8ecc9](https://github.com/sliortega295-ops/DMD-System/tree/97b8ecc9cf7747d93619acad1a6083f98faae084)，`agent/h100-wan13b-rankcache-20261015` | 计时版本是父提交 7a；97b 保留后续 norm 移植、私有编译缓存和离线 import 修复，但**新 norm 的完整周期计时 NOT_RUN**。不能把 37.670s 归给新 norm。 |
| W2 Helios No-GAN | [8fc70e8c](https://github.com/sliortega295-ops/DMD-System/tree/8fc70e8ce9f6e0243604812120119c9f4c078e55) | `agent/h100-helios-nograd-retain-20261014` | 保留 G 展开权重及内部 rollout fence 单变量对照；原外层计时同步保留。 |
| W3 Helios GAN-on | [624515e7](https://github.com/sliortega295-ops/DMD-System/tree/624515e7c8d1511aa688e95e69cf94f1a2a273f7) | `agent/h100-helios-official-lowvram-20261014` | 官方角色切换基线和优化路径；与 A100 的 116s／N70 修正批次不是同一实验。 |
| W4 Self-Forcing／LiveWan | [4d7c589a](https://github.com/sliortega295-ops/DMD-System/tree/4d7c589acd01a4a6c90ca141da266b3090df2b68) | [1b8236c2](https://github.com/sliortega295-ops/DMD-System/tree/1b8236c243323ff21023935f3b8192ab3e5d136d)，`agent/h100-livewan-teacher-resident-20261015` | 4d 是 Teacher 常驻的计时源码；1b 后续增加硬件回执复核、重复 NVTX 范围和 collective 计数修复，**没有更快的新计时**。H100 Resident 结果不能代替 A100 C51。 |
| W5 H3 全序列／Grid1000 | [9bfd3eda](https://github.com/sliortega295-ops/H3-DMD-for-A100/tree/9bfd3eda959bd0c8415fe71ba64d3ad5f7e93e71) | `H3-DMD-for-A100:agent/h100-h3-three-arm-20261014` | 同源 FSDP 控制与 GPU Attention 输出／LSE 缓存；外部执行 packet 见下。两臂均为 **Grid1000 近似 sigma**，不是 continuous 对 Grid 的数值等价结论。 |
| W6 H3 AR／DiffSynth | [0419b96b](https://github.com/sliortega295-ops/DMD-System/tree/0419b96b9ed9a8dd76737042e6c5020c80f6673a) | `agent/h100-w6-resident-adaln-staging-20261015` | 原生 AR8、continuous sigma、G1/F1、LoRA64，**FA2** 路径；包含受准入约束的 Resident AdaLN CPU staging 容量候选。FA3 完整模型 gate 仍 FAIL，不能当作成功路径。 |

W5 的 CPU checkpoint-boundary Resident 容量回退另发在 [d81ebb20](https://github.com/sliortega295-ops/H3-DMD-for-A100/tree/d81ebb2014af9ea3a84011e3be198818e63cb1e7)，分支 `agent/h100-h3-resident-boundary-contract-20261014`。它不是纯 GPU Resident，也不是与两项 9bfd 路径同 revision 的 placement-only 对照。

## 2. 可追溯的历史时间，不是此次重测

| Workload | 本行对照定义 | 对照 (s) | 优化 (s) | 单对观察比值 |
|---|---|---:|---:|---:|
| W1 | Resident reference | 52.066607 | 37.669795 | 1.382× |
| W2 | 同源 retained-G，原内部同步／移除内部同步 | 95.534631 | 94.311433 | 1.013× |
| W3 | 官方角色切换／优化实现 | 192.986046 | 117.343507 | 1.645× |
| W4 | Resident／Teacher 常驻 | 45.361398 | 45.257227 | 1.002× |
| W5 | 同源 Grid1000 FSDP／GPU Attention cache | 638.522138 | 539.718383 | 1.183× |
| W6 | 同源 native FSDP／FA2 candidate | 320.666055 | 278.886442 | 1.150× |

W2 较早官方切换基线 138.371948s 对最新 94.311433s 的 1.467× 仅是**跨 revision 描述性比值**，不是上表同源单变量收益。
W5 CPU-boundary Resident 为 637.204307s；纯 Resident 的 Fake2 backward OOM 保留，不能隐去。

所有行保留各自计时范围、更新合同、n=1 和数值 REVIEW／NOT_RUN；**不能写为严格全状态或长轨迹等价已通过，也不能把各行的不同基线统一改名 FSDP2。**
机器可读 [CSV](../../artifacts/publications/20261019/h100/measured_results.csv) 与 [source manifest](../../artifacts/publications/20261019/h100/source_manifest.json) 给出精确 source、scope 和原证据路径。

## 3. Git 外实际使用的执行代码也归档

[source_packets/campaign](../../artifacts/publications/20261019/h100/source_packets/campaign/) 保存原脚本的逐字节副本，SHA256、来源及每行证据绑定在 manifest；不只上传计时报告：

- W1：`scripts/run_wan_resident_matched_r1.py` 与原 Resident 两臂 launcher／资格逻辑。
- W2／W3：`scripts/run_helios_official_swap_matched_r1.py`，以及 `artifacts/helios_h100_rollout_fence_probe_20261015_r1/scripts/run_retained_fence_single_pair_r1.py`。
- W4：`artifacts/w4_h100_teacher_resident_20261015_r1/scripts/run_resident_teacher_matched_r2.py`、计时硬件复核器及原 Reader/Writer 修复。
- W5：`scripts/run_h3_h100_three_arm_r4.py`、`scripts/run_h3_cuda_attention_cache_r1.py`、`scripts/h3_h100_cuda_attention_recording_entry_r1.py`、同源 guard、CPU-boundary 回退与来源复核脚本。
- W6：`artifacts/w6_resident_adaln16_staging_20261015_r1/scripts/run_finish_same_revision_native_matrix_r1.py` 及对应状态／计时复核脚本；原生 worker 位于 0419 分支 `scripts/distillschedule/probe_h3_diffsynth_h100_world8.py`。

目录也保留同批失败／取消候选源码以便审查，**不是目录内每个候选都通过或都应该运行**。以上路径均相对于 source packet 根目录。
脚本保留原始绝对路径、source/asset SHA 与容量 gate；不是可直接对任意机器执行的一键脚本。模型、真实输入、环境、checkpoint、18GiB Grid 表和大型 Nsight 数据保留远端。迁移资产或路径必须显式重新准入，不能删除验证或改冻结回执。
Helios 基础 runtime 的源码快照与来源表位于各 DMD 分支 `third_party/helios_compat/`；它不替代冻结依赖／安装环境的 hash 检查。

## 4. 发布验证与停止边界

```bash
python3 scripts/distillschedule/check_h100_source_publication.py
python3 -m unittest discover -s tests -p test_h100_source_publication.py
```

这些只验证归档完整性／范围，不导入训练运行器、不初始化 CUDA，也不表示新 source 在 H100 上重新通过。
原 H100 队列保持 **USER_STOPPED_NO_AUTOMATIC_RESUME**；[R48 停机交接](../../artifacts/publications/20261019/h100/evidence/campaign/h100_campaign_handoff_20261016_r48.md) 覆盖 R47 的续跑建议。此次发布不恢复队列，不改变训练源码、默认值、数值门或 main。
历史记录中的 `push=false` 是当时状态，不追改；本次后续发布与历史测量严格分开。
