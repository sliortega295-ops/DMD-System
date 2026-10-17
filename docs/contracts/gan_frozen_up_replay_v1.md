# Opt-in GAN frozen FFN-up replay retention v1

Candidate bundle: `gan-fa3-down40-up2-nomerge-v1`.
Parent: `gan-fa3-down40-nomerge-v1`. The parent ID, profiles, numerical gates,
capacity policy, source pins and default-off behavior are unchanged.

The only added retained state is the frozen BF16 FFN-up `addmm` output in
Critic blocks38 and39. The existing per-checkpoint PyTorch selective cache owns
it; original forward records it, backward recomputation consumes it in FIFO
order, and the original per-graph drain releases all references. Trainable LoRA
matmuls are not cached, and the existing down40/even-block FA3 caches remain.
There is no merged weight, CPU transfer, extra device fence or collective.

Frozen contract: native world16 GAN, B1/global16, original trace/checkpoint/RNG,
precision, two continuous cycles, G1/C5/EMA1 per cycle and unchanged optimizer
ordering. The SHA-pinned native critic uses finite-difference R1 with one
backward per graph; analytic second backward is outside this candidate's scope.
Actual operands must be frozen BF16 local-GPU leaves with original named owner,
shape, stride, storage identity and graph/version binding. Extra cache counters
must be40 original and40 replay over two cycles, in addition to800 each of the
unchanged frozen-down counts. Native capacity and numerical admission still apply.

CPU mechanism/contract tests do not establish world16 numerical equivalence or
capacity. Native qualification and an unprofiled same-source C2 pair remain
required. No historical runtime may be inherited as this candidate's cost.
