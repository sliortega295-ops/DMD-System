# Generic two-role training: capability boundary and current progress

The workload provides **training facts**, not a layout: a generator function,
score objective, optimizer factories, input/condition loading and explicitly
versioned update boundaries. `RoleTask.program()` compiles producer forward,
latent transfer, score input VJP, gradient return and generator parameter commit
using the same versioned `Program` graph. An optional Score optimizer commit is
strictly after all Generator VJPs; currently the executable Score train path
supports **one** such update, not arbitrary GAN/DistillSchedule protocols.

`role_program.enumerate_layouts()` derives contiguous G/Score rank splits and
bounded pipeline-window candidates from **allocated rank capacities**. The
factory supplies no 8+8 rule, no optimization rankings and no model-name
whitelist. Unsupported topology/size combinations are rejected, not remapped.

`RolePoolExecutor` is an actual two-role distributed **execution primitive**:
G and Score models are created only on their assigned ranks, explicitly
initialized process groups own each role, tensor activation is sent from G to
Score, Score computes `autograd.grad(score_loss, latent_leaf)`, the resulting
input-gradient is sent back, and G applies `latent.backward(VJP)` to its own
original forward graph. Generator parameters are never stepped until ALL input
VJPs have completed. Role-local gradients are SUM-reduced after normalization by
global logical microbatch count. A separate optional Score parameter update
runs after the Generator commit. Nonblocking point-to-point sends and a bounded
in-flight G window enable G-forward/Score-evaluate overlap on disjoint ranks.
NCCL requires tag-free send/recv; messages are ordered by each source/target
pair. Runtime enforces output shapes, dtype, finite values, disjoint physical
pool ranks and agreement among role-model replicas.

**Verified:** actual 2-, 3- and 4-process CPU/Gloo transport, 1:1 and asymmetric
pool splits, pipeline windows 1/2, both frozen-guidance and separate Score
optimizer paths. Final model tensors are checked against an independently
computed single-process global-microbatch reference. These are real PyTorch
forward, input autograd, backward, Adam and point-to-point collectives, not
mocked callbacks.

**Not yet:** this primitive is not by itself the common cost-driven selected
plan. Production factory/CLI onboarding, per-layout cost calibration, measured
search/selection, native CUDA/NCCL validation, cross-node 16-GPU performance,
full original model semantics, CPU offload/prefetch, and asynchronous
communication/copy profiling require their own admission. Do not call the
supported CPU transport proof a measured A100/H100 speedup.
