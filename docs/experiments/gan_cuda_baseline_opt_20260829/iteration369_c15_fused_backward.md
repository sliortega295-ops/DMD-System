# Iteration 369: C15 fused modulated-norm backward

## Contract

C15 starts from the promoted C14 world16 GAN-on substrate. Its only runtime
change is a default-off Triton implementation of the already-explicit
modulated-norm backward: BF16 input/upstream-gradient loads, FP32 affine-free
LayerNorm grad-input math, deterministic scale/shift component reductions, and
BF16 output stores. The forward, FSDP2 placement, operation trace, RNG, losses,
model calls, checkpoint policy, update order, residency and Full5 timer are
unchanged.

## Evidence

| Gate | Run | Result |
|---|---|---:|
| A100 B1/B3 canary | `iter368_gan_cuda_c15_fusedbackward_canary_r5_20260830` | PASS; B1 `2.1677x`, B3 `2.9164x` |
| validate-only | `iter369_gan_cuda_c15_fusedbackward_validate_r1_20260830` | 16/16 PASS |
| correctness/capacity | `iter369_gan_cuda_c15_fusedbackward_correctness_r1_20260830` | 16/16 PASS |
| formal unprofiled Full5 | `iter369_gan_cuda_c15_fusedbackward_timing_r1_20260830` | **131.562027140 s** |

Code HEAD is `c2ccd577b4e94a1ea64cf538cd7626373206ebfa`. Summary SHA-256 values are:

- validate: `4fd7b245a56ee7d0f3a67579bf40896d15b6b5f6bad26afcc8463b854151cff4`;
- correctness: `a4926221f33d05697aba70a00b692e09653d1010d8e42b212d8c44894b797cd0`;
- timing: `fc0558053030efa9d1e86607881f40e82312c02c96be984d2e702e716a2c7092`.

Every rank records G1/C5/EMA1 and final logical versions G201/C1005/EMA801.
The Generator receipt records 1,120 fused backward calls, 4,720 source-order
chunks, 5,840 hidden-gradient launches, 5,840 deterministic component-gradient
launches, and zero reference fallback. Maximum allocated and reserved memory
remain `24.963570118 GiB` and `38.294921875 GiB`; CUDA and cgroup OOM deltas are
zero.

## Decision

C15 improves C14 by `4.522654637 s` (`3.323411995%`) and improves the fresh
GAN-on baseline by `41.452455864 s` (`23.958951380%`, `1.315079182x`). It is
therefore promoted as the new validated substrate. The registered target is
still not met: `131.562027140 s` is `10.451889037 s` above
`121.110138103 s`, so the iteration remains active and the next candidate must
be selected from a current overlap-aware profile rather than by extending C15
blindly.
