# Helios GAN-on optimization pause summary

## Formal result at the pause point

The frozen world16 GAN-on Full5 denominator improves from
`173.014483004 s` to `125.258735931 s`:

- absolute saving: `47.755747073 s`;
- reduction: `27.602167312%`;
- speedup: `1.381256818x`;
- registered 30%-reduction target: `121.110138103 s`;
- remaining gap to that target: `4.148597828 s`.

The current formal winner is C25. C27 passed only a one-A100 kernel canary and
was paused before world16 integration, so it does not alter the formal result.

## Frozen comparison contract

Every promoted timing below keeps world16 FSDP2 FULL_SHARD, 2x8 A100-40GB,
384x640 BF16, logical steps 1000-1004, B1/global batch16, Score packing off,
CUDA Generator/Critic residuals, CUDA EMA, no parameter offload/streaming/model
swap, G1/C5/EMA1 and final G201/C1005/EMA801. A profiled run is never used as a
formal timing denominator.

## Accepted trajectory

The delta column compares each accepted point with the preceding accepted
point, not with a rejected or diagnostic arm.

| Point | Mechanism | Full5 (s) | Delta from preceding accepted point |
|---|---|---:|---:|
| C0 | Fresh CUDA-resident baseline | `173.014483004` | -- |
| C1.1 | Exact residual/modulation/LoRA fusions extended to the official packed-B3 GAN-D call | `151.851949039` | `-21.162533965` (`-12.2317%`) |
| C2 | Materialize optimizer dispatch before the timer, then restore parameters and Adam state exactly | `144.621745256` | `-7.230203783` (`-4.7614%`) |
| C3 | Pin the validated local FlashAttention-2 backend | `143.379703687` | `-1.242041569` (`-0.8588%`) |
| C4 | Increase exact modulation chunk size from 512 to 2048 | `143.206713110` | `-0.172990577` (`-0.1207%`) |
| C5 | Selective Flash checkpoint replay, stride 2 | `142.295341636` | `-0.911371474` (`-0.6364%`) |
| C8 | Reuse exact Generator no-grad cross-attention text K/V | `141.235767135` | `-1.059574501` (`-0.7446%`) |
| C11 | Replace native-autograd reconstruction with an explicit exact modulated-norm backward | `138.462140212` | `-2.773626923` (`-1.9638%`) |
| C12 | Remove timed-path rollout synchronizations and `empty_cache` calls | `138.176082544` | `-0.286057668` (`-0.2066%`) |
| C14 | Fuse BF16-input LayerNorm and modulation materialization | `136.084681777` | `-2.091400767` (`-1.5136%`) |
| C15 | Fuse the modulated-norm backward path | `131.562027140` | `-4.522654637` (`-3.3234%`) |
| C17 | Move unused Critic scalar logging out of the formal path | `130.630471052` | `-0.931556088` (`-0.7081%`) |
| C18 | Remove the official Generator allocator flush from the timed path | `129.441379467` | `-1.189091585` (`-0.9103%`) |
| C21 | Keep Generator parameters unsharded across the five no-grad rollout calls | `125.814004444` | `-3.627375023` (`-2.8023%`) |
| C25 | Fuse exact history/current add-and-cat staging | `125.258735931` | `-0.555268513` (`-0.4413%`) |

In plain terms, the largest gains came from deleting repeated tensor
materialization and tiny pointwise/cast/norm kernels, moving one-time optimizer
setup outside the timer without changing state, and avoiding repeated FSDP
parameter materialization across the five Generator rollouts. Attention backend
selection and host-path cleanup were useful but much smaller.

## Important rejected or bounded paths

- C6 node-HSDP: `144.716706512 s`, slower than its C5 parent; rejected.
- C13 static rollout-context cache: `138.243935311 s`, slightly slower than
  C12; rejected rather than rationalized as noise.
- C23 persistence without allocator flush: real CUDA OOM; rejected.
- C24 expandable segments plus no-flush: `127.619803944 s`, slower than C21;
  rejected.
- C26 paired RoPE: the fast direct kernel changed bitwise results; the exact
  repair was only about `1.0085x` in the microbenchmark, below the registered
  gate; no world16 run.
- C27 frozen cross-attention LayerNorm: actual-source B1/B3 canary passed the
  unchanged P0 output/gradient gate and measured `2.9772x`/`6.8783x` for that
  microkernel, but world16 integration and timing are `NOT_RUN` because the user
  paused the task.

## Diagnostic boundary

The valid C22 diagnostic profile reports about `5.337 s` exposed NCCL,
`0.353 s` exposed copy activity and `4.900 s` with no traced GPU activity over
its profiled Full5 range. These are overlap-aware diagnostic quantities, not
directly removable time and not a speedup denominator. After C21, dense model
compute dominates; blind NCCL/offload changes are no longer justified.

Detailed receipts remain in `benchmark.csv`, `candidates.jsonl`, the per-
iteration Markdown files, and `preflight.md`. Large raw traces remain on the
remote experiment storage rather than being copied into GitHub.

