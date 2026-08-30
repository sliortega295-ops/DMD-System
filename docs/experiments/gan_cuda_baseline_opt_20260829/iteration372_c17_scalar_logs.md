# Iteration 372: C17 elide unused official Critic scalar logs

## Contract

C17 starts from the promoted C15 world16 GAN-on workload and changes only the
four diagnostic scalar conversions at the end of each official GAN Critic
update. The installed `_critic_loss` source is pinned by SHA-256 and rewritten
in memory; the shared Helios source is not modified. Correctness mode retains
all official `.detach().item()` conversions, while formal timing returns an
explicit omission sentinel for values that the P0 caller does not consume.
Losses, backward, optimizer commits, tensors, collectives, update order and the
Full5 timer are unchanged.

## Live preflight

- DMD-System HEAD: `b295c5e02e1555f06f4d7acf82e151eeac84c467`, clean on both nodes.
- Helios transformer SHA-256:
  `5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886`.
- Validated Torch 2.10 environment archive receipt:
  `d5532f80bcf1e74e1028c93b0400897259e72a1b547f4f7a92a5ae508cff64c7`.
- Both nodes: eight A100s, zero pre-existing compute processes, free atomic
  lock, and cgroup `oom=0`, `oom_kill=0`.

## Evidence

| Gate | Run | Result |
|---|---|---:|
| focused tests | exact environment source install + focused unit suite | PASS |
| validate-only | `iter372_gan_cuda_c17_scalarlogs_validate_r1_20260830` | 16/16 PASS |
| correctness | `iter372_gan_cuda_c17_scalarlogs_correctness_r1_20260830` | 16/16 PASS |
| formal unprofiled Full5 | `iter372_gan_cuda_c17_scalarlogs_timing_r1_20260830` | **130.630471052 s** |

The validate, correctness and timing summary SHA-256 values are respectively:

- `03c46038a52cf1c4949fa038f7dea66e48f4a7f50f27b474e52b16bdce94b66e`;
- `1dd2043932ad5826e85428d38c4f2a48bea77c932c78d578d46547ee526fd641`;
- `b29e54eceaaef5e92e2c684912aceaf0b52bd62fe85218512dfe631d3eac68d8`.

Every correctness rank records five official Critic calls, 20 collected scalar
values and zero elided values. Every formal timing rank records five calls,
zero collected values and exactly 20 elided values. All ranks finish with
G1/C5/EMA1 and logical versions G201/C1005/EMA801. Maximum allocated and
reserved HBM remain `24.963570118 GiB` and `38.294921875 GiB`. Both nodes end
with no GPU processes or lock holder and zero cgroup OOM counters.

## Decision

C17 improves C15 by `0.931556088 s` (`0.708073681%`) and improves the fresh C0
baseline by `42.384011952 s` (`24.497378032%`, `1.324457315x`). It is promoted
as the current fastest validated substrate. The registered 30%-faster target
is `121.110138103 s`; C17 remains `9.520332949 s` above it and still needs a
`7.287987919%` reduction. This source-qualified host-fence cleanup is complete;
further work returns to the remaining compute/NCCL timeline rather than
removing more correctness data blindly.
