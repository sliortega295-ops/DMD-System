# Iteration 365: C12 explicit rollout-fence cleanup

## Contract

C12 changes only one default-off policy on top of C11: omit the one
Generator-update `empty_cache` and six application-owned post-rollout
`cuda.synchronize` calls.  Full5 start/end synchronization, world16 FSDP2,
GAN-on steps 1000--1004, B1/global-B16, trace/RNG/loss/physical calls,
attention/checkpoint/cache/residency, optimizer/EMA ordering and all numerical
gates are frozen.

## Evidence

| Gate | Run | Result |
|---|---|---|
| validate-only | `iter365_gan_cuda_c12_nosync_validate_r1_20260830` | 16/16 PASS |
| correctness | `iter365_gan_cuda_c12_nosync_correctness_r1_20260830` | 16/16 PASS |
| formal timing | `iter365_gan_cuda_c12_nosync_timing_r1_20260830` | `138.176082544 s` |

Correctness and timing independently record:

- G1/C5/EMA1 and final G201/C1005/EMA801;
- Generator-update empty-cache count = 0;
- Generator-update post-rollout sync count = 0;
- five Critic-input rollout sync count = 0;
- outer Full5 boundary synchronization preserved;
- maximum allocated/reserved = `24.963356495/38.3125 GiB` per rank;
- zero CUDA and cgroup OOM delta.

Summary SHA-256 values are:

- validate: `fba1c359aa197fd5391fa74c6cb7067193f23465c9464e8cbf489502dedaef81`;
- correctness: `16ce5afb0628264db0897e38eca1058f8e8c95fdcd69e731475cb33e6bdd8441`;
- timing: `075acaeffc4719e2c3e4c2c496646b747c599619d2e8d27d9966ac109f5e026e`.

## Result boundary

C12 saves `0.286057668 s` (`0.206596%`) versus C11 and
`34.838400460 s` (`20.136118%`) versus C0.  The small isolated gain falsifies
the hypothesis that the application-owned rollout fences account for most of
the profile's synchronization API wall.  C12 is the fastest valid substrate,
but the requested 30% threshold is still `17.065944441 s` away.
