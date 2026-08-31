# LiveWan cross-scale SF-DMD G1/C5 world16 contract

## Purpose and claim boundary

This is a controlled **systems-performance** workload derived from pinned
`JonathanColetti/LiveWan` SF-DMD math. It is not the published 8-H200,
effective-batch-64, G1/C1 training denominator and is not a convergence claim.
The sole sequential denominator and every persistent-pool candidate consume the
same immutable bootstrap and operation-keyed trace.

## Frozen logical workload

- Source: LiveWan `1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5`.
- Wan source: `9737cba9c1c3c4d04b33fcad41c111989865d315` plus only the pinned LiveWan attention/config overlays.
- Hardware: 2 nodes x 8 A100-SXM4-40GB, world16.
- Student/Generator: Wan2.1-T2V-1.3B, full-parameter training, source FP32 parameter storage and BF16 autocast math.
- Fake Score/Critic: independent Wan2.1-T2V-1.3B BF16 frozen base plus trainable LoRA rank 32.
- Real Score/Teacher: frozen Wan2.1-T2V-14B BF16.
- Shape: B1 per logical sample, global B16, 368x640, latent `[B,16,21,46,80]`, 512 text tokens; no VAE decode.
- Rollout: block=3, four denoising evaluations per generated block, world choice in `{3,6,9}`, three selected differentiable blocks.
- DMD: Teacher CFG=5; continuous shifted-uniform sigma in `[0.02,0.98]`, shift=5.
- Full5: step 0 performs one Generator commit and one Critic commit; steps 1-4 perform one Critic commit each. Final versions are G1/C5/EMA0.
- EMA is excluded from Full5 and must be validated separately.

Each sample performs one clean world-prefix Student DiT, five Student DiTs per
new block (four denoise plus clean commit), and the fixed score/backward calls.
Checkpoint replay is not counted as an application-level DiT call.

## Sequential denominator

All 16 ranks own all roles. Student uses world16 FSDP2 with
`reshard_after_forward=false`, Teacher uses world16 FSDP2 with
`reshard_after_forward=true`, and the LoRA Critic is resident and replicated.
There is no role overlap, model swap, parameter/activation offload, streaming,
or external rollout engine.

## Persistent 8G+8Score candidate

- ranks 0-7: persistent resident Student/Generator;
- ranks 8-15: persistent resident Critic plus Teacher FSDP2 on the 8-rank pool;
- logical B16 is two B1 waves per role rank;
- GPU P2P transports the generated latent and one raw text context per sample;
- DMD latent gradients return over GPU P2P on step 0;
- two local losses are weighted by 1/2 before an 8-rank mean, preserving global-B16 mean semantics;
- no 14B/1.3B redistribution, ACK, or internal world barrier is allowed.

## Correctness before timing

Timing is eligible only after 16/16 receipts, exact source/bootstrap/trace
identity, G1/C5/EMA0, exact call/update inventories, no cgroup/CUDA OOM, 80
operation-keyed rollout comparisons under the repository P0 multi-metric gate,
and compatible final trainable-state fingerprints all pass.
