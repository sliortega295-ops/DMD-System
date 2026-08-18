# LightX2V MiniMax-H3 world16 DMD baseline

## Question

This baseline answers a narrow systems question: how fast does the public
LightX2V MiniMax-H3 DMD trainer execute one minimax update cycle on the same
2-node, 16×A100-40GB machine used by DMD-System?

It is also the source baseline for a later **matched MiniMax-vs-DistillSchedule
MiniMax** comparison. The existing Helios PR16 result is useful absolute
hardware context but is not a valid speedup denominator.

## Frozen upstream

```text
repository     ModelTC/LightX2V
commit         d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be
trainer        minimax_h3_t2av_dmd
hardware       2 nodes × 8 A100-40GB
parallel       FSDP2 world16, SP=1
Student        LoRA rank 128
Fake           LoRA rank 128
Teacher        frozen
resolution     768 × 1344
frames         124, joint video/audio
```

LightX2V constructs Student, Fake, and Teacher as separate physical
MiniMax-H3 Transformers. Each is then sharded with the upstream FSDP2 plan.

## One benchmark cycle

The upstream outer DMD iteration is:

```text
Student forward/DMD backward/update
        ↓
Fake update #1
Fake update #2
Fake update #3
Fake update #4
Fake update #5
```

Thus the benchmark has exactly:

```text
Student updates = 1
Fake updates    = 5
EMA updates     = 0
global batch    = 16 per update (B1 × world16)
```

MiniMax normally samples the rollout endpoint randomly, causing 1–4 model
evaluations per back simulation. Formal timing fixes `end_step_idx=3`, i.e.
four model evaluations every time. The same fixed endpoint must be used by any
future DistillSchedule MiniMax candidate.

This is what “same 5-step” means in this integration: five ordered Fake updates
after one Student update. It does **not** redefine MiniMax's four-evaluation
distilled sampler as a Helios denoising schedule.

## Timing boundary

Cold source checkout, model loading, FSDP construction, dataset setup, and
trainer/optimizer construction are excluded.

The measured range begins after setup and a world16 barrier, immediately before
the Student update, and ends after the fifth Fake optimizer update plus CUDA
completion and a world16 barrier. Each rank records a receipt; the formal wall
is the world16 MAX wall.

Correctness/acceptance requires:

- 16/16 rank receipts;
- exact upstream commit and benchmark-patch identity;
- world16;
- Student=1 and Fake=5;
- fixed endpoint index 3;
- finite DMD/Fake losses;
- no node cgroup OOM/OOM-kill increment;
- zero process failure.

Formal performance requires at least three independent unprofiled runs.

## Optional segmented-checkpoint canary

The no-offload capacity canary may replace the upstream per-block activation
checkpoint with a bounded group of consecutive blocks.  FSDP2 remains one
`fully_shard` boundary per transformer block; only the autograd checkpoint
boundary changes.  Production values are limited to `segment=2,4,5,8` and the
canary is not a timing result.

This helper is fail-closed against the exact prepared Diffusers source:

```text
Diffusers version:       0.40.0.dev0
Diffusers source commit: 9284607295a09f759aadd65ed08f48b35feea6d9
MiniMax forward SHA-256: 3e041a3400d58cf73d69e4eb540317d45b5f9631a73bb8e765447ca446ee1d41
```

`LIGHTX2V_MINIMAX_SEGMENT_DEBUG=1` is diagnostic-only.  It records segment and
block enter/exit/error events, allocator counters, and any saved-tensor scalar
census supplied by the runtime; it does not synchronize, copy, or retain a
tensor.  A CPU semantic test executes the native reference (`segment=1`) and
the production candidates, checking outputs, trainable gradients, per-block
hook counts, and checkpoint-call counts (`50, 25, 13, 10, 7` respectively).

## Comparison boundary

The current Helios PR16 result is approximately 141.5 s for a different
workload: Helios No-GAN, 384×640, G1/C5/EMA5. MiniMax-H3 uses a different
architecture, 768×1344 joint audio/video, G1/F5/no EMA.

Therefore the repository may answer **which absolute 5-update window is
shorter on the same machine**, but it must label that as cross-workload context,
not a speedup.

The claimable systems comparison is:

```text
LightX2V upstream MiniMax contract
            vs
DistillSchedule MiniMax, exact same contract
```

Only that second comparison can support a DistillSchedule speedup claim.
