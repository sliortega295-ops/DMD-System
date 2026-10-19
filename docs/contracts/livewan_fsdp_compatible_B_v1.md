# LiveWan R2 FSDP-compatible B subset v1

This is a separate opt-in arm, `only_B_fsdp_compatible`, not full `only_B`.
The existing full-B rejection and the baseline/only_A/full/Cxx contracts remain
unchanged. Its effective plan ID is
`livewan-full-system-factorial-r2-only_B_fsdp_compatible-c4`.

## Frozen execution and semantics

Use the original baseline parent, execution mode `baseline`, variant `b1`,
all-role world16 FULL_SHARD with reshard-after-forward, physical B1, unchanged
max-noop Student collective padding, GPU optimizers and no offload. No role pool,
shadow, runahead, lookahead or batch/gradient strategy from A is introduced.
The original 368x640x81-frame, global-B16, continuous sigma, seed 20260830,
trace/bootstrap/source, dtype/autocast, loss, G1/C5/EMA0 and numerical/capacity
policies remain fixed. One initialization executes four complete cycles; C1 is
qualification/warmup and C2-C4 use the unchanged baseline CUDA completion timer.

## Enabled compatible mechanisms

- Generator fused AdamW, with original learning rate/betas/epsilon/weight decay,
  on the actual sharded DTensor parameters. The original worker materializes
  its scalar step and DTensor moments. Critic remains original foreach AdamW.
- Existing scale-one Critic LoRA BF16-autocast input-cast elision.
- Existing BF16 Score modulation, frozen BF16 RMSNorm and BF16 RoPE operators,
  for Critic and Teacher only, with all original source/dtype/shape/autograd
  guards unchanged.

The original installers are reused, not bypassed. Their process-global function
patches are wrapped by explicit Critic/Teacher call scopes on every rank. Outside
these scopes every Student call dispatches to the exact original function.
The Critic scope surrounds the complete forward and backward, including native
checkpoint re-entry. Nested scopes and exceptional exits restore the previous
role. No FSDP parameter view is retained, pinned or cached across calls.

## Disabled and interaction-dependent mechanisms

Original computation is retained for text-K/V caching, rollout cast caching,
Generator RMS/modulation/gate/residual-layout, specialized Generator attention
and readonly split-KV, Score FA3/checkpoint-output reuse and the pinned Score
gate/LayerNorm composition. Their resident/causal-layer/parameter-lifetime,
B2-query19320 or composition contracts have not been adapted here. These are
explicit A-by-B or unadapted interactions, not silently enabled metadata.
Original 512 context lengths, native checkpoint recomputation and original RNG
handling are retained. No Graph backend, policy search or offload is added.

## Evidence and acceptance

The plan lowers from the exact SHA-pinned baseline and separately binds the
compatible operator contracts and disabled-mechanism matrix. Both node effective
plans and all sixteen rank receipts must match source/arm/seed/plan SHA. The
collector uses baseline physical/logical census and world-completion endpoints.
Mandatory observations include actual DTensor placement, actual fused optimizer
flag/parameters, Critic/Teacher scope entries, positive completed kernel counters,
original Student dispatches, original checkpoint body/re-entry counts, finite
states, resources and unchanged OOM counters.

Scope dispatch attempts are labeled separately from completed operator counters:
non-reentrant checkpoint early-stop may interrupt an entered wrapper before its
optimized leaf. A completed count must be positive and no greater than dispatch
attempts. This does not relax the exact model/update/checkpoint-body census.

CPU tests mock CUDA/model leaves and identify synthetic receipts as
`SYNTHETIC_TEST_ONLY`; the native CLI must reject them. Real CPU checkpoint and
tiny DTensor optimizer tests establish only their stated CPU mechanisms, not
CUDA kernel correctness or distributed timing. Target Torch2.10 CPU acceptance
and the supervisor-owned native allocation remain required. Numerical status
stays REVIEW, never strict equivalence, and this subset never represents full B.
