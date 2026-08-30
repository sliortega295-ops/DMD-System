# BAAI LiveWan cross-scale world16 runbook

Read `docs/contracts/livewan_crossscale_g1c5_world16.md` first. Do not substitute
another checkout, Python archive, trace, or checkpoint.

## Required identities

- DMD-System: exact clean experimental HEAD recorded in the run manifest.
- LiveWan submodule: `1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5`.
- Wan2.1 submodule: `9737cba9c1c3c4d04b33fcad41c111989865d315`.
- Python: validated Torch-2.10 archive extracted at
  `/tmp/distillschedule-helios-official-torch210` on both nodes.
- Python overlay: immutable user-owned dependency directory with a PASS receipt;
  it supplies only missing LiveWan Python packages (currently EasyDict 1.13)
  through `PYTHONPATH` and never edits the validated Torch environment.
- Student/Critic base: pinned Wan2.1-T2V-1.3B model root.
- Teacher: pinned Wan2.1-T2V-14B model root.
- Inputs: at least 80 source-compatible 21-latent-frame samples and the exact
  LiveWan prompt embedding bundle.

## Artifact preparation

Prepare the immutable source overlay without installing into the training env:

```bash
scripts/distillschedule/prepare_livewan_crossscale_source.sh \
  --output "$LIVEWAN_WAN21_SOURCE" \
  --receipt "$LIVEWAN_SOURCE_RECEIPT"
```

Create the rank-qualified trace and shared systems-only bootstrap:

```bash
python scripts/distillschedule/build_livewan_crossscale_trace.py \
  --data "$LIVEWAN_TEACHER_LATENTS" --prompts "$LIVEWAN_PROMPTS" \
  --output "$LIVEWAN_OPERATION_TRACE"

python scripts/distillschedule/build_livewan_crossscale_bootstrap.py \
  --student-base "$WAN13B" --critic-base "$WAN13B" --teacher-base "$WAN14B" \
  --student-state "$LIVEWAN_STUDENT_STATE" \
  --trace "$LIVEWAN_OPERATION_TRACE" \
  --source-receipt "$LIVEWAN_SOURCE_RECEIPT" \
  --output "$LIVEWAN_SYSTEMS_BOOTSTRAP"
```

## Ordered gates

1. Run focused tests and plan render locally.
2. Run controller `--run-kind validate` for baseline and pipeline. The node
   launcher holds an atomic GPU lock and runs the model-free world16 topology
   probe; pipeline also runs the exact two-wave P2P probe.
3. Run one baseline correctness/capacity trial.
4. Run one 8G+8Score correctness/capacity trial.
5. Generate the paired correctness receipt. Only a PASS receipt unlocks timing.
6. Run one unprofiled baseline denominator and one matched candidate timing.
7. If speedup is below 1.30x, profile the candidate and change only placement,
   legal ordering/overlap, transport, or exact kernels. Do not silently change
   the workload.

The controller defaults to SSH port 2222 and copies only small summaries to
`/home/lyy/Helios/research_artifacts/DMD-System/livewan_crossscale_20260830`.
Pass `--remote-output-root` explicitly; raw model/run/profile artifacts remain
under that persistent shared-storage directory and are never written into the
clean source checkout.
