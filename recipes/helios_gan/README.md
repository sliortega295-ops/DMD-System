# W3 — Helios14B GAN-on

**Original protocol:** DMD + GAN-G/GAN-D; G1/C5/EMA1.

The optimized bundle uses world16 FSDP2, chunk2048/window5, SM80 FA3,
frozen-down replay, separate GAN heads and LoRA-B fusion. LoRA stays unmerged;
runtime merge counters must remain zero.

## Choose a training path

- [`training.json`](training.json): `helios_gan.fa3_down40_nomerge.v1`, the
  original two-cycle fixed-input comparison preset.
- [Native stream](NATIVE_STREAM.md): full-bundle continuous updates, complete
  G/C/GAN-head state, checkpoint and restart.
- [CPU reference](REFERENCE_TRAINING_CPU.md): original model and GAN update
  logic through the common training lifecycle.

Supply the matching model, input and hardware configuration. Use the native
stream for long training and checkpoint recovery.

## Inspect and run

```bash
python scripts/distillctl.py workloads candidates --workload helios14b_gan_best --include-helios-gan-manual
```

The [native runbook](../../docs/runbooks/native_planner_workflow.md) covers
manual and saved-plan execution. [Native output](NATIVE_OUTPUT.md) covers the
current unmerged Student, Wan2.1 VAE and new-prompt video generation; the
[CPU output path](PORTABLE_INFERENCE_CPU.md) is documented separately.

## Implementation and results

The code is in `src/distillschedule/search/helios_model_adapter.py` and
`src/distillschedule/runtime/helios_gan_manual/`.
[Direct presets](../../docs/guides/native_presets.md) describe the source and
configuration bindings.

See [current performance and validation](../../docs/guides/performance_validation.md).
Earlier N70-on/off trials keep their original measurements in the research
records; their different LoRA settings must not be combined into one comparison.
