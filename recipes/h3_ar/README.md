# W6 — Autoregressive Audio-Video DMD (Original DiffSynth H3)

**Original protocol:** Previous+current AV chunks; continuous sigma; 8-step; G1/F1; LoRA64.

W6 uses the original DiffSynth AR trainer. Its A100 ZeRO-3/native-AP and
H100 native-FSDP/FA2 paths have separate hardware configurations.

## Choose a training path

- [`training.json`](training.json): `h3_ar.zero3_ap.v1`, the original
  two-update native comparison preset.
- [Native continuation](NATIVE_CONTINUATION.md): `h3_ar.zero3_ap_stream.v1`,
  fresh rank-qualified inputs, up to 100 cycles and native ZeRO-3 checkpoint/restart.
- [Cached-AV CPU reference](CPU_REFERENCE.md): original cached model source
  through the common Program lifecycle and complete-state recovery.

Use the original previous/current AV inputs, source and capacity settings.
The two-update benchmark preset and the finite-stream lifecycle are distinct
entrypoints; choose the latter for long training.

## Inspect and prepare inputs

```bash
python scripts/distillctl.py workloads native validate --task configs/distillschedule/original_h3_ar/w6_reference.json
```

The [native runbook](../../docs/guides/native_workloads.md#original_h3_ar_native) covers execution.
[Raw AV preparation](../../docs/guides/data.md#h3_audio_video_encode) converts frame
directories, stereo32kHz PCM WAV and preencoded text to the original cached-input
representation. Use the supplied `encoder.native.template.json` and
`raw.native.template.json` with your local model assets.

## Implementation and results

The code is in `src/distillschedule/strategies/h3_ar_cli.py`, `h3_ar_session.py`
and `h3_ar_planning.py`. See [direct presets](../../docs/guides/native_workloads.md#native_presets),
[current-Student output](../../docs/guides/training.md#h3_ar_native_inference) and
[current performance and validation](../../docs/guides/performance_validation.md).
Earlier A100/H100 trials retain their original source and measurement scope;
W6 does not use W5's Grid1000 sampler.
