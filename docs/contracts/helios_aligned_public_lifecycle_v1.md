# Helios aligned public lifecycle

W2 and W3 use the existing `benchmark_v4_strict_global` request to select the
previously measured synchronized rollout scope. The explicit request travels
through the public input producer, each cohort, runtime input admission,
checkpoint identity and current-Student export. `source_normal_only` remains
the default. The native rank-local history extension keeps its separate input
kind and cannot be combined with this request.

Prepare inputs with the public commands:

```sh
distillctl data prepare-helios-stream --sources sources.json \
  --training-config native-nogan.yaml --output prepared \
  --stage-schedule-policy benchmark_v4_strict_global

distillctl data prepare-helios-gan-stream --sources sources.json \
  --training-config native-gan.yaml --output prepared \
  --stage-schedule-policy benchmark_v4_strict_global
```

The task file used by training must contain the same explicit
`stage_schedule_policy`. All other request fields retain the original
384x640,33-frame,three-stage222,world16,B1/global16,G1/C5 contract. W2 preserves
EMA5; W3 preserves its original physical/curriculum clock mapping and EMA1.

Input preparation reads every original per-rank feature with the original
history, noise and scheduler implementations. It preserves sampled history,
including all-zero history. Each rank keeps its corresponding doubled native
scheduler grid when history is zero, while the selected exits retain the
historical globally synchronized scope. The worker rechecks these controls
against the actual history payload before model construction. No sample is
removed and no newly captured native rank-local stream is relabeled.

Training uses the existing public native lifecycle. For a short functional
check, retain a two-cycle input horizon, save and stop at complete cycle1,
start a fresh process from that checkpoint, then finish cycle2. Every saved
checkpoint includes both optimizers/schedulers, sparse EMA, all rank RNG,
input cursor and original version clocks. `distillctl export` restores the
complete terminal boundary and exports the current unmerged Student with
zero training or EMA updates. Use the existing standalone inference config
for generation with independently prepared prompt features.

W2's `helios_nogan.safe_stream_ed8.v1` and W3's
`helios_gan.fa3_down40_nomerge_stream.v1` retain their existing execution
policies. The aligned request does not activate the native-history
whole-Generator retention extension. Native capacity and output execution
are checked on the target GPUs.
