# Direct native optimization presets

The six `recipes/*/training.json` files select versioned native implementations
without requiring a saved plan or calibration campaign. Use an editable source
installation and the matching model/input assets. Current reported results are
summarized in [Performance and validation](performance_validation.md).

## Select and launch

For example, the `execution` section of [W3](../../recipes/helios_gan/training.json):

```json
{
  "adapter": "native",
  "mode": "manual",
  "preset": "helios_gan.fa3_down40_nomerge.v1",
  "task": "task.json",
  "assets": "qualified-native-assets.json"
}
```

Replace asset placeholders and choose a new output directory outside the source
checkout, then inspect and launch on the allocated devices:

```bash
distillctl doctor --config /project/recipe/training.json
distillctl train /project/recipe/training.json --dry-run
distillctl train /project/recipe/training.json
```

W1/W6 use world16 torchrun across two eight-GPU nodes. W2–W5 use the native
two-node dispatcher and the supplied node manifest.

## Preset mapping

| Workload / preset | Main mechanisms | Continuation |
| --- | --- | --- |
| W1 `wan_ar.incumbent21a.v1` | Resident, chunk2048, LoRA/norm fusion, G rollout/cast and C gradient graphs; retain4 disabled | [Incumbent stream](../../recipes/wan_ar/INCUMBENT_STREAM.md) |
| W2 `helios_nogan.safe_ed8.v1` | Node-HSDP, chunk512/window5, SafeBundle normalization/LoRA/replay reuse; warm No-GAN bootstrap | [Safe stream](../../recipes/helios_nogan/SAFE_STREAM.md) |
| W3 `helios_gan.fa3_down40_nomerge.v1` | World16 FSDP, chunk2048/window5, SM80 FA3, down40 replay, separate GAN heads; LoRA unmerged | [GAN stream](../../recipes/helios_gan/NATIVE_STREAM.md) |
| W4 `self_forcing.c51.v1` | Full C51 8G+8Score pipeline with C34 prerequisites | [C51 continuation](../../recipes/self_forcing/C51_CONTINUATION.md) |
| W5 `h3_fullseq.cpu_staged_trim.v1` | Legacy Grid1000 CPU-staged trimmed FA3 replay, depth2/N128; recorded sampled-backward issue | [Original-reference lifecycle](../../recipes/h3-grid-original-training.md) |
| W6 `h3_ar.zero3_ap.v1` | ZeRO-3, selected-cache/unit prefetch, CPU-staged replay, fusion and communication overlap | [Native AR continuation](../../recipes/h3_ar/NATIVE_CONTINUATION.md) |

The original six presets require exactly two native cycles,
`checkpoint_every: 0`, no resume and the original fixed-input contract. They
provide quick comparisons and benchmark reproduction. For long training and
checkpoint recovery, choose the corresponding continuation entry above.
The W5 reference lifecycle is separate from the legacy optimized candidate;
that candidate retains its numerical check and recorded issue.

## Binding and checks

`src/distillschedule/training/native_presets.py` owns the versioned metadata
and typed factories. W2–W5 resolve the selected implementation against current
sources/assets and use the same execution checks as saved-plan mode. W1 and W6
pass their typed strategy to the existing native worker. Numerical, state,
source, capacity and topology checks remain in those workers.

Historical origin fields describe the source experiment. They are not a new
validation result produced by `doctor` or a status for every later recipe.
The original records remain available alongside the current project summary.

Use `mode: manual` with a preset. Do not combine it with `plan`, `implementation`
or `transport`. For a saved selection, use `mode: plan`, supply the plan and
omit the preset. Choose a reference recipe explicitly when that execution path
is needed. See [native workflows](native_workloads.md) for detailed inputs,
continuation limits and output commands.
