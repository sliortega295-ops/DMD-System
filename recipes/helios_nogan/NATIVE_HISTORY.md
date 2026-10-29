# Original W2 rank-local history continuation

Use this versioned route when native history dropout/section selection produces
an all-zero-history sample. It preserves that sample's independent amplified
rollout; it does not force every rank onto a normal or identical exit.

Use the same content-pinned assets, original bootstrap, input config and native
feature format as [reference training](REFERENCE_TRAINING.md). Prepare new
inputs with one explicit additional policy:

```sh
distillctl data prepare-helios-stream \
  --sources /absolute/project/sources.reference.json \
  --training-config /absolute/project/inputs.reference.yaml \
  --history-policy source_native_amplified_v1 \
  --output /absolute/project/prepared-native-history
```

An explicit `--input-policy distillschedule.helios_record_epochs.v1` can be
combined with this option for the existing finite feature-pool policy. It does
not change source draw order, introduce feature aliases or manufacture data.

Copy [training.native-history.json](training.native-history.json) into your
project and replace its prepared/assets/output paths. Keep the original
`task.json` and `mode=reference`; the new input ABI separately binds the phase
capability, not a broadened benchmark request.

```sh
distillctl config validate /absolute/project/training.native-history.json
distillctl train /absolute/project/training.native-history.json --dry-run
```

The existing two-node world16 operator launcher and public worker still own real
training. There is no second trainer or GPU launcher. Saves and fresh-process
resume use the same complete-cycle lifecycle, with an independently bound native
history checkpoint policy. A checkpoint from another stream/source cannot be
relabeled as this one.

This route materializes the complete Generator at each aligned action boundary,
retains its parameters through G backward, and releases them before the original
optimizer step. It is not the existing detached-rollout persistence switch.
Full-size memory capacity is not established by CPU admission. See the
[versioned execution contract](../../docs/contracts/helios_nogan_native_history_v1.md)
for required topology, work census and validation scope. The old reference and
SafeBundle routes retain their original input contracts and checks.
