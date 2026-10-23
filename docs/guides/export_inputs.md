# New inputs from portable weights, without training assets

`generate --weights` consumes prepared inputs. For adapters with an explicit
export-input capability, the existing `data prepare --preparer` entrypoint can
now create those inputs **from the export itself**, including its actual text
encoders, tokenizers and VAE. It does not reopen a training bundle or checkpoint.
Choose only trusted exports and adapter code, as with all explicit Python plugins.

## Supported extent

The implemented native capabilities are FLUX.2 klein Base T2I, original
Qwen-Image T2I, Wan2.2 TI2V-5B's **T2V subpath**, and HunyuanVideo-1.5 T2V.
They retain their existing pinned `modern-flow`, CPU1/FP32/B1, unmerged-LoRA,
four-step generation and native geometry/tokenizer/codec admission contracts.
Other adapters (including SD, official Self-Forcing and LTX) have no such hook
yet and **refuse**, even if their separate asset-based preparer works.

Actual tiny upstream model classes are tested end to end: train an updated
Student, export Student plus codecs, copy the package/export to a non-repository
directory, **delete all original assets and training checkpoints**, prepare a
never-used prompt and newly created raw media through public commands, then
decode with the current Student. The decoded result matches the native restored
current Student exactly. No hand-published conditional features are used.
This is **SYNTHETIC_TEST_ONLY** CPU functionality. Full pretrained execution,
full-model capacity, production geometries, GPU/AMP/distributed execution,
performance, convergence and quality are **NOT_RUN**. It is not mature-release
or pretrained-family acceptance. Existing family configuration limits still apply.

## 1. Make an explicit new-input manifest

Save `new-inputs.json` next to `export/`. There is no model-name field; the
validated export supplies its factory and native encoding capability.

```json
{
  "schema_version": "distillschedule.export_inputs.v1",
  "weights": "export",
  "mode": "prompts",
  "records": [
    {"id": "new-001", "prompt": "a violet kite", "latent_shape": [1, 4, 4, 4]}
  ]
}
```

The example shape is **only the tiny image fixture's** native batch/channel/
height/width geometry, not a universal image shape. Supply an explicit geometry
compatible with the exported architecture. Native video shapes use **BCFHW**:
the tiny Wan fixture uses `[1,4,2,2,2]`; the tiny Hunyuan fixture uses
`[1,4,2,4,4]`. No dimensions, null conditions, crops, resize or frame rates are
guessed. Prompt-mode zero latents are explicit geometry/noise templates, not
invented images or conditional features. Empty text is allowed only as an
explicit string and is encoded by the real native empty-prompt path.

For `"mode": "raw"`, replace `latent_shape` with exactly one native media
reference: `"image": {"path": "new.png", "sha256": "ACTUAL_FILE_SHA256"}` for
images, or `"video": {"path": "new-frames", "sha256": "ACTUAL_INVENTORY_SHA256"}`
for video files/frame directories. Retain `id` and `prompt`. Frame inventory
hashes use the existing `data.manifest.source_identity(path, frames=True)`
contract (sorted supported filenames and file hashes), not a hash of directory
metadata. Raw RGB and native spatial/temporal compression divisibility are
required; no automatic color conversion, truncation, resizing or resampling.
Input references are local only and relative to the manifest.

## 2. Prepare and validate using the public entrypoint

```sh
distillctl data prepare new-inputs.json --output new-prepared \
  --preparer distillschedule.data.export_inputs:prepare_inputs
distillctl data validate new-prepared/manifest.json --sources
```

The full export inventory, completion marker, factory source, runtime, effective
metadata, module layout, safe weight tensors and native callback are admitted
before publication. Native conditions come only from restored codecs. Python,
NumPy and Torch CPU RNG are restored on success or failure. Encoding is
inference-only; exported modules remain content/ownership guarded. Preparation
is streaming and atomic, never replaces an existing output, and removes its
staging directory/lock on failure. Wrong shapes, unknown conditions, duplicate
IDs, missing capability, inconsistent callback metadata, changed raw media or
export inventory and invalid weights refuse without a prepared publication.

Prepared provenance records the export manifest/files/factory, native encoder
source/contract, new record IDs, inline prompt hashes and actual raw inventories;
it does not name removed training bundles as the new input source. Moving an
export/package is supported when code bytes and runtime match; prepare from the
new location. Later `--sources` validation requires the recorded export/code/raw
paths to remain available. Ordinary feature loading validates prepared hashes
without re-encoding. Explicit local `.py` factory paths must still be present;
prefer an installed importable adapter module for portable deployments.

## 3. Generate from the same export

Create an ordinary `distillschedule.training.v1` inference configuration. Copy
the `model.factory` and **exact** `model.parameters` from `export/export.json`,
use `algorithm: {"name":"dmd"}`, `data.prepared` pointing to the new manifest,
`execution: {"mode":"reference"}`, and a CPU trainer with an independent seed,
one cycle and an unused output directory. This is not the old training request;
do not add asset paths, Teacher/Fake, base reloads or training checkpoint state.

```sh
distillctl generate inference.json --weights export \
  --prepared new-prepared/manifest.json --output generated --output-kind image
# For the admitted video adapters, use --output-kind video.
```

Images are actual decoded PNGs. Video output is an actual decoded **PNG frame
sequence** plus THWC uint8 tensors: no FPS, playable-container or quality claim
is invented. See the [shared image recipe](../../recipes/flow_image/README.md),
[Wan recipe](../../recipes/wan22_dmd/README.md), and
[Hunyuan recipe](../../recipes/hunyuan_video_dmd/README.md).

## Adapter extension contract

`ModelComponents.encode_input` is optional and defaults to `None`. The common
preparer calls it only on fully restored local inference state:

```text
encode_input(state, record, {mode, source_dir, model_metadata})
    -> {features: native CPU tensor tree,
        metadata: nonempty finite JSON object, sources: raw source identities}
```

The callback owns exact record fields, geometry, native tokenization/codec use,
feature validation and prepared schema/binding metadata. `sources` maps actual
image/video/audio/text to the existing source-identity schemas. Metadata must
agree across every row; `_export_inputs` is reserved for the common export
receipt. Callbacks must not mutate state, inputs or external files. Violations
refuse publication; arbitrary trusted-plugin external side effects are not
rolled back. Adding another family requires its own capability and native
proofs, **not** changes to this preparer, the trainer or planner. Existing
asset-based preparers and all frozen W1–W6 contracts remain separate/unchanged.

Verification: `tests/test_data_export_inputs.py` covers the unknown-plugin
extension, transaction/refusal and RNG contracts;
`tests/test_data_export_inputs_native_cpu.py` covers all four native public
CLI-only export/new-input/decoded-output paths in the opt-in modern-flow lane.
