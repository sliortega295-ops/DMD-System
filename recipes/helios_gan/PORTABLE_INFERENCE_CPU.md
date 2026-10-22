# Current full Generator: portable CPU native latent diagnostic

This is the inference companion to [CPU reference training](REFERENCE_TRAINING_CPU.md),
not a replacement solver or a new native benchmark. It exports the **actual
restored current, full, unmerged G**: every frozen base and trained PEFT tensor,
not EMA, C, Adam, training data or RNG. The existing named-module export format
(`distillschedule.inference_export.v1`) provides safe tensor loading, strict
key/shape/dtype/finiteness checks and atomic hash-bound publication.

## Export once, while restart inputs are available

```bash
python scripts/distillctl.py export /absolute/project/train.json \
  --checkpoint /absolute/project/run/checkpoints/cycle-000002 \
  --output /absolute/project/current-generator
```

Export first admits/restores the ordinary common training checkpoint, so its
original same-topology assets and prepared training inputs are required **at
export time**. Projection references the live restored G without changing its
weights, modes, gradient flags, adapters or optimizer. It does not run G or
merge/apply EMA. The export's model identity is the **inference factory's**
source and metadata; the entire training identity is separate
`identity.training_provenance`, never an inference dependency.

The exported G then stands alone with respect to all training payloads:
prepared/raw training data, initial base-weight files, negative-teacher
conditioning, training Args JSON, restart checkpoints and run directories may
be unavailable. This is **not** a self-contained Python executable: the pinned
original source package, public provider/backend and compatible installed
runtime dependencies are still required. The current original Helios provider
requires an **editable DMD-System checkout** with its admitted original helper
tree; the lightweight framework wheel alone is not this native source bundle.
That checkout may be relocated independently of the old training payloads.
Inference identity pins source **content**, not old absolute
source/dependency/base/dataset paths. Installed
relocation is distinct from same-topology training restart.

## New prompt inputs, not five unused training rows

Use `data import` with a manifest like:

```json
{
  "schema_version": "distillschedule.data.preencoded.v1",
  "records": [{
    "id": "new-prompt-001",
    "features": {"path": "prompt.pt", "sha256": "<actual SHA256>"}
  }]
}
```

Each safe Torch feature file is **exactly** `{"prompt_embeds": Tensor}`. Supply
real original-native prompt embeddings, already CPU FP32, with the exact
`export.json.model.parameters.geometry.prompt_shape` (`[1,text_length,text_dim]`).
There is no tokenizer/text-encoder fallback, prompt projection, dtype cast,
reshape, GT-reference fabrication or reuse of the old training rows. No
negative conditioning is needed: this original Generator rollout does not use
the negative-teacher branch. New inputs are explicitly independent of training
input provenance; fixture weights remain fixture evidence even with real
prompt inputs.

```bash
python scripts/distillctl.py data import /absolute/project/new-inputs/inputs.json \
  --output /absolute/project/new-inputs/prepared
```

Copy `inference.reference.cpu.template.json`, then replace its entire `model`
section with these exact values from `export.json` (do not guess/rewrite them):

```python
import json
from pathlib import Path
artifact = json.loads(Path('/absolute/project/current-generator/export.json').read_text())
request = json.loads(Path('/absolute/project/inference.json').read_text())
request['model'] = {key: artifact['model'][key] for key in ('factory', 'parameters')}
Path('/absolute/project/inference.json').write_text(json.dumps(request, indent=2) + '\n')
```

Set `data.prepared` to the **new** prepared manifest. The template's empty
parameters intentionally refuse execution until filled. Keep the explicit
`refuse_training` algorithm binding: generation never imports/constructs the
algorithm, and training/checkpoint evaluation from this inference-only config
clearly refuses. `trainer.cycles` and `output_dir` are required by the existing
config envelope, not a request to train. `trainer.seed` is the generation noise
seed; the exported factory's stored seed remains exact model identity and is
not reseeded by its constructor.

## Explicit diagnostic, never silently called video

```bash
python scripts/distillctl.py generate /absolute/project/inference.json \
  --weights /absolute/project/current-generator \
  --prepared /absolute/project/new-inputs/prepared/manifest.json \
  --output-kind latents --output /absolute/project/native-latents
```

The API equivalent is `generate_from_export(..., output_kind='latents')`.
The default remains `video`: it refuses without an actual decoder returning
THWC uint8 RGB/RGBA. No fake VAE, RGB reshape, PNG frames, or media-quality
claim is supplied. The explicit latent path saves safe `generation.pt` and a
receipt declaring finite named FP32 `latents` in **BCTHW** layout, exact shape,
`LATENT_DIAGNOSTIC_NOT_MEDIA_QUALITY`, and `quality_validation=NOT_RUN`.

The inference factory constructs only the bounded original full G plus the
same PEFT, initializes original nonpersistent RoPE through the original CPU
constructor, freezes parameters and disables gradient checkpointing. This is
essential because original `run_generator` internally re-enables gradients.
All full exported weights must pass staged strict admission before load; no
missing key may use constructor/random/base/Hub fallback.
Architecture admission is CPU-bounded before real construction (including a
128-layer limit, declared parameter/loaded-byte limits and native shape checks).

The actual original `utils_helios_post.run_generator` and scheduler own all
noise, pyramid/denoising/history operations. This bounded first lane is
**explicit original zero-history**, last-step-per-stage, one latent section,
cache/corruption/saturation off. Native stage budgets and dynamic-shift Args
are frozen into the export. It is not the GT-history training rollout, decoded
media, held-out quality, pretrained-quality proof or a performance benchmark.
CPU1 FP32/dense SDPA only; CUDA, all-six workloads, topology conversion,
training restart from inference weights, EMA export and decoded video remain
unsupported here. Tests use an honestly labelled saved nontrivial original tiny
fixture; the original constructor's legitimate cold-zero fixture stays distinct.
