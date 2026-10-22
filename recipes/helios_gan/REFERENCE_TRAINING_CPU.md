# Original Helios GAN: public CPU reference lifecycle

This opt-in recipe is **not** the native W3 benchmark in `training.json`.
It uses the ordinary public `train` / checkpoint / resume lifecycle, the
existing Program dispatcher, the complete original Helios class, real PEFT,
and the original Helios loss/three-stage rollout helpers. It does not choose a
benchmark winner or replace the planner/solver.

**Supported:** one CPU process, FP32, explicit dense `torch-sdpa`, resident
physical/global batch 1, accumulation 1, fixed native ratio 5, one latent section,
original GT history `[16,2,1]`, final GAN head, no GAN hooks/cache. Both original
separate-gradient policies are identity-bound. The engine adapter's name does
not imply FSDP wrapping. CUDA, distributed training, auto/manual/saved-plan
modes, epochs/shuffling, periodic validation and replay are not admitted
here. Portable current-G native latent inference uses the separate
[inference-only projection](PORTABLE_INFERENCE_CPU.md). Existing W2 and native W3 configurations are unchanged.

Install the separate public runtime with `python -m pip install -e '.[train,helios-reference]'`.
It includes the actual original Args/config dependencies; pytest and the
`test` extra are not required for public training. Do not combine its Diffusers
0.37.1 environment with the helper0.34 or native-Wan0.35 groups. Dedicated CI
adds tests independently and keeps the qualification alias identical.

## Supply local assets, never a fallback model

Copy the two `*.reference.cpu.template.json` manifests into your project and
replace every placeholder with actual files and pins. These templates fail
closed as supplied. There are **no bundled pretrained small weights, training
features or implicit personal paths**. A real functional run requires your own
valid local full-source-class weights/features that fit the explicit CPU budget.
No real-data or pretrained-quality run is claimed by this recipe's tests.

The asset manifest must contain:

- Exact original provider/source identity (`reference_model_identity` with
  `backend='torch-sdpa'`) and imported dependency versions/paths
  (`helios_public_assets.runtime_identity`). Source packages must be installed
  at the same admitted locations for same-topology restart.
- One complete effective original Args JSON file, not a partial set of
  overrides. It is merged against the source `helios.utils.train_config.Args`
  schema and required to equal that complete result. The manifest binds the
  entire file, including PEFT, optimizer, clipping, LR horizon, scheduler and
  GAN/EMA curriculum. Unsupported branches are rejected, not silently ignored.
- G and C architecture JSON plus an explicit list of **every** safetensors
  shard. Each file pin has absolute canonical local `path`, positive `bytes`,
  and SHA256. No implicit index scanning, missing-key initialization or Hub
  lookup occurs. Supply full C GAN-head weights too; this first contract does
  not admit missing pretrained head keys. Duplicate, missing, extra, nonfinite
  or non-FP32 base weights are errors. Constructors run on meta storage; all
  real base tensors must come from validated local weights. Nonpersistent RoPE
  buffers are reconstructed by the actual original deterministic positional
  constructor, with checked dtype/layout and no RNG draw; `to_empty` is not
  used to leave them uninitialized.
- A safe negative-conditioning file containing exactly `negative_prompt_embeds`
  with the declared `[1, text_length, text_dim]` FP32 shape. Every safetensors
  payload must declare `evidence_kind` in its immutable header metadata.
- An explicit total parameter budget and loaded-byte budget; budgets include
  G+C and LoRA capacity. This bounded reference refuses a parameter budget over
  100 million or a loaded-byte budget over 2 GiB, not a claim that such sizes
  have been benchmarked or are fast. Geometry must fit original pyramid/history
  operations; spatial sizes are multiples of 16. Geometry is never resized.

Factory parameters accept only the explicit **absolute** `asset_manifest` path
and ordinary session-supplied `device` / `seed`. They never contain dataset
paths, tensors, checkpoints or captured model objects. The factory does not
reseed and does not initialize a random fallback base. Adam/LR/EMA start at
actual zero, not the W2 Step10 bootstrap.

## Prepare original native features

The input manifest lists a finite multiple of five distinct source files. Each
safe Torch file contains original `vae_latent` `[sections,16,frames,H,W]`,
`prompt_embed` `[text_length,text_dim]` and `prompt_raw` string. An optional
`evidence_kind` field must agree with the manifest. Native sources are loaded
with `weights_only=True`; unsafe Python objects are not accepted. Source path,
file hash and native tensor content must all be distinct; changing only an ID,
seed, prompt label or container encoding is not fresh data.

Each record identifies its seed, epoch and sequential `native_index`. The
source-pinned **original** `BucketedFeatureDataset.prepare_stage1_latent` runs
with `return_secondary=True` and its actual hardcoded history `[16,2,1]`.
Primary target/history/x0 feed G and real GAN data; secondary target/history/x0
feed C. Original GAN conditioning feeds both conditioning slots. Selection
indices and file/content identities are recorded as metadata. Only an explicit
CPU FP32 conversion of native floating features is allowed; no crop, resize,
embedding projection, precomputed G rollout, DMD timestep/noise or corruption is
prepared here.

```bash
python scripts/distillctl.py data prepare-helios-gan-reference \
  --sources /absolute/local/helios-reference/inputs.json \
  --output /absolute/local/helios-reference/prepared
```

The result is the existing immutable PreparedDataset format: one tensor-only
five-row record per public cycle, with fixed-size identity/index tensors and
provenance metadata. It is not W2's timestep-captured `prepare-helios-stream`.
The public model factory never opens the dataset.

## Train and resume

Copy `training.reference.cpu.json`; use actual absolute project paths. Set the
target cycles no larger than the existing finite inventory. The original Args
`max_train_steps` is the **native-step** capacity, a positive multiple of five,
and stays fixed for LR/restart identity. The inventory may be smaller than the
capacity; no session recycles it or silently extends the scheduler.

```bash
python scripts/distillctl.py train /absolute/local/helios-reference/train.json
```

For restart, set `trainer.resume_from` to a completed common checkpoint,
`trainer.output_dir` to a **new** directory, and a larger target `cycles` within
the same finite inventory. Only output directory, checkpoint cadence, target
cycle count and the resume pointer are relocatable. Initial base assets and
source features are still required for this same-topology restart. Source,
dependency, Args, dataset, precision, seed or horizon changes are not migrations.

Each public cycle is five ordered nonreplayable `external_grad` Calls, each
writing only the declared `native_step` owner. At source step `t`, G updates
iff `t % 5 == 0`; full C executes at **every** step and performs its genuine
fresh three-stage G rollout. GAN starts iff `t >= gan_start_step`. EMA follows
`t < ema_start_step or train_generator`, including original optimization
counters. Program `native_step` version counts atomic native entries, not G/C
role versions. At the boundary, native `global_step = 5 * completed_cycles`,
Program version is `5 * completed_cycles`, but the common prepared cursor is
`completed_cycles`. Both native and outer Python/Torch/NumPy RNG are restored.

A partial native failure poisons the owner: it may not replay or checkpoint.
The common failure receipt retains the prior complete checkpoint. The atomic
writer never replaces an earlier complete output or publishes partial success.
Inner state and outer progress are staged and checked before live restore;
merely rehashing a contradictory checkpoint container does not admit it.

Finite-v1 `HeliosStepOwner(inputs=..., cohort=...).step()/save()/restore()` is
still available unchanged. The separate explicit stream-v2 owner receives one
row per call, retains only consumed IDs/content chain (not tensors), and uses
only the public session for checkpoint progress. Standalone v2 save/restore is
refused to prevent confusing native-step and prepared-cycle cursors.

## Evidence boundaries and what is next

`FULL_ORIGINAL_TINY_CPU_FIXTURE_NOT_PRETRAINED_OR_QUALITY` is the immutable
mechanism-test label. Tests create deterministic **test-only** local tiny
weights, including a nonzero frozen output fixture for real LoRA updates and a
separate unchanged constructor-zero case. Production only loads these pinned
bytes; it never manufactures gradients, substitutes losses or edits output
weights. These fixtures cannot enter the real-data lane by changing a label.
The fixture curriculum GAN3/EMA2 exercises real boundaries; canonical
GAN1000/EMA750 arithmetic is not evidence that 1000 steps were trained.

Real supplied files use `LOCAL_PRETRAINED_NATIVE_FEATURES_UNVERIFIED_QUALITY`;
this is accountable file provenance, not a proof that the file's creator made
an honest pretrained-quality claim. Functional full-class CPU execution,
pretrained/media quality, decoded video, GPU/world16 execution and performance
are separate evidence levels. Quality validation remains `NOT_RUN`; no speedup
is claimed.

A training checkpoint intentionally omits full frozen G/C tensors and
reconstructs them from pinned local assets on restart. The separate
[current-G export](PORTABLE_INFERENCE_CPU.md) provides full unmerged inference
weights without C/Adam/training data, through the existing export lifecycle.
The training factory itself is not the portable inference loader; the
inference-only factory supplies original native latent diagnostics, not a
fake VAE/video or quality claim.
