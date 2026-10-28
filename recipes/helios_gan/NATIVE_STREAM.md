# W3 full native bundle: fresh-cohort continuation (research opt-in)

**Source/CPU integration only. Native world16 GPU continuation, numerical parity,
capacity and 100-cycle reliability have NOT been accepted.** The underlying
FA3/down40 bundle remains `REVIEW`, never a stable/recommended default. This is
not the independent CPU reference recipe and does not inherit its model-quality
or GPU qualifications.

## Preserved execution

`training.stream.json` selects the same original world16 FSDP/C2048/window5
implementation: pinned SM80 FA3, named frozen-down40 SAC replay, separate five
GAN-head FSDP units, original LoRA-B/replay-scope/norm/history/KV/residual
mechanisms and one persistent-unshard scope around all five B1 rollouts.
**Merged-LoRA/N70 stays forbidden.** Each complete cycle uses the original
Program `_prepare_cycle`, G1/C5/EMA1, original GAN/DMD helpers, Adam and constant
native scheduler. There is no second trainer or replacement loss.

The systems-only bootstrap has physical G2/C10/EMA800 and separately identified
curriculum G200/C1000/EMA800. It is **not an official checkpoint-1000**. Fresh
cycle `i` consumes original steps `1000+5*i .. 1004+5*i`; native optimizer clocks
are never relabelled as curriculum clocks. For 100 complete cycles the physical
final clocks are G102/C510/EMA900 and curriculum clocks G300/C1500/EMA900.

## Prepare explicit local assets and inputs

1. Fill `assets.stream.template.json` with your local model, precast, logical
   bootstrap, sparse EMA800, negative prompt, native GAN config and FA3 package.
   No private run ID or maintainer directory is required. The existing
   `tools/assets/build_systems_gan_bootstrap.py` and precast tooling own bootstrap
   construction; this recipe does not invent a cold start. Keep every actual
   asset SHA. The native worker verifies all used master/moment/EMA/precast
   payloads, the original source config/index and FA3 package bytes.
2. The GAN config must match
   `configs/distillschedule/templates/helios_stage3_gan_384x640_systems.yaml`.
   Only the admitted input random-drop/history-clean-probability settings may
   differ. The four legacy `model_config` locator fields
   (`pretrained_model_name_or_path`, `transformer_model_name_or_path`,
   `real_score_model_name_or_path`, `reward_model_name_or_path`) and
   `data_config.gan_data_root` may be omitted, null, or replaced with other
   appropriately typed locator values. **They do not participate in worker
   loading:** native P0 uses the explicit local asset document and prepared
   stream instead. No other keys are ignored; loss/head/precision/update/
   optimizer/EMA/sampling settings retain their original admission rules.
   The template's benchmark `max_train_steps` is not extended: the original P0
   engine actually constructs an unbounded **constant** scheduler. The finite
   prepared stream determines this invocation's horizon; no new decay schedule
   is inferred from its length.
3. Supply a JSON list of cohorts. Each has `seed` (an explicit 63-bit integer)
   and `samples` (five lists of sixteen absolute `.pt`/`.safetensors` native
   feature paths). Each feature uses original `vae_latent`, `prompt_embed` and
   `attention_mask` fields. All `80 * cycles` paths and payload hashes must be
   globally distinct; changing a seed or copying bytes is not fresh input.
   For the agreed one-seed acceptance, use the same seed in all 100 cohorts;
   cohort/rank/step/phase retain their own deterministic native streams.
   Alternatively, use the explicit [finite feature-pool epoch policy](../../docs/guides/helios_feature_epochs.md)
   to reuse at least 80 source records across cycles without copying features
   or replaying captured noise. The list format above remains the default.

Generate a complete, public configuration without copying the template or its
personal-path text. This small helper removes **only** those five unused system
locators, keeps every original algorithm/input setting, and refuses to overwrite
an existing file:

```sh
python - /absolute/project/native-gan.yaml <<'PY'
import sys
from pathlib import Path
import yaml
from distillschedule.training.helios_gan_inputs import portable_config

with Path(sys.argv[1]).open("x") as target:
    yaml.safe_dump(portable_config(), target, sort_keys=False)
PY
```

Use that same generated file for data preparation and the `gan_config` asset.
The whole file's SHA remains bound to the prepared stream; accepting unused
locator text does not waive source/config/asset identity checks. Any later file
change requires a matching freshly prepared stream and asset pin.

```sh
distillctl data prepare-helios-gan-stream \
  --sources /absolute/project/cohorts.json \
  --training-config /absolute/project/native-gan.yaml \
  --output /absolute/project/prepared-gan-stream
```

The producer calls the original dataset/history/scheduler primitives. It writes
new step-qualified conditioning, primary/secondary source tensors, rollout and
score noise, native GAN crops and R1 epsilon. Normal globally synchronized
33-frame/222/world16/B1 is the admitted shape. Amplified all-zero history and
rank-local exception branches are rejected, not resampled or silently replaced.

## Public entry and same-topology restart

Fill `training.stream.json`; `trainer.cycles` must equal the complete finite
stream length (100 for the later acceptance), not an arbitrary two/64-cycle cap.
Run the ordinary `distillctl train` validation first, then enter training on each
rank of an already-authorized original two-node/eight-ranks-per-node torchrun.
The public API neither reserves GPUs nor launches remote nodes. Preserve the
native Torch 2.10.0+cu128/CUDA 12.8/A100 and local pinned FA2/FA3 dependencies;
`LOCAL_KERNELS=kernels-community/flash-attn2=<absolute-existing-package>` is still
required by the unchanged native FA2 identity gate before FA3 installation.
No package/model download is performed by this recipe.

Checkpoints are published only after all required ranks save and validate their
local shards at complete, drained G1/C5/EMA1 boundaries. They bind G, Critic and
all optimizer-owned GAN-head tensors, named Adam, constant scheduler, EMA,
physical clocks, data cursor, current source/assets/stream/layout and rank RNG.
Frozen bases are rebuilt from verified assets. GPU input caches, SAC stores,
kernel caches and persistent-unshard scopes are **not** checkpoint payloads.

For restart use a new output directory, the exact same committed source,
assets/input stream/world16/policy, and set `trainer.resume_from` to the published
`checkpoints/cycle-000050` directory. A target-complete checkpoint is not a new
longer stream: source/input identities and the finite horizon are immutable.
The runtime restores native owners before qualifying the next cohort and builds
fresh per-graph caches. Partial shards, stale source/shape, pending gradients,
live FIFOs, disabled adapters or a changed native layout are hard failures.

## Stop at a durable checkpoint without changing the stream

For a prepared 100-cycle stream, keep `trainer.cycles: 100` and set
`trainer.checkpoint_every: 50` plus `trainer.stop_after_cycle: 50`. The first
process executes only cycles 1–50 and returns after the complete all-rank
checkpoint is durable. It does not execute cycles 51–100 and replay them later.
The boundary remains G1/C5/EMA1; N70 remains OFF.

Start a new process with the same source/assets/full prepared stream and a new
output directory. Set `trainer.resume_from` to `checkpoints/cycle-000050`, then
**remove `trainer.stop_after_cycle`** (or set it to a later checkpoint boundary,
such as 100). Never change `cycles` to 50 or shorten/reprepare the stream. An
explicit stop at or before the restored cycle is rejected before model build.

The stop must be an integer inside the unchanged horizon with a positive
checkpoint cadence. An intermediate stop must be divisible by that cadence;
the final horizon is always a valid save boundary. Rank-disagreeing stop
controls, missing/corrupt shards and incomplete checkpoint publications fail
closed. Without the stop field, the existing full-horizon behavior is unchanged.
Public export discards this training-only stop control and restores the explicit
export checkpoint without running training updates.

A graceful early stop has a distinct `STOPPED_*_AT_CHECKPOINT` status, not the
complete-run status. `training_segment` records the full horizon, start,
requested stop, completed cycle and actual executed count. Neither an early
segment nor a resumed segment is formal performance or 100-cycle GPU evidence;
native GPU reliability/quality acceptance remains **NOT_RUN**.

## Evidence boundaries and remaining gates

- New CPU tests cover actual public/source/asset admission, fresh metadata/shape
  qualification, strict original storage/FIFO scope selection, source-cache
  switching, Program/cycle/state publication and independent-process restart.
- The orchestration fixture uses real native Engine/Adam/LR/EMA/checkpoint state
  but **synthetic P0 operations and kernel counters**. It is not GAN math or GPU
  proof. A separate target-dependency test exercises one actual new cohort's
  original producer and official score/critic input adapters; missing native
  dependencies are explicitly `NOT_RUN`, not counted as a pass.
- The versioned stream uses the existing reviewed `fsdp2_static.py` overlay
  `0b939331` / SHA `dd2fd086...`. The old down40 source pin and old two-cycle
  refusal rules are unchanged. Fresh source integration identities are explicit;
  no frozen source-layout lock or historical receipt is rewritten.
- Future native acceptance must prove actual per-cycle FA3/down40/head counters,
  persistent-unshard release (also on failure), same-topology all-rank restart,
  numerical/capacity qualification, then the agreed one-seed/100-complete-cycle
  continuation. Dynamic new-input counters are not the old fixed trace's
  literal timing profile and carry no speedup credit.
- Native decoded output and portable current-Student export remain unsupported
  by this stream recipe. Their separate work is not claimed complete here.
