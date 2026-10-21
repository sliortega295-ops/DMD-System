# A complete, tiny model-extension path

**`TINY_MEDIA_TRAINING_FIXTURE_NOT_PRETRAINED_QUALITY`**

This example proves plumbing and real CPU autograd, not useful generated media.
Its tiny Student, Fake and frozen Teacher are **randomly initialized**. The
Teacher is not pretrained. It is neither W1–W6, a large-model parity test, a GPU
recipe nor performance evidence. Decoded video is only a coarse diagnostic;
the audio output is **features, not a reconstructed waveform**.

The packaged `distillschedule.examples.tiny_av` module works from an installed
wheel. Training needs PyTorch; raw PPM-frame preparation also needs NumPy and
Pillow. From a source checkout, install the public training/test extras in your
chosen environment (skip this when already provisioned):

```sh
python -m pip install -e '.[train,test]'
```

There are no model downloads or remote services in this example. Its runtime
commands neither install dependencies nor allocate a GPU.

## Prepare → train → evaluate → export → new-input generation

Run these public commands with `distillctl` and Python from the same installed
environment. `DEMO` is a new, disposable directory; the builder refuses to
overwrite a nonempty directory.

```sh
DEMO="$(mktemp -d -t distillschedule-tiny-av.XXXXXX)"
python -m distillschedule.examples.tiny_av --output "$DEMO" --samples 6

distillctl data prepare "$DEMO/raw/manifest.json" --output "$DEMO/prepared"
distillctl train "$DEMO/train.json"

CP="$DEMO/run/checkpoints/cycle-000004"
distillctl evaluate "$DEMO/train.json" --checkpoint "$CP" --output "$DEMO/generated"
distillctl export "$DEMO/train.json" --checkpoint "$CP" --output "$DEMO/exported"

python -m distillschedule.examples.tiny_av --output "$DEMO/new-inputs" --samples 2
distillctl data prepare "$DEMO/new-inputs/raw/manifest.json" --output "$DEMO/new-prepared"
distillctl generate "$DEMO/train.json" --weights "$DEMO/exported" \
  --prepared "$DEMO/new-prepared/manifest.json" --sample-index 1 --output "$DEMO/new-generated"
```

The builder creates three real 8×8 PPM RGB frames, a 16-bit mono 8 kHz WAV tone
and a UTF-8 caption per record. Their paths are relative to the raw manifest:

```json
{
  "schema_version": "distillschedule.data.raw.v1",
  "encoder": {
    "factory": "distillschedule.examples.tiny_av:encode",
    "parameters": {"frames": 2, "size": 4}
  },
  "records": [{
    "id": "sample-000000",
    "video": "sample-000000/frames",
    "audio": "sample-000000/tone.wav",
    "text": {"path": "sample-000000/caption.txt"},
    "metadata": {}
  }]
}
```

Preparation decodes actual files, invokes the configured trusted Python encoder
and saves its tensor feeds. Here the encoder pools RGB frames and temporal audio
energy into `latent: [1,4,2,4,4]`; a four-bin UTF-8 histogram and audio statistics
produce `condition: [1,4,1,1,1]`. It is fixed deterministic feature extraction,
not a pretrained video/audio/text embedding model. Changing any modality can
change these feeds. The preparer does not invent latents or replace model math.

The generated `train.json` contains the ordinary user-facing request:

```json
{
  "schema_version": "distillschedule.training.v1",
  "model": {
    "factory": "distillschedule.examples.tiny_av:build_model",
    "parameters": {"student_width": 12, "fake_width": 16, "teacher_width": 24,
                   "student_lr": 0.002, "fake_lr": 0.003}
  },
  "data": {"prepared": "prepared/manifest.json"},
  "algorithm": {
    "factory": "distillschedule.examples.tiny_av:build_algorithm",
    "parameters": {"fake_updates": 2, "fake_sample_policy": "reuse_student",
                   "schedule_steps": 16, "guidance_scale": 1.5}
  },
  "execution": {"mode": "reference"},
  "trainer": {"output_dir": "run", "cycles": 4, "seed": 7,
              "checkpoint_every": 2, "device": "cpu"}
}
```

Paths in that request are relative to its own directory. This is explicitly
**Student1/Fake2/EMA0**, with real AdamW updates on Student and Fake and no Teacher
updates. One algorithm-factory call returns exactly one cycle. The root trainer
owns the four-cycle loop, data progress, checkpointing and execution selection.

Check `run/summary.json` and `run/metrics.jsonl` for the run result rather than
inferring success from file creation. Complete-cycle checkpoints live under
`run/checkpoints/`. Evaluation loads the trained Student and invokes its real
generation/decode callbacks. Export saves module weights and accompanying
metadata; it does not turn these tiny weights into a pretrained model.
`generate` then restores the hash-bound exported module weights on explicitly
new prepared inputs without reading the original training dataset/checkpoint or
constructing the algorithm. The original factory source and non-device
parameters must still match. Its `generation.pt` and PNG `frames/` preserve the
actual decoder output, with quality validation **NOT_RUN**, not invented
high-resolution media. Exported weights are not a resumable checkpoint.

Continue to a total of six cycles from the complete four-cycle checkpoint:

```sh
distillctl train "$DEMO/train.json" --resume "$CP" --cycles 6 --output "$DEMO/resumed"
```

## The math belongs to the model extension

The inspectable hooks in `tiny_av.py` use the existing
`runtime.dmd_boundary` flow-noising, flow-to-clean conversion, guided Teacher
prediction, normalized Fake-minus-Teacher input gradient, detached Student
surrogate and Fake denoising objective. The Student output retains its autograd
graph. Teacher/Fake scoring for that update runs detached; Fake's own subsequent
denoising forward is differentiable. Noise is seeded and explicit, **not a random
loss standing in for DMD**. This is meaningful lifecycle/gradient evidence even
though a random frozen Teacher provides no pretrained knowledge or quality.

`reuse_student` means Fake uses the detached sample from *before* the Student
update. `fresh_student` instead regenerates from the updated Student for every
Fake step. These are different workload semantics, not execution optimizations.
The example chooses neither checkpoint policies, placement layouts nor a solver.

## Copy the external new-model template

[`newmodel.py`](newmodel.py) is a runnable, external alternative architecture,
intended to be copied from this source checkout into your own model project.
It uses the same tiny I/O and loss contract, with different channel MLPs. You do
not need a registry entry, a new CLI mode, planner code or a workload name.

```sh
cp examples/framework/newmodel.py "$DEMO/newmodel.py"
python - "$DEMO" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
config = json.loads((root / "train.json").read_text())
config["model"]["factory"] = "newmodel.py:build_model"
config["algorithm"]["factory"] = "newmodel.py:build_algorithm"
config["trainer"]["output_dir"] = "newmodel-run"
(root / "train-newmodel.json").write_text(json.dumps(config, indent=2) + "\n")
PY
distillctl train "$DEMO/train-newmodel.json"
```

Adapt the three independently owned contracts for a real model:

1. **Encoder** — `encode(record, parameters) -> dict` of tensor feeds. The decoded
   record contains `id`, THWC uint8 RGB `video`, `audio` with TxC float32 `samples`
   and integer `sample_rate`, literal `text`, and `metadata`. Supply your own
   actual pretrained encoders if needed; changing this encoder requires preparing
   the raw data again. Missing modalities need an explicit model/data contract,
   not invented random tensors.
2. **Model factory** — `build_model(parameters) -> ModelComponents`. The trainer
   supplies explicit `device` and `seed`. Bind models and persistent optimizers
   in `state`; name immutable modules, and optionally provide
   `generate(state, feeds, parameters)`, `decode(value, parameters)` and paired
   custom `export`/`load_export` callbacks if needed. The decoder returns explicit
   THWC uint8 video; the framework does not reinterpret latent tensors as pixels.
   Preserve any EMA/scaler/scheduler state your algorithm requires.
   This tiny fixture intentionally has no EMA or scheduler.
3. **Algorithm factory** —
   `build_algorithm(components, feeds, parameters, cycle, seed) -> Workload`
   for one cycle. Supply your own inspectable `DMDHooks` and
   `build_dmd_workload`, or use the existing `Program`/`Call` ABI for different
   update semantics. Access models only through `ctx.state`; declare additional
   mutable state effects. Do not capture a second model/optimizer in a closure.

Only load factory references you trust: these are executable Python, not a
sandboxed interchange format. The template's imported tiny hooks are appropriate
only while their exact latent, guidance, detach and optimizer contract still
matches your algorithm. Do not rename the fixture and claim real-model parity.

## Another workflow: history, EMA and schedulers

[`newworkflow.py`](newworkflow.py) is a **different stateful workflow**, while
`newmodel.py` above demonstrates a different architecture. This external plugin
adds previous-cycle Student feature conditioning and an explicit post-cycle
commit of Student EMA, next-cycle history and two exponential learning-rate
schedulers, all through the existing `Program`/`Call` ABI. Student surrogate and
Fake denoising losses remain genuine tiny DMD losses, not dummy random losses.
The generic planner/runtime needs no model-name or workflow-name edits.

From the source checkout, run:

```sh
DEMO="$(mktemp -d -t distillschedule-newworkflow.XXXXXX)"
python examples/framework/newworkflow.py --output "$DEMO" --samples 6 --validation-samples 2
distillctl data prepare "$DEMO/raw/manifest.json" --output "$DEMO/prepared"
distillctl data prepare "$DEMO/raw/validation-manifest.json" --output "$DEMO/validation-prepared"
distillctl train "$DEMO/train.json" --cycles 2 --output "$DEMO/first"
distillctl train "$DEMO/train.json" --resume "$DEMO/first/checkpoints/cycle-000002" \
  --cycles 6 --output "$DEMO/resumed"
distillctl train "$DEMO/train.json" --cycles 6 --output "$DEMO/uninterrupted"
```

The builder generates the complete runnable config with explicit external `.py`
factories, `data.validation_prepared` and
`trainer.validation={"every_cycles":2,"samples":2}`. The two validation records
are disjoint from the six training records. Every second complete cycle measures
held-out `student_teacher_flow_mse`, equal-record mean flow agreement at t=0
against the **random frozen Teacher**. This is a meaningful deterministic
adapter diagnostic, **not** FVD, VBench, a quality threshold or pretrained quality.
It never advances the training cursor. Logs are `validation.jsonl`,
`validation-summary.json` and the training summary's separate `validation` field;
`quality_validation` remains `NOT_RUN`.

The template is explicitly CPU/single-process only, including validation.
Distributed validation requests fail before construction/updates/output; this
does not change the existing distributed training capabilities when validation
is absent. Validation is read-only and runs under eval/no-grad with exact mode
and training RNG restoration even on failure. Mutation rejects without silently
rolling persistent model/optimizer state back. Changed validation records or
settings reject checkpoint resume.

All parameters, Adam state, EMA, history, schedulers, RNG, cursor and logical
versions are carried in complete training checkpoints. The fresh-process test
below compares interrupted/restarted continuation with uninterrupted training,
including exact held-out metrics. Portable `generate`/`export` instead uses
**current Student with zero history**, explicitly recorded in model metadata;
it does not pretend the default module-only export includes EMA/history trees.

## Focused proof

```sh
python -m pytest -q tests/test_training_example.py tests/test_open_workload_dmd_builder.py
python -m pytest -q tests/test_training_validation.py tests/test_training_workflow_extension.py
```

The example tests actual modality-dependent encoding, global-RNG isolation,
frozen Teacher state, nonzero autograd gradients, Student/Fake parameter and full
AdamW-state updates against an independent sequential DMD cycle, both Fake sample
policies, reference/checkpointed open-runtime execution, actual fixture files and
the external template path. These are CPU correctness tests, not quality or
speedup measurements.
