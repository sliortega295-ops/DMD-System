# DiffSynth H3 cached-AV reference derivative

This is a **maintained cached-only derivative**, not a pristine or complete
DiffSynth distribution. The retained Apache-2.0 license is reproduced verbatim
in `LICENSE.Apache-2.0` (SHA-256
`d43b96112d60d83595395649e61c6bc215f568052a979320489556c6c1262eb1`).
Both the original source metadata and retained support metadata declared
Apache-2.0. The original full matching distribution/revision is not established;
no repository HEAD is invented for either archived source.

## Origins and changes

- The H3 DMD2 world16 reference-source archive retained September numerical
  model, attention, gradient-checkpoint, scheduler, DMD2 and training sources.
  `distillschedule.reference.diffsynth_h3_cached/source_manifest.json` records
  every original relative filename, whole-file SHA-256, selected symbol/span,
  symbol source hash, normalized AST hash, output hash and allowed change.
- The complete H3 model, attention, gradient, flow scheduler and optional ZeRO3
  modules are byte-identical. The complete `h3_dmd2.py` changes only four
  top-level imports to the explicit maintained namespace. All numerical bodies
  and original defaults, including critic_updates=5 and GAN start=500, remain.
- Parser, packing/model functions, weight loading and training/checkpoint helpers
  are statically selected original definitions, not runtime AST execution or
  replacement algorithms. The unused pipeline annotation is postponed; the
  optional checkpoint ZeRO3 import is relative. No missing ControlNet, VAE,
  encoder, full pipeline or full training-module implementation is invented.
- Actual older `PipelineUnit` and `transfer_data_to_device` support is retained
  from the H3-SFT4AR August runner baseline. The latter is exposed in an honestly
  limited, newly maintained `CachedTensorMover`, not a complete upstream
  `DiffusionTrainingModule`. Older numerical modules are **not** overlaid.
- The lazy admission provider and minimal import bridges are new maintained
  code. They validate packaged bytes and imports, require an explicit dense CPU
  backend, reject optimized/remote attention contamination, and compile admitted
  source rather than executing possibly mismatched bytecode. The provider's
  own code is review/package-owned; the numerical manifest does not self-hash.

No model weights, caches, downloads, personal machine paths, original private
defaults, or underlying media/dataset rights are included. This source provider
does not alter the frozen source16/source83 native benchmark or expose a new
public training lifecycle. CPU tiny-model functional/autograd/Adam/restart
characterization is not full-50-block, world16, CUDA, ZeRO3, pretrained-quality,
FSDP-equivalence, capacity or performance evidence.

## Checkpoint trust

`training_api.load_checkpoint` is the exact original **trusted local pickle**
API (`weights_only=False`). It is never called on discovery or made a default
public untrusted-input route. Only load checkpoints you created and trust.
The original tensor-cache dataset uses `weights_only=True`; its fingerprint is
metadata-based, not a content hash. Public lifecycle admission is separate work.
