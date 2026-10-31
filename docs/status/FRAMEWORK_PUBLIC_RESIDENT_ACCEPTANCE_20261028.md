# Public resident training acceptance — 2026-10-28

`distillctl train` now connects explicit native checkpoint policies, measured
automatic selection and saved plans to the same persistent training lifecycle.
The first admission uses a capacity-bounded serial snapshot; subsequent cycles
reuse the owner with fresh inputs. Complete-cycle checkpoints support a new
process on the same topology, and export retains the current Student.

## Actual model scope

| Model | Direct policy and fresh-process resume | Automatic selection and continuation | Output |
|---|---|---|---|
| Official DMD2 SD1.5 | Three cycles; exact complete state | Three cycles; exact native state | Current Student, decoded RGB image |
| Official DMD2 SDXL | Three cycles; exact complete state | Three cycles; exact native state | Current Student, decoded RGB image |
| Official Self-Forcing | Three cycles; exact complete state | Three cycles; exact native state | Current Student, decoded RGB video |

The shared saved-plan path is tested separately with SD1.5: auto and saved
execution produce the same training state, resume is exact, no implicit search
runs during replay, and a changed saved plan is rejected. Self-Forcing keeps
causal Generator checkpointing ON; its candidate toggles only the noncausal
Fake checkpoint implementation.

These are actual upstream classes at their recipe's randomly initialized tiny
FP32 geometry, under the explicit deterministic test policy. Hardware was A100
40 GiB; one device performed training, another checked RNG isolation. They are
not pretrained, distributed, quality or throughput results.

## Verification and provenance

- Implementation base: `ca23a540a22b0878df3d0bd20c57048c45ec9320`; the public
  lifecycle changes are carried by the commit adding this document.
- Target stack: Python 3.10.20, Torch 2.10.0+cu128, Diffusers 0.37.1,
  Transformers 4.51.3, PEFT 0.17.1 and Accelerate 1.14.0. Pinned upstream source
  and recipe assets were checked before model construction.
- Target CPU groups passed with zero skips: public integration 53,
  preconstruction 23, continuation 7 and shared-control regression 33. These
  are overlapping focused groups, not a sum of distinct tests.
- Seven logical CUDA scenarios are complete. The first invocation passed all
  three auto scenarios and completed SD1.5 training/resume/export, then its
  test compared a CPU export directly to a CUDA tensor. The checker was fixed
  to serialize the current tensor to CPU without changing equality. A bounded
  follow-up ran only the three unfinished scenarios and the saved SD1.5 output
  assertions; no completed training was repeated. Production code was identical
  across those two invocations.
- Both owned invocations released their processes and locks, with no cgroup
  OOM or OOM-kill increment. All original failed and passing receipts remain.
- A broader historical refactor run timed out after 1,062 logged passes and
  45 skips, without logged failures; this is not a complete-suite pass.

Tests: `tests/test_training_resident_lifecycle.py` and
`tests/gpu/test_training_resident_public_gpu.py`. The latter is opt-in and
requires the pinned sources and a CUDA-qualified owner. Evidence index:
`framework_maturation_20261023/r19_public_resident_lifecycle/reports/conclusion.json`.

## Operational checks

Inspect `construction-admission.json`, `owner-admission.json` or
`plan-admission/execution.json`, per-cycle metrics and a complete checkpoint.
Healthy execution advances exactly one logical cycle and its native clocks per
accepted input, retains the same parameter ownership, and honors capacity
reserves. A failed admission restores its snapshot; a failed ordinary update
does not publish a new checkpoint. Resume from the last complete checkpoint
rather than reusing an invalidated owner.

Public pretrained capacity and the final installed-release/100-cycle gates
remain separate. The existing no-budget CUDA refusal and the native distributed
contracts are unchanged.
