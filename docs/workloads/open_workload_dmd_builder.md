# User-owned DMD math, shared open-program planner

`distillschedule.open_workload.dmd.build_dmd_program` builds the existing
`Program`/`Call` graph for a **bounded Student1/FakeN detached-score DMD loop**.
It is a semantic onboarding helper, not another solver or a model registry.
`build_dmd_workload` additionally binds user models and optimizers into an
ordinary `Workload`; return that object from the same external factory used by
`workloads native program auto`. No CLI, calibrator, capability or planner changes
are needed. Factory construction still requires the existing explicit execution
authorization. Nothing here downloads weights or allocates GPUs.

## Supply math, not a winner

The caller supplies inspectable functions (or `DMDHook(function, reads=...,
writes=..., replay_safe=...)`) in `DMDHooks`:

| Hook | `dmd_inputs(ctx)` semantic values, in addition to `feeds` | Gradient mode |
| --- | --- | --- |
| `student_forward` | optional prior-cycle `history` | grad for Student update; no-grad for fresh Fake samples |
| `history` (optional) | detached `current`, optional `previous` history | no-grad |
| `noise` | detached `generated`, optional current `history` | no-grad |
| `teacher_forward` | `generated`, `noise`, optional `history` | no-grad |
| `fake_forward` | `generated`, `noise`, optional `history` | no-grad for Student scoring; grad for Fake training |
| `student_loss` | **live** `generated`, detached `noise`/`teacher`/`fake`, optional `history` | grad |
| `fake_loss` | detached `generated`/`noise`, **live** `fake`, optional `history` | grad |
| `student_backward`, `fake_backward` | `loss` | grad; defaults to `loss.backward()` |
| `student_update`, `fake_update` | `backward` completion token | no-grad; defaults to optimizer step then zero-grad(None) |

Every hook also receives `ctx.attributes['cycle']`, `['phase']` (`student` or
`fake`), and `['fake_index']` (None for Student). `feeds` is the complete supplied
feed mapping. Hook return values may be tensor trees/mappings, so noise packages,
conditioning, conditional/unconditional predictions and custom objectives are
not forced into a particular tensor signature. The builder contains **no**
noise schedule, DMD gradient formula, guidance scale, loss reduction or surrogate
objective. The CPU tests bind the existing `runtime.dmd_boundary` math explicitly.

Minimal factory wiring, after constructing the user's models, optimizers, inputs
and Context hooks:

```python
from distillschedule.open_workload.dmd import DMDHook, DMDHooks, build_dmd_workload

hooks = DMDHooks(
    student_forward=DMDHook(my_student, replay_safe=True),
    fake_forward=DMDHook(my_fake, replay_safe=True),
    teacher_forward=my_teacher,
    noise=my_noising_contract,
    student_loss=my_dmd_objective,
    fake_loss=my_denoising_objective,
    history=my_previous_current_history,  # optional, output must be read-only
)
workload = build_dmd_workload(
    student=student, fake=fake, teacher=already_frozen_teacher,
    student_optimizer=student_optimizer, fake_optimizer=fake_optimizer,
    hooks=hooks, feeds=feeds, cycles=2, fake_updates=5,
    fake_sample_policy='fresh_student', history_feed='initial_history',
    # replica_batch_size=local_batch_size,  # optional equal-rank-mean contract
)
return workload
```

Inside a hook, use `dmd_inputs(ctx)` instead of hard-coding graph call IDs, and use
`ctx.state['student']`, `['fake']` or `['teacher']` for models. Context exposes only
declared/inferred state reads. Do not capture another model, optimizer or tensor
in a closure/global. `replay_safe` is a caller purity/RNG assertion, not automatic
proof; default is false. Optional hook state effects must name `extra_state`
entries (or other existing logical state) explicitly. Extra state must use the
normal backend's supported signature types (tensors, modules, optimizers,
mappings/sequences and scalar data), not opaque scaler/scheduler objects. State
that cannot be represented by that ABI requires another backend/contract. The
default updates do **not** silently add clipping, scaling, EMA or decay changes.
Preserve such algorithmic behavior in declared custom hooks.

For direct ABI use, call `build_dmd_program(hooks=hooks, feed_names=tuple(feeds),
...)`, then bind `hooks.bindings(program)` and all declared state in `Workload`.
The reserved logical state names are `student`, `fake`, `teacher`,
`student_optimizer`, `fake_optimizer`, `student_grad`, `fake_grad`. Optional
`initial_versions` bind the current logical versions; default is zero.

## Two explicit, different workload contracts

`fake_sample_policy` is required, never selected as an optimization:

- `reuse_student`: each cycle's Fake updates reuse the detached Student sample
  from **before** that cycle's Student optimizer update.
- `fresh_student`: after the Student update, every Fake substep calls the updated
  Student under no-grad. Its noise/forward/loss hooks receive that fresh sample.

Student scoring reads the current Fake version. Fake updates have their own
separate backward and optimizer calls. Teacher has no updates. G/F/T can have
different internal widths/depths; compatible latent/noise/loss I/O is the caller's
contract, not inferred from architecture names. The helper rejects a trainable
Teacher or optimizers that do not exactly own each role's trainable parameters.

## Detached previous/current history, not causal-cache resume

An optional history hook combines the detached current Student output and prior
history. All history/noising/scoring outputs cross explicit no-grad graph nodes;
the backend recursively detaches returned tensors. Student's live output is
passed separately to its loss. History returned by the hook is read-only and may
share storage: this is **detach**, not an implicit clone or permission to mutate
the live activation. Student consumes the *previous cycle's* history; fresh Fake
rollouts use that same previous context. Their temporary histories do not replace
the main history that feeds the next cycle. Final main history and every scalar
loss are program outputs.

There is no persistent KV cache, causal attention state reconstruction, checkpoint
resume, or hidden multi-call buffer lifecycle. A real persistent causal workload
needs its own declared state and parity proof. `build_dmd_workload` also requires
all tensors nested in the supplied initial `history_feed` to have
`requires_grad=False` and no `grad_fn`; even a grad-enabled leaf is rejected at
construction. It never silently detaches caller-supplied external history graphs.
External autograd graphs and in-flight gradients are still rejected by the
existing backend. The compiler
rejects backward/replay across a changed live model version; no-grad history must
not be relabeled differentiable to bypass that guard.

## Validation and limits

`tests/test_open_workload_dmd_builder.py` specifies two complete CPU cycles with
Student1/Fake1 and Student1/Fake3, both sample policies, cross-scale model widths,
frozen Teacher, detached histories, real DMD boundary math, parameter updates and
full Adam state against an independent sequential loop. Ordinary no-history
cases exercise both sample policies through that same oracle. It also exercises
baseline/full-checkpoint variants, nested initial-history admission and stale
provenance rejection. These are test specifications, not an execution receipt.
It adds no
target-GPU, large-model, distributed-numerical or performance claim.

The helper instantiates at most 511 calls, as required by `Program`. It describes
this explicit loop, not all DMD variants. GAN branches, other update orderings,
dynamic ratios, differentiable score paths, stateful forward-buffer mutations,
nonstandard backward/optimizer lifecycles or additional Teacher updates may need
direct `Program`/`Call` construction. The existing external factory path remains
available for those real workloads; no registry/solver edit is necessary.
