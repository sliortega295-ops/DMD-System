# Open workload program: semantics, not a prewritten optimization recipe

## Public entry

A user Python factory returns `Workload(Program(...), bindings, state, feeds)`.
It describes model calls, loss/backward/update functions, logical reads/writes,
backward provenance and actual inputs. It does NOT provide measured winners,
optimization candidates, placement choices or a model-specific scheduler.
Python loops/branches instantiate the actual workload from parameters before
planning. Unresolved data-dependent control flow is not guessed.

For a bounded Student1/FakeN detached-score loop, the optional
[DMD semantic builder](open_workload_dmd_builder.md) constructs these same calls
from user-owned forward, noising, loss and history hooks; it selects no strategy.

```bash
python scripts/distillctl.py workloads native program auto \
  --factory examples/open_workloads/video_input_gradient.py:build \
  --parameters examples/open_workloads/video_parameters.json \
  --output /tmp/open-video-run --memory-limit-bytes 1000000 \
  --samples 1 --max-program-runs 8 --execute
```

The example uses small random Conv3d weights and actual input gradients/Adam on
CPU. It is an interface example, not an original DMD performance result.
To change frames, width or update count, change the JSON; no planner edit or new
launcher is required. `describe`, `calibrate`, metadata-only `plan`, and `run`
are available separately. Factories are trusted user code and model construction
requires explicit --execute; no implicit Hub download or cluster allocation.

## What the shared system does

- `workloads/program.py` compiles calls into the existing ExecutionGraph. It
  derives version RAW/WAR edges and keeps backward/replay dependencies alive.
- The generic Torch backend matches immutable parameter identity (content,
  shape, dtype, layout) and explicitly replay-safe gradient-bearing regions.
  It can choose physical immutable aliases and nonreentrant checkpointing.
  Adapters/optimizers remain separate; mutable parameter aliases are rejected.
- Calibration runs the actual supplied math from paired cloned model/Adam
  states. Numerical mismatches are REVIEW, never relabeled PASS.
- Native save-attention/save-matmul variants require **observed** matching
  operator calls and retained bytes on every rank; unavailable options are
  recorded as inapplicable rather than ranked as optimizations. Checkpoint
  replay callbacks are not required: PyTorch 2.14 may reuse saved native
  results without replay dispatch. Native forward-operation counts plus full
  gradient/optimizer parity test that recomputation is genuinely avoided.
- `search/service.py` reuses the existing event-frontier DP/beam across admitted
  candidates. It is independent of model names, loss functions and update ratios.
- The selected sequence runs through PreparedRolePlan. Stored timing and memory
  predictions are reconciled against measured components before execution.
  Logical work, including Python/Torch RNG, remains tied to call IDs.
- Costs bind code, parameters, state, feeds, dtype/device and environment. New
  parameters require matching calibration. Budget truncation is explicit.

## Actual boundaries

The open callable backend now executes local programs and explicitly allocated
multi-process data-parallel programs. For admitted module-forward contracts it
matches replicated post-backward reduction, DDP asynchronous reduction and FSDP2
reshard/keep-forward layouts, plus topology-admitted HSDP2 and per-module layout
assignments. These use the SAME Program, calibrator, search
service and call executor, with extra completion edges for reducer lifetimes.
See [Distributed open workloads](open_workload_distributed.md) for exact
contracts, CPU/Gloo coverage, examples and remaining restrictions. The separate
[RoleTask/RoleFunctions route](open_role_pools.md) implements bounded role-pool
partitioning and cross-pool VJP transport; arbitrary Workload graphs are not
automatically lowered to it. A100/H100 validation of these generic capabilities
is NOT_RUN. Original workload-specific routes remain available; no five-model
GPU parity is implied.

`immutable_modules` and `replay_safe` are semantic promises verified where
possible, not automatic inference of all Python side effects. All relevant
state must pass via ctx.state/inputs. No external autograd graph, hidden model
closure, in-flight accumulated gradients or silent optimizer reset is admitted.
Use backward_of for differentiated values, including input gradients through a
frozen critic. Incorrect state declarations are outside the ABI.

CPU memory is tracked tensor/storage accounting, not an RSS guarantee. CUDA
allocator accounting is implemented but NOT GPU validated here. Calibration
clones model state, so peak preparation capacity differs from steady execution;
large-model lifecycle/offload requires its appropriate backend. CPU timings are
not claimed GPU speedups. A successful small-model test is not full pretrained
Wan/Helios/H3/LiveWan numerical or quality evidence.

This branch is stacked on PR #33 and preserves its original-protocol routes.
The full five-workload capability migration and GPU validation remain unsatisfied goals.
