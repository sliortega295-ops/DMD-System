
## Iteration 281c — DP8 ZeRO-3 exact-trace control-order audit

### Milestone / Gate

Continuous Full5 Route-B, Step10-A backend correction gate.

### Question

Does the restored DP8 ZeRO-3 Step10-A backend execute the frozen rank-qualified Step10 trace safely inside the combined world16 process group?

### Hypothesis

After subgroup-scoped DeepSpeed initialization and BF16 resident-target EMA representation fixes, the exact Step10 trace will reach its first payload and complete the validated 8G+4S+4T update path.

### Evidence and Result

The run `rb16_full5_cont_zero3step10a_2g6c_pipeline_correctness_20260810_003` initialized all eight Generator ranks as ZeRO-3 with DP8, MB1, GAS2, global batch 16 and three audited subgroup barrier redirects. All sixteen ranks reached `step10a_and_future_cpu_models_ready`, with no new cgroup OOM or OOM-kill event. It then produced no Step10 input payload for more than 1,500 seconds; node0 GPU utilization remained effectively zero after the ready boundary. The job was stopped fail-closed with return code 130 rather than treating the delay as performance.

The frozen Step10 control manifests are not physically collective-safe for naive DP8 ZeRO-3 execution. In wave 0, rank call counts are `4,4,4,8,4,4,6,7`; in wave 1 they are `4,3,7,7,4,6,9,8`. This variation comes from the source-confirmed rank-local amplified GT-zero-history exception. ZeRO-3 parameter all-gathers require every DP8 rank to issue the same Transformer call order. The missing boundary markers mean this attempt alone does not prove the exact blocking line, but the trace audit independently rejects the naive execution plan.

The historical `48.5516197304 s` Iteration-237 Step15-A result is not a valid exact-workload denominator: it generated synthetic data online, rejected zero-history ranks, and used rank-zero synchronized rollout control. It did not replay the current operation-keyed rank-local exception schedule.

### Correctness

Not Run beyond initialization and trace compatibility. No Step10 output, loss, backward, optimizer, redistribution, Critic, or EMA result was produced.

### Performance

Not Run. Neither the >1,500-second wait nor the earlier ZeRO-1 `93.563083 s` is accepted as the corrected ZeRO-3 Step10-A performance.

### Files Changed

- `artifacts/manifests/iteration281_zero3_step10a_real16_attempt003_rank_divergent_trace_stall.json`
- append-only `research_log.md`

### Failure Signature / Recovery

Failure signature: all ranks ready, no payload, no GPU activity, exact trace has rank-divergent physical DiT call counts under a DP8 ZeRO-3 parameter group. Do not retry the same naive path and do not fall back to ZeRO-1 as the performance candidate. The next run requires a fail-closed trace/collective compatibility gate plus an exact collective-safe padded-slot implementation (or another separately preregistered exact-control method).

### Gate Status

NO-GO for another real naive DP8 ZeRO-3 launch. GO for CPU/unit-level padded-slot semantics and a tiny subgroup forward/backward gate.


## Iteration 282 — ZeRO-3 exact-control gate and process-world decision

### Milestone / Gate

Route-B continuous Full5 Step10-A backend and collective-control gate.

### Question

Can the operation-keyed rank-divergent Step10 trace execute under ZeRO-3 without reverting to the slower ZeRO-1 backend, and can that ZeRO-3 engine remain a subgroup of the continuous world16 process graph?

### Hypothesis

Deferring every rank's true selected call to a common final physical slot, with discarded no-grad padding before that slot, will align both parameter collective order and grad lifecycle. If the continuous world16 subgroup remains unsafe, the validated recovery is a separate default-world8 Generator phase rather than further DeepSpeed patch escalation.

### Measured Results

The corrected helper passed 6 focused tests and the complete Runtime suite (`322 passed`). The DMD focused suite passed 17 tests and the full suite passed `212 passed, 1 skipped`. A DP2 ZeRO-3 subgroup inventory proved that both parameter and optimizer groups were exactly global ranks `[0,1]`; nevertheless both ranks stalled inside their first no-grad forward for 120 seconds in a default-world4 process graph (return code 124, no OOM delta). This rejects initialization-only subgroup repair.

The same divergent-exit schedule then passed in a default-world2 ZeRO-3 engine: logical exits `[0,2]`, physical calls `[3,3]`, one selected call and one optimizer step on each rank. Rank 0 executed `[padding,padding,selected]`; rank 1 executed `[prefix,prefix,selected]`. A preceding diagnostic failed only because the toy network returned width 8 and was incorrectly reused as a width-16 next-step input; it is retained as a harness failure and did not change policy or tolerance.

### Correctness

CPU semantic parity and the default-world ZeRO-3 toy forward/backward/update gate passed. Real-14B Step10-A output, G2-to-G3, redistribution, Critic, EMA and queue correctness remain **Not Run**.

### Performance

**Not Run.** The historical 48.551620-second result remains evidence that the default-world8 ZeRO-3 backend is faster than the 93.563083-second ZeRO-1 diagnostic, but it is not an exact current-trace denominator.

### Interpretation

The user's backend correction is confirmed: Step10-A must not use the Attempt005 ZeRO-1 path. The single-world16 ZeRO-3-subgroup design is now a bounded negative result. The next implementation must preserve `8G+4S+4T` while making the eight Generator ranks their own default ZeRO-3 world, then gather G3, destroy the obsolete sharded engine, reinitialize the target world16 process graph, and broadcast only logical G3 trainables into the per-node `2G+6C` Generator pool. This reuses the validated source-to-target representation rule and avoids in-place ZeRO engine conversion.

### Files Changed

- `artifacts/manifests/iteration282_zero3_collective_schedule_gate_result.json`
- `artifacts/manifests/iteration282_zero3_padded_default_world_gate.py`
- preserved subgroup/default-world gate outputs under `artifacts/profile/`
- append-only `research_log.md`

### Failure or Limitation

No real 14B claim is active. Exact padding expands the frozen Step10 trace to 22 physical DiT calls per Generator rank across two waves, so even a correct ZeRO-3 run may not reproduce 48.55 seconds. The overhead must be measured rather than assumed.

### Gate Status

NO-GO for more ZeRO-3 subgroup patching. GO for one preregistered phase-reinitialization toy gate before modifying or launching the real-14B Full5 path.

### Next Highest-Value Question

Can two independent node-local Phase-A worlds safely destroy/reinitialize into one global Phase-B world while preserving a tensor/version payload and leaving no old NCCL work or ZeRO handles?


## Iteration 283 Preregistration — Phase-local default world to target world16

### Milestone / Gate

Route-B Full5 process-world and parameter-redistribution gate.

### Question

Can the validated default-world ZeRO-3 substrate be used in Step10-A and then safely relinquish its process group before the same live processes form the target Critic-phase world?

### Hypothesis

Gathering updated logical trainables before destroying the obsolete ZeRO-3 engine, then destroying the node-local default groups and initializing one new global world, will preserve the versioned state without in-place parameter reinterpretation.

### Allowed Changes

Toy gate, process-world lifecycle, logical tensor gather/broadcast, and evidence only. No real-14B workload, trace, loss, tolerance, target 2G+6C policy, or timing run is authorized before PASS.

### Gate Status

GO for one two-node toy gate.


### Iteration 283 Attempt001 bounded harness failure

Both node-local Phase-A worlds reached their ready markers with no OOM, but all four Phase-B ranks timed out as TCPStore clients. Source inspection found a deterministic launcher-state cause: `torchrun --standalone` exported `TORCHELASTIC_USE_AGENT_STORE=True`, so PyTorch forced `is_master=False` even for explicit Phase-B global rank 0 on the new port. Attempt002 is preregistered with one change only: remove that stale Phase-A agent-store hint after destroying Phase A and before explicit Phase-B initialization. No model, state, rank, group, or numerical contract changes.


### Iteration 283 Attempt002 bounded harness failure

Removing the stale elastic-store hint worked: all ranks created the explicit Phase-B world4. Node1 then failed a fail-closed DeepSpeed-world assertion because it had no Phase-A DeepSpeed engine and therefore no `deepspeed.comm.cdb` wrapper. Attempt003 changes only initialization plumbing already validated in Candidate B: after native torch Phase-B initialization, all ranks call `deepspeed.init_distributed(..., dist_init_required=False)` before constructing the target ZeRO-1 subgroup. No state or workload contract changes.


### Iteration 283 Attempt003 bounded harness failure

The new process-world lifecycle passed through global world4 and target group creation. Rank 0 then hit PyTorch's leaf in-place guard while injecting the already-gathered G1 tensor into the target model. Attempt004 adds only the `torch.no_grad()` guard already used by the production materialization helper. If Attempt004 exposes a nontrivial process-world, collective, or state mismatch, this candidate closes rather than receiving further patch escalation.


### Iteration 283 Attempt004 result — PASS

The two-node gate passed with return code 0 on both node launchers and no OOM delta. Node0 executed a default-world2 ZeRO-3 padded update and gathered one logical G1 state; node1 used an independent default-world2 control. All Phase-A groups were destroyed. The same four live processes then formed a new global world4, reattached DeepSpeed, created target groups, broadcast identical G1 state to resident Generator ranks 0/2, executed one DP2 ZeRO-1 Critic update on ranks 1/3, and globally flushed. Final versions were Generator `[1,1]` and Critic `[1,1]`.

This closes the process-lifecycle uncertainty but remains a toy real-NCCL/DeepSpeed gate. Real-14B Full5 is still Not Run. GO is limited to refactoring the production launcher/executor with the same gather-before-destroy rule; no in-place ZeRO reinterpretation and no timing claim yet.

## Iteration 284 Preregistration — Production default-world ZeRO-3 Full5 transition

### Milestone / Gate

Route-B continuous Full5 production process-world refactor.

### Question

Can the real-14B production executor keep Step10-A on the correct default-world8 DP8 ZeRO-3 backend, then destroy that obsolete process/parameter layout and form the global world16 per-node 2G+6C Critic phase without changing trace, loss, batch, version, Critic, or EMA semantics?

### Hypothesis

The already-passed Iteration-283 mechanism—gather logical G3 before destruction, destroy the node-local default groups, explicitly initialize global world16, then broadcast G3 into separately preconstructed resident targets—will remove the erroneous 93.563083-second ZeRO-1 Step10-A path without in-place shard reinterpretation.

### Preflight / Prior Evidence

Live DMD HEAD was `d79fdf23c7e83eff7abc40fb3d52d555d80dbd63`; Runtime HEAD was `fad336ae874dcf52451e0fbfa0e8e82a47b888e5`; both trees were clean. Both nodes had no GPU process and no new cgroup OOM increment. Iteration 282 closes larger-default-world ZeRO-3 subgroup forward; Iteration 283 validates the node-local-default-world to global-world recovery. The historical 48.551620-second result is not the exact-current-trace denominator because the current padded trace has 22 physical calls per Generator rank.

### Allowed Changes

Process-world lifecycle, global-rank mapping, lifecycle file barriers, launcher topology, tests, and evidence only. No model, trace, batch, loss, optimizer, EMA, target placement, tolerance, or ACK policy change.

### Gate Status

GO for production refactor and tests. Real-14B correctness and timing remain **Not Run** until the code gate passes.

### Iteration 284 code gate result — PASS

The production executor now launches two independent node-local default worlds for Phase A. Node0's default world8 owns the Step10-A DP8 ZeRO-3 Generator; node1's default world8 owns the four Student/four Teacher roles. After logical G3 is gathered and the obsolete engine is destroyed, the Phase-A Gloo/default groups are explicitly destroyed, the same live processes initialize a global world16, target `4G+12C` groups are constructed, and only G3 logical trainables are broadcast before the existing ACK-free Critic/EMA phase.

Tests passed: focused `19`, complete DMD `215`, complete Runtime `322` (using `pytest tests`; unrestricted repository-root discovery includes preserved source snapshots with duplicate test module names and is not the project test command). Trace, batch, loss, optimizer, EMA, target placement, numerical tolerance, and ACK policy were unchanged.

Real-14B correctness and formal timing remain **Not Run**. The next authorized action is one correctness launch; no speed claim is active.

## Iteration 284 real-14B Attempt001 — bounded Phase-A control barrier failure

The first production correctness launch, rb16_f5_zero3_phaseworld_correctness_20260810_001, passed both node input gates and the real world16 topology probe, but stopped before model construction in an extra Phase-A node-local Gloo control barrier. GPU memory remained about 422 MiB/rank, no rank result file was produced, and neither cgroup recorded a new OOM. Both launchers were terminated fail-closed with return code 143. Correctness and performance are **Not Run**. The preserved run root is artifacts/runs/rb16_f5_zero3_phaseworld_correctness_20260810_001. Recovery is initialization-only: use the already-validated Phase-A default NCCL world barrier and retain global Gloo only after Phase-B reinitialization.

## Iteration 285 Preregistration — strict rank-synchronized normal-path trace

### Milestone / Gate

Continuous Full5 trace-contract correction before another real-14B launch.

### Question

Can the active operation trace and executor enforce the user-frozen official normal path in which every rank performs exactly the same per-stage and total DiT calls for each logical operation and step?

### Hypothesis

Removing the amplified GT-zero-history rank-local exception from the measured experiment scope, replaying the rank-zero synchronized exit on every rank, and rejecting any call-count divergence before model construction will eliminate the erroneous padding workload without changing model, loss, batch, optimizer, EMA, or version semantics.

### Prior Evidence / Failure Signature

The immutable v3 bundle contains 268 local-exception entries. Step10-A wave call counts are 4,4,4,8,4,4,6,7 and 4,3,7,7,4,6,9,8; the collective padding path expands this to 22 physical calls/rank. For the synchronized Step10 exit [1,0,0], the correct normal-path workload is 4 calls/wave and 8 calls/rank across GAS2. The upstream special branch remains source evidence but is explicitly excluded from this experiment contract.

### Allowed Changes

Trace generator/validator, current Full5 binding/executor, tests, current launcher trace default/hash, append-only evidence, and the previously bounded initialization-only Phase-A barrier recovery. The read-only upstream source is unchanged. No model/loss/batch/optimizer/EMA/version/tolerance change and no timing run are authorized.

### Correctness / Performance

Not Run. Trace correction is a correctness repair and can never be counted as speedup.

### Gate Status

GO for strict trace generation and CPU/unit validation. Real-14B remains blocked until cross-rank per-operation physical/logical call parity passes fail-closed.

### Iteration 285 initialization-policy amendment — concurrent construction

The user confirmed that historical construction_wave=4 is not part of the
current baseline contract and need not be retroactively applied to preserved
measurements. Future continuous Full5 2G+6C launches now use mixed-BF16 direct
loading with all eight local ranks constructing concurrently
(--construction-wave 8, equivalent to normalized
construction_wave_size=0). Future-target preconstruction no longer serializes
local ranks and has one post-construction barrier.

This changes cold-start admission/turnaround only. It does not change the
workload, model state, trace, batch, optimizer, EMA, or steady-state timing
denominator and may not be counted as a speedup. Historical artifacts remain
unchanged. Focused tests passed 20/20; real-14B concurrent-construction
admission is **Not Run**. An actual OOM must fail closed and be recorded rather
than silently restoring wave 4.

### Iteration 285 initialization cleanup — historical construction-wave control removed

After confirming that the active value 8 produced exactly one construction
interval covering every local rank, the obsolete construction-wave control was
removed from the active continuous Full5 2G+6C launcher and executor. Source
construction now directly executes on every local rank, future-target
preconstruction remains concurrent, and a single post-construction barrier is
retained. The launch manifest records only
construction_policy=concurrent_all_local_ranks.

This is algebraically equivalent to the previous active wave size 8 and touches
only cold-start admission. Model, loss, optimizer, trace, EMA, phase
transition, swap policy, and steady-state timing are unchanged. Preserved
historical candidate launchers and run artifacts were not rewritten.

Validation: launcher bash syntax PASS, executor py_compile PASS, focused tests
21/21 PASS, full DMD tests 216 PASS and 1 SKIP. The first unrestricted full-test
invocation omitted PYTHONPATH=src:. and failed collection; the documented
project invocation then passed. Real-14B all-rank construction admission remains
**Not Run**.

### Iteration 285 strict trace code gate — PASS

The active continuous Full5 launcher is now bound to immutable trace
syn_feature_trace_rank16_v4_strict_sync_steps20 with bundle hash
654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3.
The trace contains 16 rank-qualified tensor streams but one globally
synchronized stage-exit schedule per logical operation. Cross-rank initial
exits, executed exits, per-stage calls, and total calls are fail-closed exact.
There are zero local-exception entries.

For Step10-A the strict schedule is [1,0,0], exactly four physical/logical DiT
calls per wave per rank and eight calls across GAS2, with zero dummy calls. The
active executor no longer passes a padded schedule. The optional collective
padding seam and its padding-only test were removed from the maintained Helios
copy and DMD wrapper so future code cannot silently recover divergent traces by
adding work.

The low-cost Route-B input gate passed for v4 without loading tensor payloads.
The preserved v3 bundle failed closed on forbidden local-exception fields. v3
artifacts remain immutable historical evidence but are not the active input.

Validation: Runtime 320 tests PASS; DMD 217 PASS and 1 SKIP; focused strict
suites 9 and 29 PASS; diff check PASS. Tensor payload hashes remain exact with
v3. This is a correctness repair, not a performance improvement. Real-14B
correctness and formal timing remain **Not Run**.

Runtime strict-trace commit: ec1462bfb84ca444b3beae216c66632744e0d369.


### Iteration 285 real-14B Attempt001 — bounded exchange-directory wait failure

The first strict-v4 real-14B correctness launch, `rb16_f5_stricttrace_v4_pipeline_correctness_20260810_001`, passed both input gates and the world16 topology probe. It then stopped before any rank result or model construction. Each node-local rank 0 called the file-only `_wait_for_path()` on the `execution/exchange` **directory**, so `Path.is_file()` could never pass; the other fourteen ranks waited in the node-local NCCL barrier. The resulting 100% utilization on GPUs 1–7 was a barrier spin kernel, not Helios computation.

This is an initialization-only harness failure, not a trace, model, memory, numerical, or performance result. The run was terminated fail-closed. Node0/node1 cgroup peaks were 27.304/19.516 GiB, per-GPU process memory was about 1,100 MiB, and neither node recorded a new OOM or OOM-kill. All launcher logs and the incomplete run root remain preserved.

A prior-signature search found no matching directory/file-type mechanism. Iteration 284 Attempt001 involved an extra Phase-A Gloo barrier and is materially different. The bounded repair adds `_wait_for_directory()` only for exchange-directory publication while retaining `_wait_for_path()` for file payload fences. Model, loss, batch, optimizer, EMA, trace v4/hash, rank mapping, process worlds, and schedule are unchanged.

Validation after the repair: py_compile PASS; focused Route-B suites `41 passed`; complete DMD suite `218 passed, 1 skipped`; diff check PASS. GO is limited to one retry of the identical real-14B strict-v4 correctness gate. Correctness and performance remain **Not Run** for Attempt001.


### Iteration 285 real-14B Attempt002 — bounded DP-group identity assertion failure

After the directory-fence fix, `rb16_f5_stricttrace_v4_pipeline_correctness_20260810_002` passed input/topology gates, created all sixteen rank records, constructed the real 14B Step10-A source models, and reached ZeRO-3 initialization. All eight Generator ranks then failed the same local assertion: `Step10-A optimizer escaped the DP8 group`.

Source inspection closed the cause without changing the backend: DeepSpeed `groups._get_data_parallel_group()` deliberately returns `_clone_world_group()` when no MPU is present. The optimizer therefore owns the same logical default-world8 rank set through a different Python ProcessGroup object. Object identity is not a valid communication-domain invariant. No collective mismatch, trace mismatch, OOM, or OOM-kill occurred. Node0/node1 cgroup peaks were 78.490/73.402 GiB; maximum per-GPU process memory was 31,762/33,334 MiB. Correctness and performance remain **Not Run**.

The bounded repair replaces only object identity with fail-closed membership checks on both `engine.data_parallel_group` and `engine.optimizer.dp_process_group`; both must resolve exactly to ranks `[0..7]`. The validated ranks are recorded in every initialization result. No model, trace, loss, batch, ZeRO stage, optimizer, EMA, schedule, or tolerance changed.

Validation: py_compile PASS; focused Route-B suites `42 passed`; complete DMD suite `219 passed, 1 skipped`; diff check PASS. GO is limited to one same-contract real-14B retry. A further nontrivial failure closes this patch sequence for review rather than triggering unbounded repair.


### Iteration 285 real-14B Attempt003 — strict trace advances to post-backward optimizer API gate

`rb16_f5_stricttrace_v4_pipeline_correctness_20260810_003` passed the repaired directory and DP-group gates. All eight real 14B Step10-A Generator ranks restored G2 under DP8 ZeRO-3 with exact engine/optimizer rank membership `[0..7]`, then entered both GAS2 rollout waves under trace v4 (`[1,0,0]`, four DiT calls/wave/rank, eight calls/rank total). The executor reached `engine.step()` on every Generator rank without a collective stall or trace/call-count assertion. This is real execution evidence that the strict synchronized trace removes the previous rank-divergent ZeRO-3 control failure; it is not yet an end-to-end Full5 correctness or timing result.

The run failed inside the first resumed PyTorch AdamW step with `state_steps argument must contain a list of singleton tensors`. The logical ZeRO-3 loader restored the correct scalar value 2 but encoded it as a Python integer based on a stale DeepSpeedCPUAdam assumption. The configured concrete optimizer is `torch.optim.AdamW`, whose current functional API requires a singleton Tensor. This failure occurs after both rollout/backward waves and is independent of trace scheduling. No new OOM or OOM-kill occurred. Diagnostic-only cgroup peaks were 336.046 GiB (node0) and 74.782 GiB (node1); maximum per-GPU process memory was 31,986/36,754 MiB.

The bounded compatibility repair preserves the logical step value and selects the native representation: singleton FP32 Tensor for PyTorch Adam/AdamW, integer for other CPU-optimizer implementations. The restore unit test now executes a real AdamW step and proves the counter advances 2→3; the previous test inspected state only and therefore missed this API boundary. Focused tests: `31 passed`; complete DMD suite: `219 passed, 1 skipped`; diff check PASS.

GO is limited to one final same-contract correctness retry. No further nontrivial patch escalation is authorized in Iteration 285. Correctness and performance remain **Not Run** end to end.


### Iteration 285 real-14B Attempt004 — repeated optimizer signature; bounded stop

The final same-contract retry `rb16_f5_stricttrace_v4_pipeline_correctness_20260810_004` again completed both strict-v4 Step10-A rollout/backward waves on all eight DP8 ZeRO-3 Generator ranks and again failed only inside the first resumed AdamW optimizer step with the identical `state_steps` singleton-Tensor signature. There was no trace/call-count assertion, collective-order stall, OOM, or OOM-kill. Diagnostic-only node cgroup peaks were 333.884/74.573 GiB and maximum per-GPU process memory was 31,986/36,754 MiB.

The attempted shallow type repair in `ed3f7000bbd800873be48209736e98f48232bdd8` is rejected: `_unwrap_optimizer()` stops at the ZeRO-3 wrapper because that wrapper already exposes `state`, so the direct-AdamW unit fixture did not represent the production wrapper chain. Production code and that misleading unit-only assertion were returned to the last validated substrate; the failed repair commit and all run artifacts remain in history.

Per the frozen repeated-failure rule, the same mechanism and magnitude after one bounded repair ends patch escalation. No more 16-GPU launches are authorized in this iteration. A future checkpoint-restore iteration must separately preregister nested concrete-optimizer discovery and pass an actual ZeRO-3-wrapper optimizer-step gate before another 14B run.

### Iteration 285 strict trace gate result — PARTIAL

The trace question itself is answered positively within a precise boundary. The immutable v4 bundle has zero rank-local exception entries, Step10-A exit `[1,0,0]`, exactly four DiT calls/wave/rank and eight across GAS2, with no dummy padding. In two independent real-14B attempts, every Generator rank advanced through both rollout/backward waves to `engine.step()` without the former rank-divergent ZeRO-3 collective stall. Therefore strict rank synchronization fixes the targeted model-call-order defect.

End-to-end Full5 correctness remains **Not Run** because G2→G3 optimizer commit, role transition, Critic steps, and EMA did not execute. Performance remains **Not Run** and trace correction is never counted as speedup. Gate status is PARTIAL: trace synchronization validated; continuous Full5 blocked by a separate logical-checkpoint optimizer-wrapper compatibility defect.


## Iteration 286 Preregistration — ZeRO-3 logical optimizer resume gate

### Milestone / Gate

Continuous Full5 checkpoint-restore compatibility gate, separate from the closed Iteration 285 trace gate.

### Question

Can the already-validated ZeRO-3 compute path restore canonical logical G2 master weights, Adam moments and step counter into the real nested DeepSpeed optimizer wrapper, then complete its first resumed optimizer update?

### Hypothesis

Earlier ZeRO-3 success used fresh/native optimizer initialization and therefore received the optimizer's native step representation automatically. The current custom logical restore maps master/m/v correctly but stores step 2 as a Python integer. The actual nested PyTorch Adam-family optimizer requires a singleton Tensor. Wrapper-chain discovery plus native representation restoration should fix this boundary without changing any training or schedule semantics.

### Importance

This is the only boundary currently blocking G2→G3 after both strict-v4 real-14B GAS2 rollout/backward waves. It is checkpoint-layout plumbing, not a ZeRO-3 compute failure.

### Evidence Needed

A real DP2 DeepSpeed ZeRO-3 wrapper gate must report the wrapper chain, restore exact master/m/v/step=2, execute backward and optimizer step, and prove 2→3. Only then may one same-contract real-14B retry run.

### Refutation Condition

The DP2 wrapper gate still fails after one representation fix, changes master/m/v unexpectedly, or requires a model/trace/loss/batch/optimizer/schedule change.

### Allowed Changes

`runtime/critic_pool_checkpoint.py`, `tests/test_routeb_checkpoint.py`, one DP2 wrapper-level gate, manifests and this append-only log. No model, trace, batch, loss, optimizer hyperparameter, EMA, role-placement or timing change.

### Commands

Preflight revalidated both live nodes, clean DMD/Runtime trees, idle GPUs, current cgroup events, assets, and full DMD tests (`220 passed`). Exact command records will be included in the result manifest.

### Correctness

Not Run.

### Performance

Not Run.

### Gate Status

GO for bounded DP2 wrapper introspection and restore-and-step gate. NO-GO for another 16-GPU launch until that gate passes.


### Iteration 286 Measured Results — wrapper restore-and-step gate PASS

Read-only DP2 introspection under the exact production environment found the real chain:

`DeepSpeedZeroOptimizer_Stage3 -> torch.optim.adamw.AdamW`

The outer and nested optimizers expose the same state mapping, but the previous `_unwrap_optimizer()` stopped at the outer wrapper. This explains why the direct AdamW unit fixture passed while the real wrapper retained an integer step and failed. It also closes the historical comparison: Iterations 229/230 proved ZeRO-3 compute with freshly initialized optimizer state; they did not restore canonical logical G2 master/m/v/step across layouts.

The bounded repair discovers the deepest concrete optimizer and reconstructs only the native representation of logical step 2 as a singleton FP32 Tensor. Logical master weights, Adam moments, learning rate, trace, model, loss, batch, schedule and checkpoint values are unchanged. Unsupported concrete optimizer classes fail closed.

DP2 Attempt001 already proved the API fix: both ranks restored exact master/m/v, advanced optimizer 2→3, engine 2→3 and scheduler 2→3. It returned nonzero only because an over-strong test demanded every local FP32 shard change bitwise; rank1's arbitrary toy update rounded to no representable master change. The harness was corrected to require a global update plus finite optimizer state.

DP2 Attempt002 passed on both ranks: exact restore before step, optimizer 2→3, engine 2→3, scheduler 2→3, global master changed, optimizer state finite, and no cgroup OOM/OOM-kill increment. Focused tests passed 7/7; the complete DMD suite passed 220/220; `git diff --check` passed.

### Correctness

PASS for the real DeepSpeed ZeRO-3 wrapper-level logical restore and first resumed optimizer step. Real-14B G2→G3 and later Full5 phases remain Not Run after this repair.

### Performance

Not Run. The DP2 gate is diagnostic and supplies no timing claim.

### Interpretation

The old ZeRO-3 backend was reusable; the failure was not ZeRO-3 forward/backward or strict trace. The missing piece was a new cross-layout logical-checkpoint resume boundary absent from the historical fresh-optimizer runs.

### Gate Status

PASS. GO for exactly one same-contract strict-v4 real-14B correctness retry after commit and two-node source verification.

### Iteration 286 real-14B Attempt001 — Step10-A passes; premature source-release check fails closed

The authorized same-contract retry
`rb16_f5_stricttrace_v4_z3resume_correctness_20260810_001` used the frozen
official 384x640 workload, strict trace v4/hash
`654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3`,
logical G2/C10/EMA10 package, DP8 ZeRO-3 Step10-A and the planned 2G+6C
target. It passed input/topology gates, both strict GAS2 rollout/backward waves,
the repaired AdamW optimizer commit G2 to G3, and collective G3 extraction.
This closes the optimizer-resume question on the real 14B path.

All eight Step10-A Generator ranks then failed the same fail-closed lifetime
assertion: `obsolete ZeRO-3 source storage did not decrease`. The check ran
inside `_destroy_obsolete_zero3_engine()` while its caller still owned the
last `step10a_engine` alias. The object therefore could not be collected before
the snapshot. This is an ownership-observation-order defect in the transition
harness, not a model, trace, optimizer, numerical, loss, batch, or scheduling
failure. Node1 was waiting for the missing Phase-B peer and was explicitly
terminated; no new cgroup OOM or OOM-kill occurred.

Diagnostic resource peaks were 350.335 GiB / 73.289 GiB cgroup memory on
node0/node1. Maximum per-GPU process memory was 31,986 MiB on node0 and
36,754/35,786 MiB on node1. Phase-B Critic steps, EMA, end-to-end Full5
correctness and performance are **Not Run** for this attempt. All original run
artifacts remain preserved.

### Iteration 286 ZeRO-3 source-release ownership gate — PASS

The bounded repair changes only observation order. Teardown first detaches the
obsolete engine; the caller then clears its final owner; only afterward does a
separate verifier run GC, empty the allocator cache, synchronize, and require
allocated storage to decrease. No trace, model, loss, batch, optimizer state,
EMA, placement, schedule, tolerance, or performance contract changed.

A real DP2 DeepSpeed ZeRO-3 + CPU-optimizer gate exercised the production
helpers. Attempt001 exposed retained diagnostic loss/parameter aliases.
Attempt002 passed release but failed while formatting output after the engine
was intentionally cleared. Attempt003 passed on both ranks: exact logical
master/m/v restore, optimizer/engine/scheduler 2 to 3, finite optimizer state,
global master change, and positive allocated-storage release on every rank
(1,024 bytes/rank for the tiny fixture). Cgroup OOM counters did not change.

Focused tests passed 31/31; complete DMD tests passed 219 with one environment
skip; `git diff --check` passed. Prior-signature search found no earlier run of
this exact ownership-order assertion; the check had not been reached before
the optimizer-resume blocker was fixed. Gate decision: **GO** for exactly one
same-contract real-14B strict-v4 correctness retry. Performance remains
**Not Run**.

### Iteration 286 real-14B Attempt002 — caller alias fix is insufficient; DeepSpeed source liveness identified

The same-contract retry
`rb16_f5_stricttrace_v4_z3resume_correctness_20260810_002` again passed strict
Step10-A GAS2, the G2 to G3 AdamW commit, and collective G3 extraction. It
failed closed on the same source-storage assertion after the caller alias had
been removed. The local coordinator immediately terminated the peer node
instead of leaving eight ranks waiting indefinitely. Return codes were 1/143;
there was no new OOM or OOM-kill. Node cgroup peaks were 356.353/74.113 GiB and
per-GPU peaks matched the prior attempt (31,986 MiB on node0;
36,754/35,786 MiB on node1). Critic, EMA, Full5 correctness, and performance
remain **Not Run**.

This repeated signature invalidates the first tiny-fixture inference that the
caller alias was the only owner. No additional 14B launch was made while the
mechanism remained unexplained. A bounded DP2 referrer audit supplied material
new evidence: DeepSpeed 0.18 forward-hook closures retained the Engine;
`DeepSpeedEngine._set_client_model()` retained the client module in both
`_modules["module"]` and `__dict__["module"]`; and ZeRO Init plus generated
pre/post-backward classes retained Parameter objects and their `ds_tensor`
CUDA shards. This is a physical-representation lifetime problem, not a
workload or optimizer-update defect.

### Iteration 286 obsolete ZeRO-3 partition-drain gate — PASS

The bounded repair removes the Engine forward hooks, clears both client-module
owners plus parameter-offload/compression owners, and explicitly replaces only
the obsolete CUDA `ds_tensor`/secondary shards with empty CPU tensors. This
occurs after exact G3 extraction and before target materialization; the source
representation has no legal future read. Trainable values, optimizer state,
trace, batch, loss, EMA, placement, and execution order are unchanged.

DP2 Attempt008 passed on both ranks. It retained exact logical master/m/v,
advanced optimizer/engine/scheduler 2 to 3, kept state finite, collected the
Engine and optimizer, drained both local low-precision partitions (10 bytes in
the tiny fixture), and reduced CUDA allocated storage. Module/Parameter Python
metadata cycles remained alive but no longer owned CUDA parameter storage.
Focused tests passed 31/31. Gate decision is **GO** for one final same-contract
real-14B correctness retry because the new referrer evidence materially differs
from the earlier caller-alias hypothesis. Performance remains **Not Run**.

## Iteration 286 Final Result — real-14B continuous Full5 correctness PASS

### Milestone / Gate

Route-B continuous Full5 `8G+4S+4T -> per-node 2G+6C`, final same-contract
correctness retry after the obsolete ZeRO-3 source-partition drain gate.

### Question

Can the frozen real-14B strict-trace Pipeline run continuously from canonical
`G2/C10/EMA10`, commit Step10-A under DP8 ZeRO-3, release that obsolete source
layout, materialize the target `4G+12C` layout, then complete five Critic and
five EMA updates without OOM, stale state, queue leakage or call-count drift?

### Hypothesis

After exact G3 extraction, removing DeepSpeed Engine owners and draining only
the obsolete no-future-use CUDA partition shards will cross the transition
capacity boundary without changing any logical tensor or training semantic.

### Importance

This is the first end-to-end real-14B validation of the live Step10-A to
Critic-phase representation transition. Earlier Candidate runs either started
from an isolated phase, used the slower ZeRO-1 Step10-A, or stopped before
Critic/EMA completion.

### Evidence Needed

Sixteen PASS rank records; exact trace and state identities; one `G2 -> G3`
update; `C10 -> C15`; `EMA10 -> EMA15`; exact G3 preservation; 60 generated
and consumed payloads; strict per-step cross-rank call parity; empty queues and
transport; positive obsolete-source storage release; zero new cgroup OOM; and
a fail-closed summary with no error.

### Refutation Condition

Any rank failure, trace/call mismatch, wrong update count or version, G3
mutation/gradient, non-exact payload identity, EMA mismatch, unflushed queue,
new OOM, or non-positive source-layout release.

### Allowed Changes

No workload change. The only code entering this retry was the already-gated
physical teardown of the obsolete DP8 ZeRO-3 source layout. Model, 384x640
shape, strict trace v4, logical checkpoint, batch, loss, optimizer, EMA,
placement, Pipeline order, ACK policy and numerical contract remained frozen.

### Commands

- launched both nodes through
  `scripts/distillschedule/launch_routeb_full5_2g6c_continuous_node.sh` with
  run ID `rb16_f5_stricttrace_v4_z3resume_correctness_20260810_003`;
- used DMD HEAD `0fb0363d2772947df1446ce6e742ec551684dcbf` and Runtime HEAD
  `ec1462bfb84ca444b3beae216c66632744e0d369`;
- ran `summarize_routeb_full5_2g6c_continuous.py --schedule pipeline
  --mode correctness`;
- ran a trace-backed post-run call-count audit against each rank's immutable
  v4 operation trace;
- reran the complete DMD regression suite: `219 passed, 1 skipped`.

### Measured Results

**Measured.** Both node launchers returned zero and all 16 rank JSON records
reported PASS. The fail-closed summary reported:

- 16 Step10-A source identities;
- 60 generated / 60 consumed target payloads and 60 unique target trace
  identities;
- one `G2 -> G3` update, five `C10 -> C15` updates and five
  `EMA10 -> EMA15` updates;
- 1,919 obsolete partition tensors and 3,890,967,568 bytes drained per DP8
  source rank, with 3.624506 GiB/rank allocated CUDA release;
- node cgroup peaks 330.793976 / 158.991283 GiB and zero new OOM/OOM-kill;
- target Generator peak allocated/reserved 33.757590/34.800781 GiB;
- target Critic peak allocated/reserved 34.410658/35.048828 GiB.

Correctness-mode diagnostic maxima were Step10-A 124.320293 s, transition
50.655504 s, Critic phase 168.497340 s, and continuous Full5 344.846019 s.
These include SHA and detailed instrumentation and are not formal timing.

### Correctness

**PASS for internal continuous-Full5 correctness.** G3 SHA
`cc9ce4dd39981b5213a40dc119f8965e0c475f7ba806dfcdad60e1a6d3f4003f`
was exact on all four target Generator ranks before and after the Critic phase;
no Generator trainable or frozen gradient was retained. EMA used official
early-bootstrap decay 0.0 and matched G3 exactly at every update. Producer and
consumer slots were empty and outstanding NCCL work was zero at the Step15-A
boundary.

Step10-A executed four logical/physical calls per GAS wave and zero dummy
calls on every Generator rank. Critic rollout calls were rank-synchronized
within each step and matched the immutable trace: steps 10--14 used
`6/6/5/4/6` calls with stage patterns
`[2,2,2]/[2,2,2]/[2,2,1]/[2,1,1]/[2,2,2]`, all no-grad. An initial post-run
audit incorrectly assumed every Critic step had six calls; that audit contract
was superseded by the trace-backed v2 audit and is not an execution failure.

### Performance

**Not Run.** This was correctness mode, not formal low-overhead timing. No
Sequential pair, same-layout numerical comparison, matched-batch16 comparison,
or causal Pipeline speedup is claimed.

### Interpretation

The source-release blocker was a physical lifetime defect in an obsolete
DeepSpeed representation, not a ZeRO-3 compute, trace, optimizer, model or
training-semantic failure. Exact G3 extraction followed by explicit obsolete
partition drain is sufficient for the real transition and admits the resident
target pools. The result closes functional feasibility only; performance
remains a separate preregistered experiment.

### Files Changed

- `artifacts/manifests/iteration286_real14b_attempt003_full5_correctness_pass.json`
- `docs/experiments/rb16_full5_2g6c_continuous_20260810.md`
- `docs/contracts/experiment_names.md`
- append-only `research_log.md`
- immutable run summary under
  `artifacts/runs/rb16_f5_stricttrace_v4_z3resume_correctness_20260810_003/summary/`

### Failure or Limitation

The final run has no functional failure. It is one instrumented Pipeline run;
the frozen same-layout Sequential/Pipeline pair and repeated formal timing are
still Not Run. Target Critic global batch is 12, so this point is not a direct
batch16 speedup denominator. The first ad-hoc call audit encoded a wrong
constant-six-call assumption and is preserved as superseded diagnostic output.

### Gate Status

**PASS** for continuous real-14B Pipeline internal correctness. **NO CLAIM**
for formal performance or causal speedup.

### Next Highest-Value Question

After a separate preregistration, does the exact same contract beat a
same-layout Sequential execution under low-overhead repeated timing, or is the
50.656-second representation transition and Critic bottleneck dominant?

### Next Command

Not Run in this iteration. Freeze the passed correctness package before any
new Sequential/timing launch.

## Iteration 287 — official Step10-A backward semantics audit

### Milestone / Gate

Route-B Step10-A causal attribution before any additional timing or optimization.

### Question

Does the official Helios baseline require the second DP8/GAS2 microbatch to
backpropagate only selected call [3], as in the historical 48.551620-second
anchor, or selected calls [1,2,3], as in strict-v4?

### Hypothesis

The source world16 operation samples one rank-zero stage-exit schedule and
broadcasts it to all 16 ranks. When the same global batch is represented as
DP8/MB1/GAS2, both production rounds must replay that same logical-operation
schedule. For strict-v4 Step10 exit [1,0,0], both rounds retain all three
selected stage calls.

### Importance

A different backward-connected call set changes the executed model workload.
The 48.55-second historical anchor cannot diagnose a code regression or serve
as a speed denominator unless this control and gradient-lineage contract
matches.

### Evidence Needed

Official source sites for exit sampling, rank synchronization, selected-call
grad enablement and Generator backward; the strict-v4 rank-qualified trace; and
the preserved old anchor gradient-lineage records.

### Refutation Condition

Official source samples independently per sample within the world16 logical
batch, or strict-v4 does not represent the frozen source operation, or exit
[1,0,0] severs either downstream selected-stage graph.

### Allowed Changes

Source/artifact audit and append-only documentation only. No model, loss,
trace, optimizer, precision, distributed policy, runtime schedule or GPU run.

### Commands

Inspected:

- /home/shuchenweng/xty/Helios-main/helios/utils/utils_helios_post.py
  lines 575--587, 991--995, 1038--1045, 1158--1161 and 1259--1324;
- /home/shuchenweng/xty/Helios-main/train_helios.py lines 1528--1625;
- official stage_3_post.yaml lines 90--93, 107--108 and 237--251;
- strict-v4 trace bundle hash
  654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3;
- preserved old result
  step15a_zero3_pipeline_precast_wave8_nsys_result_001.json.

### Measured Results

**Source-verified / preserved measured evidence.** Official
generate_and_sync_flag() samples on rank zero and broadcasts. One exit call
in each of the three pyramid stages is executed with gradients for the single
gradient-bearing section; prefix calls are no-grad. The final Generator loss is
then backwarded.

The frozen world16 Step10 operation has synchronized exits [1,0,0]. It
therefore executes call 0 as the stage-0 no-grad prefix and selected calls
[1,2,3] in stages 0/1/2. Stages 1 and 2 have no no-grad prefix, so they do not
sever the upstream graph. In the DP8/GAS2 mapping, source ranks 0--7 and 8--15
are two halves of the same world16 logical operation; both rounds must use
[1,0,0]. Each Generator rank therefore executes two backward calls, each
connected to three selected stage calls (six connected selected calls total).

The old 48.551620-second artifact recorded [1,2,3] for wave 0 but only [3]
for wave 1. That artifact is internally valid for its own different control
trace, but it is not workload-equivalent to strict-v4.

### Correctness

**PASS** for the source/trace semantic audit. Current strict-v4 is aligned on
this dimension. The old second-wave lineage is rejected as an exact-baseline
mapping for the frozen operation.

### Performance

**Not Run.** The 48.55-second value is demoted to a historical
different-control-trace anchor. It cannot be used to claim either a speedup or
a regression against the current strict-v4 implementation.

### Interpretation

Helios does not have one trace-independent constant saying that only the final
pyramid stage backpropagates. Every stage exit forward is grad-enabled, while
the final backward reaches the contiguous selected-stage suffix left connected
after downstream no-grad prefix calls. For the frozen [1,0,0] trace, that
suffix is all three stages in both GAS rounds.

### Files Changed

- artifacts/manifests/iteration287_step10a_official_backward_semantics_audit.json
- append-only research_log.md

### Failure or Limitation

This audit establishes official-normal-path semantics and denominator validity;
it does not explain the remaining strict-v4 runtime overhead and does not
supply a new timing measurement. The literal upstream GT-zero-history
rank-local resampling exception remains outside the user-frozen normal-path
baseline.

### Gate Status

**PASS / GO** for continued causal profiling on strict-v4. **NO-GO** for using
48.55 seconds as the matched official-baseline denominator.

### Next Highest-Value Question

With the workload now fixed, which strict-v4 runtime mechanism accounts for
the excess pageable activation/checkpoint-boundary copies and exposed wait?

### Next Command

No GPU launch in this audit. Continue only with a same-trace, same-precision,
same-layout bounded A/B after preregistration.

## Iteration 288 — legacy Route-A world16 backward-lineage audit

### Milestone / Gate

Baseline semantic reconciliation after the official Step10-A source audit.

### Question

Did the accepted Route-A world16 baseline execute the same Step10 Generator
backward-connected selected calls as strict-v4?

### Hypothesis

No. The accepted Route-A artifact predates the strict normal-path correction
and is bound to trace v3, which contains rank-local amplified GT-zero-history
exit schedules.

### Evidence Needed

The accepted run's compiled trace root, DP/GAS contract and all 16 rank-qualified
Step10 Generator operations.

### Refutation Condition

The accepted run is bound to strict-v4, or every rank has exit [1,0,0] and
three backward-connected selected calls.

### Allowed Changes

Preserved-artifact audit and append-only documentation only. No workload,
baseline artifact, trace, runtime code or GPU execution change.

### Commands

Inspected the accepted run
routea_attempt005_full5_world16_20260808_002, its compiled run/evidence
contract, and every rank's Step10 Generator operation in
syn_feature_trace_rank16_v3_sync_steps20.

### Measured Results

**Preserved measured-artifact audit.** The accepted Route-A run used
world16/MB1/GAS1/global batch 16 and trace v3. Every rank invoked Generator
backward once, but the connected selected-call count was not uniform:

- 9 ranks backpropagated through 3 selected stage calls;
- 3 ranks backpropagated through 2;
- 4 ranks backpropagated through only 1;
- mean 2.3125 connected selected calls/sample;
- physical Generator calls ranged from 3 to 9 per rank.

Only ranks 0,1,2,4,5,8 and 12 had the strict-v4 exit [1,0,0]. Rank 9 and
rank 13 also retained three selected stages under different exits. The remaining
ranks retained a shorter suffix because a downstream no-grad prefix severed
earlier graphs.

### Correctness

**PASS under the frozen legacy-v3 Route-A contract.** This audit does not
invalidate its functional G2/C10/EMA10 to G3/C15/EMA15 result.

**FAIL as a strict-v4 matched denominator.** It did not execute the same
rank-synchronized normal-path workload.

### Performance

No new timing. Existing Route-A timing remains diagnostic for v3 only. It may
not be directly compared with strict-v4 Step10-A.

### Interpretation

"One backward call per rank" does not imply equal backward work. The graph
reachable by that backward depended on each rank's v3 exit schedule. The later
strict-v4 repair made the exit [1,0,0] global and therefore requires three
connected selected stages for every sample.

### Files Changed

- artifacts/manifests/iteration288_routea_v3_backward_lineage_audit.json
- docs/status/routea_world16.md
- append-only research_log.md

### Failure or Limitation

No direct gradient-output hook was stored by Route-A. Connected lineage is
reconstructed deterministically from the preserved stage exits and official
no-grad boundary rule already validated by the historical retained-gradient
artifact. Formal strict-v4 Route-A timing is Not Run.

### Gate Status

Functional Route-A v3 baseline remains preserved. Direct Route-A versus
strict-v4 performance comparison is NO-GO until Route-A is replayed with the
same strict-v4 trace.

### Next Highest-Value Question

What is the strict-v4 Route-A Step10 and Full5 timing under the already
validated Route-A lifecycle policy?

### Next Command

Not Run in this audit. Any rerun must use strict-v4 without counting the trace
repair as a speedup.

## Iteration 289 — fail-closed performance comparability ledger

### Milestone / Gate

Reconcile all preserved timings after discovering trace-v3 versus strict-v4
backward-lineage differences.

### Question

Which existing measurements currently form a valid causal performance pair?

### Hypothesis

Only the isolated Critic-only same-layout Sequential/Pipeline formal trials are
paired. Historical Step15-A, Route-A Full5, current strict-v4 Step10-A and
Route-B Full5 are standalone or mismatched evidence.

### Evidence Needed

Trace/state binding, batch/update scope, layout, instrumentation mode and
correctness status for every proposed numerator and denominator.

### Refutation Condition

Another pair has exact source/config/trace/state/batch/precision/output scope
and repeated formal timing, or the Critic-only timing manifests lack their
later PASS numerical-contract binding.

### Allowed Changes

Evidence audit and documentation only. No workload, runtime or GPU launch.

### Measured Results

The isolated Critic-only formal trials are the only valid pair:

- Sequential: 354.891836, 365.309444 and 348.746184 s; median 354.891836 s.
- Pipeline: 258.777335, 254.509573 and 270.570229 s; median 258.777335 s.
- Median ratio 1.371418x; makespan reduction 27.0828%.

Their manifests share execution trace hash
9e282f4bc24eb589d259e5e5603cae02fa5cd0c6cdd0f06f7c5416461e4564d0
and bind to the later PASS same-layout numerical contract. The scope is the
isolated Critic-only subgraph, not Full5.

No matched strict-v4 Step10-A or Full5 performance pair exists. The current
approximately 123-second strict-v4 Step10-A is a valid standalone observation.
The 48.551620-second historical Step15-A, Route-A v3 Full5 and Route-B
strict-v4 correctness run are not its denominators.

### Correctness

PASS for the Critic-only timing binding under the user-authorized same-layout
noise contract. Full5 strict-v4 comparison remains Not Run.

### Performance

One valid isolated result: Pipeline reduces the four-wave Critic-only makespan
by 27.08% versus same-contract Sequential. No official Full5 speedup claim.

### Interpretation

Previous discussion mixed valid standalone data with causal comparisons.
The new ledger fails closed: if a numerator and denominator are not explicitly
paired under one frozen contract, no speedup is reported.

### Files Changed

- docs/status/performance_comparability_ledger.md
- append-only research_log.md

### Failure or Limitation

The isolated Critic-only trace is not a substitute for an official strict-v4
Full5 comparison. Route-B target Critic global batch 12 also prevents direct
comparison with Route-A global batch 16.

### Gate Status

NO-GO for any current official Full5 speedup. GO only for reporting the isolated
Critic-only 27.08% result within its stated scope.

### Next Highest-Value Question

Can a strict-v4 Route-A baseline and a strict-v4 candidate be run with identical
logical batch/update and low-overhead repeated timing?

### Next Command

Not Run. Freeze a matched official Full5 comparison manifest before launching.

## Iteration 290 — recover the completed Score4 Full5 result in the comparability ledger

### Milestone / Gate

Historical performance-evidence reconciliation after the user identified an
omitted completed offload-four Full5 run.

### Question

Was the approximately 400-second offload-four result a real completed Full5
run, and what comparisons does its preserved contract permit?

### Hypothesis

Yes. The result is `RA16-F5-GLP-SCORE4`, a functional world16 Full5 PASS that
selectively streams four **Score** tail blocks. It is a valid legacy-v3
standalone result and a diagnostic comparison against the legacy-v3 all-40
reference, but not a strict-v4 or formal repeated timing result.

### Evidence Needed

The immutable result manifest, frozen contract, rank traces, update/version
counts, timing mode, exact streamed block set and historical denominator.

### Refutation Condition

The run did not finish steps 10--14, did not preserve G/C/EMA semantics, used a
different global batch from its denominator, was actually Generator-offload4,
or had formal repeated paired timing.

### Allowed Changes

Preserved-artifact audit and append-only documentation only. No workload,
runtime, trace or GPU execution change.

### Commands

Inspected `ra16_f5_glp_score4_20260809_001_result_summarizer_v2.json`,
`iteration_264_ra16_f5_glp_score4_result.json`, the Score4 frozen contract and
its experiment report; reconciled them with the performance ledger.

### Measured Results

`RA16-F5-GLP-SCORE4` completed world16, official 384x640, checkpoint-10,
steps 10--14, MB1/GAS1/global batch 16, trace v3, ZeRO-1 CPU optimizer and
A32 activation offload. Its policy includes Generator g12, typed Generator
LP-state phase offload, and Score blocks `[36,37,38,39]` streamed with window 1.
It is therefore **Score4**, not the failed Generator partial-offload4 attempt.

The observed Full5 window was **404.041535 s**:

- step 10: 161.524501 s;
- step 11: 66.580910 s;
- step 12: 62.646466 s;
- step 13: 62.034698 s;
- step 14: 65.487268 s;
- weighted logical step: 80.808307 s;
- throughput: 0.197999 samples/s.

The preserved same-v3 historical all-40 reference is 688.725684 s. The single
observation is 41.3349% shorter, or 1.704x throughput-equivalent. This remains
diagnostic because repeated randomized paired timing was Not Run.

### Correctness

**Functional PASS.** All 16 ranks completed the exact observed transitions
`G2->G3`, `C10->C15` and `EMA10->EMA15`; checkpoint-15 passed structural
validation. No CUDA/cgroup OOM increment was observed.

### Performance

**Measured diagnostic, not formal.** The 404.041535-second result may be
reported as a legacy-v3 Full5 capacity/performance point and compared
motivationally with its same-v3 688.725684-second all-40 reference. It cannot be
used against strict-v4 Step10-A, strict-v4 Route-B, or as a repeated formal
speedup.

### Interpretation

Iteration 289 was incomplete: it correctly found no strict-v4 Full5 pair, but
its wording omitted this valid legacy-v3 diagnostic pair. The key naming
clarification is that the successful four-block policy streams Score blocks
36--39; the earlier Generator `g4` Full5 attempt was a different failed path.

### Files Changed

- docs/status/performance_comparability_ledger.md
- append-only research_log.md

### Failure or Limitation

The legacy trace v3 has rank-dependent Step10 backward lineage and cannot serve
as the official-normal-path strict-v4 denominator. The run is one
instrumented acceptance trial, so variance and formal median speedup are
unknown.

### Gate Status

**PASS** for restoring the omitted result to the evidence ledger. **NO-GO** for
promoting it to a strict-v4 or formal repeated performance claim.

### Next Highest-Value Question

Can the frozen Score4 lifecycle plan and its all-40 comparator be replayed as a
matched strict-v4 repeated pair without changing any other contract field?

### Next Command

Not Run. Any GPU launch requires a new strict-v4 paired preregistration.

## Iteration 291 — audit strict-v4 Step10-A and live G3 representation transition

### Milestone / Gate

Route-B continuous Full5 optimization audit after strict-v4 functional correctness; matched
formal Full5 timing remains closed.

### Question

Which measured or source-proven mechanisms explain the approximately 123.12-second
Step10-A and 50.66-second live representation transition, and which bounded optimizations
can reduce them without changing the frozen strict-v4 workload?

### Hypothesis

Step10-A is limited mainly by repeated ZeRO-3 AllGather plus pageable checkpoint-boundary
traffic, while transition time is limited mainly by serialized process-group creation and
target-runtime initialization rather than by G3 tensor extraction alone.

### Importance

These two phases contribute about 174 seconds to the diagnostic 344.85-second continuous
Full5 run. Optimizing Critic-only overlap cannot compensate for a slow Generator update and
layout conversion.

### Evidence Needed

Existing low-overhead Step10-A timing, Nsight kernel/copy intervals and memory kinds,
per-rank transition records, exact source group/broadcast loops, live repository/GPU state,
and a clean test result.

### Refutation Condition

Gradient-file wait or CPU optimizer step dominates Step10-A; transition time tracks only
G3 gather/broadcast ranks; or the current source no longer matches the preserved artifacts.

### Allowed Changes

Evidence audit, report, manifest and append-only research log only. No runtime policy,
trace, placement, loss, optimizer, GPU launch or performance claim.

### Commands

Revalidated both remote nodes and repository state; ran `PYTHONPATH=src:. python -m pytest
-q`; parsed Iteration 286/287 manifests and Nsight summaries; audited Step10-A ZeRO-3
configuration, strict-v4 execution, transition group creation, G3 broadcast and Student
initialization; inspected logical LoRA and frozen-checkpoint size inventories.

### Measured Results

- Live HEAD `02c059d73e98d13532d15a10e6c59c63c5923e9c`, clean, no GPU processes,
  no current cgroup OOM increment; tests: 222 passed, 1 skipped.
- Low-overhead Step10-A: 123.115506 s median; two rollouts 20.675323/21.036667 s;
  two backwards 38.368997/39.909053 s; engine step 1.150879 s; gradient wait 0.009794 s.
- Nsight summed phase medians: compute 28.239920 s, NCCL union 49.235247 s, copy union
  27.814135 s, uncovered 29.408873 s. These overlap and are diagnostic only.
- Rank0 copy inventory: 57.747357 GiB D2H to pageable host, 29.185779 GiB pageable H2D,
  approximately zero pinned D2H. The three largest repeated D2H sizes match BF16
  checkpoint-boundary hidden shapes; call-site ownership remains unproven.
- Transition: 50.655504 s max. G3 extraction is 1.67--3.46 s and G3 broadcast 2.58--7.79 s,
  but retained node1 Score ranks still take about 47.825 s.
- Timed transition source performs 30 serialized NCCL subgroup creations and about 31
  Gloo barriers, then 818 individual G3 broadcasts.
- A 2M-element ZeRO-3 persistence threshold covers 737 Generator trainables / 1.789 GiB
  BF16, while adding only about 0.019 GiB of frozen checkpoint tensors below that threshold.

### Correctness

No execution policy changed. Existing continuous correctness remains 16/16 PASS under
strict-v4; this audit introduces no new correctness result.

### Performance

No new timing launch and no new speedup. Existing numbers remain diagnostic. Proposed
85--105 s Step10-A and 10--25 s transition ranges are falsifiable targets, not measurements.

### Interpretation

The first Step10 optimization should be bounded ZeRO-3 persistence/prefetch plus attribution
of pageable boundary copies, not file transport or deletion of selected calls. The first
transition optimization should be subphase timing followed by communicator collapse and
bucketed G3 broadcast. Most transition time is shared setup because ranks that do no G3
work still wait approximately 47.8 s.

### Files Changed

- docs/experiments/rb16_step10a_transition_optimization_audit_20260810.md
- artifacts/manifests/iteration291_step10a_transition_optimization_audit.json
- append-only research_log.md

### Failure or Limitation

Nsight has profiling overhead and no runtime call chains for the pageable copies. Transition
subphases other than G3 extraction/broadcast are not separately timed. Formal matched
Full5 performance remains Not Run.

### Gate Status

GO for one profile-only transition-instrumentation iteration and small communicator gate.
NO-GO for a Full5 speedup claim or for changing the strict-v4 graph/backend contract.

### Next Highest-Value Question

How much of the common 47.8-second transition is process-group creation versus DP12 engine
initialization/restore and target-model H2D?

### Next Command

Add low-overhead transition subphase events with no policy change, run the existing small
distributed group gate, then authorize one correctness-bound real14B transition profile.


## Iteration 292 — preregister strict-v4 Step10-A and live-transition optimization

### Milestone / Gate

Continuous Route-B Full5 optimization after the strict-v4 functional PASS. Formal repeated Full5 timing remains closed until optimized-path correctness passes.

### Question

Can lifecycle/communication scheduling alone move strict-v4 Step10-A from 123.115506 s toward 85--105 s and the live representation transition from 50.655504 s toward 10--25 s?

### Hypothesis

Step10-A can remove repeated small-trainable ZeRO-3 materialization with a bounded 2M-element persistence threshold; transition time can remove serialized communicator setup and 818 small G3 broadcasts while preserving the exact logical tensor inventory.

### Importance

The two phases contribute about 174 seconds to the only continuous strict-v4 Full5 functional run. They must improve before the 4G+12C Critic pipeline can be assessed as an end-to-end schedule.

### Evidence Needed

Transition subphase timestamps, a bounded distributed communicator gate, exact G3 inventory/SHA validation, real allocator admission, strict-v4 x0/gradient identities, and matched low-overhead timing.

### Refutation Condition

Group creation and G3 launch count are not material; MP-size-1 without singleton groups or global-world P2P is unsafe; 2M persistence OOMs or does not materially improve Step10-A; or any exact semantic fingerprint changes.

### Allowed Changes

Low-overhead timing instrumentation; small communicator gate; validated communicator collapse; bounded bucketed G3 broadcast; optional 2M ZeRO-3 persistence candidate; tests/launcher plumbing/manifests/docs. No trace, graph, batch, loss, optimizer, EMA, placement, precision, or backend-stage change.

### Commands

Live preflight checked both nodes, HEAD/branch/clean state, GPU processes, cgroup events and assets. `PYTHONPATH=src:. /tmp/distillschedule-helios-official-torch210/bin/python -m pytest -q` passed 222 tests with 1 skip.

### Measured Results

Preflight only. Both nodes are idle and clean at `8f62e6deca7706cd5b4213d12e2678e561819ad6`; current cgroup OOM counters are zero, with historical `oom_kill=1`.

### Correctness

No runtime change yet. Frozen reference remains Iteration 286 strict-v4 16/16 PASS.

### Performance

Not Run. Targets are hypotheses, not results.

### Interpretation

GO for timing instrumentation and one bounded communicator gate. Known ZeRO-2, in-place shard reinterpretation, meta/swap and reduced-backward paths remain closed.

### Files Changed

- artifacts/manifests/iteration292_step10a_transition_optimization_prereg.json
- append-only research_log.md

### Failure or Limitation

A bare pytest invocation failed because the source-layout package was not on `PYTHONPATH`; the correct repository invocation passed and no code/environment repair was needed.

### Gate Status

GO.

### Next Highest-Value Question

Can a low-overhead transition timeline prove which common setup subphase dominates the 47.8--50.7 s transition?

### Next Command

Implement subphase instrumentation and its unit tests before any policy change or GPU launch.


## Iteration 292 — preregister strict-v4 Step10-A and live-transition optimization

### Milestone / Gate

Continuous Route-B Full5 optimization after the strict-v4 functional PASS. Formal repeated Full5 timing remains closed until optimized-path correctness passes.

### Question

Can lifecycle/communication scheduling alone move strict-v4 Step10-A from 123.115506 s toward 85--105 s and the live representation transition from 50.655504 s toward 10--25 s?

### Hypothesis

Step10-A can remove repeated small-trainable ZeRO-3 materialization with a bounded 2M-element persistence threshold; transition time can remove serialized communicator setup and 818 small G3 broadcasts while preserving the exact logical tensor inventory.

### Importance

The two phases contribute about 174 seconds to the only continuous strict-v4 Full5 functional run. They must improve before the 4G+12C Critic pipeline can be assessed as an end-to-end schedule.

### Evidence Needed

Transition subphase timestamps, a bounded distributed communicator gate, exact G3 inventory/SHA validation, real allocator admission, strict-v4 x0/gradient identities, and matched low-overhead timing.

### Refutation Condition

Group creation and G3 launch count are not material; MP-size-1 without singleton groups or global-world P2P is unsafe; 2M persistence OOMs or does not materially improve Step10-A; or any exact semantic fingerprint changes.

### Allowed Changes

Low-overhead timing instrumentation; small communicator gate; validated communicator collapse; bounded bucketed G3 broadcast; optional 2M ZeRO-3 persistence candidate; tests/launcher plumbing/manifests/docs. No trace, graph, batch, loss, optimizer, EMA, placement, precision, or backend-stage change.

### Commands

Live preflight checked both nodes, HEAD/branch/clean state, GPU processes, cgroup events and assets. `PYTHONPATH=src:. /tmp/distillschedule-helios-official-torch210/bin/python -m pytest -q` passed 222 tests with 1 skip.

### Measured Results

Preflight only. Both nodes are idle and clean at `8f62e6deca7706cd5b4213d12e2678e561819ad6`; current cgroup OOM counters are zero, with historical `oom_kill=1`.

### Correctness

No runtime change yet. Frozen reference remains Iteration 286 strict-v4 16/16 PASS.

### Performance

Not Run. Targets are hypotheses, not results.

### Interpretation

GO for timing instrumentation and one bounded communicator gate. Known ZeRO-2, in-place shard reinterpretation, meta/swap and reduced-backward paths remain closed.

### Files Changed

- artifacts/manifests/iteration292_step10a_transition_optimization_prereg.json
- append-only research_log.md

### Failure or Limitation

A bare pytest invocation failed because the source-layout package was not on `PYTHONPATH`; the correct repository invocation passed and no code/environment repair was needed.

### Gate Status

GO.

### Next Highest-Value Question

Can a low-overhead transition timeline prove which common setup subphase dominates the 47.8--50.7 s transition?

### Next Command

Implement subphase instrumentation and its unit tests before any policy change or GPU launch.


## Iteration 293 — measure the strict-v4 live-transition causal profile

### Milestone / Gate

Continuous Route-B Full5 transition optimization, instrumentation-only gate.

### Question

Which common subphase actually dominates the 47.8--50.7-second live transition?

### Hypothesis

The prior source audit predicted serialized process-group creation and target runtime setup; low-overhead boundary timers can distinguish group creation, H2D, DeepSpeed initialization, optimizer restore and waiting.

### Importance

A wrong causal target would add communicator complexity without moving end-to-end time.

### Evidence Needed

A real 16-rank strict-v4 Full5 correctness run with no policy change and per-rank subphase timestamps.

### Refutation Condition

Group creation is below 10% of transition time, or another measured interval dominates.

### Allowed Changes

Timing instrumentation only; no workload, schedule, resource policy or numerical-contract change.

### Commands

Launched `rb16_f5_iter292_transition_instrumentation_correctness_20260810_001` on both nodes at source `8bef5a7`, summarized all 16 rank records and checked launcher/cgroup outcomes.

### Measured Results

Step10-A 124.911580 s; transition 49.839245 s; Critic phase 165.794245 s; Full5 341.819356 s. Group creation max 0.666031 s. Switching Critic H2D max 33.168625 s; target Generator H2D max 28.799422 s; G3 broadcast max 10.868301 s.

### Correctness

PASS: 16/16 ranks, exact frozen identities, G3/C15/EMA15, 60 generated/consumed payloads, queue flush, no new OOM.

### Performance

Diagnostic correctness/profile only; no formal speedup.

### Interpretation

The group-creation hypothesis is refuted. Pageable whole-module target H2D is dominant. Retained-rank 37--38 s DS initialization is mostly DP12 arrival wait because switching ranks show 3.6--5.8 s local initialization after 30--33 s H2D.

### Files Changed

- docs/experiments/rb16_transition_causal_profile_20260810.md
- artifacts/manifests/iteration293_transition_causal_profile_result.json
- append-only research_log.md

### Failure or Limitation

Formal repeated timing and optimized-policy correctness are Not Run. Timers are host wall at existing boundaries, not pure CUDA activity.

### Gate Status

GO for bounded pinned target materialization plus bucketed G3. NO-GO for production communicator collapse at this transition gate.

### Next Highest-Value Question

Can bounded pinned ping-pong H2D reduce the 30--33 s target materialization without changing any registered state or exceeding memory?

### Next Command

Implement the same-source selectable candidate, pass unit/CUDA mini tests, then run one exact strict-v4 Full5 correctness trial.


## Iteration 294 — preregister bounded pinned transition materialization

### Milestone / Gate

Continuous Route-B Full5 transition lifecycle candidate after Iteration 293 causal attribution.

### Question

Can bounded pinned ping-pong target H2D plus fixed-order G3 buckets move the 49.839245-second transition into the 10--25-second target without changing semantics?

### Hypothesis

The current 22.8--33.2-second pageable-source `module.to()` paths are launch/pageable limited. Two 64-MiB pinned slots per encountered dtype can overlap CPU staging and H2D; 128-MiB G3 buckets can replace 818 collectives while preserving named tensors.

### Importance

This directly addresses the measured dominant transition intervals. Group collapse is rejected here because its measured ceiling is below 2%.

### Evidence Needed

Unit/CUDA-mini exactness, bounded arena accounting, real allocator admission, 16/16 strict-v4 correctness, G3 SHA, final C15/EMA15, queue flush and per-rank subphase times.

### Refutation Condition

Any alias/layout guard, exact identity, version, queue, OOM or leak failure; or transition improvement below 10%.

### Allowed Changes

Selectable transition H2D policy, bounded pinned transfer helper, selectable G3 bucket size, tests and launcher plumbing. Step10 persistence remains zero; communicators, graph, batch, optimizer, EMA and placement remain frozen.

### Commands

Not Run yet.

### Measured Results

Baseline: transition 49.839245 s; switching Critic H2D 33.168865 s max; Generator H2D 28.799422 s max; G3 broadcast 10.868068 s max.

### Correctness

Not Run.

### Performance

Not Run.

### Interpretation

GO for one exact correctness/profile trial after CPU and CUDA-mini tests.

### Files Changed

- artifacts/manifests/iteration294_bounded_pinned_transition_prereg.json
- append-only research_log.md

### Failure or Limitation

Pinned benefit under ten concurrent switching ranks is unknown; formal timing is outside this trial.

### Gate Status

GO.

### Next Highest-Value Question

Does bounded pinned materialization preserve official 14B state and achieve real H2D bandwidth under concurrent ranks?

### Next Command

Land selectable primitives, run all tests including CUDA mini, then launch one strict-v4 correctness trial with persistence=0, pinned64x2 and G3 bucket128.

## Iteration 294 — result: bounded pinned transition with 128-MiB G3 buckets

### Milestone / Gate

Continuous Route-B Full5 transition optimization, first measured lifecycle candidate.

### Question

Does bounded pinned target materialization plus 128-MiB G3 bucketing reduce the strict-v4 transition by at least 10% without semantic drift?

### Hypothesis

Pinned ping-pong staging hides pageable-source overhead and fixed-order buckets reduce G3 collective overhead.

### Importance

The prior causal profile assigned 22.8--33.2 s to target H2D and 10.9 s to G3 redistribution.

### Evidence Needed

One exact real-14B correctness/profile run with bounded memory and per-rank subphases.

### Refutation Condition

Any exact failure/OOM/leak, or transition improvement below 10%.

### Allowed Changes

Only the preregistered transition H2D and G3 redistribution primitives; all strict-v4 semantics and placement frozen.

### Commands

Ran `rb16_f5_iter294_pinned64x2_g3b128_correctness_20260810_001` on both nodes at `7b67932`, then ran the fail-closed Full5 summarizer.

### Measured Results

PASS. Step10-A 120.921975 s; transition 46.086928 s; Critic phase 167.261416 s; Full5 335.253502 s. Target H2D max 29.745484 s versus 33.168865 s baseline (-10.32%). G3 broadcast max 15.400286 s versus 10.868068 s (+41.71%). Transition improved only 7.53%. Pinned arena was bounded at 256 MiB per switching rank. Node cgroup peaks: 357,551,136,768 and 171,137,269,760 bytes; OOM delta 0.

### Correctness

16/16 PASS; exact frozen trace/input identities; Step10-A 16 samples; 60 generated/consumed target payloads; G3/C15/EMA15; queues and runtime handles flushed.

### Performance

Diagnostic correctness/profile only. The joint candidate missed the preregistered >=10% transition gate and is a bounded negative. No formal speedup claim.

### Interpretation

Pinned materialization is valid and locally helpful, but node-0 concurrent target loads still set a 29.7 s critical path. The 128-MiB bucket experiment is refuted: reducing collective count increased the critical-path broadcast. Reuse legacy per-tensor G3 redistribution for the next one-variable trial.

### Files Changed

- docs/experiments/rb16_transition_pinned_result_20260810.md
- artifacts/manifests/iteration294_bounded_pinned_transition_result.json
- append-only research_log.md

### Failure or Limitation

Single correctness/profile run only. Copy activity is measured by CUDA events around runtime-owned copies, not Nsight. No NUMA policy was changed.

### Gate Status

NO-GO for pinned64x2 + bucket128 as the final transition plan; GO for one bounded pinned64x2 + legacy-broadcast isolation run.

### Next Highest-Value Question

Does removing the measured bucket regression let the already-correct pinned materializer cross the 10% transition gate?

### Next Command

Preregister and launch one strict-v4 correctness/profile run with pinned64x2, G3 bucket size 0, and Step10 persistence 0.

## Iteration 295 — preregister pinned transition with legacy G3 redistribution

### Milestone / Gate

Continuous Route-B Full5 transition optimization, one-variable isolation.

### Question

Was the remaining Iteration 294 transition miss caused by the newly measured 128-MiB G3 bucket regression?

### Hypothesis

Keeping exact pinned64x2 materialization but restoring the validated legacy fixed-order per-tensor broadcast removes about 4.5 s from the critical path and crosses the 10% transition gate.

### Importance

It reuses two independently validated primitives and avoids expanding scope before testing the direct measured regression.

### Evidence Needed

One 16-rank strict-v4 correctness/profile run with transition subphases, exact identities, versions, queues and OOM gates.

### Refutation Condition

Any exact/OOM/leak failure; transition >=44.855321 s; or G3 redistribution remains materially above the Iteration 293 legacy interval.

### Allowed Changes

Configuration only: set G3 bucket size from 128 MiB to 0. Pinned64x2 remains; Step10 persistence, communicators, workload, placement and all semantics remain frozen.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run.

### Performance

Not Run.

### Interpretation

GO. NUMA binding and Step10 persistence remain separate future hypotheses.

### Files Changed

- artifacts/manifests/iteration295_pinned_legacy_g3_prereg.json
- append-only research_log.md

### Failure or Limitation

A correctness/profile trial is not formal repeated timing.

### Gate Status

GO.

### Next Highest-Value Question

Can the pinned primitive deliver a >=10% transition improvement once the bucket regression is removed?

### Next Command

Launch strict-v4 Full5 with pinned64x2, G3 bucket0, persistence0, then summarize fail-closed.

## Iteration 295 — result: pinned transition with legacy G3 redistribution

### Milestone / Gate

Continuous Route-B Full5 transition optimization, one-variable isolation result.

### Question

Was the remaining Iteration 294 transition miss caused by the 128-MiB G3 bucket regression?

### Hypothesis

Restoring the validated legacy per-tensor broadcast while retaining pinned64x2 crosses the 10% transition gate.

### Importance

This tests the most direct measured regression without expanding the action space.

### Evidence Needed

One exact real-14B strict-v4 correctness/profile run and fail-closed summary.

### Refutation Condition

Any correctness/OOM/leak failure, transition >=44.855321 s, or broadcast critical path remains elevated.

### Allowed Changes

Configuration only: G3 bucket 128 MiB -> 0. All workload and placement semantics frozen.

### Commands

Ran `rb16_f5_iter295_pinned64x2_g3legacy_correctness_20260810_001` on both nodes at `f4f2800`; summarized all 16 rank records.

### Measured Results

PASS. Step10-A 124.918693 s; transition 47.881013 s; Critic 170.207090 s; Full5 346.689093 s. Target H2D max 28.955584 s (-12.70% from Iter293), but G3 broadcast wall max 14.097394 s (+29.72%). Transition improved only 3.93%. Node peaks 363,020,435,456 / 172,666,871,808 bytes; OOM delta 0.

### Correctness

16/16 PASS; exact frozen identities; G3/C15/EMA15; 60/60 payloads; queues flushed.

### Performance

Diagnostic correctness/profile only. Below the preregistered 10% transition gate; bounded negative.

### Interpretation

The bucket policy is not the root cause. Iterations 294--295 show that bounded pinned H2D is correct and lowers materialization, but concurrent node-0 loads plus rendezvous variability dominate the global transition. Stop bucket-size tuning.

### Files Changed

- docs/experiments/rb16_transition_pinned_legacy_g3_result_20260810.md
- artifacts/manifests/iteration295_pinned_legacy_g3_result.json
- append-only research_log.md

### Failure or Limitation

No formal repeated timing. NUMA-local placement and role-preserving transition remain Not Run.

### Gate Status

NO-GO for more G3 bucket tuning. GO for the previously preregistered Step10-A 2M persistence question.

### Next Highest-Value Question

Can 2M ZeRO-3 trainable persistence reduce repeated parameter materialization in strict-v4 Step10-A?

### Next Command

Preregister a one-run low-overhead Step10-only admission/timing trial with persistence=2,000,000.

## Iteration 296 — preregister Step10-A 2M ZeRO-3 persistence

### Milestone / Gate

Strict-v4 Step10-A lifecycle optimization.

### Question

Can bounded 2M per-parameter persistence reduce repeated ZeRO-3 trainable all-gathers enough to move 123.115506 s toward 85--105 s?

### Hypothesis

The 737 trainable tensors under 2M elements are repeatedly materialized across two selected graphs and two checkpoint backwards. Keeping their 1.789 GiB BF16 state resident is admitted by historical headroom and removes materialization without changing math.

### Importance

Nsight attributed 49.235 s kernel union to NCCL, dominated by ZeRO-3 AllGather; rollout/backward, not detached-gradient wait, dominates Step10-A.

### Evidence Needed

One low-overhead stop-after-Step10 run with unchanged trace/calls/backwards/update, real allocator admission, GPU peak and no OOM. If promising, a later exact Full5 correctness run.

### Refutation Condition

OOM/guard/fingerprint failure or Step10-A >=110.804 s.

### Allowed Changes

Configuration only: `stage3_param_persistence_threshold: 0 -> 2,000,000`. Prefetch/live/reuse/buckets/overlap and all semantics remain unchanged.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run; the timing launcher will bind an immutable prior strict-v4 PASS evaluation and preserve workload fingerprints. Exact candidate Full5 correctness remains required for a final claim.

### Performance

Not Run. Positive target <=105 s; partial 105--110.804 s; negative >=110.804 s.

### Interpretation

GO for one bounded admission/timing trial.

### Files Changed

- artifacts/manifests/iteration296_step10a_2m_persistence_prereg.json
- append-only research_log.md

### Failure or Limitation

This first run does not independently establish exact candidate state equivalence.

### Gate Status

GO.

### Next Highest-Value Question

Does 2M persistence trade <=1.81 GiB extra static model state for >=10% Step10-A time reduction?

### Next Command

Launch one two-node Step10-only timing run with persistence2M and a hash-bound strict-v4 PASS evaluation.

## Iteration 296 — result: Step10-A 2M ZeRO-3 persistence

### Milestone / Gate

Strict-v4 Step10-A lifecycle optimization, bounded persistence result.

### Question

Does 2M persistence remove enough repeated ZeRO-3 trainable materialization to reduce Step10-A by >=10%?

### Hypothesis

Keeping 1.789 GiB of small trainables resident reduces repeated AllGather without OOM or semantic change.

### Importance

Nsight identified AllGather as the dominant communication kernel family.

### Evidence Needed

One real-14B low-overhead Step10-only admission/timing run with exact boundary fingerprints.

### Refutation Condition

OOM/fingerprint failure or Step10-A >=110.804 s.

### Allowed Changes

Only `stage3_param_persistence_threshold: 0 -> 2,000,000`.

### Commands

Ran `rb16_step10a_iter296_persistence2m_timing_20260810_001` on both nodes at `979d774`, bound to immutable PASS summary SHA `0f1d8bed...`.

### Measured Results

Generator-rank median 116.829453 s versus 123.115506 s (-5.11%); world max 118.859191 s versus 129.003622 s (-7.86%). Wave medians: rollout0 19.705991, backward0 36.111826, rollout1 19.686933, backward1 39.858838 s. DeepSpeed persisted 1,432 objects / 967,692,352 elements. Node0 peak 31,762 MiB/GPU and 356,686,761,984 cgroup bytes. OOM delta 0.

### Correctness

All eight Generator ranks/two waves matched trace, x0 SHA, gradient SHA, and surrogate loss exactly; G2->G3 once. Full post-update tensor/Adam export was omitted by low-overhead mode, so combined Full5 exactness is Not Run.

### Performance

A real small improvement, but below the preregistered 10% gate and outside 85--105 s. Bounded negative for persistence alone.

### Interpretation

Small trainable persistence is effective but insufficient. The remaining frozen-block AllGather path still uses a 16.8M-element prefetch bucket and 268M-element live cap, smaller than one 351.4M-element block.

### Files Changed

- docs/experiments/rb16_step10a_2m_persistence_result_20260810.md
- artifacts/manifests/iteration296_step10a_2m_persistence_result.json
- append-only research_log.md

### Failure or Limitation

Single preliminary timing run; no full-state export and no repeated formal timing.

### Gate Status

NO-GO for 2M persistence alone as final plan; GO for one preregistered bounded prefetch/live candidate.

### Next Highest-Value Question

Can one-block-ahead prefetch with a two-block live cap hide frozen-block AllGather without exceeding A100 memory?

### Next Command

Preregister exact element caps, add fail-closed configuration plumbing and unit tests, then run one Step10-only timing/admission trial.

## Iteration 297 — preregister one-block prefetch / two-block live Step10-A candidate

### Milestone / Gate

Strict-v4 Step10-A ZeRO-3 materialization scheduling.

### Question

Can a prefetch bucket larger than one 351,394,304-element frozen block and a live cap larger than two blocks hide exposed AllGather after 2M persistence?

### Hypothesis

The current 16.8M prefetch bucket and 268M live cap are smaller than a 351.4M-element block. Setting prefetch=402,653,184 and live=805,306,368 elements permits one-block-ahead materialization while current-block compute runs.

### Importance

Persistence alone saved only 5.11%; the frozen backbone remains the high-volume repeated materialization path.

### Evidence Needed

Fail-closed config resolution, unit tests, one real-14B Step10-only timing/admission run, exact boundary fingerprints and peak memory.

### Refutation Condition

OOM/fingerprint failure or Step10-A >=110.804 s.

### Allowed Changes

Expose optional ZeRO-3 prefetch/live element caps; set only those fields plus the already validated 2M persistence. Graph, batch, update, overlap_comm, allgather bucket and reuse distance stay frozen.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run.

### Performance

Not Run. Target <=105 s; partial 105--110.804 s.

### Interpretation

GO. This is the final preregistered synchronous ZeRO-3 window candidate before a new causal profile is required.

### Files Changed

- artifacts/manifests/iteration297_step10a_prefetch_live_prereg.json
- append-only research_log.md

### Failure or Limitation

DeepSpeed may not overlap prefetch with `overlap_comm=false`; the run is explicitly falsifiable.

### Gate Status

GO.

### Next Highest-Value Question

Does a block-sized window change measured Step10-A rather than merely its configured values?

### Next Command

Implement fail-closed knobs and tests, run the full CPU test suite, then launch one Step10-only candidate.

## Iteration 297 — result: one-block prefetch / two-block live Step10-A

### Milestone / Gate

Strict-v4 Step10-A ZeRO-3 materialization scheduling.

### Question

Does a block-sized synchronous prefetch/live window reduce Step10-A by >=10%?

### Hypothesis

A one-block prefetch bucket and two-block live cap reduce exposed frozen-parameter AllGather.

### Importance

The old prefetch/live caps were smaller than one 351.4M-element frozen block.

### Evidence Needed

One real-14B low-overhead Step10-only run with exact boundary fingerprints and memory admission.

### Refutation Condition

OOM/fingerprint failure or Step10-A >=110.804 s.

### Allowed Changes

2M persistence retained; prefetch/live element caps changed exactly as preregistered. All other config and semantics frozen.

### Commands

Implemented fail-closed knobs, passed 40 focused and 230 full tests (1 skipped), then ran `rb16_step10a_iter297_persist2m_prefetch384m_live768m_timing_20260810_001` at `295d13c`.

### Measured Results

Generator median 115.489513 s (-6.19% from baseline, -1.15% from Iter296); world max 120.909677 s (-6.27%). Wave medians 19.275301/36.082621/21.213110/37.354568 s. Node0 GPU peak 31,762 MiB and cgroup peak 359,993,237,504 bytes; OOM delta 0.

### Correctness

Exact trace/x0/gradient/loss identities for 8 ranks x 2 waves; G2->G3 once. Full-state export Not Run in timing mode.

### Performance

Below 10% gate and above 105 s; bounded negative.

### Interpretation

Source audit shows `overlap_comm=false` puts prefetch all-gathers on the default stream. The larger synchronous window cannot hide communication. Stop synchronous-window tuning.

### Files Changed

- runtime/deepspeed_backend.py and Step10 launcher/executor knobs
- tests
- docs/experiments/rb16_step10a_prefetch_live_result_20260810.md
- artifacts/manifests/iteration297_step10a_prefetch_live_result.json
- append-only research_log.md

### Failure or Limitation

Single preliminary timing run; no full-state export or formal repeats.

### Gate Status

NO-GO for further synchronous window tuning. GO for one one-variable async DeepSpeed communication-stream candidate.

### Next Highest-Value Question

Does `overlap_comm=true` hide the now block-sized prefetch without OOM or identity drift?

### Next Command

Preregister overlap-only candidate, add fail-closed launcher plumbing/tests, then run one Step10-only trial.

## Iteration 298 — preregister Step10-A dedicated communication-stream prefetch

### Milestone / Gate

Strict-v4 Step10-A asynchronous ZeRO-3 parameter prefetch.

### Question

Can `overlap_comm=true` overlap the admitted block-sized prefetch with model compute?

### Hypothesis

Iteration 297's window is large enough but used the default stream. A dedicated all-gather stream hides part of the 49.2 s AllGather-dominant NCCL union.

### Importance

This tests actual overlap rather than another bucket-size guess.

### Evidence Needed

Fail-closed resolved config, unit tests, real allocator admission, exact boundary identities, per-wave timing and GPU peak.

### Refutation Condition

OOM/identity failure or Step10-A >=110.804 s.

### Allowed Changes

Expose one Step10 boolean and set `overlap_comm=true`; every other Iter297 policy and all workload semantics remain frozen.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run.

### Performance

Not Run.

### Interpretation

GO for one bounded run. No further repair escalation if it reproduces a known failure.

### Files Changed

- artifacts/manifests/iteration298_step10a_overlap_comm_prereg.json
- append-only research_log.md

### Failure or Limitation

Async ordering can change floating reduction order; exact boundary checks remain fail-closed.

### Gate Status

GO.

### Next Highest-Value Question

Can a dedicated communication stream turn configured lookahead into end-to-end overlap?

### Next Command

Implement the boolean with tests, run the full suite, and launch one Step10-only trial.

## Iteration 298 — result: Step10-A dedicated communication-stream prefetch

### Milestone / Gate

Strict-v4 Step10-A async ZeRO-3 materialization.

### Question

Does `overlap_comm=true` convert the block-sized lookahead into useful compute/communication overlap?

### Hypothesis

A dedicated all-gather stream hides a material part of the AllGather-dominant interval.

### Importance

It is the final direct test after synchronous persistence/window candidates.

### Evidence Needed

One admitted real-14B run with exact boundary identity, phase timing and GPU peak.

### Refutation Condition

OOM/identity failure or Step10-A >=110.804 s.

### Allowed Changes

Only `overlap_comm: false -> true`; all Iter297 and workload fields frozen.

### Commands

Added fail-closed plumbing, passed 40 focused and 230 full tests (1 skipped), then ran `rb16_step10a_iter298_persist2m_prefetch384m_live768m_overlap_timing_20260810_001` at `87a3528`.

### Measured Results

Generator median 120.465448 s; world max 124.963757 s. This is only 2.15% faster than baseline and 4.31% slower than Iter297. Wave medians 20.041639/37.008138/20.544714/41.349006 s. Node0 GPU peaks 33,314--33,514 MiB and cgroup peak 358,040,174,592 bytes; no OOM.

### Correctness

Exact trace/x0/gradient/loss identities for 8 ranks x 2 waves; G2->G3 once.

### Performance

Bounded negative; outside the 85--105 s target and below 10% gate.

### Interpretation

Extra in-flight buffers and synchronization/contended streams outweigh any hidden AllGather, especially in backward wave1. Stop Step10 persistence/window/overlap knob tuning.

### Files Changed

- Step10 overlap launcher/executor/tests
- docs/experiments/rb16_step10a_overlap_comm_result_20260810.md
- artifacts/manifests/iteration298_step10a_overlap_comm_result.json
- append-only research_log.md

### Failure or Limitation

Single preliminary run; no full-state export/formal repeats.

### Gate Status

Bounded NO-GO for this Step10 optimization family.

### Next Highest-Value Question

Can persistent pinned target backing remove the transition's pageable staging critical path?

### Next Command

Preregister persistent-pinned transition ownership using existing unbound Iter151 evidence; do not reintroduce process-wide NUMA binding.

## Iteration 299 — preregister persistent-pinned switch-target backing

### Milestone / Gate

Strict-v4 live representation transition host ownership.

### Question

Can persistent pinned CPU backing for the ten preconstructed switch targets reduce transition H2D into the 10--25 s target?

### Hypothesis

Cold-prepared pinned registered storage removes the repeated pageable->small-arena staging that remained critical under eight concurrent node0 loads.

### Importance

Iterations 294--295 proved bounded staging exact but sub-threshold; prior Iter151 proved unbound persistent pinning is a real 14B performance primitive.

### Evidence Needed

Pinned-byte inventory, host/cgroup admission, exact real-14B Full5 correctness, direct H2D CUDA interval, transition and final release.

### Refutation Condition

Pinned allocation/OOM, exact failure/leak, or transition >=44.855321 s.

### Allowed Changes

Add selectable persistent-pinned target host ownership and direct nonblocking H2D. Pinning occurs only during excluded cold prep. Baseline Step10, legacy G3, communicators and all semantics stay frozen.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run.

### Performance

Not Run. Target <=25 s; partial if it crosses 10% but remains above 25 s.

### Interpretation

GO. The previously failed whole-process NUMA binding is explicitly forbidden and not retested.

### Files Changed

- artifacts/manifests/iteration299_transition_persistent_pinned_prereg.json
- append-only research_log.md

### Failure or Limitation

Estimated node0 pinned ownership is about 249.49 GB; actual OOM is the admission authority per user contract.

### Gate Status

GO.

### Next Highest-Value Question

Can direct DMA from persistent pinned target state remove enough of the multi-rank transition critical path?

### Next Command

Implement narrow pin/direct-transfer primitives with CPU/CUDA-mini tests, run full suite, then launch one exact Full5 trial.

## Iteration 299 — result: persistent-pinned switch-target backing

### Milestone / Gate

Strict-v4 live representation transition host ownership.

### Question

Can persistent pinned CPU backing for the ten preconstructed targets reduce transition H2D into the 10--25 s target?

### Hypothesis

Cold-prepared pinned state removes pageable staging and makes direct DMA the transfer path.

### Importance

Iterations 294--295 proved bounded staging exact but sub-threshold; target H2D remained the transition bottleneck.

### Evidence Needed

Pinned inventory, real cgroup admission, exact 16-rank Full5 correctness, direct H2D intervals, release accounting and transition time.

### Refutation Condition

OOM/exact failure/leak or transition >=44.855321 s; positive target <=25 s.

### Allowed Changes

Persistent private pinned backing during excluded cold prep and direct H2D only. Step10-A, G3, graph, trace, backend, placement, optimizer and EMA frozen.

### Commands

Added the persistent-pinned primitive, passed 34 focused and 232 full tests (1 skipped), committed `9885508`, ran `rb16_f5_iter299_persistent_pinned_transition_correctness_20260810_001`, and summarized all 16 rank records.

### Measured Results

Transition 30.887268 s versus 49.839245 s (-38.03%); Step10-A 132.899935 s; Critic 160.708690 s; Full5 328.480532 s. Direct H2D CUDA max 2.769706 s, but synchronous allocator trim max 11.277463 s. Cold pin prep max 42.414934 s was excluded. Pinned ownership was 264,383,106,048 bytes on node0 and 62,372,217,600 on node1. Cgroup peaks 739,236,196,352 / 178,212,835,328 bytes; OOM delta 0.

### Correctness

PASS 16/16: exact frozen identities, G3/C15/EMA15, 60/60 payloads and flushed queues. Summary SHA `9940fb0a0610391306ea9486ab528c96a3b25b433ccfce7ce762372439ecec01`.

### Performance

Partial positive. It crosses the 10% gate but misses the <=25 s target. Formal repeats Not Run.

### Interpretation

Persistent pinning solved pageable staging, but synchronous release of 31--34 GB of cached pinned storage per switching process is now exposed. Cleanup has no target-readiness dependency and can legally overlap the much longer Critic phase if it is joined before Full5 completion.

### Files Changed

- runtime persistent-pinned transfer primitive and tests
- docs/experiments/rb16_transition_persistent_pinned_result_20260810.md
- artifacts/manifests/iteration299_transition_persistent_pinned_result.json
- append-only research_log.md

### Failure or Limitation

Node0 host peak reached 688.47 GiB. Step10-A regressed in this single correctness run; attribution and formal repeat remain open.

### Gate Status

PARTIAL; GO for one bounded deferred-cleanup scheduling trial.

### Next Highest-Value Question

Can pinned allocator cleanup be hidden under the Critic phase while preserving exact state and bounded host memory?

### Next Command

Preregister deferred cleanup, implement a bounded cleanup handle joined before final Full5 completion, run tests, then one exact Full5 trial.

## Iteration 300 — preregister deferred pinned-cache cleanup

### Milestone / Gate

Strict-v4 transition safe-release scheduling.

### Question

Can the 6.13--11.28 s pinned allocator cleanup run concurrently with Critic work rather than block target readiness?

### Hypothesis

After direct H2D completion, old pinned backing has no semantic consumer. A bounded cleanup task can start then, overlap the 160.71 s Critic phase, and be joined before Full5 success.

### Importance

Iteration 299 measured a causal transition upper bound of about 19.61 s after removing synchronous trim from the readiness path, inside the 10--25 s target.

### Evidence Needed

Unit tests for cleanup lifecycle, exact 16-rank Full5 correctness, transition/cleanup timing, cgroup/OOM accounting and no outstanding cleanup at completion or exception.

### Refutation Condition

Any semantic/OOM/leak failure, transition >=27.798541 s, or cleanup exposed at Full5 completion.

### Allowed Changes

Only cleanup scheduling for the already-validated persistent-pinned primitive. Host ownership, transfer, Step10-A, G3 and all workload semantics frozen.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run.

### Performance

Not Run. Target transition <=25 s.

### Interpretation

GO for one bounded implementation and exact Full5 trial.

### Files Changed

- artifacts/manifests/iteration300_transition_deferred_cleanup_prereg.json
- append-only research_log.md

### Failure or Limitation

The single Iter299 Step10-A regression remains un-attributed and cannot be hidden by transition cleanup.

### Gate Status

GO.

### Next Highest-Value Question

Does safe cleanup overlap turn the direct-DMA partial result into a <=25 s transition without increasing actual OOM risk?

### Next Command

Implement cleanup handle/start/drain, test normal and exception paths, commit, and launch one exact Full5 trial.

## Iteration 300 — result: immediate background pinned-cache cleanup

### Milestone / Gate

Strict-v4 transition safe-release scheduling.

### Question

Can cleanup launched immediately after H2D hide under remaining transition and Critic work?

### Hypothesis

Pinned cleanup has no data dependency after H2D and will not delay target readiness.

### Importance

Iteration 299 exposed up to 11.28 s of synchronous cleanup.

### Evidence Needed

Exact Full5, cleanup lifecycle, subphase timing, host peak and OOM evidence.

### Refutation Condition

Any exact/OOM/leak failure, transition >=27.798541 s, or exposed cleanup at completion.

### Allowed Changes

Cleanup scheduling only; persistent backing, transfer and all strict-v4 semantics frozen.

### Commands

Added deferred cleanup handles, passed 35 focused and 233 full tests (1 skipped), committed `162eef5`, ran `rb16_f5_iter300_persistent_pinned_deferred_cleanup_correctness_20260810_001`, and summarized all ranks.

### Measured Results

Transition 40.980166 s, a 32.68% regression from Iter299. Direct H2D stayed 2.946535 s max, but G3 broadcast rose to 19.244619 s, Student initialization to 27.207748 s and target-ready fence to 11.048340 s. Cleanup took 6.47--26.18 s; final join wait was negligible. Step10-A 124.916147 s, Critic 163.099019 s, Full5 333.102958 s. Cgroup peaks 716,753,629,184 / 171,579,322,368 bytes; no OOM.

### Correctness

PASS 16/16 with exact identities, G3/C15/EMA15 and flushed queues. Summary SHA `76b7bf788ed88d703a3fbe3ad7dc89ffcef8c48830dab0695800e7b6c886c2ef`.

### Performance

Bounded negative. Formal repeats Not Run.

### Interpretation

The cleanup dependency was legal but scheduled too early: host deregistration contended with G3 and DP12 setup. Stop immediate-background cleanup. New evidence supports one late-start trial after global target readiness.

### Files Changed

- deferred cleanup runtime and tests
- docs/experiments/rb16_transition_immediate_background_cleanup_result_20260810.md
- artifacts/manifests/iteration300_transition_immediate_background_cleanup_result.json
- append-only research_log.md

### Failure or Limitation

One diagnostic run; it does not isolate how much cleanup would affect Critic CPU optimizer when started later.

### Gate Status

NO-GO for immediate background cleanup; GO for one materially different late-start schedule.

### Next Highest-Value Question

Can cleanup start after target setup and hide under Critic without delaying Full5 completion?

### Next Command

Preregister late-start cleanup, add a fail-closed start point at Critic-phase entry, test, then run one exact Full5 trial.

## Iteration 301 — preregister late-start pinned cleanup

### Milestone / Gate

Strict-v4 late safe-release scheduling.

### Question

Does starting cleanup only at Critic-phase entry avoid setup contention and achieve transition <=25 s?

### Hypothesis

The direct target transfer is already 2.95 s or less. Delaying host deregistration until global target readiness removes it from G3/DP12 setup, while the 160+ s Critic phase provides overlap.

### Importance

This is the final materially different cleanup schedule justified by Iter300 causal evidence; it avoids blind thread tuning.

### Evidence Needed

Exact Full5, start-point proof, cleanup completion/join, transition, Critic/Full5, host peak and OOM.

### Refutation Condition

Any correctness/OOM/leak failure, transition >=27.798541 s, or Full5 >344.904558 s.

### Allowed Changes

Only defer cleanup launch from per-rank H2D completion to Critic-phase entry. Everything else frozen.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run.

### Performance

Not Run. Target transition <=25 s.

### Interpretation

GO for one bounded implementation and run; no further cleanup-thread tuning after it.

### Files Changed

- artifacts/manifests/iteration301_transition_late_cleanup_prereg.json
- append-only research_log.md

### Failure or Limitation

Cleanup may contend with Critic CPU optimizer; that cost must remain in Critic/Full5 timing.

### Gate Status

GO.

### Next Highest-Value Question

Can release be hidden in a phase where it has no setup dependency and sufficient compute lead?

### Next Command

Implement late-deferred start at Critic timing entry, test lifecycle, then one exact Full5 trial.

## Iteration 301 — result: late-start pinned cleanup

### Milestone / Gate

Strict-v4 late safe-release scheduling.

### Question

Can cleanup start after global target readiness and achieve transition <=25 s without end-to-end regression?

### Hypothesis

Late cleanup avoids G3/DP12 setup contention and hides under Critic GPU work.

### Importance

It is the final materially different safe-release schedule justified by Iter300.

### Evidence Needed

Exact Full5, start point, cleanup join, transition, Critic/Full5 and host/OOM data.

### Refutation Condition

Any exact/OOM/leak failure, transition >=27.798541 s, or Full5 >344.904558 s.

### Allowed Changes

Cleanup launch point only; all strict-v4 semantics and physical contract frozen.

### Commands

Added late-deferred start at Critic entry, passed 36 focused and 234 full tests (1 skipped), committed `2375d10`, ran `rb16_f5_iter301_persistent_pinned_late_cleanup_correctness_20260810_001`, and summarized all ranks.

### Measured Results

Transition 23.837731 s, meeting the target and improving 52.17% over pageable. G3 max 6.578637 s and Student init max 9.787566 s. Cleanup took up to 20.585752 s with negligible join wait. Critic rose to 184.964888 s and Full5 to 350.064454 s, 6.57% slower than Iter299. Step10-A was 141.102216 s. Cgroup peaks 714,701,598,720 / 173,750,935,552 bytes; no OOM.

### Correctness

PASS 16/16, exact identities, G3/C15/EMA15, 60/60 payloads and flushed queues. Summary SHA `74199aec519b2bde82e77d7d5b300dd11af571515c28e2875c3ef76f882cee6c`.

### Performance

Transition mechanism positive; end-to-end bounded negative due Full5 guardrail failure. Formal repeats Not Run.

### Interpretation

The 10--25 s transition is achievable without semantic change, but not free: host deregistration moved into Critic and slowed it. Synchronous Iter299 remains the best measured one-cycle Full5 point. Close cleanup timing variants.

### Files Changed

- late cleanup runtime and tests
- docs/experiments/rb16_transition_late_cleanup_result_20260810.md
- artifacts/manifests/iteration301_transition_late_cleanup_result.json
- append-only research_log.md

### Failure or Limitation

Single diagnostic run and large Step10 variance under pinned host pressure; no formal repeat.

### Gate Status

Transition target PASS; candidate overall NO-GO.

### Next Highest-Value Question

What owns the large Step10-A D2H tensors whose sizes/counts match checkpoint boundaries?

### Next Command

Preregister and run a no-policy-change saved-tensor/checkpoint-boundary census before any activation pinning.

## Iteration 302 — preregister Step10-A checkpoint-boundary census

### Milestone / Gate

Strict-v4 Step10-A copy ownership.

### Question

Do the three dominant Nsight D2H size/count signatures correspond one-to-one to checkpoint-boundary saved tensors?

### Hypothesis

Two GAS waves times forty blocks produce 80 saves for each selected stage shape: 113,049,600, 46,694,400 and 30,105,600 BF16 bytes.

### Importance

The ZeRO-3 knob family is closed below target. Pinning or changing activation lifetime before proving ownership would be blind optimization.

### Evidence Needed

Pass-through pack/unpack metadata, exact shape/count summary, closed hooks, unchanged x0/gradient/loss/version and no OOM.

### Refutation Condition

Any semantic/lifetime failure or mismatch between saved-tensor and Nsight signatures.

### Allowed Changes

Instrumentation and summarizer only; pack/unpack returns original tensor and performs no copy.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run.

### Performance

Not applicable; diagnostic run only.

### Interpretation

GO. No activation policy is authorized until the census closes.

### Files Changed

- artifacts/manifests/iteration302_step10a_saved_tensor_census_prereg.json
- append-only research_log.md

### Failure or Limitation

Size/count correlation alone may still require a policy-isolation run to prove the CUDA copy owner.

### Gate Status

GO.

### Next Highest-Value Question

Can the checkpoint boundary be pinned or recomputed differently after ownership is proven, without changing graph semantics?

### Next Command

Implement metadata-only saved-tensor census, tests, run full suite, then one bounded Step10-only diagnostic.

## Iteration 302 — Step10-A checkpoint-boundary census result

### Milestone / Gate

Strict-v4 Step10-A copy ownership.

### Question

Do the three dominant Nsight D2H size/count signatures correspond one-to-one to checkpoint-boundary saved tensors?

### Hypothesis

Each rank will save each selected-stage boundary 80 times: two GAS waves times forty blocks.

### Importance

A pinned or async policy is only defensible after proving the owner and exact lifetime group.

### Evidence Needed

Eight-rank metadata census, closed hooks, frozen trace gate, no OOM and unchanged pass-through semantics.

### Refutation Condition

Any signature count mismatch, tensor-reference leak, failed rank or OOM.

### Allowed Changes

Metadata-only profiler and summarizer; no model or offload policy.

### Commands

Ran focused tests, full suite, then `rb16_step10a_iter302_saved_tensor_census_20260810_001` on two nodes and summarized all 16 rank records.

### Measured Results

PASS. Every Generator rank observed 80 saves for BF16 `[1,11040,5120]` (113,049,600 bytes), `[1,4560,5120]` (46,694,400 bytes), and `[1,2940,5120]` (30,105,600 bytes). Per rank: six scopes, 1,302 packs, 1,298 unpacks, 40 signatures and 22.891823 GiB aggregate logical saved storage. Node cgroup peaks were 379,505,057,792 / 79,576,207,360 bytes; OOM delta zero.

### Correctness

16/16 rank status PASS; pack returned the original Tensor, unpack returned the same Tensor, census retained zero Tensor references. The timing-mode launch was fail-closed against the prior exact Iter295 correctness summary and frozen trace hash.

### Performance

Not applicable. Detailed profiler events were enabled; no timing claim.

### Interpretation

The checkpoint-boundary ownership hypothesis is verified. A selective synchronous pinned policy for only these three signatures is authorized; generic full activation offload and async prefetch remain unauthorized.

### Files Changed

- `runtime/saved_tensor_census.py`
- census tests, launcher integration and summarizer
- `docs/experiments/rb16_step10a_saved_tensor_census_result_20260810.md`
- Iteration 302 result and Iteration 303 preregistration manifests

### Failure or Limitation

Shape/count equality is strong ownership evidence but does not yet measure how much of Nsight pageable traffic the selective policy replaces. Four saved tensors were not unpacked per rank; raw evidence is retained.

### Gate Status

PASS; GO to one bounded synchronous pinned policy-isolation candidate.

### Next Highest-Value Question

Can synchronous pinned checkpoint-boundary pack/unpack push the strongest legal Step10-A window below 110.804 seconds without affecting Full5 correctness?

### Next Command

Implement exact-signature allowlisting, test CPU/CUDA parity, then run strict-v4 Full5 correctness before any timing.

## Iteration 303 — preregister selective synchronous pinned Step10-A boundary policy

### Milestone / Gate

Strict-v4 Step10-A copy policy.

### Question

Does pinning only the three proven boundary classes reduce exposed pageable copy overhead?

### Hypothesis

The selective policy preserves exact recomputation inputs while replacing pageable checkpoint-boundary movement with synchronous pinned copies.

### Importance

The strongest legal Step10-A candidate remains 115.489513 seconds, above the 85--105 second target.

### Evidence Needed

Unit/CUDA parity, full strict-v4 state/identity correctness, pinned/alias/live-byte audit, admission, and only then preliminary timing.

### Refutation Condition

Any exact correctness failure, OOM/leak, or preliminary median at least 110.804 seconds.

### Allowed Changes

Exact saved-tensor signature filter and Step10-only pinned synchronous pack/unpack. Trace, graph, loss, precision, optimizer, placement and ZeRO backend are frozen.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run.

### Performance

Not Run.

### Interpretation

GO. Use the Iter297 persistence/prefetch/live window so the new primitive is added to the strongest measured static Step10 substrate.

### Files Changed

- `artifacts/manifests/iteration303_step10a_boundary_pinned_sync_prereg.json`
- append-only `research_log.md`

### Failure or Limitation

This is synchronous policy isolation; async prefetch and formal repeats are explicitly Not Run.

### Gate Status

GO.

### Next Highest-Value Question

Whether the pinned boundary policy is correct and admitted on the real 14B Full5 path.

### Next Command

Implement allowlist and tests, run full suite, then launch exact Full5 correctness.

### Iteration 303 preregistration amendment 01 — fail-closed host admission

Before launch, static admission rejected concurrent Iter299 target persistence plus the new
boundary-pinned handles. Node0 `memory.max` is 816,043,786,240 bytes; Iter299 peaked at
739,236,196,352 bytes. The three allowlisted boundary classes require 15,187,968,000 bytes
per Generator rank, or 121,503,744,000 bytes for eight ranks. A conservative combined peak
is 860,739,940,352 bytes, exceeding the cgroup by 44,696,154,112 bytes. To avoid a known
OOM, the exact Full5 correctness run uses the existing `module_to` transition solely to
isolate Step10 policy correctness. Workload semantics and Step10 thresholds are unchanged;
the joint pinned-transition candidate is Not Run until pinned capacity can be reused after
boundary release.

## Iteration 303 — selective synchronous pinned boundary result

### Milestone / Gate

Strict-v4 Step10-A copy policy.

### Question

Can pinning only the three proven checkpoint boundaries improve the strongest legal Step10-A configuration?

### Hypothesis

Selective pinned pack/unpack will replace exposed pageable movement without changing graph semantics.

### Importance

Step10-A remains above the 85--105 second engineering target.

### Evidence Needed

Exact continuous Full5 correctness, bounded pinned state, no OOM/leak, then one low-overhead Step10 diagnostic.

### Refutation Condition

Any exact failure or Step10 median at least 110.804 seconds.

### Allowed Changes

Exact-signature synchronous pinned saved-tensor policy only, on the Iter297 ZeRO-3 window.

### Commands

Ran CPU and CUDA parity tests, full suite, exact Full5 correctness, summarizer, then one correctness-bound Step10-only timing run.

### Measured Results

Correctness run PASS 16/16, G3/C15/EMA15, 60/60 payloads, no OOM. Per rank 235--240 CPU copies, 13.891--14.145 GiB peak pinned boundary bytes, final live bytes zero. Preliminary Step10 Generator phase-sum median was 133.831677 s (range 133.637197--133.904982); world max 141.008760 s. Node0 peak 510,986,096,640 bytes; GPU peak 31,762 MiB.

### Correctness

PASS; summary SHA `1f636d7e23d1a439ebb828a1fb5bd61baca8a09b177cfdb17790db3b3c2dddcb`.

### Performance

Bounded negative: 15.88% slower than Iter297 115.489513 s and 8.70% slower than strict-v4 123.115506 s.

### Interpretation

The selective hook adds synchronous pinned allocation/copy rather than replacing the existing dominant copy path. Close this family; do not proceed to async on a regressing synchronous substrate.

### Files Changed

- exact-signature filter in `runtime/activation_offload.py`
- launcher/executor policy binding and tests
- Iteration 303 result report/manifest
- Iteration 304 preregistration

### Failure or Limitation

One preliminary timing run; formal repeats Not Run. A new Nsight capture could further attribute the duplicate path but is not needed to reject the candidate.

### Gate Status

NO-GO for Step10 boundary pinning; return to strongest Iter297 Step10 substrate.

### Next Highest-Value Question

Can the transition pinned target cache remain alive until after the five-step boundary so cleanup no longer slows Critic?

### Next Command

Implement a fail-closed post-Full5 cleanup policy and run one exact strict-v4 Full5 experiment.

## Iteration 304 — preregister persistent target cache through Full5

### Milestone / Gate

Strict-v4 representation transition.

### Question

Can delayed final cleanup preserve the 23.84-second transition while recovering Critic time?

### Hypothesis

Cleanup contention, not pinned residency itself, caused Iter301's Critic regression.

### Importance

Iteration301 met the transition target but violated the Full5 performance guardrail.

### Evidence Needed

Exact Full5 correctness, no OOM, host peak, final cleanup, transition/Critic/Full5 timings.

### Refutation Condition

OOM/leak, any exact mismatch, transition above 25 s, Critic above 168.75 s, or Full5 above 328.480532 s.

### Allowed Changes

Only the cleanup boundary of already validated persistent pinned target backing. Step10 uses Iter297 and no boundary activation pinning.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run.

### Performance

Not Run.

### Interpretation

GO. Static headroom is approximately 101 GB using Iter301 peak versus live cgroup maximum, but actual admission remains required.

### Files Changed

- Iteration 304 preregistration manifest
- append-only research log

### Failure or Limitation

One-cycle boundary experiment; repeated-cycle cache reuse remains Not Run.

### Gate Status

GO.

### Next Highest-Value Question

Whether keeping the cache avoids both Critic contention and OOM.

### Next Command

Add `post_full5` cleanup policy, tests, full suite, then one exact Full5 run.

## Iteration 304 attempt 01 — post-Full5 cleanup-thread failure

### Milestone / Gate

Strict-v4 representation transition cache lifetime.

### Question

Can the persistent pinned target cache survive through Full5 and be released safely after the Step15-A boundary?

### Hypothesis

Moving cache cleanup after Full5 will preserve the fast transition without Critic contention.

### Importance

The prior late-deferred cleanup met the transition target but slowed Critic.

### Evidence Needed

Exact Full5 completion, final cleanup, no OOM/leak, and transition/Critic/Full5 timings.

### Refutation Condition

Any OOM, cleanup failure, exact mismatch, or missed performance guardrail.

### Allowed Changes

Only the target pinned-cache cleanup boundary; workload and model policies are frozen.

### Commands

Launched exact two-node world16 run `rb16_f5_iter304_persistent_cache_post_full5_correctness_20260810_001`.

### Measured Results

Before cleanup, rank 4 observed Step10-A 118.332 s, transition 23.714 s, Critic 164.520 s, and Full5 310.431 s; the world-max Step10-A was 122.904 s and transition 23.716 s. Node cgroup peaks were 713,661,657,088 and 257,071,157,248 bytes. The cgroup OOM counters did not increment.

### Correctness

INVALID. G3/C15/EMA15 work completed on the ranks that reached cleanup, but ranks 4 and 7 failed during mandatory cleanup and the remaining ranks blocked at the final barrier.

### Performance

Diagnostic only. The pre-cleanup numbers cannot be promoted because the run did not return PASS.

### Interpretation

The failure is not a Step10 rollout OOM. The post-Full5 branch started a host allocator cleanup thread and immediately joined it; ranks 4 and 7 reported CUDA OOM from that cleanup path after the frozen Full5 timer.

### Files Changed

- preserved run root
- `artifacts/manifests/iteration304_transition_post_full5_attempt001_cleanup_thread_oom.json`
- append-only log

### Failure or Limitation

The exact source of the PyTorch pinned-cache CUDA error is not proven. Per the repeated-failure rule, only one non-invasive main-thread verification is allowed.

### Gate Status

NO-GO for attempt 01; no performance claim.

### Next Highest-Value Question

Does identical post-Full5 trimming pass when executed synchronously on the CUDA-owning main thread?

### Next Command

Replace the pointless start-and-immediate-join thread at the post-Full5 boundary with a synchronous trim, run tests, and perform one exact rerun.

## Iteration 304 amendment 01 — synchronous post-Full5 trim preflight

### Milestone / Gate

Strict-v4 representation transition cache lifetime.

### Question

Can the same cleanup operation run safely on the main thread after Full5?

### Hypothesis

The failure is caused by the cleanup-thread execution context, not by target-cache residency or Full5 semantics.

### Importance

The candidate already met the transition target before an invalid teardown.

### Evidence Needed

All-rank exact PASS, cleanup result and wall time, no OOM, and no remaining GPU process.

### Refutation Condition

Any repeat cleanup failure, exact mismatch, OOM, or transition above 25 seconds.

### Allowed Changes

Only execution context of the post-Full5 host allocator trim. The cleanup boundary, workload, Step10 window and transition policy remain frozen.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run.

### Performance

Not Run.

### Interpretation

GO for one bounded non-invasive verification. Cleanup wall time will remain separately visible.

### Files Changed

- `artifacts/manifests/iteration304_transition_post_full5_cleanup_amendment01.json`
- append-only log

### Failure or Limitation

Formal repeats remain Not Run. A second cleanup failure closes this candidate.

### Gate Status

GO.

### Next Highest-Value Question

Whether main-thread cleanup makes the otherwise completed candidate valid.

### Next Command

Implement and test synchronous post-Full5 cleanup, then launch attempt 02.

## Iteration 304 attempt 02 — synchronous post-Full5 trim result

### Milestone / Gate

Strict-v4 representation transition cache lifetime.

### Question

Can main-thread post-Full5 cleanup validate the fast persistent-pinned transition?

### Hypothesis

The attempt-01 failure came from the pointless cleanup-thread context, not the cache-residency plan.

### Importance

Attempt 01 met the transition target before invalid teardown.

### Evidence Needed

Exact all-rank PASS, G3/C15/EMA15, 60/60 payloads, no OOM, successful cleanup, and frozen boundary timings.

### Refutation Condition

Any exact failure, OOM, transition above 25 s, Critic above 168.75 s, or Full5 above 328.481 s.

### Allowed Changes

Only execute the same post-Full5 allocator trim on the main thread. Workload, model policy, Step10 window and transition policy are frozen.

### Commands

Ran the full 239-test suite, then launched `rb16_f5_iter304_persistent_cache_post_full5_correctness_20260810_002` on both nodes and ran the exact summarizer.

### Measured Results

PASS 16/16. Step10-A 124.811303 s, transition 22.142147 s, Critic 156.062761 s, and Full5 306.015029 s. Post-boundary main-thread cleanup was at most 12.605171 s and passed. Node cgroup peaks were 716,569,948,160 and 257,567,371,264 bytes; GPU peaks were 39,250 and 37,002 MiB. OOM deltas were zero.

### Correctness

PASS: G3/C15/EMA15, 16 Step10-A samples, five Critic/EMA updates, 60/60 payloads, exact frozen identities. Summary SHA `9c79353fec00519045c7fc70e4ea35b0b0233ba5896209adaeee3a667a0da366`.

### Performance

Preliminary positive. Transition fell from the Iter293 49.839245 s anchor to 22.142147 s (55.57% less time). Full5 to the Step15-A boundary fell from 341.819356 s to 306.015029 s (10.47% less time). Including the separately measured serial cleanup upper bound gives 318.620200 s (6.79% less time). Formal repeats are Not Run.

### Interpretation

The transition engineering target is achieved without semantic change. The cleanup thread, not persistent target residency, caused attempt 01's teardown failure. Step10-A remains outside its 85--105 s target.

### Files Changed

- main-thread post-Full5 cleanup implementation and tests
- exact run and summary
- Iteration 304 result manifest/report
- append-only log

### Failure or Limitation

One exact correctness/profile run only. Cleanup is outside the Step15-A semantic timer and is reported separately; it must not be silently omitted from repeated-cycle lifecycle accounting.

### Gate Status

Positive for the transition mechanism; formal repeated timing remains Not Run.

### Next Highest-Value Question

Can ZeRO-3 fine-grained module fetch overhead be reduced without increasing live parameters enough to OOM?

### Next Command

Profile the real Generator module-granularity distribution, preregister one bounded threshold candidate, and test Step10-A only before any Full5 promotion.

## Iteration 305 — preregister ZeRO-3 module-granularity census

### Milestone / Gate

Strict-v4 Step10-A ZeRO-3 fetch overhead.

### Question

Does the real Generator contain fine-grained module groups that can reduce separate parameter all-gathers?

### Hypothesis

DeepSpeed's module-granularity leaf mechanism can coarsen fetch hooks without changing the model graph.

### Importance

Nsight observed 35,187 NCCL kernels and 49.235 s of NCCL interval union; overlap_comm and pinned boundary policies are closed negatives.

### Evidence Needed

Exact CPU module inventory, minimum effective threshold, selected grouped modules and static live-parameter admission.

### Refutation Condition

No effective candidate or static memory rejection.

### Allowed Changes

Read-only CPU census code and tests only. No training policy, trace, graph, or GPU timing change.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run.

### Performance

Not Run.

### Interpretation

GO. This iteration only selects a bounded candidate; it cannot produce a speedup claim.

### Files Changed

- preregistration manifest
- append-only log

### Failure or Limitation

DeepSpeed leaf grouping still requires later exact Step10 and Full5 validation.

### Gate Status

GO.

### Next Highest-Value Question

Which one threshold is both effective and statically admitted?

### Next Command

Implement a DeepSpeed-compatible CPU granularity census, test it, and run it on the official Generator.

## Iteration 305 — ZeRO-3 module-granularity census result

### Milestone / Gate

Strict-v4 Step10-A ZeRO-3 fetch overhead.

### Question

Which real Generator modules can be fetched as integral ZeRO-3 leaf units?

### Hypothesis

Transformer blocks are fine-grained under DeepSpeed's measured granularity formula.

### Importance

Coarser integral fetches directly target the 35,187 NCCL-kernel symptom without revisiting closed overlap or activation-pinning paths.

### Evidence Needed

Real model inventory, minimum threshold, leaf parameter size, and static live-window admission.

### Refutation Condition

No block-level threshold that fits the configured prefetch/live caps.

### Allowed Changes

CPU-only read-only census.

### Commands

Constructed the real Helios-Distilled Generator plus LoRA on CPU and mirrored DeepSpeed 0.18.4's module-granularity calculation.

### Measured Results

613 finite records; minimum effective threshold 10,240. Each of the 40 HeliosTransformerBlock modules has granularity 7,641,303 and 382,065,152 logical parameters. Threshold 8,000,000 selects 47 parent-first leaves including all 40 blocks. One block fits the 402,653,184-element prefetch bucket; two blocks total 764,130,304 elements, below the 805,306,368 max-live cap.

### Correctness

Profile script and tests PASS; full suite 241 passed, one skipped. Raw profile SHA `883cfc6bb235ed36f52047907f5a7e40afff0c9ff9aa7bf7102a5569e0331be0`.

### Performance

Not Run.

### Interpretation

Static GO for exactly one 8,000,000 threshold candidate. This does not yet prove actual DeepSpeed classification, admission, numerical parity, or speed.

### Files Changed

- `runtime/zero3_module_granularity.py`
- profile launcher and tests
- raw profile and result manifest
- append-only log

### Failure or Limitation

The selected-parameter sum is not a simultaneous-live estimate; max leaf and DeepSpeed live-window caps provide the relevant conservative admission check.

### Gate Status

GO for one bounded candidate.

### Next Highest-Value Question

Does block-integral fetching reduce real Step10 time while preserving exact Full5 semantics?

### Next Command

Expose threshold 8,000,000 through the fail-closed launcher, test, then run one exact Full5 candidate.

## Iteration 306 — preregister block-integral ZeRO-3 fetch

### Milestone / Gate

Strict-v4 Step10-A ZeRO-3 fetch granularity.

### Question

Can one fetch per Transformer block replace thousands of fine-grained fetch hooks?

### Hypothesis

Threshold 8,000,000 reduces NCCL launch/host overhead and moves Step10-A below the 115.490 s preliminary best.

### Importance

Step10-A remains the only engineering target not met.

### Evidence Needed

Actual leaf inventory, exact G3/C15/EMA15 and payload identities, no OOM, then low-overhead Step10 timing.

### Refutation Condition

Any exact/admission failure or low-overhead Step10 world-max at least 115.490 s.

### Allowed Changes

Only DeepSpeed's built-in stage3_module_granularity_threshold=8,000,000. All other policies and semantics are frozen.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run.

### Performance

Not Run.

### Interpretation

GO for one candidate, not a sweep.

### Files Changed

- Iteration 306 preregistration
- append-only log

### Failure or Limitation

A later Nsight trace is conditional and Not Run.

### Gate Status

GO.

### Next Highest-Value Question

Whether actual DeepSpeed leaf classification and exact Full5 execution pass.

### Next Command

Implement fail-closed threshold plumbing, tests, and exact Full5 run.

## Iteration 306 — scalar module-granularity result

### Milestone / Gate

Strict-v4 Step10-A ZeRO-3 fetch granularity.

### Question

Can threshold 8,000,000 make Transformer blocks integral ZeRO-3 leaves?

### Hypothesis

The built-in threshold reduces fine-grained fetches without changing numerical execution.

### Importance

It directly targets the measured NCCL launch count.

### Evidence Needed

Exact Full5 correctness before timing.

### Refutation Condition

Any exact failure, OOM, or dtype change.

### Allowed Changes

Only `stage3_module_granularity_threshold=8,000,000`.

### Commands

Ran tests, then launched `rb16_f5_iter306_module_leaf8m_correctness_20260810_001`.

### Measured Results

DeepSpeed classified the predicted 47 leaves, including all 40 Transformer blocks. All eight Generator ranks failed at the first wave before any update.

### Correctness

FAIL: `condition_embedder.time_embedder.linear_1` reached a base Linear with Float input and BF16 weight. No performance run was authorized.

### Performance

Not Run.

### Interpretation

The scalar threshold unavoidably selects lower-granularity TimestepEmbedding and PixArt modules before it reaches the 7,641,303-granularity Transformer blocks. This mixed-dtype collateral selection invalidates the threshold mechanism; neighboring-threshold sweeps cannot solve the ordering.

### Files Changed

- fail-closed threshold plumbing and tests
- preserved failure run
- result manifest and append-only log

### Failure or Limitation

The result rejects scalar threshold selection, not block-integral fetching itself.

### Gate Status

NO-GO; scalar module-granularity threshold family closed.

### Next Highest-Value Question

Can DeepSpeed's explicit leaf API mark only Transformer blocks and avoid the proven embedding failure?

### Next Command

Preregister one explicit-block-only candidate; do not alter any dtype or model math.

## Iteration 307 — preregister explicit block-only ZeRO-3 leaves

### Milestone / Gate

Strict-v4 Step10-A block-integral fetch.

### Question

Can explicit block-only leaves reduce fetch overhead without collateral mixed-dtype modules?

### Hypothesis

Only HeliosTransformerBlock should be integral; condition embeddings retain their validated fine-grained hooks and dtype behavior.

### Importance

This isolates the intended mechanism exposed by Iteration306 rather than widening repair.

### Evidence Needed

Exactly 40 actual leaves, exact Full5 PASS, no OOM, then low-overhead timing.

### Refutation Condition

Any other leaf, exact failure, OOM, or no timing improvement over 115.490 s.

### Allowed Changes

Only pre-initialize `set_z3_leaf_modules(model, ["HeliosTransformerBlock"])`; threshold is zero and all other contracts are frozen.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Not Run.

### Performance

Not Run.

### Interpretation

GO for one final block-leaf candidate. No further leaf repair is allowed if it fails.

### Files Changed

- Iteration307 preregistration
- append-only log

### Failure or Limitation

Manual leaf marking remains a physical ZeRO fetch policy and needs exact validation.

### Gate Status

GO.

### Next Highest-Value Question

Whether block-only grouping preserves exact mixed-dtype execution.

### Next Command

Implement fail-closed explicit block leaf selection, tests, and one exact Full5 run.

## Iteration 307 — explicit block-only ZeRO-3 leaf result

### Milestone / Gate

Strict-v4 Step10-A block-integral fetch.

### Question

Can block-only leaf grouping preserve exact execution and reduce Step10 time?

### Hypothesis

Keeping embeddings fine-grained avoids the Iter306 dtype failure while grouping block-internal parameter fetches.

### Importance

This is the first candidate that directly reduces the measured fetch-launch granularity.

### Evidence Needed

Exactly 40 leaves, exact Full5 PASS, no OOM, and a low-overhead Step10 result below 115.490 s.

### Refutation Condition

Any other leaf, exact failure, OOM, or no improvement.

### Allowed Changes

Only explicit HeliosTransformerBlock leaf marking.

### Commands

Ran 244 tests, one exact continuous Full5, the exact summarizer, and one correctness-bound low-overhead Step10 diagnostic.

### Measured Results

Exact Full5 PASS: Step10 Generator max 102.884 s, world max 106.780 s, transition 21.522 s, Critic 168.798 s, Full5 300.395 s. Low-overhead Step10 Generator median was 87.995 s and world max 92.606 s. No OOM.

### Correctness

PASS 16/16, exactly 40 block leaves on every Generator rank, G3/C15/EMA15 and 60/60 payloads. Summary SHA `1ff4b4a11d81053f3149d0a7f24bf9b7d6b51bc0bfc7f18ad8b22ea969f12377`.

### Performance

Preliminary positive. Low-overhead Generator time is 23.81% below Iter297's 115.490 s preliminary best; world max is 28.21% below the 129.004 s strict anchor. The 85--105 s target is met.

### Interpretation

Fetch granularity, not only bucket volume, was a major Step10 bottleneck. Class-specific grouping is necessary because a scalar threshold conflates blocks with mixed-dtype embeddings.

### Files Changed

- explicit block leaf runtime/launcher policy and tests
- exact and timing artifacts
- result report/manifest/table
- append-only log

### Failure or Limitation

One exact run and one timing diagnostic; formal repeat distribution is Not Run.

### Gate Status

Positive; proceed to three frozen formal repeats.

### Next Highest-Value Question

Are both engineering targets stable across independent low-overhead Full5 launches?

### Next Command

Run the preregistered three independent combined-candidate timing launches.

## Iteration 308 — preregister combined formal repeats

### Milestone / Gate

Strict-v4 combined Step10 and transition optimization.

### Question

Are the 92.606 s Step10 boundary and <=25 s transition repeatable?

### Hypothesis

Three independent launches retain both target ranges without correctness-binding drift or OOM.

### Importance

Single-run results cannot support the final performance claim.

### Evidence Needed

Every raw run, median, dispersion, resources and failure retention.

### Refutation Condition

Any wrapper/binding failure, OOM, median Step10 above 105 s, or transition above 25 s.

### Allowed Changes

None. Candidate and correctness binding are frozen.

### Commands

Not Run yet.

### Measured Results

Not Run.

### Correctness

Bound to exact Iter307 summary; timing wrappers still must PASS.

### Performance

Not Run.

### Interpretation

GO for three independent launches.

### Files Changed

- Iteration308 preregistration
- append-only log

### Failure or Limitation

This remains a batch12 Pareto result, not a matched-batch16 Route-A speedup.

### Gate Status

GO.

### Next Highest-Value Question

Formal stability of the combined candidate.

### Next Command

Launch repeats 1--3 serially and preserve all outcomes.

## Iteration 308 — combined candidate formal result

### Milestone / Gate

Strict-v4 combined Step10-A and transition optimization.

### Question

Does the exact-passing frozen candidate keep Step10-A in 85--105 s and transition in 10--25 s across three independent low-overhead Full5 launches?

### Hypothesis

Explicit block-only ZeRO-3 leaves plus the persistent pinned transition cache are stable without semantic drift or OOM.

### Importance

This closes the two user-authorized engineering targets with repeated real-14B evidence rather than a best single run.

### Evidence Needed

All raw rounds, exact binding, target checks, dispersion, resources, and preserved failures.

### Refutation Condition

Any wrapper/binding failure, OOM, median Step10 above 105 s, or median transition above 25 s.

### Allowed Changes

None during timing. The frozen candidate used exactly 40 HeliosTransformerBlock leaves and post-Full5 main-thread cleanup.

### Commands

Launched three independent two-node Full5 timing jobs from the same G2/C10/EMA10 state and summarized every outcome. Final verification used the repository/runtime `PYTHONPATH` and passed 245 tests.

### Measured Results

R1/R2/R3 Step10-A: 104.775/100.689/104.687 s; transition: 21.683/17.776/17.393 s; Critic: 167.179/150.816/158.494 s; Full5: 290.584/265.316/279.934 s. Median Full5 was 279.934 s. Post-boundary cleanup was 13.643/12.034/13.082 s, giving median inclusive time 293.016 s. Reported median throughput was 0.214336 samples/s (0.013396 samples/s/GPU).

### Correctness

PASS. Timing is bound to exact summary `1ff4b4a11d81053f3149d0a7f24bf9b7d6b51bc0bfc7f18ad8b22ea969f12377`: 16/16 ranks, exactly 40 block leaves, G3/C15/EMA15, 60/60 payloads, no queue/trace/version error. All timing wrappers PASS.

### Performance

Both preregistered targets passed in every round. Relative to the matched low-overhead strict-v4 Step10 world-max anchor 129.004 s, formal median Step10 decreased 18.85% (1.232x). The transition median 17.776 s is 2.804x below the Iter293 49.839 s historical profile anchor, but that anchor is not a paired formal low-overhead denominator.

### Interpretation

Step10 was limited materially by fine-grained ZeRO fetch hook/all-gather launch structure; block-integral leaves remove that overhead without changing graph semantics. Transition was limited by pageable target H2D plus allocator cleanup on the ready critical path; cold-prepared pinned direct H2D and post-boundary trim remove that cleanup from the transition timer without deleting it.

### Files Changed

- formal result report
- aggregate JSON/CSV table
- formal result manifest
- append-only research log

### Failure or Limitation

This is a batch12 resource-throughput Pareto, not a matched-batch16 Route-A speedup. Cleanup remains 12--14 s outside the Full5 timer. The rejected scalar 8M threshold failure is preserved. One unqualified final `pytest` invocation omitted the external runtime from `PYTHONPATH` and failed collection with seven import errors; this was an invocation error, not a model/runtime result, and the corrected frozen-environment command passed 245 tests.

### Gate Status

PASS_FORMAL_TARGETS_MET.

### Next Highest-Value Question

A paired same-mode transition baseline is required only if a publication-grade causal transition speedup is needed; otherwise stop tuning this candidate and move to the next preregistered system question.

### Next Command

Freeze the candidate at this commit; do not change semantics or reopen the scalar-threshold family.
