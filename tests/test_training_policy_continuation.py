"""CPU_TEST_ONLY: persistent policy admission and reducer ownership on real Gloo."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.open_workload.torch_backend import Session, candidates, compare, signature
from distillschedule.training.policy import CyclePolicy, cycle_structure
from distillschedule.workloads.program import Call
from test_open_workload_multirole_replica import build_workload, role_a


def _workload(state=None, versions=None, *, rank=0, global_batch=False):
    workload = build_workload(2, 4, 3, rank=rank, global_batch=global_batch)
    # A valid no-grad interface may be declared later without changing the
    # Program or its code. It must still change the bound replica contract.
    workload.program = replace(workload.program, calls=(*workload.program.calls,
        Call("observe", "observe_a", ("input",), ("role_a",))))
    workload.bindings = {**workload.bindings, "observe_a": role_a}
    if state is not None:
        workload.state = state
    if versions is not None:
        workload.program = replace(workload.program, initial_versions=versions)
    return workload


def _change_semantics(workload, field):
    semantics = workload.replica_semantics
    value = {"module_bindings": {**semantics.module_bindings, "observe_a": "role_a"},
             "backward_calls": semantics.backward_calls[:-1],
             "local_batch_size": semantics.local_batch_size + 1,
             "objective": "sum_instead_of_equal_rank_mean"}[field]
    workload.replica_semantics = replace(semantics, **{field: value})


@pytest.mark.parametrize("field", ["module_bindings", "backward_calls", "local_batch_size", "objective"])
def test_later_cycle_binds_every_replica_semantics_field(tmp_path, field):
    policy = CyclePolicy({"mode": "reference"})
    session, _ = policy.run(_workload(), tmp_path)
    workload = _workload(session.state, session.receipt["dispatcher"]["final_versions"])
    _change_semantics(workload, field)
    # The first three variations remain valid standalone Workloads; their
    # declaration must not silently replace the retained reducer's contract.
    if field != "objective":
        workload.validate()
    assert cycle_structure(workload) != policy.bound["structure"], field
    before = signature(session.state)
    with pytest.raises(ValueError, match="structure changed"):
        policy.run(workload, tmp_path)
    assert signature(session.state) == before


def _admission_worker(rank, rendezvous, output, fault, resume):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=10))
    root = Path(output)
    try:
        # This group belongs to the embedding caller, not the training context.
        policy = CyclePolicy({"mode": "reference"})
        session, _ = policy.run(_workload(rank=rank), root)
        workload = _workload(session.state, session.receipt["dispatcher"]["final_versions"], rank=rank)
        if resume:
            resumed = CyclePolicy(policy.execution)
            resumed.restore(policy.bound)
            policy = resumed
        if rank == 1:
            if fault == "shape":
                workload.feeds = {"input": workload.feeds["input"][:, :2]}
                workload.validate()
            elif fault == "ownership":
                workload.state = {**workload.state, "opt_a": torch.optim.Adam(
                    workload.state["role_a"].adapter.parameters(), lr=.001)}
                workload.validate()
            elif fault == "versions":
                workload.program = replace(workload.program, initial_versions={
                    key: value + 1 for key, value in workload.program.initial_versions.items()})
                workload.validate()
            elif fault == "zero_gradient":
                parameter = workload.state["role_a"].adapter.weight
                parameter.grad = torch.ones_like(parameter)
            elif fault == "mode":
                policy.execution["mode"] = "manual"
            elif fault == "source":
                policy.bound["source"] = "CPU_TEST_ONLY: stale saved source"
            else:
                _change_semantics(workload, fault)
                if fault != "objective":
                    workload.validate()
        before = signature(session.state)
        started = time.monotonic()
        error = None
        try:
            policy.prepare_resume(workload) if resume else policy.run(workload, root)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        unchanged = signature(session.state) == before
        # Do not let a post-failure test collective accidentally match a peer's
        # still-pending lifecycle gather in the broken implementation.
        (root / f"observed-{rank}.json").write_text(json.dumps({"error": error, "elapsed": elapsed}))
        deadline = time.monotonic() + 15
        while not (root / f"observed-{1-rank}.json").exists():
            if time.monotonic() >= deadline:
                raise AssertionError("peer did not leave policy admission within the bounded deadline")
            time.sleep(.02)
        group_error = None
        probe = torch.tensor(rank + 1)
        try:
            dist.all_reduce(probe)
        except Exception as exc:
            group_error = f"{type(exc).__name__}: {exc}"
        (root / f"rank-{rank}.json").write_text(json.dumps({
            "error": error, "elapsed": elapsed, "unchanged": unchanged,
            "group_error": group_error, "probe": probe.item(), "status": "CPU_TEST_ONLY"}))
    finally:
        dist.destroy_process_group()


def _run_workers(root, worker, *args):
    root.mkdir()
    processes = mp.spawn(worker, args=(str(root / "rdzv"), str(root), *args), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not processes.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("CPU_TEST_ONLY Gloo workers exceeded the bounded 60s deadline")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
        for process in processes.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    return [json.loads((root / f"rank-{rank}.json").read_text()) for rank in range(2)]


@pytest.mark.parametrize("fault", ["shape", "ownership", "versions", "zero_gradient", "mode", "source",
                                   "module_bindings", "backward_calls", "local_batch_size", "objective"])
def test_rank_local_later_cycle_rejection_preserves_caller_owned_group(tmp_path, fault):
    rows = _run_workers(tmp_path / "workers", _admission_worker, fault, False)
    assert all(row["error"] and "policy admission" in row["error"] for row in rows), rows
    assert rows[0]["error"] == rows[1]["error"], rows
    assert all(row["elapsed"] < 5 and row["unchanged"] for row in rows), rows
    assert all(row["group_error"] is None and row["probe"] == 3 for row in rows), rows


@pytest.mark.parametrize("fault", ["shape", "mode", "source"])
def test_rank_local_resume_policy_rejection_precedes_layout_collectives(tmp_path, fault):
    rows = _run_workers(tmp_path / "workers", _admission_worker, fault, True)
    assert all(row["error"] and "policy admission" in row["error"] for row in rows), rows
    assert rows[0]["error"] == rows[1]["error"], rows
    assert all(row["elapsed"] < 5 and row["unchanged"] for row in rows), rows
    assert all(row["group_error"] is None and row["probe"] == 3 for row in rows), rows


def _telemetry_worker(rank, rendezvous, output, reduction):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        local = {"backend": "torch-callable-resident-v1", "share_immutable": False,
                 "checkpoint_bindings": []}
        implementation = {"backend": "torch-callable-replicated-v1", "local": local,
                          "reduction": reduction, "bucket_cap_mb": 1}
        if reduction == "per_module":
            implementation.update(bucket_cap_mb=128,
                                  module_layouts={"role_a": "ddp_overlap", "role_b": "post_backward"})
        policy = CyclePolicy({"mode": "manual", "implementation": implementation})
        oracle = _workload(global_batch=True)
        reference = Session(oracle, candidates(oracle, limit=1)["candidates"][0],
                            copy_state=False, persistent=True)
        state = versions = owned = None
        rows = []
        for cycle in range(1, 7):
            workload = _workload(state, versions, rank=rank)
            session, outputs = policy.run(workload, output)
            state = session.state
            versions = session.receipt["dispatcher"]["final_versions"]
            if cycle > 1:
                reference.bind_cycle(_workload(reference.state,
                    reference.receipt["dispatcher"]["final_versions"], global_batch=True))
            expected = reference.result_state(reference.run())["state"]
            gate = compare(expected, session.result_state(outputs)["state"], rtol=3e-5, atol=3e-7)
            assert gate["status"] == "PASS", (cycle, gate)
            assert versions == reference.receipt["dispatcher"]["final_versions"]
            ownership = {"objects": {key: id(value) for key, value in state.items()},
                         "parameters": {key: tuple(id(p) for p in value.parameters())
                                        for key, value in state.items() if isinstance(value, torch.nn.Module)},
                         "optimizers": {key: tuple(id(p) for group in value.param_groups for p in group["params"])
                                        for key, value in state.items() if isinstance(value, torch.optim.Optimizer)},
                         "hooks": tuple(id(hook) for hook in session.lifecycle.hooks)}
            if owned is None:
                owned = ownership
            assert ownership == owned, cycle
            assert session.effects["lifecycle_installations"] == 1
            assert session.effects["parameter_backward_calls"] == cycle * 3
            rows.append({"cycle": cycle, "effects": deepcopy(session.effects),
                         "collectives": session.lifecycle.collectives,
                         "gradient_bytes": session.lifecycle.gradient_bytes,
                         "hook_entries": [(len(hook.bucket_bytes), len(hook.bucket_indices))
                                          for hook in session.lifecycle.hooks]})
        Path(output, f"rank-{rank}.json").write_text(json.dumps({"cycles": rows, "status": "CPU_TEST_ONLY"}))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("reduction", ["ddp_overlap", "per_module"])
def test_persistent_ddp_telemetry_is_bounded_and_cumulative_without_reinstall(tmp_path, reduction):
    ranks = _run_workers(tmp_path / "workers", _telemetry_worker, reduction)
    backward_calls = 3 if reduction == "ddp_overlap" else 1
    for rank in ranks:
        assert len(rank["cycles"]) == 6
        for row in rank["cycles"]:
            assert all(entries == [0, 0] for entries in row["hook_entries"]), row
            expected = row["cycle"] * backward_calls
            # These fixture roots each have one 4x4 FP32 trainable Parameter,
            # hence one 64-byte native DDP bucket per declared backward.
            assert row["collectives"] == expected, row
            assert row["gradient_bytes"] == expected * 64, row
            effects = row["effects"]
            if reduction == "ddp_overlap":
                assert effects["gradient_collectives"] == expected
                assert effects["communicated_gradient_bytes"] == expected * 64
            else:
                assert effects["ddp_bucket_completions"] == expected
                assert effects["reference_collectives"] == row["cycle"] * 2
