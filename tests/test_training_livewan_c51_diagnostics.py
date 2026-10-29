"""CPU scalar/control-chain evidence, never native CUDA capacity qualification."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_training_livewan_c51_continuation import (
    original_source, original_lora, stream, score_continuation_fixture,
)


FLAG = "DISTILLSCHEDULE_LIVEWAN_C51_PHASE_RECEIPTS"
RUN = "SYNTHETIC_TEST_ONLY_phase_receipts"


class BoundaryStop(RuntimeError):
    pass


class ScalarCuda:
    """Exactly the existing helper's scalar API; no tensor/device execution."""
    def __init__(self):
        self.samples, self.calls, self.resets = 0, [], 0

    def mem_get_info(self, device):
        self.samples += 1
        self.calls.append("mem_get_info")
        return 10000-self.samples, 20000

    def counter(self, name, base):
        self.calls.append(name)
        return base+self.samples

    def install(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "mem_get_info", self.mem_get_info)
        for name, base in (("memory_allocated", 100), ("memory_reserved", 200),
                           ("max_memory_allocated", 300), ("max_memory_reserved", 400)):
            monkeypatch.setattr(torch.cuda, name, lambda device=None, name=name, base=base: self.counter(name, base))
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", self.reset)

    def reset(self, device):
        self.resets += 1  # The original worker's one reset, not a phase reset.


def identity(horizon=2, input_sha256="2"*64):
    from distillschedule.training import livewan_c51_training as training
    return dict(schema_version=training.SCHEMA, source_head="f85dfda02b7958a383a3391205f4df588ab1dd5b",
                source_files=training.source_files(), assets_sha256="1"*64,
                input_sha256=input_sha256, horizon=horizon)


def binding(rank):
    inventory = {str(i): f"00000000-0000-0000-0000-{rank//8*8+i+1:012d}" for i in range(8)}
    return dict(rank=rank, node_rank=rank//8, cuda_device=rank % 8, physical_device=rank % 8,
                uuid=inventory[str(rank % 8)], physical_inventory=inventory)


def continuation(tmp_path, rank=8):
    from distillschedule.training import livewan_c51_training as training
    inputs = SimpleNamespace(cycles=2, global_batch=16, rows=[], sha256="3"*64)
    value = training.Continuation(config=SimpleNamespace(trainer=dict(checkpoint_every=1)),
        admitted={}, identity=identity(), output=tmp_path/"output", bootstrap=None,
        loader=SimpleNamespace(inputs=inputs), header=None, capacity_check=lambda: None)
    value.device_binding = binding(rank)
    return value


def setup_worker(value, monkeypatch, tmp_path, *, rank=8, stop="before_bind", build_error=False,
                 invocation=None, runner=None, catch=True):
    """Enter actual typed run_worker through setup; native dependencies stand in.

    Unlike a receipt-builder-only test this executes the production setup read
    and the exact producer callsites. No native model/kernel/distributed work.
    """
    import torch
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.runtime import livewan_cuda_setup, livewan_generator_splitkv
    from distillschedule.training import distributed, livewan_c51_training as training
    from distillschedule.training.livewan_c51_state import physical_versions
    pairs = tuple((i, i+8) for i in range(8))
    value.bootstrap = SimpleNamespace(versions=physical_versions(rank, 0))
    plan = training.make_plan(dict(native=dict(models={name: dict(root="/explicit/"+name)
                                                      for name in ("student", "critic", "teacher")})), value.output)
    path = tmp_path/"effective-plan.json"
    if invocation is None:
        path.write_text(json.dumps(plan))
        invocation = dict(plan_path=path, mode="pipeline", run_kind="continuation", repo_root=tmp_path,
                          continuation=value)
    runner = runner or worker.run_worker
    built = []
    bundle = SimpleNamespace(student=None, critic=None, critic_shadow=None, teacher=None, metadata={})
    def build(**kwargs):
        built.append(True)
        if build_error:
            raise BoundaryStop("synthetic model-construction exception")
        return bundle
    def bound(**kwargs):
        if stop == "before_bind":
            raise BoundaryStop("synthetic public-bind exception")
        # A resumed actual owner restores this clock only at public bind,
        # never by relabeling its pre-bind construction snapshots.
        kwargs["versions"].update(physical_versions(rank, value.start_cycle))
        return value.loader
    def stop_evidence():
        raise BoundaryStop("synthetic after-bind exception")
    with monkeypatch.context() as patch:
        for key, val in dict(RANK=str(rank), LOCAL_RANK=str(rank % 8), WORLD_SIZE="16").items():
            patch.setenv(key, val)
        package = ModuleType("wanstreamer")
        package.blockcausal = SimpleNamespace(ckpt=None)
        patch.setitem(sys.modules, "wanstreamer", package)
        patch.setattr(torch.distributed, "is_initialized", lambda: True)
        patch.setattr(distributed, "rank_world", lambda: (0, 1))
        patch.setattr(torch.cuda, "set_device", lambda device: None)
        patch.setattr(livewan_cuda_setup, "materialize_blas_handles", lambda *args: {})
        patch.setattr(livewan_generator_splitkv, "enabled", lambda *args, **kwargs: False)
        patch.setattr(worker, "_validate_plan_identity", lambda *args, **kwargs: None)
        patch.setattr(worker, "install_geometry", lambda *args, **kwargs: None)
        patch.setattr(worker, "_continuous_pipeline_contract", lambda *args, **kwargs: None)
        patch.setattr(worker, "_publish_effective_plan", lambda *args, **kwargs: None)
        patch.setattr(worker, "_trajectory_critic_batch_ablation_enabled", lambda *args, **kwargs: False)
        for name in vars(worker).copy():
            if name.startswith("_") and name.endswith("_enabled") and callable(getattr(worker, name)):
                patch.setattr(worker, name, lambda *args, **kwargs: False)
        patch.setattr(worker, "build_models", build)
        patch.setattr(worker, "_warm_communicators", lambda *args, **kwargs: {})
        patch.setattr(worker, "cgroup_oom_counts", lambda: dict(oom=0, oom_kill=0))
        patch.setattr(value, "configure_sources", lambda: dict(source_receipt_sha256="4"*64))
        patch.setattr(value, "validate_constructed", lambda: None)
        patch.setattr(value, "bind", bound)
        if stop == "after_bind":
            patch.setattr(worker, "RunEvidence", stop_evidence)
        if catch:
            with pytest.raises(BoundaryStop): runner(**invocation)
        else:
            runner(**invocation)
    return built


def receipts(value):
    rank = value.device_binding["rank"]
    return json.loads((value.output/"capacity/phase-receipts"/f"rank-{rank:05d}.json").read_text())


@pytest.fixture
def scalar(monkeypatch):
    value = ScalarCuda()
    value.install(monkeypatch)
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ID", RUN)
    monkeypatch.setenv(FLAG, "v1")
    return value


@pytest.mark.parametrize("rank", [0, 8], ids=["generator-node0", "score-node1"])
def test_real_worker_persists_existing_setup_sample_before_public_bind(tmp_path, monkeypatch, scalar, rank):
    value = continuation(tmp_path, rank)
    assert setup_worker(value, monkeypatch, tmp_path, rank=rank) == [True]
    result = receipts(value)
    rows = result["observations"]
    assert [row["phase"] for row in rows] == ["construction_entry", "construction_exit", "setup_ready"]
    assert scalar.samples == 3  # two extra boundaries + ONE existing setup read
    assert rows[-1]["memory"] == dict(allocated_bytes=103, reserved_bytes=203, max_allocated_bytes=303,
                                      max_reserved_bytes=403, driver_free_bytes=9997, driver_total_bytes=20000)
    assert rows[-1]["memory_origin"] == "EXISTING_SETUP_MEMORY_NO_REQUERY"
    assert scalar.resets == 1
    assert all(row["wave"] is None and row["physical_versions"] == dict(generator=0, critic=0, ema=0) for row in rows)
    assert result["identity"]["run_id"] == RUN and result["identity"]["gpu_uuid"] == binding(rank)["uuid"]
    assert result["identity"]["node_rank"] == rank//8
    assert result["status"] == "DIAGNOSTIC_ONLY_PARTIAL_HISTORY" and not result["formal_timing_eligible"]
    assert not (value.output/"continuation/rank-00008.json").exists()
    assert not list(value.output.rglob("_SUCCESS"))


def test_construction_exception_leaves_only_the_reached_entry(tmp_path, monkeypatch, scalar):
    value = continuation(tmp_path)
    setup_worker(value, monkeypatch, tmp_path, build_error=True)
    assert [row["phase"] for row in receipts(value)["observations"]] == ["construction_entry"]
    assert scalar.samples == 1 and scalar.resets == 1


def test_default_off_real_worker_has_no_extra_reads_or_writes(tmp_path, monkeypatch, scalar):
    from distillschedule.training import livewan_c51_training as training
    value = continuation(tmp_path)
    monkeypatch.delenv(FLAG)
    monkeypatch.setattr(training, "source_files", lambda: pytest.fail("disabled diagnostic source read"))
    setup_worker(value, monkeypatch, tmp_path)
    assert scalar.samples == 1 and scalar.resets == 1  # unchanged incumbent setup read/reset
    assert not (value.output/"capacity/phase-receipts").exists()


def test_default_off_does_not_import_diagnostic_module_or_inspect_its_identity(tmp_path, monkeypatch, scalar):
    value = continuation(tmp_path)
    monkeypatch.setenv(FLAG, "0")
    monkeypatch.setitem(sys.modules, "distillschedule.training.livewan_c51_diagnostics", None)
    value.identity = value.device_binding = object()  # Disabled diagnostics may not inspect either.
    value.configure_phase_diagnostics(rank=8)
    value.phase_receipt(torch=object(), rank=-1, device=object(), phase="not-a-diagnostic-phase",
                        cycle=-1, versions=object(), memory=object())
    assert scalar.samples == 0 and not value.output.exists()


def public_config(tmp_path, prepared="/explicit/input.json"):
    from distillschedule.training.config import TrainingConfig
    raw = dict(schema_version="distillschedule.training.v1", model=dict(workload="W4"),
        algorithm=dict(name="original_self_forcing_c51"),
        data=dict(input_kind="original_livewan_c51_teacher_stream.v1", prepared=str(prepared)),
        execution=dict(adapter="native", mode="manual", task="/explicit/task.json", assets="/explicit/assets.json",
                       capacity_policy="/explicit/capacity.json", preset="self_forcing.c51_stream.v1"),
        trainer=dict(cycles=4, seed=31415, checkpoint_every=1, stop_after_cycle=1, output_dir=str(tmp_path/"public-output")))
    path = tmp_path/"training.json"
    path.write_text(json.dumps(raw))
    return TrainingConfig.from_dict(raw, source=path)


def test_public_descriptor_and_checkpoint_source_closure_include_the_actual_helper(tmp_path):
    from distillschedule.training.native import describe_native
    from distillschedule.training.livewan_c51_training import source_files
    from distillschedule.workloads.best_known.registry import file_sha256
    helper = "training/livewan_c51_diagnostics.py"
    assert helper in describe_native(public_config(tmp_path))["preset"]["implementation_paths"]
    package = Path(__file__).parents[1]/"src/distillschedule"
    assert source_files()[helper] == file_sha256(package/helper)


def test_typed_public_config_native_runner_reaches_actual_diagnostic_setup(
        stream, tmp_path, monkeypatch, scalar):
    """Actual public dispatch; admission/device/assets are CPU-only stand-ins.

    InputLoader, TrainingConfig, training identity, original output reservation,
    native.run_native -> run_training -> typed worker setup and failure receipt
    are real. No clean-source/GPU/admitted-model evidence is claimed here.
    """
    from contextlib import nullcontext
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.search import resources
    from distillschedule.training import distributed, native, livewan_c51_training as training, livewan_training as reference
    from distillschedule.workloads.best_known.registry import file_sha256
    config = public_config(tmp_path, stream[0])
    assets = dict(native=dict(sources=dict(livewan=stream[1]["source"]),
        models={name: dict(root="/SYNTHETIC_TEST_ONLY/"+name) for name in ("student", "critic", "teacher")}))
    admitted = dict(assets=assets, inputs=stream[1], capacity_policy={})
    source = dict(identity=dict(head="f85dfda02b7958a383a3391205f4df588ab1dd5b", dirty=False),
                  scope="SYNTHETIC_TEST_ONLY_ADMISSION_NOT_A_GIT_CLEANLINESS_RECEIPT")
    monkeypatch.setattr(native, "_admit", lambda _: ("W4", source, dict(output=config.trainer["output_dir"]), admitted))
    monkeypatch.setattr(distributed, "world16_reference_process_group", lambda _: nullcontext())
    monkeypatch.setattr(resources, "ResourceObserver", lambda *args, **kwargs: object())
    monkeypatch.setattr(reference, "_bind_devices", lambda *args, **kwargs: binding(0))
    monkeypatch.setattr(reference, "_capacity", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference, "_admit_model_layout", lambda *args, **kwargs: {})
    monkeypatch.setattr(training, "admit_runtime", lambda _: {})
    monkeypatch.setattr(training.C51Assets, "load", lambda _: SimpleNamespace(versions=dict(generator=0, critic=0, ema=0)))
    monkeypatch.setattr(training.Continuation, "configure_sources", lambda _: dict(source_receipt_sha256="4"*64))
    actual_worker, observed = worker.run_worker, []
    def enter(**invocation):
        value = invocation["continuation"]
        observed.append(value)
        setup_worker(value, monkeypatch, tmp_path, rank=0, invocation=invocation, runner=actual_worker, catch=False)
    monkeypatch.setattr(worker, "run_worker", enter)
    with pytest.raises(BoundaryStop, match="public-bind"):
        native.run_native(config)
    assert len(observed) == 1 and type(observed[0]) is training.Continuation
    value = observed[0]
    helper = "training/livewan_c51_diagnostics.py"
    package = Path(training.__file__).parents[1]
    assert value.identity["source_files"][helper] == file_sha256(package/helper)
    result = json.loads((value.output/"capacity/phase-receipts/rank-00000.json").read_text())
    assert result["identity"]["source_files"][helper] == value.identity["source_files"][helper]
    assert result["identity"]["input_sha256"] == stream[1]["input_sha256"]
    assert result["observations"][-1]["phase"] == "setup_ready" and scalar.samples == 3
    failure = json.loads((value.output/"continuation/rank-00000.failed.json").read_text())
    assert failure["status"] == "FAIL" and failure["last_locally_completed_cycle"] == 0
    assert failure["native_cuda_numerical_qualification"] == "NOT_RUN"
    assert not (value.output/"continuation/rank-00000.json").exists() and not list(value.output.rglob("_SUCCESS"))


@pytest.mark.parametrize("damage", ["partial_source", "changed_source", "missing_head", "input", "asset", "rank", "uuid", "run"])
def test_enabled_identity_fails_closed_before_memory_or_output(tmp_path, monkeypatch, scalar, damage):
    value = continuation(tmp_path)
    if damage == "partial_source": value.identity["source_files"].pop("training/checkpoint.py")
    elif damage == "changed_source": value.identity["source_files"]["training/checkpoint.py"] = "0"*64
    elif damage == "missing_head": value.identity.pop("source_head")
    elif damage == "input": value.identity["input_sha256"] = "partial"
    elif damage == "asset": value.identity["assets_sha256"] = "partial"
    elif damage == "rank": value.device_binding["rank"] = 0
    elif damage == "uuid": value.device_binding["uuid"] = binding(0)["uuid"]
    else: monkeypatch.delenv("DISTILLSCHEDULE_RUN_ID")
    with pytest.raises(ValueError, match="phase diagnostic"):
        value.configure_phase_diagnostics(rank=8)
    assert scalar.samples == 0 and not value.output.exists()


def test_invalid_flag_and_output_only_are_explicitly_refused(tmp_path, monkeypatch, scalar):
    value = continuation(tmp_path)
    monkeypatch.setenv(FLAG, "1")
    with pytest.raises(ValueError, match="phase diagnostic"):
        value.configure_phase_diagnostics(rank=8)
    monkeypatch.setenv(FLAG, "v1")
    value.native_output = object()
    with pytest.raises(ValueError, match="phase diagnostic"):
        value.configure_phase_diagnostics(rank=8)
    assert scalar.samples == 0 and not value.output.exists()


def test_unwritable_diagnostic_fails_before_model_construction(tmp_path, monkeypatch, scalar):
    from distillschedule.runtime import livewan_crossscale_worker as worker
    value = continuation(tmp_path)
    def deny(*args, **kwargs):
        raise PermissionError("synthetic unwritable diagnostic destination")
    monkeypatch.setattr(worker, "atomic_json", deny)
    with pytest.raises(PermissionError, match="unwritable"):
        setup_worker(value, monkeypatch, tmp_path)
    assert not list(value.output.rglob("_SUCCESS"))


@pytest.mark.parametrize("damage", ["rank", "cycle", "clock", "exit_without_entry", "unknown_phase"])
def test_wrong_rank_cycle_or_phase_never_samples_or_replaces_history(tmp_path, monkeypatch, scalar, damage):
    import torch
    value = continuation(tmp_path)
    setup_worker(value, monkeypatch, tmp_path, stop="after_bind")
    path = value.output/"capacity/phase-receipts/rank-00008.json"
    before, count = path.read_bytes(), scalar.samples
    kwargs = dict(torch=torch, rank=8, device=torch.device("cuda:0"), phase="full5_entry", cycle=1,
                  versions=dict(generator=0, critic=0, ema=0))
    if damage == "rank": kwargs["rank"] = 0
    elif damage == "cycle": kwargs["cycle"] = 2
    elif damage == "clock": kwargs["versions"]["critic"] = 5
    elif damage == "exit_without_entry": kwargs["phase"] = "full5_exit"
    else: kwargs["phase"] = "invented_completed_cycle"
    with pytest.raises(ValueError, match="phase diagnostic"):
        value.phase_receipt(**kwargs)
    assert scalar.samples == count and path.read_bytes() == before


def test_atomic_replace_failure_preserves_last_receipt_and_no_fake_cycle(tmp_path, monkeypatch, scalar):
    import torch
    value = continuation(tmp_path)
    setup_worker(value, monkeypatch, tmp_path, stop="after_bind")
    path = value.output/"capacity/phase-receipts/rank-00008.json"
    before = path.read_bytes()
    def interrupted(*args):
        raise OSError("synthetic interrupted atomic replacement")
    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(OSError, match="atomic"):
        value.phase_receipt(torch=torch, rank=8, device=torch.device("cuda:0"), phase="full5_entry", cycle=1,
                            versions=dict(generator=0, critic=0, ema=0))
    assert path.read_bytes() == before
    assert len(value.phase_diagnostics.observations) == 4
    assert receipts(value)["observations"][-1]["phase"] == "public_bind_exit"
    assert not list(value.output.rglob("_SUCCESS"))


def test_boundary_observation_does_not_consume_rng_or_read_tensor_state(tmp_path, monkeypatch, scalar):
    import numpy as np
    import torch
    from test_training_livewan_state import equal
    value = continuation(tmp_path)
    setup_worker(value, monkeypatch, tmp_path, stop="after_bind")
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    def forbidden(*args, **kwargs): pytest.fail("diagnostic touched tensor/synchronization/cache/RNG API")
    for name in ("synchronize", "empty_cache", "get_rng_state", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    value.phase_receipt(torch=torch, rank=8, device=torch.device("cuda:0"), phase="full5_entry", cycle=1,
                        versions=dict(generator=0, critic=0, ema=0))
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert before[0] == after[0] and before[1][0] == after[1][0]
    assert np.array_equal(before[1][1], after[1][1]) and before[1][2:] == after[1][2:]
    assert equal(before[2], after[2]) and not torch.cuda.is_initialized()
    assert len(scalar.calls) == 5*scalar.samples


def test_maximum_horizon_is_bounded_and_cannot_fabricate_cycle101(tmp_path, monkeypatch, scalar):
    import torch
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.training.livewan_c51_state import physical_versions
    value = continuation(tmp_path)
    value.identity["horizon"] = value.inputs.cycles = value.end_cycle = value.cycles_executed = 100
    setup_worker(value, monkeypatch, tmp_path, stop="after_bind")
    # Atomic replacement behavior is covered separately; suppress 400 large
    # redundant file rewrites while checking the complete bounded state machine.
    writes = []
    monkeypatch.setattr(worker, "atomic_json", lambda path, doc: writes.append(len(doc["observations"])))
    for cycle in range(1, 101):
        for phase in ("full5_entry", "full5_exit", "checkpoint_pack_entry", "checkpoint_pack_exit"):
            value.phase_receipt(torch=torch, rank=8, device=torch.device("cuda:0"), phase=phase, cycle=cycle,
                versions=physical_versions(8, cycle-1 if phase == "full5_entry" else cycle))
    assert value.phase_diagnostics.limit == len(value.phase_diagnostics.observations) == writes[-1] == 404
    assert len(writes) == 400 and scalar.resets == 1
    count = scalar.samples
    with pytest.raises(ValueError, match="phase diagnostic"):
        value.phase_receipt(torch=torch, rank=8, device=torch.device("cuda:0"), phase="full5_entry", cycle=101,
                            versions=physical_versions(8, 100))
    assert scalar.samples == count and len(writes) == 400


@pytest.mark.parametrize("failure", [None, "full5", "packing", "capacity"])
@pytest.mark.parametrize("start", [0, 2], ids=["fresh", "restored"])
def test_actual_continuous_owner_and_packing_chain_persists_only_reached_phases(
        stream, original_source, original_lora, tmp_path, monkeypatch, scalar, failure, start):
    import torch
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.training import checkpoint
    checks = []
    def capacity():
        checks.append(True)
        if failure == "capacity" and len(checks) == 2:
            raise ValueError("synthetic unchanged-capacity-reserve guard failure")
    value = score_continuation_fixture(stream, original_source, original_lora, tmp_path, monkeypatch,
                                       start=start, end=start+1, cadence=1, capacity_check=capacity)
    cont = value.cont
    cont.identity = identity(cont.inputs.cycles, cont.inputs.input_sha256)
    cont.device_binding = binding(8)
    setup_worker(cont, monkeypatch, tmp_path, stop="after_bind")
    timeline = value.timeline
    native_atomic, native_pack = worker.atomic_json, checkpoint.pack_state
    def atomic(path, document):
        if "phase-receipts" in str(path):
            timeline.append(("phase", document["observations"][-1]["phase"]))
        return native_atomic(path, document)
    def pack(state):
        timeline.append(("pack-start",))
        if failure == "packing": raise ValueError("synthetic exact-pack failure")
        result = native_pack(state)
        timeline.append(("pack-return",))
        return result
    def body(**kwargs):
        timeline.append(("full5-body",))
        if failure == "full5": raise ValueError("synthetic original-Full5 failure")
        value.step()  # real original tiny LoRA/Adam; NOT Wan/CUDA math
    monkeypatch.setattr(worker, "atomic_json", atomic)
    monkeypatch.setattr(checkpoint, "pack_state", pack)
    monkeypatch.setattr(worker, "_run_pipeline_8x8", body)
    value.transport.finish = lambda: None
    value.transport.receipt = lambda: dict(scope="CPU_TRANSPORT_STANDIN")
    value.shadow.finish = lambda: None
    value.shadow.receipt = lambda: dict(scope="CPU_SHADOW_STANDIN")
    def execute():
        worker._run_pipeline_8x8_continuous(torch=torch, rank=8, trace=cont.inputs, loader=value.loader,
            bundle=value.current.bundle, streamers=value.streamers, score_context=None, run_kind="continuation",
            evidence=value.evidence, versions=value.actual, device=torch.device("cuda:0"),
            runahead_host_transport=value.transport, critic_shadow_hot_sync=value.shadow,
            slot_pairs_by_lane=value.loader.pairs, full_context_lens_elision=True,
            critic_step1_split_overlap=False, critic_b1_attribution=False, inventory=value.inventory,
            continuation=cont)
    if failure:
        with pytest.raises(ValueError, match="synthetic"): execute()
    else:
        execute()
    phases = [row["phase"] for row in receipts(cont)["observations"]]
    assert phases[:4] == ["construction_entry", "construction_exit", "setup_ready", "public_bind_exit"]
    observed = receipts(cont)["observations"]
    assert all(row["physical_versions"] == dict(generator=0, critic=0, ema=0) for row in observed[:3])
    assert observed[3]["physical_versions"] == dict(generator=0, critic=5*start, ema=0)
    assert observed[4]["cycle"] == start+1 and observed[4]["global_step"] == 5*start
    assert timeline.index(("phase", "full5_entry")) < timeline.index(("full5-body",))
    if failure == "full5":
        assert phases[4:] == ["full5_entry"] and value.loader.cycle == start
    else:
        assert timeline.index(("full5-body",)) < timeline.index(("phase", "full5_exit")) < timeline.index(("drain", start+1))
        assert timeline.index(("phase", "checkpoint_pack_entry")) < timeline.index(("pack-start",))
        if failure == "packing":
            assert phases[4:] == ["full5_entry", "full5_exit", "checkpoint_pack_entry"]
        else:
            assert timeline.index(("pack-return",)) < timeline.index(("phase", "checkpoint_pack_exit"))
            assert phases[4:] == ["full5_entry", "full5_exit", "checkpoint_pack_entry", "checkpoint_pack_exit"]
    complete = cont.output/"continuation/rank-00008.json"
    assert complete.exists() is (failure is None)
    assert bool(list(cont.output.rglob("_SUCCESS"))) is (failure is None)
    assert receipts(cont)["status"] == "DIAGNOSTIC_ONLY_PARTIAL_HISTORY"
    assert not receipts(cont)["claimable_speedup"] and not torch.cuda.is_initialized()
