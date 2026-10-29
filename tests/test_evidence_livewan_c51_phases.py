"""SYNTHETIC_TEST_ONLY scalar/device identities; no CUDA/model/native gate.

Every accepted document is written by the actual PhaseReceipts.record and
worker.atomic_json. Corruption cases modify those real producer documents.
"""
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


RUN_ID = "SYNTHETIC_TEST_ONLY_phase_consumer"
FLAG = "DISTILLSCHEDULE_LIVEWAN_C51_PHASE_RECEIPTS"
ZERO = {"generator": 0, "critic": 0, "ema": 0}


class ScalarCuda:
    """The original memory helper's six scalar calls, not a CUDA interface."""
    def __init__(self):
        self.samples = 0

    def mem_get_info(self, device):
        self.samples += 1
        return 10000-self.samples, 20000

    def memory_allocated(self, device):
        return 100+self.samples

    def memory_reserved(self, device):
        return 200+self.samples

    def max_memory_allocated(self, device):
        return 300+self.samples

    def max_memory_reserved(self, device):
        return 400+self.samples


def device_binding(rank):
    inventory = {str(i): f"00000000-0000-0000-0000-{rank//8*8+i+1:012d}" for i in range(8)}
    return dict(rank=rank, node_rank=rank//8, cuda_device=rank % 8, physical_device=rank % 8,
                uuid=inventory[str(rank % 8)], physical_inventory=inventory)


def produce(output, *, rank=8, start=0, end=1, interrupted=None, cadence=1, binding_override=None):
    from distillschedule.training.livewan_c51_diagnostics import PhaseReceipts
    from distillschedule.training.livewan_c51_training import source_files
    from distillschedule.training.livewan_c51_state import physical_versions
    from distillschedule.runtime.livewan_crossscale_worker import memory_snapshot
    identity = dict(schema_version="distillschedule.livewan_c51_continuation.v1", source_head="c"*40,
                    source_files=source_files(), assets_sha256="1"*64, input_sha256="2"*64,
                    horizon=2, checkpoint_every=cadence, evidence_scope="SYNTHETIC_TEST_ONLY")
    binding = device_binding(rank) if binding_override is None else binding_override
    producer = PhaseReceipts(identity=identity, output=output, binding=binding, rank=rank,
        start_cycle=start, end_cycle=end, current_source_files=source_files(), run_id=RUN_ID)
    scalar = ScalarCuda()
    torch = SimpleNamespace(cuda=scalar)
    phases = [(name, start, ZERO) for name in ("construction_entry", "construction_exit", "setup_ready")]
    phases.append(("public_bind_exit", start, physical_versions(rank, start)))
    for cycle in range(start+1, end+1):
        phases.extend([("full5_entry", cycle, physical_versions(rank, cycle-1)),
                       ("full5_exit", cycle, physical_versions(rank, cycle))])
        if cadence and cycle % cadence == 0:
            phases.extend((name, cycle, physical_versions(rank, cycle))
                          for name in ("checkpoint_pack_entry", "checkpoint_pack_exit"))
    for phase, cycle, versions in phases:
        kwargs = dict(torch=torch, rank=rank, device=f"cuda:{rank % 8}", phase=phase, cycle=cycle, versions=versions)
        if phase == "setup_ready":
            kwargs["memory"] = memory_snapshot(torch, f"cuda:{rank % 8}")
        producer.record(**kwargs)
        if phase == interrupted:
            break
    expected = dict(output_root=output, training_identity=identity, environment={FLAG:"v1", "DISTILLSCHEDULE_RUN_ID":RUN_ID},
                    run_id=RUN_ID, binding=binding, expected_pid=os.getpid(), namespace_pids=[os.getpid()],
                    construction_versions=dict(ZERO), start_cycle=start, end_cycle=end)
    return SimpleNamespace(path=producer.path, output=output, expected=expected, scalar=scalar)


def consume(packet, **changes):
    module = importlib.import_module("distillschedule.evidence.livewan_c51_phases")
    return module.read_phase_receipt(packet.path, **{**packet.expected, **changes})


def corrupt(packet, operation):
    from distillschedule.runtime.livewan_crossscale_worker import atomic_json
    value = json.loads(packet.path.read_text())
    operation(value)
    atomic_json(packet.path, value)


@pytest.fixture
def produced(tmp_path):
    return produce(tmp_path/"SYNTHETIC_TEST_ONLY-output")


@pytest.mark.parametrize("rank", [0, 8], ids=["synthetic-generator", "synthetic-score"])
def test_real_producer_complete_chain(tmp_path, rank):
    packet = produce(tmp_path/"SYNTHETIC_TEST_ONLY-output", rank=rank)
    before = packet.path.read_bytes()
    result = consume(packet)
    assert result["history_state"] == "COMPLETE_DECLARED_PHASE_CHAIN"
    assert result["observation_count"] == 8 and packet.scalar.samples == 8
    assert result["diagnostic_only"] is True and result["formal_timing_eligible"] is False
    assert result["native_cuda_numerical_qualification"] == result["capacity_qualification"] == "NOT_RUN"
    assert result["endpoint_qualification"] == "NOT_ESTABLISHED"
    assert result["pid_evidence_scope"] == "CALLER_NAMESPACE_PID_MEMBERSHIP_NOT_BIRTH_IDENTITY"
    assert "birth_start" not in result and "minimum_free_bytes" not in result and "completed_cycles" not in result
    assert packet.path.read_bytes() == before


@pytest.mark.parametrize("rank", [0, 8])
def test_c2_restored_setup_keeps_real_bootstrap_clocks(tmp_path, rank):
    packet = produce(tmp_path/"SYNTHETIC_TEST_ONLY-C2", rank=rank, start=1, end=2)
    result = consume(packet)
    rows = result["observations"]
    assert all(row["physical_versions"] == ZERO for row in rows[:3])
    from distillschedule.training.livewan_c51_state import physical_versions
    assert rows[3]["physical_versions"] == physical_versions(rank, 1)
    assert result["history_state"] == "COMPLETE_DECLARED_PHASE_CHAIN"


@pytest.mark.parametrize("phase", ["construction_entry", "setup_ready", "full5_entry", "full5_exit", "checkpoint_pack_entry"])
def test_real_producer_interrupted_prefix(tmp_path, phase):
    packet = produce(tmp_path/"SYNTHETIC_TEST_ONLY-interrupted", interrupted=phase)
    result = consume(packet)
    assert result["history_state"] == "INTERRUPTED_PREFIX"
    assert result["observations"][-1]["phase"] == phase
    assert result["endpoint_qualification"] == "NOT_ESTABLISHED"


def test_actual_producer_without_checkpoint_cadence(tmp_path):
    packet = produce(tmp_path/"SYNTHETIC_TEST_ONLY-no-checkpoint", cadence=0)
    assert consume(packet)["history_state"] == "COMPLETE_DECLARED_PHASE_CHAIN"
    assert consume(packet)["observation_count"] == 6


def test_actual_producer_two_cycle_scalar_interval(tmp_path):
    packet = produce(tmp_path/"SYNTHETIC_TEST_ONLY-two", end=2)
    assert consume(packet)["observation_count"] == 12
    assert consume(packet)["history_state"] == "COMPLETE_DECLARED_PHASE_CHAIN"


IDENTITY_DRIFTS = {
    "source_head": "d"*40, "source_files": {"SYNTHETIC_TEST_ONLY.py":"4"*64},
    "source_files_sha256":"4"*64, "assets_sha256":"4"*64, "input_sha256":"4"*64,
    "training_identity_sha256":"4"*64, "run_id":RUN_ID+"-other", "rank":0, "node_rank":0,
    "local_rank":1, "pid":1, "gpu_uuid":"00000000-0000-0000-0000-000000000099",
    "physical_device":1, "physical_inventory":{}, "role":"generator", "start_cycle":1, "requested_stop_cycle":2,
}


@pytest.mark.parametrize("field", IDENTITY_DRIFTS)
def test_reject_identity_drift_in_actual_producer_file(produced, field):
    corrupt(produced, lambda value: value["identity"].__setitem__(field, IDENTITY_DRIFTS[field]))
    with pytest.raises(ValueError):
        consume(produced)


PROMOTIONS = {"status":"PASS", "diagnostic_only":False, "formal_timing_eligible":True,
              "claimable_speedup":True, "native_cuda_numerical_qualification":"PASS",
              "capacity_qualification":"PASS", "unsynchronized_host_samples":False,
              "peak_scope":"RESET_PER_PHASE", "completed_cycles":1}


@pytest.mark.parametrize("field", PROMOTIONS)
def test_reject_forged_promotion(produced, field):
    corrupt(produced, lambda value: value.__setitem__(field, PROMOTIONS[field]))
    with pytest.raises(ValueError):
        consume(produced)


ROW_DRIFTS = {
    "phase":"forged", "sequence":99, "cycle":2, "global_step":123, "physical_versions":ZERO,
    "wave":0, "wave_scope":"CUDA_INTERVAL", "memory_origin":"RECONSTRUCTED",
    "timing_class":"FORMAL", "observation":"C1_ACCEPTED", "monotonic_ns":True, "wall_time_ns":True,
}


@pytest.mark.parametrize("field", ROW_DRIFTS)
def test_reject_bad_phase_order_sequence_clock_or_labels(produced, field):
    corrupt(produced, lambda value: value["observations"][-1].__setitem__(field, ROW_DRIFTS[field]))
    with pytest.raises(ValueError):
        consume(produced)


@pytest.mark.parametrize("kind", ["negative", "boolean", "missing", "extra", "driver_bounds"])
def test_reject_memory_drift(produced, kind):
    def operation(value):
        memory = value["observations"][-1]["memory"]
        if kind == "negative": memory["allocated_bytes"] = -1
        elif kind == "boolean": memory["allocated_bytes"] = True
        elif kind == "missing": memory.pop("reserved_bytes")
        elif kind == "extra": memory["invented_cuda_interval"] = 1
        else: memory["driver_free_bytes"] = memory["driver_total_bytes"]+1
    corrupt(produced, operation)
    with pytest.raises(ValueError):
        consume(produced)


@pytest.mark.parametrize("kind", ["reorder", "decreasing_monotonic", "empty", "wrong_limit", "partial_json", "duplicate_json"])
def test_reject_invalid_history_document(produced, kind):
    if kind == "partial_json": produced.path.write_text('{"schema_version":')
    elif kind == "duplicate_json":
        produced.path.write_text(produced.path.read_text().replace('"status":', '"status":"PASS", "status":', 1))
    else:
        def operation(value):
            if kind == "reorder": value["observations"][0],value["observations"][1] = value["observations"][1],value["observations"][0]
            elif kind == "decreasing_monotonic": value["observations"][-1]["monotonic_ns"] = 0
            elif kind == "empty": value["observations"] = []
            else: value["observation_limit"] += 1
        corrupt(produced, operation)
    with pytest.raises(ValueError):
        consume(produced)


@pytest.mark.parametrize("kind", ["flag", "run_id", "pid_membership", "exact_pid", "segment", "source", "binding", "construction_clock"])
def test_reject_caller_binding_drift(produced, kind):
    changes = deepcopy(produced.expected)
    if kind == "flag": changes["environment"][FLAG] = "0"
    elif kind == "run_id": changes["environment"]["DISTILLSCHEDULE_RUN_ID"] += "-different"
    elif kind == "pid_membership": changes["namespace_pids"] = [os.getpid()+1]
    elif kind == "exact_pid":
        changes["expected_pid"] += 1
        changes["namespace_pids"].append(changes["expected_pid"])
    elif kind == "segment": changes["start_cycle"],changes["end_cycle"] = 1,2
    elif kind == "source": changes["training_identity"]["source_head"] = "e"*40
    elif kind == "binding": changes["binding"]["physical_inventory"]["0"] = "00000000-0000-0000-0000-000000000099"
    else: changes["construction_versions"]["critic"] = 5
    with pytest.raises(ValueError):
        consume(produced, **changes)


@pytest.mark.parametrize("kind", ["outside_root", "file_symlink", "directory_symlink"])
def test_reject_path_escape_or_symlink(produced, tmp_path, kind):
    if kind == "outside_root":
        with pytest.raises(ValueError):
            consume(produced, output_root=tmp_path/"different-output")
        return
    if kind == "file_symlink":
        saved = tmp_path/"SYNTHETIC_TEST_ONLY-real.json"
        produced.path.rename(saved)
        produced.path.symlink_to(saved)
    else:
        folder = produced.path.parent
        saved = tmp_path/"SYNTHETIC_TEST_ONLY-real-directory"
        folder.rename(saved)
        folder.symlink_to(saved, target_is_directory=True)
    with pytest.raises(ValueError):
        consume(produced)


def world(tmp_path, *, interrupted=None):
    output = tmp_path/"SYNTHETIC_TEST_ONLY-world16"
    packets = [produce(output, rank=rank, interrupted=interrupted) for rank in range(16)]
    first = packets[0].expected
    common = {k:first[k] for k in ("training_identity", "environment", "run_id", "start_cycle", "end_cycle")}
    common["rank_bindings"] = {rank:{k:packet.expected[k] for k in ("binding", "expected_pid", "namespace_pids", "construction_versions")}
                               for rank,packet in enumerate(packets)}
    return output, packets, common


@pytest.mark.parametrize("interrupted", [None, "checkpoint_pack_entry"])
def test_actual_all16_history_collection_is_not_endpoint_success(tmp_path, interrupted):
    output, packets, common = world(tmp_path, interrupted=interrupted)
    module = importlib.import_module("distillschedule.evidence.livewan_c51_phases")
    result = module.read_phase_receipts(output, **common)
    assert result["rank_count"] == 16
    assert result["all_phase_chains_complete"] is (interrupted is None)
    assert result["endpoint_qualification"] == "NOT_ESTABLISHED"
    assert result["capacity_qualification"] == "NOT_RUN"


@pytest.mark.parametrize("kind", ["missing_rank", "extra_rank", "missing_binding"])
def test_all16_inventory_is_exact(tmp_path, kind):
    output, packets, common = world(tmp_path)
    if kind == "missing_rank": packets[15].path.unlink()
    elif kind == "extra_rank": (packets[0].path.parent/"rank-00016.json").write_bytes(packets[0].path.read_bytes())
    else: common["rank_bindings"].pop(15)
    module = importlib.import_module("distillschedule.evidence.livewan_c51_phases")
    with pytest.raises(ValueError):
        module.read_phase_receipts(output, **common)


def test_all16_rejects_duplicate_physical_uuid_under_gpu_prefix_alias(tmp_path):
    output, packets, common = world(tmp_path)
    inventory = device_binding(0)["physical_inventory"]
    for rank in range(8,16):
        binding = device_binding(rank)
        binding["physical_inventory"] = dict(inventory)
        binding["uuid"] = "GPU-"+inventory[str(rank % 8)]
        packet = produce(output, rank=rank, binding_override=binding)
        common["rank_bindings"][rank] = {k:packet.expected[k] for k in ("binding", "expected_pid", "namespace_pids", "construction_versions")}
    module = importlib.import_module("distillschedule.evidence.livewan_c51_phases")
    with pytest.raises(ValueError):
        module.read_phase_receipts(output, **common)


def test_consumer_import_is_torch_free_even_when_torch_import_is_forbidden():
    source = Path(__file__).resolve().parents[1]/"src"
    code = '''import builtins,sys
sys.path.insert(0,sys.argv[1])
original=builtins.__import__
def no_torch(name,*args,**kwargs):
 if name.split('.')[0]=='torch':raise RuntimeError('TORCH_IMPORT_FORBIDDEN')
 return original(name,*args,**kwargs)
builtins.__import__=no_torch
import distillschedule.evidence.livewan_c51_phases
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable,"-B","-I","-c",code,str(source)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
