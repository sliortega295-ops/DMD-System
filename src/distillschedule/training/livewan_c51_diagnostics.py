"""Explicit v1 host-boundary scalar diagnostics for the existing C51 owner.

No polling thread, tensor/state inspection, synchronization or peak reset.
An observation proves only that this rank reached a host boundary; neither a
complete cycle nor continuous GPU headroom follows from these samples.
"""
import os
from pathlib import Path
import re
import time

from distillschedule.workloads.best_known.registry import fingerprint
from .livewan_reference_assets import require


SCHEMA = "distillschedule.livewan_c51_phase_receipts.v1"
FLAG = "DISTILLSCHEDULE_LIVEWAN_C51_PHASE_RECEIPTS"
MEMORY_FIELDS = frozenset(("allocated_bytes", "reserved_bytes", "max_allocated_bytes", "max_reserved_bytes",
                           "driver_free_bytes", "driver_total_bytes"))
SETUP_PHASES = ("construction_entry", "construction_exit", "setup_ready", "public_bind_exit")
CYCLE_PHASES = ("full5_entry", "full5_exit", "checkpoint_pack_entry", "checkpoint_pack_exit")


def _hash(value, length=64):
    return type(value) is str and re.fullmatch("[0-9a-f]{"+str(length)+"}", value) is not None


def from_environment(continuation, *, rank):
    flag = os.environ.get(FLAG, "0")
    require(flag in {"0", "v1"}, "phase diagnostic flag must be 0 or explicit v1")
    if flag == "0":
        return None  # No source reads, identity probes, scalar calls or writes.
    require(continuation.native_output is None, "phase diagnostic v1 supports training, not output-only export")
    from .livewan_c51_training import source_files
    return PhaseReceipts(identity=continuation.identity, output=continuation.output,
        binding=continuation.device_binding, rank=rank, start_cycle=continuation.start_cycle,
        end_cycle=continuation.end_cycle, current_source_files=source_files(),
        run_id=os.environ.get("DISTILLSCHEDULE_RUN_ID", ""))


class PhaseReceipts:
    """Bounded per-rank atomic history, not an execution/checkpoint owner."""
    def __init__(self, *, identity, output, binding, rank, start_cycle, end_cycle, current_source_files, run_id):
        from .livewan_training import _gpu_uuid
        require(type(rank) is int and 0 <= rank < 16, "phase diagnostic rank is invalid")
        require(type(run_id) is str and re.fullmatch(r"[A-Za-z0-9._-]{1,128}", run_id) is not None
                and run_id.strip("."), "phase diagnostic explicit run identity is missing/invalid")
        require(type(identity) is dict and identity.get("schema_version") == "distillschedule.livewan_c51_continuation.v1"
                and _hash(identity.get("source_head"), 40) and _hash(identity.get("assets_sha256"))
                and _hash(identity.get("input_sha256")) and type(identity.get("source_files")) is dict
                and identity["source_files"] and identity["source_files"] == current_source_files
                and all(type(k) is str and _hash(v) for k, v in identity["source_files"].items()),
                "phase diagnostic complete source/asset/input identity differs")
        require(type(identity.get("horizon")) is int and 2 <= identity["horizon"] <= 100
                and type(start_cycle) is int and type(end_cycle) is int
                and 0 <= start_cycle < end_cycle <= identity["horizon"], "phase diagnostic cycle interval is invalid")
        require(type(binding) is dict and binding.get("rank") == rank and binding.get("node_rank") == rank//8
                and type(binding.get("cuda_device")) is int and binding["cuda_device"] == rank % 8
                and type(binding.get("physical_device")) is int and binding["physical_device"] == rank % 8
                and type(binding.get("physical_inventory")) is dict
                and set(binding["physical_inventory"]) == {str(i) for i in range(8)},
                "phase diagnostic physical rank binding differs")
        inventory = binding["physical_inventory"]
        try:
            valid_uuid = (_gpu_uuid(binding.get("uuid")) == inventory[str(rank % 8)]
                          and all(_gpu_uuid(value) == value for value in inventory.values())
                          and len(set(inventory.values())) == 8)
        except ValueError:
            valid_uuid = False
        require(valid_uuid, "phase diagnostic GPU UUID/inventory differs")
        self.rank, self.start_cycle, self.end_cycle = rank, start_cycle, end_cycle
        self.path = Path(output)/"capacity/phase-receipts"/f"rank-{rank:05d}.json"
        self.identity = dict(run_id=run_id, training_identity_sha256=fingerprint(identity),
            source_head=identity["source_head"], source_files=dict(identity["source_files"]),
            source_files_sha256=fingerprint(identity["source_files"]), assets_sha256=identity["assets_sha256"],
            input_sha256=identity["input_sha256"], rank=rank, node_rank=rank//8, local_rank=rank % 8,
            pid=os.getpid(), gpu_uuid=binding["uuid"], physical_device=binding["physical_device"],
            physical_inventory=dict(inventory), role="generator" if rank < 8 else "score",
            start_cycle=start_cycle, requested_stop_cycle=end_cycle)
        self.observations = []
        self.limit = 4+4*(end_cycle-start_cycle)  # <=404: setup + two Full5/two pack markers per cycle.
        self._setup, self._returned, self._active, self._packing, self._packed = 0, start_cycle, None, None, None

    def descriptor(self):
        return dict(schema_version=SCHEMA, path=str(self.path), diagnostic_only=True,
                    formal_timing_eligible=False, claimable_speedup=False)

    def record(self, *, torch, rank, device, phase, cycle, versions, memory=None):
        from .livewan_c51_state import physical_versions
        from distillschedule.runtime.livewan_crossscale_worker import atomic_json, memory_snapshot
        require(type(rank) is int and rank == self.rank and str(device) == f"cuda:{rank % 8}"
                and os.getpid() == self.identity["pid"], "phase diagnostic rank/device/process binding differs")
        require(phase in SETUP_PHASES+CYCLE_PHASES and type(cycle) is int
                and type(versions) is dict and set(versions) == {"generator", "critic", "ema"}
                and all(type(v) is int and v >= 0 for v in versions.values()),
                "phase diagnostic phase/cycle/physical clock is invalid")
        setup, returned, active, packing, packed = self._setup, self._returned, self._active, self._packing, self._packed
        if phase in SETUP_PHASES:
            require(setup < 4 and phase == SETUP_PHASES[setup] and cycle == self.start_cycle,
                    "phase diagnostic setup ordering/cycle differs")
            if phase == "public_bind_exit":
                require(versions == physical_versions(rank, cycle), "phase diagnostic restored physical clock differs")
            setup += 1
        else:
            require(setup == 4 and self.start_cycle < cycle <= self.end_cycle,
                    "phase diagnostic cycle before setup or outside admitted interval")
            if phase == "full5_entry":
                require(active is None and packing is None and cycle == returned+1
                        and versions == physical_versions(rank, cycle-1), "phase diagnostic Full5 entry/cycle/clock differs")
                active = cycle
            elif phase == "full5_exit":
                require(active == cycle and versions == physical_versions(rank, cycle),
                        "phase diagnostic Full5 exit without matching entry/physical clock")
                active, returned = None, cycle
            elif phase == "checkpoint_pack_entry":
                require(active is None and packing is None and returned == cycle and packed != cycle
                        and versions == physical_versions(rank, cycle), "phase diagnostic packing entry/cycle/clock differs")
                packing = cycle
            else:
                require(packing == cycle and versions == physical_versions(rank, cycle),
                        "phase diagnostic packing exit without matching entry/physical clock")
                packing, packed = None, cycle
        require((memory is not None) == (phase == "setup_ready") and len(self.observations) < self.limit,
                "phase diagnostic setup-memory origin or bounded observation count differs")
        memory = memory_snapshot(torch, device) if memory is None else memory
        require(type(memory) is dict and set(memory) == MEMORY_FIELDS
                and all(type(v) is int and v >= 0 for v in memory.values())
                and memory["driver_free_bytes"] <= memory["driver_total_bytes"],
                "phase diagnostic requires exactly the existing six scalar memory fields")
        global_step = 5*(cycle-1) if phase == "full5_entry" else 5*cycle
        row = dict(sequence=len(self.observations), phase=phase, cycle=cycle, global_step=global_step,
            wave=None, wave_scope="WHOLE_FULL5_BOTH_WAVES_NOT_PER_WAVE_SAMPLING", physical_versions=dict(versions),
            monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(), memory=dict(memory),
            memory_origin="EXISTING_SETUP_MEMORY_NO_REQUERY" if phase == "setup_ready" else "ORIGINAL_MEMORY_SNAPSHOT_HELPER",
            timing_class="DIAGNOSTIC_NONPERFORMANCE", observation="HOST_BOUNDARY_REACHED_NOT_CYCLE_ACCEPTANCE")
        observations = self.observations+[row]
        atomic_json(self.path, dict(schema_version=SCHEMA, status="DIAGNOSTIC_ONLY_PARTIAL_HISTORY",
            identity=self.identity, observations=observations, observation_limit=self.limit,
            peak_scope="ORIGINAL_GLOBAL_COUNTERS_NO_PHASE_RESETS", unsynchronized_host_samples=True,
            diagnostic_only=True, formal_timing_eligible=False, claimable_speedup=False,
            native_cuda_numerical_qualification="NOT_RUN", capacity_qualification="NOT_RUN"))
        # A failed/partial write cannot advance the in-memory history either.
        self.observations = observations
        self._setup, self._returned, self._active, self._packing, self._packed = setup, returned, active, packing, packed
