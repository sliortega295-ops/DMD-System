"""Default-off evidence for two iterations of the existing native H3 loop.

The denominator is rank-local current-stream GPU completion elapsed around
Student1 + Fake5, excluding setup and between-cycle correctness/state probes.
It includes dependencies and host launch gaps *within* that envelope; it is
neither CPU wall time, isolated service time, nor a formal speedup denominator.
No sampler, RNG reset, model call, or collective lives here. The explicit
host_fenced option adds only four device fences at cycle boundaries. Loss/state
and RNG host reads may wait, strictly outside the timed envelopes.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

from distillschedule.evidence.trajectory_state_sketch import (
    canonical_state_name,
    deterministic_flat_indices,
)


TIMING_SCOPE = "GPU_COMPLETION_ELAPSED_STUDENT1_FAKE5_EXCLUDING_SETUP_AND_BETWEEN_CYCLE_PROBES"
_STAGES = ("h3/student_step", "h3/critic_prepare_5xG", *("h3/critic_update_F",) * 5)
_GRID_COUNTS = dict(checkpoint_wrap_count=300, scoped_execution_count=600,
                    cache_hit_count=600, cache_miss_count=0, missing_key_count=0)
_SOURCE_ENVIRONMENT_KEYS = (
    "H3_PLANNER_TWO_CYCLE", "H3_PLANNER_CALIBRATION", "H3_BENCHMARK_SEED", "H3_ATTN_BACKEND",
    "H3_MAX_ITERS", "H3_SAVE_EVERY", "H3_ACTIVATION_POLICY", "H3_ACTIVATION_CHECKPOINT_SEGMENT",
    "H3_ACTIVATION_OFFLOAD", "H3_FA3_NOGRAD_NUM_SPLITS", "H3_FA3_GRAD_NUM_SPLITS",
    "H3_FA3_REPLAY_CACHE_BLOCKS", "H3_FA3_REPLAY_CACHE_STORAGE", "H3_FA3_REPLAY_CACHE_MAX_D2H_INFLIGHT",
    "H3_FA3_REPLAY_CACHE_TRIM_BEFORE_BACKWARD", "H3_ADALN_GRID_MANIFEST", "H3_FLASH_ATTN3_LOCAL_PATH",
    "DISTILLSCHEDULE_PYTHON", "LIGHTX2V_ROOT", "EXPECTED_INTEGRATION_HEAD",
)


def planner_two_cycle_enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    enabled = env.get("H3_PLANNER_TWO_CYCLE", "0") == "1"
    if env.get("H3_PLANNER_CALIBRATION", "0") == "1" and not enabled:
        raise ValueError("H3_PLANNER_CALIBRATION=1 requires H3_PLANNER_TWO_CYCLE=1")
    return enabled


def sample_memory(torch, device) -> dict[str, Any]:
    """Boundary samples only; cgroup counters/peaks are shared and not reset."""
    result = {"scope": "BOUNDARY_SAMPLES_NOT_CONTINUOUS_PEAK_PROOF", "hbm": {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }}
    free, total = torch.cuda.mem_get_info(device)
    result["hbm"].update(driver_free_bytes=int(free), driver_total_bytes=int(total))
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
        result["process"] = {key: int(fields[key].split()[0]) * 1024 for key in ("VmRSS", "VmHWM")}
    except (OSError, KeyError, ValueError) as error:
        result["process"] = {"status": "UNAVAILABLE", "reason": str(error)}
    try:
        relative = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                        if line.startswith("0::"))
        mount = Path("/sys/fs/cgroup")
        directory = (mount / relative.lstrip("/")).resolve()
        if not directory.is_relative_to(mount):
            raise ValueError("cgroup path is outside the mounted hierarchy")
        result["cgroup"] = {"path": str(directory), "scope": "SHARED_CGROUP_LIFETIME_COUNTERS"}
        for name in ("memory.current", "memory.peak", "memory.max", "memory.events"):
            try:
                raw = (directory / name).read_text().strip()
                result["cgroup"][name] = (
                    {k: int(v) for k, v in (line.split() for line in raw.splitlines())}
                    if name == "memory.events" else (raw if raw == "max" else int(raw))
                )
            except (OSError, ValueError) as error:
                result["cgroup"][name] = {"status": "UNAVAILABLE", "reason": str(error)}
    except (OSError, StopIteration, ValueError) as error:
        result["cgroup"] = {"status": "UNAVAILABLE", "reason": str(error)}
    return result


def capture_shard_fingerprints(trainer, *, torch, rank: int,
                               max_tensors: int = 32, samples_per_tensor: int = 16) -> dict:
    """Bounded local samples, never state_dict/full_tensor/all_gather/contiguous.

    Reuse the trajectory sketch's deterministic naming/coordinate APIs without
    its distributed reconstruction: these fingerprints are comparable only on
    the same rank and local layout, NOT full-state or cross-reference parity.
    """
    names = {id(value): canonical_state_name(name)
             for name, value in trainer.shared_model.denoiser_module().named_parameters()}
    components = {}
    for role, parameters, optimizer in (
        ("student", trainer.trainable_params, trainer.optimizer),
        ("fake", trainer.fake_trainable_params, trainer.fake_optimizer),
    ):
        parameter_values, optimizer_values = [], []
        for value in parameters:
            if id(value) not in names:
                raise RuntimeError(f"H3 {role} trainable parameter lacks a native name")
            name = names[id(value)]
            parameter_values.append((name, value))
            for key, stored in sorted(optimizer.state.get(value, {}).items()):
                optimizer_values.append((f"{name}/{key}", stored))
        for kind, values in (("parameters", parameter_values), ("optimizer", optimizer_values)):
            values.sort(key=lambda row: row[0])
            selected = deterministic_flat_indices(f"{role}/{kind}/inventory", len(values), count=max_tensors)
            rows = []
            for index in selected:
                name, value = values[index]
                if not torch.is_tensor(value):
                    if not isinstance(value, (float, int)):
                        raise RuntimeError(f"unsupported H3 optimizer state: {name}")
                    row = {"name": name, "values_fp32": [float(value)], "local_flat_indices": [0],
                           "representation": "PYTHON_SCALAR"}
                else:
                    local = value.detach().to_local() if hasattr(value, "to_local") else value.detach()
                    if not local.is_contiguous():
                        raise RuntimeError(f"noncontiguous H3 state cannot be sampled without a full copy: {name}")
                    indices = deterministic_flat_indices(name, int(local.numel()), count=samples_per_tensor)
                    selected_values = local.view(-1).index_select(
                        0, torch.tensor(indices, dtype=torch.int64, device=local.device))
                    row = {"name": name, "local_shape": list(local.shape), "stored_shape": list(value.shape),
                           "dtype": str(local.dtype), "local_flat_indices": list(indices),
                           "values_fp32": selected_values.float().cpu().tolist(),
                           "representation": "DTENSOR_LOCAL_SHARD" if hasattr(value, "to_local") else "LOCAL_TENSOR"}
                if not all(math.isfinite(v) for v in row["values_fp32"]):
                    raise RuntimeError(f"non-finite H3 state samples: {name}")
                rows.append(row)
            if not values:
                raise RuntimeError(f"H3 {role} {kind} state inventory is empty")
            components[f"{role}_{kind}"] = {
                "inventory_tensor_count": len(values), "sampled_tensor_count": len(rows), "rows": rows,
                "sample_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                "inventory_sha256": hashlib.sha256(json.dumps([name for name, _ in values]).encode()).hexdigest(),
            }
    return {"schema_version": "distillschedule.h3_local_shard_fingerprint.v1", "rank": rank,
            "status": "CAPTURED_FINITE_SAMPLES", "scope": "SAME_RANK_LOCAL_LAYOUT_SAMPLED_COORDINATES_ONLY",
            "max_tensors_per_component": max_tensors, "samples_per_tensor": samples_per_tensor,
            "full_state_finite_check": "NOT_RUN", "numerical_cross_reference": "NOT_RUN",
            "components": components}


class H3PlannerCycleObserver:
    def __init__(self, *, torch, rank: int, source_head: str, calibration: bool = False,
                 memory_probe=sample_memory, state_probe=capture_shard_fingerprints,
                 host_fenced_clock: bool = False):
        self.torch, self.rank, self.source_head = torch, rank, source_head
        self.calibration_enabled = calibration
        self.memory_probe, self.state_probe = memory_probe, state_probe
        self.updates = dict(student=0, fake=0, ema=0)
        self.cycles, self.events = [], []
        self.active = None
        self.collector = None
        self._original_nvtx = None
        self.finished = False
        self.host_fenced_clock = host_fenced_clock
        self.divergence = None
        self.action_trace = None
        self.action_program = None
        self.run_kind = os.environ.get("DISTILLSCHEDULE_RUN_KIND", "correctness")

    def validate_start(self, trainer, *, current_iter: int, resume_ckpt_path) -> None:
        if current_iter != 0 or resume_ckpt_path is not None:
            raise ValueError("H3 two-cycle observer requires start_iter=0 without a checkpoint")
        from distillschedule.search.h3_implementation import current_implementation
        self.implementation = current_implementation()
        if self.implementation is None and os.environ.get("H3_PLANNER_ROLLOUT_WINDOW"):
            raise ValueError("H3 rollout window requires generated action identity")
        self.implementation_binding = {}
        if self.implementation is not None:
            self.implementation_binding = self.implementation.receipt_binding(
                trainer.config, os.environ, mode=os.environ.get("DISTILLSCHEDULE_RUN_KIND", "correctness"))
        expected_depth = self.implementation.prefetch_depth if self.implementation is not None else 2
        expected_storage = self.implementation.storage if self.implementation is not None else "cpu_staged"
        expected_trim = self.implementation.trim_before_backward if self.implementation is not None else True
        frozen = dict(max_train_iters=2, save_every_iters=0, auto_resume=False,
                      activation_policy="none", activation_checkpoint_segment_size=1,
                      gradient_checkpointing=True, activation_offload_enabled=False,
                      adaln_grid_enabled=True, adaln_grid_size=1000,
                      fa3_replay_cache_storage=expected_storage, fa3_replay_cache_max_d2h_inflight=expected_depth,
                      fa3_replay_cache_trim_before_backward=expected_trim,
                      fa3_nograd_num_splits=2, fa3_grad_num_splits=2,
                      matched_compute_enabled=True, fake_update_ratio=5,
                      gradient_accumulation_iters=1, reorder_critic_rollouts=True)
        for key, expected in frozen.items():
            if getattr(trainer, key, None) != expected:
                raise ValueError(f"H3 two-cycle frozen contract requires {key}={expected!r}")
        if tuple(trainer.fa3_replay_cache_blocks) != tuple(range(50)):
            raise ValueError("H3 two-cycle frozen contract requires FA3 replay blocks 0-49")
        if trainer.shared_model.config["model"].get("attention_backend") != "_flash_3_hub":
            raise ValueError("H3 two-cycle frozen contract requires the current FA3 attention backend")
        if self.host_fenced_clock:
            if trainer.matched_expected_world_size != 16 or self.torch.distributed.get_world_size() != 16:
                raise ValueError("H3 native planner requires actual and configured world16")
            if (trainer.config["training"].get("seed") != 20260817
                    or os.environ.get("H3_BENCHMARK_SEED") != "20260817"):
                raise ValueError("H3 native planner seed must be 20260817")
        self.frozen_contract = frozen | {"fa3_replay_cache_blocks": list(range(50)),
                                         "attention_backend": "_flash_3_hub"}
        from .divergence_probe import from_environment
        self.divergence = from_environment(torch=self.torch, rank=self.rank, source_head=self.source_head)

    def after_setup(self, trainer) -> None:
        if not self.torch.cuda.is_available():
            raise RuntimeError("H3 two-cycle timing requires native CUDA events")
        registration = getattr(trainer, "fa3_replay_cache_registration", None)
        if (registration is None or not registration.enabled
                or registration.storage != self.frozen_contract["fa3_replay_cache_storage"]
                or tuple(registration.block_indices) != tuple(range(50))
                or registration.max_d2h_inflight != self.frozen_contract["fa3_replay_cache_max_d2h_inflight"]
                or registration.trim_before_backward != self.frozen_contract["fa3_replay_cache_trim_before_backward"]):
            raise ValueError("H3 two-cycle native FA3 cache registration changed")
        if getattr(trainer, "grid_replay_registration", None) is None:
            raise ValueError("H3 two-cycle requires the native Grid-1000 replay registration")
        self.device = trainer.shared_model.device
        self.events = [(self.torch.cuda.Event(enable_timing=True, blocking=False),
                        self.torch.cuda.Event(enable_timing=True, blocking=False)) for _ in range(2)]
        if self.implementation is not None and self.implementation.rollout_window is not None:
            from .cycle_actions import H3ActionTrace, torch_rng_snapshot
            self.action_trace = H3ActionTrace(rank=self.rank, source_head=self.source_head,
                rng_snapshot=lambda: torch_rng_snapshot(self.torch, self.device),
                value_mode="full" if self.run_kind == "correctness" else "metadata")
            self.action_trace.install_draw_hooks(trainer, self.torch)
            if self.calibration_enabled:
                from distillschedule.search.h3_action_calibration import H3ActionCalibrator
                self.collector = H3ActionCalibrator(torch=self.torch, distributed=self.torch.distributed,
                    device=self.device, rank=self.rank, source_head=self.source_head,
                    implementation=self.implementation)
                self.collector.install(trainer)
        if self.calibration_enabled and self.action_trace is None:
            from distillschedule.search.async_range_calibration import AsyncRangeCollector
            self.collector = AsyncRangeCollector(torch=self.torch, device=self.device, rank=self.rank,
                source_head=self.source_head, implementation=(self.implementation.id if self.implementation is not None
                                                            else "minimax_h3_native_student1_fake5"))
            original = trainer._nvtx
            self._original_nvtx = original

            @contextmanager
            def observed_nvtx(label):
                if label in _STAGES:
                    with self.collector.range(label), original(label):
                        yield
                else:
                    with original(label):
                        yield

            trainer._nvtx = observed_nvtx
        if self.divergence is not None:
            self.divergence.initial(trainer, self._rng_fingerprints())

    def before_cycle(self, trainer, current_iter: int) -> None:
        if self.active is not None or current_iter != len(self.cycles) or current_iter not in (0, 1):
            raise RuntimeError("H3 observer cycle sequence changed")
        self.pending = dict(cycle=current_iter + 1, updates_start=dict(self.updates),
                            grid_start=dict(trainer.grid_replay_registration.stats),
                            memory_before=self.memory_probe(self.torch, self.device))
        if self.host_fenced_clock:
            self.pending["rng_before"] = self._rng_fingerprints()
        if self.divergence is not None:
            self.divergence.begin_cycle(current_iter + 1, self.pending["rng_before"])
        if self.collector is not None:
            self.collector.cycle = current_iter + 1
        if self.action_trace is not None:
            self.action_program = None
            self.action_trace.begin_cycle(current_iter + 1)

    def prepare_actions(self, trainer, samples, grad_accum_iters, current_iter):
        if self.action_trace is None:
            return None
        from .cycle_actions import prepare_cycle
        if self.active is not None:
            raise RuntimeError("H3 plan construction belongs outside the cycle timer")
        self.action_program = prepare_cycle(trainer, samples=samples, grad_accum_iters=grad_accum_iters,
            current_iter=current_iter, rollout_window=self.implementation.rollout_window, trace=self.action_trace)
        return self.action_program

    def start_cycle(self, current_iter: int) -> None:
        if (self.active is not None or current_iter != len(self.cycles)
                or getattr(self, "pending", {}).get("cycle") != current_iter + 1):
            raise RuntimeError("H3 observer start does not match prepared cycle")
        if self.host_fenced_clock:
            # This optional native benchmark fence is common to every H3
            # planner arm. It is not a layer/operation instrumentation fence.
            self.torch.cuda.synchronize(self.device)
            self.pending["host_start_monotonic_ns"] = time.perf_counter_ns()
        self.active = current_iter
        self.events[current_iter][0].record(self.torch.cuda.current_stream(self.device))

    def stop_cycle(self, current_iter: int) -> None:
        if self.active != current_iter:
            raise RuntimeError("H3 observer stop does not match active cycle")
        self.events[current_iter][1].record(self.torch.cuda.current_stream(self.device))
        if self.host_fenced_clock:
            self.torch.cuda.synchronize(self.device)
            self.pending["host_stop_monotonic_ns"] = time.perf_counter_ns()
            self.pending["native_wall_seconds"] = (
                self.pending["host_stop_monotonic_ns"] - self.pending["host_start_monotonic_ns"]
            ) / 1e9
        self.active = None

    def note_update(self, role: str) -> None:
        if self.active is None:
            raise RuntimeError("H3 optimizer update was observed outside native cycle compute")
        self.updates[role] += 1

    def _rng_fingerprints(self):
        # Default Torch generators only, not Python/NumPy, data-loader workers
        # or user-created generators. This is a scoped diagnostic binding.
        return {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in
                (("torch_cpu", self.torch.get_rng_state()),
                 ("torch_cuda", self.torch.cuda.get_rng_state(self.device)))}

    def after_cycle(self, trainer, current_iter: int, dmd_metrics, fake_metrics) -> None:
        if self.active is not None:
            raise RuntimeError("H3 correctness probes must stay outside cycle timing")
        row = self.pending
        delta = {key: self.updates[key] - row["updates_start"][key] for key in self.updates}
        if delta != dict(student=1, fake=5, ema=0):
            raise RuntimeError(f"H3 actual optimizer update count mismatch: {delta}")
        grid = dict(trainer.grid_replay_registration.stats)
        grid_delta = {key: int(grid[key]) - int(row["grid_start"].get(key, 0)) for key in _GRID_COUNTS}
        if grid_delta != _GRID_COUNTS:
            raise RuntimeError(f"H3 cycle Grid-1000 replay mismatch: {grid_delta}")
        losses = {"student": [float(value.item()) for value in dmd_metrics],
                  "fake": [float(value.item()) for value in fake_metrics]}
        if len(losses["student"]) != 1 or len(losses["fake"]) != 5:
            raise RuntimeError("H3 native cycle loss inventory changed")
        if not all(math.isfinite(value) for values in losses.values() for value in values):
            raise RuntimeError("non-finite H3 native cycle loss")
        row.update(status="PASS_NATIVE_RUNTIME_CONTRACTS", updates=delta, cumulative_updates=dict(self.updates),
                   census=trainer.matched_cycle_census.snapshot(), grid_replay=grid, grid_replay_delta=grid_delta,
                   losses=losses, finite_losses=True, memory_after=self.memory_probe(self.torch, self.device))
        if self.implementation is not None and self.implementation.cache_policy != "legacy_cpu_staged":
            from distillschedule.search.h3_implementation import validate_cache_registration
            actual = trainer.fa3_replay_cache_registration.receipt()
            validate_cache_registration(actual, self.implementation)
            row["fa3_replay_cache_registration"] = actual
        if self.host_fenced_clock:
            row["rng_after"] = self._rng_fingerprints()
        if self.divergence is not None:
            if getattr(self.divergence, "full_state_enabled", False):
                self.divergence.after_cycle(current_iter + 1, row["rng_after"], trainer=trainer)
            else:
                self.divergence.after_cycle(current_iter + 1, row["rng_after"])
        if self.action_trace is not None:
            row["action_execution"] = self.action_trace.finish_cycle(
                self.action_program.receipt(), self.updates, row["census"])
        row.pop("updates_start")
        row.pop("grid_start")
        self.cycles.append(row)

    def finish_after_sync(self, trainer) -> dict:
        if self.finished or self.active is not None or len(self.cycles) != 2:
            raise RuntimeError("H3 two-cycle observer is unfinished or already resolved")
        if self.updates != dict(student=2, fake=10, ema=0):
            raise RuntimeError("H3 final actual update count must be G2/C10/EMA0")
        for row, (start, stop) in zip(self.cycles, self.events, strict=True):
            if not (start.query() and stop.query()):
                raise RuntimeError("H3 native final synchronization has not completed")
            elapsed = float(start.elapsed_time(stop)) / 1000
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("H3 cycle CUDA completion elapsed must be finite and positive")
            row["gpu_completion_elapsed_s"] = elapsed
        calibration = self.collector.finish() if self.collector is not None else None
        if calibration is not None and self.action_trace is None:
            for cycle in (1, 2):
                labels = tuple(r["label"] for r in calibration["rows"] if r["cycle"] == cycle)
                if labels != _STAGES:
                    raise RuntimeError(f"H3 calibration expected seven native stages at cycle {cycle}")
        final_state = self.state_probe(trainer, torch=self.torch, rank=self.rank)
        diagnostic = self.divergence.finish() if self.divergence is not None else None
        if diagnostic is not None:
            for row in self.cycles:
                for field in ("native_wall_seconds", "gpu_completion_elapsed_s",
                              "host_start_monotonic_ns", "host_stop_monotonic_ns"):
                    row.pop(field, None)
        self.finished = True
        rng_binding = {}
        if self.host_fenced_clock:
            rng_binding = {"rng_source_binding": dict(integration_head=self.source_head, rank=self.rank,
                initial_sha256=hashlib.sha256(json.dumps(self.cycles[0]["rng_before"], sort_keys=True).encode()).hexdigest(),
                final_sha256=hashlib.sha256(json.dumps(self.cycles[-1]["rng_after"], sort_keys=True).encode()).hexdigest(),
                scope="NATIVE_RNG_STATE_BEFORE_AND_AFTER_TWO_CYCLES")}
        return {"schema_version": "distillschedule.h3_native_two_cycle.v1", **self.implementation_binding, **rng_binding,
                **({"divergence_diagnostic": diagnostic} if diagnostic is not None else {}),
                "status": "PASS_NATIVE_RUNTIME_CONTRACTS", "rank": self.rank,
                "integration_head": self.source_head, "semantics": "approximate_grid1000",
                "cycles": self.cycles, "updates": dict(self.updates),
                "update_count_scope": "SUCCESSFULLY_RETURNED_NATIVE_OPTIMIZER_STEP_CALLS",
                "final_versions": {"generator": self.updates["student"], "critic": self.updates["fake"], "ema": 0},
                "ema_evidence": "NO_EMA_UPDATE_SITE_IN_NATIVE_H3_LOOP",
                "timing_scope": TIMING_SCOPE, "clock": "RANK_LOCAL_CUDA_CURRENT_STREAM",
                "native_wall_clock": "HOST_MONOTONIC_CUDA_FENCED" if self.host_fenced_clock else "NOT_RUN",
                "native_boundary_fences": 4 if self.host_fenced_clock else 0,
                "cross_rank_clock_alignment": "NOT_ESTABLISHED",
                "formal_timing_eligible": False, "calibration_only": self.calibration_enabled, "run_kind": self.run_kind,
                "numerical_cross_reference": "NOT_RUN", "state_fingerprints": final_state,
                "rng_coverage": "TORCH_DEFAULT_CPU_AND_RANK_CUDA_ONLY; OTHER_GENERATORS_NOT_OBSERVED",
                "stage_calibration": calibration,
                "extra_explicit_synchronization_calls": (4 if self.host_fenced_clock else 0) +
                    (45 if self.calibration_enabled and self.action_trace is not None else 0),
                "extra_collectives": 23 if self.calibration_enabled and self.action_trace is not None else 0,
                "probe_host_reads": ("ACTION_CORRECTNESS_FULL_DRAW_HASHES_MAY_SYNC_INSIDE_CYCLE_NOT_TIMING"
                                     if self.action_trace is not None and self.action_trace.value_mode == "full"
                                     else "TENSOR_VALUES_OUTSIDE_TIMING; ACTION_RNG_AND_METADATA_ONLY"),
                "boundary_policy": trainer.activation_policy,
                "fa3_replay_storage": trainer.fa3_replay_cache_storage,
                "frozen_contract": self.frozen_contract,
                "runtime_identity": {"python_executable": sys.executable, "python_prefix": sys.prefix,
                    "torch_version": getattr(self.torch, "__version__", None),
                    "torch_cuda_version": getattr(getattr(self.torch, "version", None), "cuda", None)},
                "source_environment": {key: value for key, value in os.environ.items()
                    if key in _SOURCE_ENVIRONMENT_KEYS or key.startswith("H3_")
                    or key == "DISTILLSCHEDULE_SEARCH_CANDIDATE_ID"}}

    def close(self, trainer) -> None:
        if self.divergence is not None:
            self.divergence.close()
        if self.calibration_enabled and self.action_trace is not None and self.collector is not None:
            self.collector.close()
        if self.action_trace is not None:
            self.action_trace.close(trainer)
        if self._original_nvtx is not None:
            trainer._nvtx = self._original_nvtx
            self._original_nvtx = None
