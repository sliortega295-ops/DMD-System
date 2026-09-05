"""Real world16 worker for the pinned LiveWan cross-scale SF-DMD workload.

The implementation intentionally imports Torch, Wan2.1, and LiveWan lazily so
plan validation and unit tests remain CPU-only.  Both placements call the same
math helpers; only placement, rank routing, transport, and legal overlap differ.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from distillschedule.evidence.trajectory_coarse import (
    AppendOnlyTrajectoryWriter,
    TrajectoryCycleReceipt,
)
from distillschedule.evidence.trajectory_state_sketch import (
    canonical_state_name,
    collect_numeric_state_sketch,
)

from distillschedule.runtime.livewan_grad_reducer import (
    GradReadyAsyncGroupReducer,
)
from distillschedule.workloads.livewan_crossscale import (
    CRITIC_UPDATES,
    FULL5_STEPS,
    GENERATOR_UPDATES,
    GLOBAL_BATCH,
    LIVEWAN_COMMIT,
    LORA_RANK,
    PIPELINE_6G10S_LAYOUT,
    PIPELINE_6G10S_NODE_LOCAL_LAYOUT,
    PIPELINE_6G10S_SCORE_LEADERS,
    PIPELINE_6G10S_SCORE_NODE_GROUPS,
    PIPELINE_8G8S_LAYOUT,
    TRACE_BALANCED_8G8S_SLOT_PAIRS,
    WAN21_COMMIT,
    WORLD_SIZE,
    balanced_slots,
    sha256_file,
    validate_slot_groups,
    validate_slot_pairs,
)


DTYPE_NAME = "torch.bfloat16"
LATENT_SHAPE = (1, 16, 21, 46, 80)
TEXT_SHAPE = (1, 512, 4096)
WORLD_CHOICES = (3, 6, 9)
BLOCK_FRAMES = 3
STEPS_PER_BLOCK = 4
GRAD_BLOCKS = 3
TIME_SCALE = 1000.0
ASYNC_CRITIC_POOL_POLICY = (
    "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_"
    "local_context_critic_async32"
)
TRACE_BALANCED_POOL_POLICY = ASYNC_CRITIC_POOL_POLICY + "_trace_balanced"
PHASE_EXPANDED_CRITIC16_POOL_POLICY = (
    "persistent_8g_8score_teacher_fsdp8_trace_balanced_"
    "phase_expanded_critic16_fsdp16"
)
B32_CONTINUOUS_NAME = "continuous_full5_b32_depth1_systems_v1"
TRAJECTORY_CONTINUOUS_NAME = "trajectory_coarse_c24_b16_depth1_v1"
TWO_CYCLE_TIMING_CONTINUOUS_NAME = "two_cycle_timing_c24_b16_depth1_v1"
TWO_CYCLE_TIMING_CYCLES = 2
TRAJECTORY_ANCHOR_CYCLES = frozenset({1, 10, 25, 50})
TRAJECTORY_ALLOWED_CYCLES = frozenset({1, 2, 5, 50})
TRAJECTORY_CRITIC_B1_ATTRIBUTION_NAME = (
    "c24_critic_b1_two_backward_single_commit_attribution_v1"
)


def _pipeline_8x8_slot_pairs(
    plan: Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    """Resolve the registered physical lane assignment without touching trace rows."""

    policy = str(plan.get("metadata", {}).get("pool_policy", ""))
    if policy in {TRACE_BALANCED_POOL_POLICY, PHASE_EXPANDED_CRITIC16_POOL_POLICY}:
        pairs = validate_slot_pairs(
            plan.get("metadata", {}).get("slot_pairs_by_lane", ())
        )
        _require(
            pairs == TRACE_BALANCED_8G8S_SLOT_PAIRS,
            "trace-balanced slot-pair policy differs from its preregistration",
        )
        return pairs
    return tuple((lane, lane + 8) for lane in range(8))


def _pipeline_8x8_slot_groups(
    plan: Mapping[str, Any],
) -> tuple[tuple[int, ...], ...]:
    """Resolve the exact B16 pair or registered B32 four-slot lane mapping."""

    batch = int(plan["workload"]["shape"]["batch_size"])
    if batch == GLOBAL_BATCH:
        return tuple(_pipeline_8x8_slot_pairs(plan))
    _require(batch == 32, "registered continuous lane mapping supports only B16/B32")
    contract = dict(plan.get("metadata", {}).get("batch32_diagnostic", {}))
    _require(
        contract
        == {
            "enabled": True,
            "name": "matched_b32_source_reuse_systems_v1",
            "source_reuse_within_step": True,
            "unique_source_samples_per_cycle": 80,
            "rank_qualified_samples_per_cycle": 160,
            "training_quality_claimable": False,
        },
        "B32 source-reuse diagnostic contract changed",
    )
    groups = validate_slot_groups(
        plan.get("metadata", {}).get("slot_groups_by_lane", ()),
        batch=32,
        slots_per_lane=4,
    )
    expected = tuple(
        (first, second, first + 16, second + 16)
        for first, second in TRACE_BALANCED_8G8S_SLOT_PAIRS
    )
    _require(groups == expected, "B32 slot groups differ from preregistration")
    return groups


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@contextmanager
def _profile_range(torch: Any, *, run_kind: str, label: str):
    """Add profile-only NVTX without changing the formal timing path."""
    if run_kind == "profile":
        with torch.cuda.nvtx.range(label):
            yield
    else:
        yield


def _canonical_without_sha(value: Mapping[str, Any], key: str) -> bytes:
    copy = dict(value)
    copy.pop(key, None)
    return json.dumps(copy, sort_keys=True, separators=(",", ":")).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class OperationTraceRow:
    logical_step: int
    logical_slot: int
    dataset_index: int
    sample_path: str
    sample_sha256: str | None
    operation_key: str
    world_frames: int
    rope_gap: int
    seeds: Mapping[str, int]
    generator_update: bool
    critic_update: bool

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        total_steps: int = FULL5_STEPS,
        cycle_length: int = FULL5_STEPS,
        global_batch: int = GLOBAL_BATCH,
    ) -> "OperationTraceRow":
        required_seeds = {
            "rollout",
            "generator_dmd_t",
            "generator_dmd_noise",
            "generator_grad_blocks",
            "critic_t",
            "critic_noise",
        }
        seeds = {str(k): int(v) for k, v in dict(value["seeds"]).items()}
        _require(set(seeds) == required_seeds, "operation trace seed streams changed")
        row = cls(
            logical_step=int(value["logical_step"]),
            logical_slot=int(value["logical_slot"]),
            dataset_index=int(value["dataset_index"]),
            sample_path=str(value["sample_path"]),
            sample_sha256=(str(value["sample_sha256"]) if value.get("sample_sha256") else None),
            operation_key=str(value["operation_key"]),
            world_frames=int(value["world_frames"]),
            rope_gap=int(value["rope_gap"]),
            seeds=seeds,
            generator_update=bool(value["generator_update"]),
            critic_update=bool(value["critic_update"]),
        )
        cycle_index = row.logical_step // cycle_length
        cycle_step = row.logical_step % cycle_length
        if total_steps == FULL5_STEPS:
            expected_operation_key = (
                f"step={row.logical_step}/slot={row.logical_slot}"
            )
        else:
            _require(
                int(value.get("cycle_index", -1)) == cycle_index
                and int(value.get("cycle_step", -1)) == cycle_step,
                "continuous trace cycle address mismatch",
            )
            expected_operation_key = (
                f"cycle={cycle_index}/step={cycle_step}/"
                f"global_step={row.logical_step}/slot={row.logical_slot}"
            )
        _require(row.operation_key == expected_operation_key, "operation key mismatch")
        _require(row.world_frames in WORLD_CHOICES, "world choice is outside source contract")
        _require(0 <= row.logical_step < total_steps, "logical step outside trace")
        _require(0 <= row.logical_slot < global_batch, "logical slot outside global batch")
        _require(
            row.generator_update == (cycle_step == 0),
            "G1/C5 per-cycle trace predicate changed",
        )
        _require(row.critic_update, "every Full5 step requires a Critic update")
        return row

    @property
    def cycle_index(self) -> int:
        return self.logical_step // FULL5_STEPS

    @property
    def cycle_step(self) -> int:
        return self.logical_step % FULL5_STEPS

    @property
    def new_blocks(self) -> int:
        return (LATENT_SHAPE[2] - self.world_frames) // BLOCK_FRAMES


@dataclass(frozen=True)
class FormalTrace:
    path: Path
    sha256: str
    prompts_path: Path
    prompts_sha256: str
    rows: tuple[OperationTraceRow, ...]
    global_batch: int = GLOBAL_BATCH
    steps: int = FULL5_STEPS
    cycles: int = 1
    source_reuse_across_cycles: bool = False

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_cycles: int = 1,
        expected_global_batch: int = GLOBAL_BATCH,
    ) -> "FormalTrace":
        resolved = Path(path).expanduser().resolve()
        value = json.loads(resolved.read_text(encoding="utf-8"))
        _require(value.get("livewan_commit") == LIVEWAN_COMMIT, "trace LiveWan revision mismatch")
        _require(value.get("wan21_commit") == WAN21_COMMIT, "trace Wan2.1 revision mismatch")
        _require(int(value.get("world_size", -1)) == WORLD_SIZE, "trace is not world16")
        global_batch = int(value.get("global_batch", -1))
        _require(
            global_batch == expected_global_batch,
            "trace global batch differs from the registered plan",
        )
        _require(global_batch in {GLOBAL_BATCH, 32}, "trace batch is not B16/B32")
        _require(expected_cycles >= 1, "trace cycle count must be positive")
        steps = int(value.get("steps", -1))
        cycles = int(value.get("cycles", 1))
        cycle_length = int(value.get("cycle_length", FULL5_STEPS))
        _require(cycle_length == FULL5_STEPS, "trace cycle length is not Full5")
        _require(cycles == expected_cycles, "trace cycle count differs from plan")
        _require(steps == cycles * FULL5_STEPS, "trace step count is not cycles x Full5")
        source_reuse = bool(value.get("source_reuse_across_cycles", False))
        if cycles == 1:
            _require(not source_reuse, "single-cycle trace unexpectedly declares source reuse")
        else:
            _require(source_reuse, "continuous trace must disclose source reuse")
        recorded = str(value.get("trace_sha256", ""))
        actual = hashlib.sha256(_canonical_without_sha(value, "trace_sha256")).hexdigest()
        _require(recorded == actual, "trace payload SHA mismatch")
        rows = tuple(
            OperationTraceRow.from_mapping(
                row,
                total_steps=steps,
                cycle_length=cycle_length,
                global_batch=global_batch,
            )
            for row in value["rows"]
        )
        pairs = {(row.logical_step, row.logical_slot) for row in rows}
        expected = {
            (step, slot) for step in range(steps) for slot in range(global_batch)
        }
        _require(
            len(rows) == steps * global_batch and pairs == expected,
            "trace is not a steps x global-batch bijection",
        )
        prompts = dict(value["prompts"])
        prompts_path = Path(prompts["path"]).expanduser().resolve()
        _require(prompts_path.is_file(), "trace prompt bundle is missing")
        _require(sha256_file(prompts_path) == prompts["sha256"], "prompt bundle SHA mismatch")
        return cls(
            resolved,
            actual,
            prompts_path,
            str(prompts["sha256"]),
            rows,
            global_batch,
            steps,
            cycles,
            source_reuse,
        )

    def row(self, step: int, slot: int) -> OperationTraceRow:
        return self.rows[step * self.global_batch + slot]


def _validate_trace_balanced_pairing(
    plan: Mapping[str, Any], trace: FormalTrace
) -> None:
    """Fail closed if the registered routing no longer balances this trace."""

    if str(plan.get("metadata", {}).get("pool_policy", "")) not in {
        TRACE_BALANCED_POOL_POLICY,
        PHASE_EXPANDED_CRITIC16_POOL_POLICY,
    }:
        return
    groups = _pipeline_8x8_slot_groups(plan)
    current = (
        tuple((lane, lane + 8) for lane in range(8))
        if trace.global_batch == GLOBAL_BATCH
        else tuple((lane, lane + 8, lane + 16, lane + 24) for lane in range(8))
    )

    def cycle_step_max(
        candidate: Sequence[Sequence[int]], cycle_index: int
    ) -> list[int]:
        offset = cycle_index * FULL5_STEPS
        return [
            max(
                sum(trace.row(offset + step, slot).new_blocks for slot in pair)
                for pair in candidate
            )
            for step in range(FULL5_STEPS)
        ]

    current_by_cycle = [
        cycle_step_max(current, cycle) for cycle in range(trace.cycles)
    ]
    candidate_by_cycle = [
        cycle_step_max(groups, cycle) for cycle in range(trace.cycles)
    ]
    current_max = current_by_cycle[0]
    candidate_max = candidate_by_cycle[0]
    metadata = plan.get("metadata", {})
    _require(
        current_max == list(metadata.get("current_step_max_new_blocks", ())),
        "trace-balanced current-load receipt differs from the immutable trace",
    )
    _require(
        candidate_max == list(metadata.get("candidate_step_max_new_blocks", ())),
        "trace-balanced candidate-load receipt differs from the immutable trace",
    )
    _require(
        sum(candidate_max) < sum(current_max),
        "trace-balanced routing no longer lowers the registered critical load",
    )
    _require(
        all(value == current_max for value in current_by_cycle)
        and all(value == candidate_max for value in candidate_by_cycle),
        "continuous trace changed the registered per-cycle routing load",
    )


@dataclass(frozen=True)
class Bootstrap:
    path: Path
    sha256: str
    value: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path, *, trace: FormalTrace) -> "Bootstrap":
        resolved = Path(path).expanduser().resolve()
        value = json.loads(resolved.read_text(encoding="utf-8"))
        recorded = str(value.get("bootstrap_sha256", ""))
        actual = hashlib.sha256(_canonical_without_sha(value, "bootstrap_sha256")).hexdigest()
        _require(recorded == actual, "bootstrap payload SHA mismatch")
        _require(value.get("kind") == "systems_only_livewan_crossscale_bootstrap", "bootstrap kind mismatch")
        _require(value["trace"]["sha256"] == trace.sha256, "bootstrap and trace differ")
        _require(int(value["critic_lora"]["seed"]) >= 0, "Critic LoRA seed missing")
        return cls(resolved, actual, value)

    @property
    def versions(self) -> dict[str, int]:
        return {key: int(value) for key, value in self.value["initial_versions"].items()}


def validate_source_receipt(bootstrap: Bootstrap, *, repo_root: Path) -> dict[str, Any]:
    item = dict(bootstrap.value["source_receipt"])
    path = Path(item["path"]).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("status") == "PASS", "source receipt is not PASS")
    _require(value.get("livewan_commit") == LIVEWAN_COMMIT, "source receipt LiveWan mismatch")
    _require(value.get("wan21_commit") == WAN21_COMMIT, "source receipt Wan2.1 mismatch")
    _require(value.get("source_receipt_sha256") == item["sha256"], "bootstrap source receipt mismatch")
    actual = hashlib.sha256(_canonical_without_sha(value, "source_receipt_sha256")).hexdigest()
    _require(actual == item["sha256"], "source receipt payload SHA mismatch")
    for row in value["files"].values():
        source_path = Path(row["path"])
        _require(source_path.is_file(), f"source receipt file missing: {source_path}")
        _require(sha256_file(source_path) == row["sha256"], f"source receipt file changed: {source_path}")
    livewan = repo_root / "third_party/livewan"
    _require(livewan.is_dir(), "pinned LiveWan submodule is missing")
    return value


def configure_python_sources(*, repo_root: Path, source_receipt: Mapping[str, Any]) -> None:
    paths = [str(repo_root / "third_party/livewan"), str(source_receipt["source_root"])]
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)


def memory_snapshot(torch: Any, device: Any) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "driver_free_bytes": int(free),
        "driver_total_bytes": int(total),
    }


def host_memory_snapshot() -> dict[str, int | None]:
    rss_pages = int(Path("/proc/self/statm").read_text().split()[1])
    page_size = os.sysconf("SC_PAGE_SIZE")
    result: dict[str, int | None] = {"rss_bytes": rss_pages * page_size}
    for name in ("memory.current", "memory.peak"):
        path = Path("/sys/fs/cgroup") / name
        try:
            result[name.replace(".", "_") + "_bytes"] = int(path.read_text().strip())
        except (FileNotFoundError, ValueError):
            result[name.replace(".", "_") + "_bytes"] = None
    return result


def cgroup_oom_counts() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.is_file():
        return {}
    rows: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        if key in {"oom", "oom_kill"}:
            rows[key] = int(value)
    return rows


@dataclass
class CallInventory:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, role: str, grad_enabled: bool, grad_checkpoint: bool) -> None:
        kind = "grad" if grad_enabled else "nograd"
        replay_capable = "checkpoint" if grad_checkpoint else "plain"
        for key in (role, f"{role}/{kind}", f"{role}/{kind}/{replay_capable}"):
            self.counts[key] = self.counts.get(key, 0) + 1

    @contextmanager
    def install(self, *, torch: Any, blockcausal: Any, roles: Mapping[int, str]):
        original = blockcausal.block_forward

        def wrapped(model: Any, *args: Any, **kwargs: Any):
            role = roles.get(id(model), "unknown")
            self.add(role, torch.is_grad_enabled(), bool(kwargs.get("grad_checkpoint", False)))
            return original(model, *args, **kwargs)

        blockcausal.block_forward = wrapped
        try:
            yield
        finally:
            blockcausal.block_forward = original


def _load_checkpoint(load_file: Any, root: Path) -> dict[str, Any]:
    one = root / "diffusion_pytorch_model.safetensors"
    if one.is_file():
        return load_file(str(one))
    result: dict[str, Any] = {}
    files = sorted(root.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    _require(bool(files), f"no Wan transformer weights under {root}")
    for path in files:
        part = load_file(str(path))
        overlap = set(result) & set(part)
        _require(not overlap, f"duplicate checkpoint keys in {path}")
        result.update(part)
    return result


def _wan_model(WanModel: Any, *, size: str, config: Any) -> Any:
    if size == "1.3B":
        return WanModel(
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            window_size=config.window_size,
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
        )
    _require(size == "14B", f"unsupported Wan size {size}")
    return WanModel(
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        num_heads=40,
        num_layers=40,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        text_len=512,
    )


def _load_student_state(torch: Any, model: Any, bootstrap: Bootstrap) -> dict[str, Any]:
    spec = bootstrap.value.get("student_state")
    if spec is None:
        return {"kind": "base_weights", "sha256": bootstrap.value["student_base"]["sha256"]}
    path = Path(spec["path"])
    _require(sha256_file(path) == spec["sha256"], "Student state SHA mismatch")
    # The pinned Stage-1 file is a complete zip-format state.  mmap keeps its
    # immutable storage file-backed while load_state_dict copies into the model;
    # it does not change any tensor value.
    blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = blob.get("model", blob)
    model.load_state_dict(state, strict=True)
    return {"kind": "stage1_state", "sha256": spec["sha256"]}


def _configure_fsdp_reduction(model: Any, *, group_size: int) -> int:
    count = 0
    for module in model.modules():
        if hasattr(module, "set_force_sum_reduction_for_comms"):
            module.set_force_sum_reduction_for_comms(True)
            module.set_gradient_divide_factor(float(group_size))
            count += 1
    _require(count > 0, "FSDP2 model has no configured shard groups")
    return count


def materialize_adam_state(torch: Any, optimizer: Any, *, step: int = 0) -> dict[str, int]:
    optimizer.state.clear()
    tensors = 0
    local_numel = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            # Fused AdamW requires its scalar step on the same device as the
            # CUDA parameters.  The reference foreach implementation keeps the
            # scalar on CPU.  This only materializes the same initial state in
            # the representation required by each PyTorch kernel.
            step_device = parameter.device if group.get("fused") else "cpu"
            state["step"] = torch.tensor(
                float(step), dtype=torch.float32, device=step_device
            )
            state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            tensors += 1
            local_numel += int(local.numel())
    return {"tensor_count": tensors, "local_numel_each_moment": local_numel, "step": step}


def _state_metrics(torch: Any, parameters: Sequence[Any]) -> dict[str, float | int]:
    """Return placement-invariant aggregate tensor statistics.

    FSDP2 may leave a full unsharded parameter buffer resident after a forward
    even though the optimizer owns a sharded DTensor.  Reducing ``to_local()``
    statistics across the mesh in that state counts every full tensor once per
    rank.  ``full_tensor()`` gives one canonical logical value on every owner;
    replicated tensors are already canonical and need no collective reduction.
    This helper is correctness-only and is never entered by a formal timing run.
    """
    tensors = [value for value in parameters if value is not None]
    _require(bool(tensors), "state metrics require at least one tensor")
    device = tensors[0].device
    values = torch.zeros(5, device=device, dtype=torch.float64)
    count = torch.zeros(1, device=device, dtype=torch.int64)
    for index, parameter in enumerate(tensors):
        logical = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
        value = logical.detach().double()
        values[0] += value.sum()
        values[1] += value.square().sum()
        values[2] = torch.maximum(values[2], value.abs().max())
        values[3] += value.sum() * float((index % 97) + 1)
        values[4] += value.square().sum() * float((index % 89) + 1)
        count += value.numel()
    cpu = values.cpu().tolist()
    return {
        "numel": int(count.cpu()),
        "sum": cpu[0],
        "sum_sq": cpu[1],
        "max_abs": cpu[2],
        "weighted_sum": cpu[3],
        "weighted_sum_sq": cpu[4],
    }


@dataclass
class ModelBundle:
    student: Any | None
    critic: Any | None
    critic_shadow: Any | None
    teacher: Any | None
    generator_optimizer: Any | None
    critic_optimizer: Any | None
    generator_optimizer_parameter_names: dict[int, str]
    critic_optimizer_parameter_names: dict[int, str]
    critic_lora_params: list[Any]
    critic_shadow_lora_params: list[Any]
    critic_gradient_reducer: GradReadyAsyncGroupReducer | None
    generator_group: Any | None
    score_group: Any | None
    critic_group: Any | None
    score_local_group: Any | None
    score_leader_group: Any | None
    runahead_host_group: Any | None
    critic_shadow_sync_group: Any | None
    score_local_ranks: tuple[int, ...]
    score_leader_ranks: tuple[int, ...]
    generator_group_size: int
    score_group_size: int
    critic_group_size: int
    metadata: dict[str, Any]


def _parameter_sequence_sha256(torch: Any, parameters: Sequence[Any]) -> str:
    """Hash an ordered CPU parameter sequence before any optimizer mutation."""

    digest = hashlib.sha256()
    count = 0
    for parameter in parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
        count += 1
    _require(count > 0, "parameter identity requires a non-empty sequence")
    return digest.hexdigest()


def _make_mesh(torch: Any, ranks: Sequence[int], group: Any) -> Any:
    from torch.distributed.device_mesh import DeviceMesh
    try:
        return DeviceMesh.from_group(group, "cuda", mesh_dim_names=("fsdp",))
    except (AttributeError, TypeError):
        return DeviceMesh("cuda", list(ranks), mesh_dim_names=("fsdp",))


def _materialize_fsdp_units(model: Any) -> int:
    """Synchronously unshard every composable-FSDP unit in module order.

    This is placement setup, not an extra model forward.  It is needed for the
    phase-expanded Critic because only Score ranks execute the two step-0 DMD
    Critic calls, while all sixteen ranks execute each later trainable Critic
    call.  Keeping the parameters gathered makes those asymmetric no-grad calls
    collective-free without changing their math or call inventory.
    """

    units = [module for module in model.modules() if hasattr(module, "unshard")]
    _require(bool(units), "FSDP materialization found no shard units")
    for module in units:
        module.unshard(async_op=False)
    return len(units)


def _plan_role_ranks(plan: Mapping[str, Any], *, mode: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if mode == "baseline":
        ranks = tuple(range(WORLD_SIZE))
        return ranks, ranks
    groups = {str(row["name"]): tuple(int(rank) for rank in row["ranks"]) for row in plan["rank_groups"]}
    generator_ranks = groups.get("generator", ())
    score_ranks = groups.get("score", ())
    teacher_ranks = groups.get("teacher", ())
    _require(bool(generator_ranks) and bool(score_ranks), "pipeline role ranks are missing")
    _require(teacher_ranks == score_ranks, "Teacher ranks must equal Score ranks")
    _require(not (set(generator_ranks) & set(score_ranks)), "pipeline role pools overlap")
    _require(
        set(generator_ranks) | set(score_ranks) == set(range(WORLD_SIZE)),
        "pipeline role pools do not cover world16",
    )
    policy = plan.get("metadata", {}).get("pool_policy")
    expected = {
        "persistent_8g_8score_teacher_fsdp8": PIPELINE_8G8S_LAYOUT,
        "persistent_8g_8score_teacher_fsdp8_runahead": PIPELINE_8G8S_LAYOUT,
        "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2": PIPELINE_8G8S_LAYOUT,
        "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_local_context": PIPELINE_8G8S_LAYOUT,
        ASYNC_CRITIC_POOL_POLICY: PIPELINE_8G8S_LAYOUT,
        TRACE_BALANCED_POOL_POLICY: PIPELINE_8G8S_LAYOUT,
        PHASE_EXPANDED_CRITIC16_POOL_POLICY: PIPELINE_8G8S_LAYOUT,
        "persistent_8g_8score_teacher_fsdp8_score_b2": PIPELINE_8G8S_LAYOUT,
        "persistent_6g_10score_teacher_fsdp10": PIPELINE_6G10S_LAYOUT,
        "persistent_6g_10score_teacher_node_local_hierarchical": PIPELINE_6G10S_NODE_LOCAL_LAYOUT,
    }.get(policy)
    _require(expected is not None, "unknown pipeline pool policy")
    _require(generator_ranks == expected.generator_ranks, "Generator rank policy mismatch")
    _require(score_ranks == expected.score_ranks, "Score rank policy mismatch")
    return generator_ranks, score_ranks


def _plan_critic_ranks(
    plan: Mapping[str, Any], *, mode: str, score_ranks: Sequence[int]
) -> tuple[int, ...]:
    if mode == "baseline":
        return tuple(range(WORLD_SIZE))
    policy = str(plan.get("metadata", {}).get("pool_policy", ""))
    groups = {
        str(row["name"]): tuple(int(rank) for rank in row["ranks"])
        for row in plan["rank_groups"]
    }
    critic_ranks = groups.get("critic", tuple(score_ranks))
    if policy == PHASE_EXPANDED_CRITIC16_POOL_POLICY:
        _require(
            critic_ranks == tuple(range(WORLD_SIZE)),
            "phase-expanded Critic ranks must cover world16",
        )
    else:
        _require(
            critic_ranks == tuple(score_ranks),
            "legacy pipeline Critic ranks must equal Score ranks",
        )
    return critic_ranks


def _split_dmd_shadow_enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    """Validate the exact C14 split-DMD execution-graph contract."""

    metadata = plan.get("metadata", {})
    contract = metadata.get("split_dmd_shadow")
    if contract is None:
        return False
    _require(mode == "pipeline", "split-DMD shadow is pipeline-only")
    _require(
        metadata.get("pool_policy") == TRACE_BALANCED_POOL_POLICY,
        "split-DMD shadow requires the validated K5b pool substrate",
    )
    _require(
        metadata.get("full_context_lens_elision") is True,
        "split-DMD shadow requires full-context elision",
    )
    expected = {
        "enabled": True,
        "version": "C0",
        "placement": "generator_rank_frozen_replica",
        "scope": "step0_fake_score_only",
        "teacher_velocity_transport": "cuda_p2p_float32",
        "critic_training_placement": "score_ranks_only",
        "source_trainable_sha_matches_score_c0": True,
        "logical_calls_preserved": True,
    }
    _require(dict(contract) == expected, "split-DMD shadow contract changed")
    return True


def _continuous_pipeline_contract(
    plan: Mapping[str, Any], *, mode: str
) -> Mapping[str, Any] | None:
    """Validate the C21 continuous-B16 execution contract."""

    value = plan.get("metadata", {}).get("continuous_pipeline")
    if value is None:
        return None
    _require(mode == "pipeline", "continuous execution is pipeline-only")
    _require(
        plan.get("metadata", {}).get("pool_policy") == TRACE_BALANCED_POOL_POLICY,
        "continuous execution requires the exact C20 pool substrate",
    )
    expected = {
        "enabled": True,
        "name": "continuous_full5_b16_depth1_v1",
        "cycles": 3,
        "cycle_length": FULL5_STEPS,
        "lookahead_depth": 1,
        "lookahead_scope": "next_cycle_step0_wave0_rollout_only",
        "source_reuse_across_cycles": True,
        "fresh_rank_qualified_rng_per_cycle": True,
        "critic_shadow_hot_sync": "paired_score_to_generator_fp32_lora_after_c5",
        "critic_shadow_sync_versions": [5, 10],
        "single_generator_commit_per_cycle": True,
        "five_critic_commits_per_cycle": True,
        "math_and_call_inventory_preserved": True,
    }
    expected_c1 = dict(
        expected,
        name="continuous_full5_b16_depth1_step1_b1_overlap_v1",
        critic_step1_physical_schedule=(
            "wave0_b1_backward_then_wave1_b1_backward_single_commit"
        ),
        critic_step1_loss_weights=[0.5, 0.5],
        critic_step1_physical_calls_per_rank=2,
    )
    expected_b32 = dict(
        expected,
        name=B32_CONTINUOUS_NAME,
        waves_per_step=4,
        critic_physical_microbatches_per_step=2,
        critic_microbatch_size=2,
        critic_loss_scales=[0.5, 0.5],
        source_reuse_within_step=True,
        training_quality_claimable=False,
    )
    if value.get("name") == TRAJECTORY_CONTINUOUS_NAME:
        cycles = int(value.get("cycles", -1))
        _require(
            cycles in TRAJECTORY_ALLOWED_CYCLES,
            "trajectory cycle count must be one of 1, 2, 5, or 50",
        )
        expected_trajectory = dict(
            expected,
            name=TRAJECTORY_CONTINUOUS_NAME,
            cycles=cycles,
            source_reuse_across_cycles=cycles > 1,
            critic_shadow_sync_versions=[
                CRITIC_UPDATES * cycle for cycle in range(1, cycles)
            ],
            trajectory_schema="trajectory_coarse_v1",
            trajectory_diagnostic_only=True,
            formal_timing_eligible=False,
        )
        _require(
            dict(value) == expected_trajectory,
            "trajectory continuous pipeline contract changed",
        )
    elif value.get("name") == TWO_CYCLE_TIMING_CONTINUOUS_NAME:
        expected_timing = dict(
            expected,
            name=TWO_CYCLE_TIMING_CONTINUOUS_NAME,
            cycles=TWO_CYCLE_TIMING_CYCLES,
            critic_shadow_sync_versions=[CRITIC_UPDATES],
            timing_schema="livewan_two_cycle_timing_v1",
            profiled=False,
            formal_timing_eligible=True,
        )
        _require(
            dict(value) == expected_timing,
            "two-cycle timing continuous pipeline contract changed",
        )
    else:
        _require(
            dict(value) in (expected, expected_c1, expected_b32),
            "continuous pipeline contract changed",
        )
    _require(_split_dmd_shadow_enabled(plan, mode=mode), "continuous pipeline needs split-DMD shadow")
    _require(
        _generator_runahead_transport_enabled(plan, mode=mode),
        "continuous pipeline needs the C19 pinned-host transport",
    )
    return value


def _apply_trajectory_contract(
    plan: dict[str, Any], *, cycles: int, mode: str
) -> None:
    """Turn one exact matched arm into a bounded trajectory plan in memory.

    The mutation is diagnostic-only and happens after rendering.  It changes
    neither the checked-in baseline nor C24 plan, and legacy run kinds never
    enter this function.
    """

    _require(
        cycles in TRAJECTORY_ALLOWED_CYCLES,
        "trajectory cycles must be one of 1, 2, 5, or 50",
    )
    metadata = dict(plan.get("metadata", {}))
    if mode == "baseline":
        _require(
            metadata.get("execution_mode") == "baseline"
            and metadata.get("pool_policy") in {None, "world16_fsdp2_sequential"},
            "trajectory baseline must start from the exact sequential plan",
        )
        _require(
            "continuous_pipeline" not in metadata,
            "trajectory baseline unexpectedly contains a pipeline contract",
        )
        metadata["trajectory_diagnostic"] = {
            "schema": "trajectory_coarse_v1",
            "cycles": cycles,
            "schedule": "world16_fsdp2_sequential",
            "source_reuse_across_cycles": cycles > 1,
            "formal_timing_eligible": False,
        }
        plan["metadata"] = metadata
        return
    _require(mode == "pipeline", "unknown trajectory execution mode")
    current = dict(metadata.get("continuous_pipeline", {}))
    _require(
        current.get("name") == "continuous_full5_b16_depth1_v1"
        and int(current.get("cycles", -1)) == 3,
        "trajectory mode must start from the exact C24 continuous-B16 plan",
    )
    score_modulation = dict(metadata.get("score_modulation_kernel", {}))
    _require(
        score_modulation.get("enabled") is True
        and score_modulation.get("name") == "score_bf16_modulation_fusion_v1",
        "trajectory mode requires the exact C24 Score modulation kernel",
    )
    current.update(
        name=TRAJECTORY_CONTINUOUS_NAME,
        cycles=cycles,
        source_reuse_across_cycles=cycles > 1,
        critic_shadow_sync_versions=[
            CRITIC_UPDATES * cycle for cycle in range(1, cycles)
        ],
        trajectory_schema="trajectory_coarse_v1",
        trajectory_diagnostic_only=True,
        formal_timing_eligible=False,
    )
    metadata["continuous_pipeline"] = current
    plan["metadata"] = metadata


def _apply_two_cycle_timing_contract(plan: dict[str, Any], *, mode: str) -> None:
    """Bind the immutable two-cycle trace to the exact sequential/C24 arms.

    This is a timing-only in-memory plan adaptation.  It changes only the
    number of consecutive cycles represented by the immutable trace; the
    checked-in plans and every per-cycle workload operation remain unchanged.
    """

    metadata = dict(plan.get("metadata", {}))
    evidence = dict(plan.get("evidence", {}))
    _require(int(evidence.get("steps", -1)) == FULL5_STEPS, "source plan is not Full5")
    evidence["steps"] = TWO_CYCLE_TIMING_CYCLES * FULL5_STEPS
    plan["evidence"] = evidence
    if mode == "baseline":
        _require(
            metadata.get("execution_mode") == "baseline"
            and metadata.get("pool_policy") in {None, "world16_fsdp2_sequential"},
            "two-cycle timing baseline must start from the exact sequential plan",
        )
        _require(
            "continuous_pipeline" not in metadata,
            "two-cycle timing baseline unexpectedly contains a pipeline contract",
        )
        metadata["two_cycle_timing"] = {
            "schema": "livewan_two_cycle_timing_v1",
            "cycles": TWO_CYCLE_TIMING_CYCLES,
            "schedule": "world16_fsdp2_sequential",
            "source_reuse_across_cycles": True,
            "profiled": False,
            "formal_timing_eligible": True,
        }
        plan["metadata"] = metadata
        return

    _require(mode == "pipeline", "unknown two-cycle timing execution mode")
    current = dict(metadata.get("continuous_pipeline", {}))
    _require(
        current.get("name") == "continuous_full5_b16_depth1_v1"
        and int(current.get("cycles", -1)) == 3,
        "two-cycle timing must start from the exact C24 continuous-B16 plan",
    )
    score_modulation = dict(metadata.get("score_modulation_kernel", {}))
    _require(
        score_modulation.get("enabled") is True
        and score_modulation.get("name") == "score_bf16_modulation_fusion_v1",
        "two-cycle timing requires the exact C24 Score modulation kernel",
    )
    current.update(
        name=TWO_CYCLE_TIMING_CONTINUOUS_NAME,
        cycles=TWO_CYCLE_TIMING_CYCLES,
        critic_shadow_sync_versions=[CRITIC_UPDATES],
        timing_schema="livewan_two_cycle_timing_v1",
        profiled=False,
        formal_timing_eligible=True,
    )
    metadata["continuous_pipeline"] = current
    plan["metadata"] = metadata


def _effective_plan_sha256(plan: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _record_two_cycle_endpoint(
    torch: Any,
    *,
    records: list[dict[str, Any]],
    versions: Mapping[str, int],
    inventory: "CallInventory",
    evidence: "RunEvidence",
) -> None:
    """Enqueue a completion marker without synchronizing the host or world.

    The event is recorded on the current compute stream after the terminal
    optimizer commit.  Any helper stream used by the runtime must already be
    joined by its ``finish``/``wait`` contract before that commit can return.
    Event elapsed times are read only after the single post-cycle2 device sync.
    """

    event = torch.cuda.Event(enable_timing=True)
    event.record(torch.cuda.current_stream())
    logical = dict(inventory.counts)
    logical.update(evidence.logical_score_calls)
    records.append(
        {
            "event": event,
            "host_completion_ns": time.perf_counter_ns(),
            "local_versions": {key: int(value) for key, value in versions.items()},
            "physical_call_inventory": dict(inventory.counts),
            "logical_call_inventory": logical,
        }
    )


def _load_two_cycle_timing_gate(
    *,
    path: str | Path,
    expected_sha256: str,
    expected_head: str,
    mode: str,
    pipeline_variant: str,
    arm_id: str,
    seed: int,
    trace: FormalTrace,
    bootstrap: Bootstrap,
    source_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact-head correctness evidence authorizing native timing."""

    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), "two-cycle timing correctness gate is missing")
    _require(
        len(expected_sha256) == 64 and sha256_file(resolved) == expected_sha256,
        "two-cycle timing correctness gate SHA256 changed",
    )
    value = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "schema_version": "distillschedule.livewan_two_cycle_timing_gate.v2",
        "status": "PASS",
        "claimable_for_two_cycle_timing": True,
        "expected_head": expected_head,
        "cycles": TWO_CYCLE_TIMING_CYCLES,
        "final_versions": {"generator": 2, "critic": 10, "ema": 0},
    }
    for key, expected in required.items():
        _require(value.get(key) == expected, f"two-cycle timing gate changed {key}")
    numerical_review = dict(value.get("numerical_review", {}))
    _require(
        numerical_review.get("policy")
        == "REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW"
        and numerical_review.get("extreme_failure_count") == 0,
        "two-cycle timing gate lost its extreme-drift policy",
    )
    _require(
        set(dict(value.get("heads", {})).values()) == {expected_head},
        "two-cycle timing gate uses a different source HEAD",
    )
    identity = dict(value.get("identity", {}))
    expected_identity = {
        "trace_sha256": trace.sha256,
        "bootstrap_sha256": bootstrap.sha256,
        "source_receipt_sha256": source_receipt["source_receipt_sha256"],
    }
    _require(identity == expected_identity, "two-cycle timing gate asset identity changed")
    _require(int(value.get("seed", -1)) == seed, "two-cycle timing seed changed")
    arm = dict(dict(value.get("arms", {})).get(mode, {}))
    _require(arm.get("arm_id") == arm_id, "two-cycle timing arm identity changed")
    _require(
        arm.get("pipeline_variant") == pipeline_variant,
        "two-cycle timing physical implementation changed",
    )
    _require(
        isinstance(arm.get("trajectory_summary_sha256"), str)
        and len(str(arm["trajectory_summary_sha256"])) == 64,
        "two-cycle timing trajectory summary SHA is missing",
    )
    comparison = dict(value.get("trajectory_comparison", {}))
    _require(
        comparison.get("schema_version")
        == "distillschedule.trajectory_coarse_comparison.v1"
        and comparison.get("contract_status") == "PASS"
        and comparison.get("complete") is True
        and isinstance(comparison.get("sha256"), str)
        and len(str(comparison["sha256"])) == 64,
        "two-cycle cross-arm trajectory comparison is not PASS",
    )
    return {
        "path": str(resolved),
        "sha256": expected_sha256,
        "schema_version": value["schema_version"],
        "trajectory_summary_sha256": arm.get("trajectory_summary_sha256"),
        "trajectory_comparison_sha256": comparison["sha256"],
        "arm_id": arm_id,
        "pipeline_variant": pipeline_variant,
        "status": value["status"],
    }


def _trajectory_critic_batch_ablation_enabled(
    plan: Mapping[str, Any], *, mode: str
) -> bool:
    """Validate the diagnostic-only C24 physical-B1 Critic contract.

    The ordinary C24 plan has no such contract and remains on its existing
    physical-B2 path.  The ablation changes only how each Score rank executes
    its two logical Critic samples: two weighted B1 backwards accumulate into
    the same asynchronous reducer and retain one optimizer commit.
    """

    metadata = plan.get("metadata", {})
    value = metadata.get("trajectory_critic_batch_ablation")
    if value is None:
        return False
    expected = {
        "enabled": True,
        "name": TRAJECTORY_CRITIC_B1_ATTRIBUTION_NAME,
        "scope": "trajectory_diagnostic_only",
        "critic_physical_batch_size": 1,
        "physical_backward_calls_per_step": 2,
        "loss_weights": [0.5, 0.5],
        "reducer_accumulation": [[0, 2], [1, 2]],
        "optimizer_commits_per_step": 1,
        "all_other_c24_execution_unchanged": True,
        "formal_timing_eligible": False,
    }
    _require(mode == "pipeline", "C24 Critic-B1 attribution is pipeline-only")
    _require(dict(value) == expected, "C24 Critic-B1 attribution contract changed")
    _require(
        int(metadata.get("critic_physical_batch_size", -1)) == 1,
        "C24 Critic-B1 attribution did not select physical B1",
    )
    _require(
        dict(metadata.get("score_modulation_kernel", {})).get("name")
        == "score_bf16_modulation_fusion_v1",
        "C24 Critic-B1 attribution lost the exact C24 Score kernel",
    )
    return True


def _critic_checkpoint_rng_elision_enabled(
    plan: Mapping[str, Any], *, mode: str
) -> bool:
    """Validate the exact C22 Critic checkpoint RNG-state contract.

    Wan's block-causal Transformer block contains no stochastic operator.  C22
    therefore removes only PyTorch's per-checkpoint CUDA RNG save/restore on
    Score ranks; Generator and frozen Teacher execution remain untouched.
    """

    contract = plan.get("metadata", {}).get("critic_checkpoint_rng")
    if contract is None:
        return False
    _require(mode == "pipeline", "Critic checkpoint RNG elision is pipeline-only")
    _require(
        _continuous_pipeline_contract(plan, mode=mode) is not None,
        "Critic checkpoint RNG elision requires the C21 continuous substrate",
    )
    expected = {
        "enabled": True,
        "name": "critic_checkpoint_rng_state_elision_v1",
        "scope": "score_rank_critic_grad_checkpoint_only",
        "use_reentrant": False,
        "preserve_rng_state": False,
        "checkpoint_block_stochastic_ops": False,
        "generator_checkpoint_unchanged": True,
        "teacher_checkpoint_unchanged": True,
        "math_and_call_inventory_preserved": True,
    }
    _require(dict(contract) == expected, "Critic checkpoint RNG contract changed")
    return True


class _CheckpointRngElisionProxy:
    """Process-local proxy for ``wanstreamer.blockcausal.ckpt``.

    Replacing the module attribute rather than the global
    ``torch.utils.checkpoint.checkpoint`` function limits the change to the
    pinned block-causal call site.  The proxy is installed only in Score-rank
    processes and fails closed if upstream starts supplying either checkpoint
    option explicitly.
    """

    def __init__(self, checkpoint_module: Any) -> None:
        self._checkpoint_module = checkpoint_module
        self.call_count = 0

    def checkpoint(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        _require(
            "preserve_rng_state" not in kwargs,
            "upstream checkpoint now controls RNG preservation explicitly",
        )
        _require(
            kwargs.get("use_reentrant") is False,
            "Critic checkpoint is no longer non-reentrant",
        )
        self.call_count += 1
        return self._checkpoint_module.checkpoint(
            function,
            *args,
            preserve_rng_state=False,
            **kwargs,
        )

    def receipt(self, *, expected_calls: int) -> dict[str, Any]:
        _require(expected_calls > 0, "Critic checkpoint expected-call count is empty")
        _require(
            self.call_count == expected_calls,
            "Critic checkpoint RNG proxy call inventory changed",
        )
        return {
            "enabled": True,
            "name": "critic_checkpoint_rng_state_elision_v1",
            "role": "score",
            "scope": "score_rank_critic_grad_checkpoint_only",
            "use_reentrant": False,
            "preserve_rng_state": False,
            "checkpoint_calls": self.call_count,
            "expected_checkpoint_calls": expected_calls,
            "execution_status": "PASS",
        }


def _generator_runahead_transport_enabled(
    plan: Mapping[str, Any], *, mode: str
) -> bool:
    """Validate the exact C19 Critic-only latent transport contract."""

    metadata = plan.get("metadata", {})
    contract = metadata.get("generator_runahead_transport")
    if contract is None:
        return False
    _require(mode == "pipeline", "host run-ahead transport is pipeline-only")
    _require(
        metadata.get("pool_policy") == TRACE_BALANCED_POOL_POLICY,
        "host run-ahead transport requires the validated C16/C18 substrate",
    )
    _require(
        metadata.get("generator_nograd_runahead") is True,
        "host transport requires Generator run-ahead",
    )
    _require(
        metadata.get("full_context_lens_elision") is True,
        "host transport requires trace-local Score context replay",
    )
    _require(
        bool(metadata.get("generator_cross_attention_kv_cache", {}).get("enabled")),
        "C19 must inherit the passing C18 Generator K/V cache",
    )
    waves_per_step = (
        4
        if dict(metadata.get("continuous_pipeline", {})).get("name")
        == B32_CONTINUOUS_NAME
        else 2
    )
    expected = {
        "enabled": True,
        "name": "pinned_cpu_gloo_latent_v1",
        "scope": "critic_only_steps_1_4_latent_only",
        "logical_steps": [1, 2, 3, 4],
        "waves_per_step": waves_per_step,
        "payload_dtype": "float32",
        "payload_shape": list(LATENT_SHAPE),
        "pinned_host": True,
        "background_sender_threads_per_rank": 1,
        "preposted_receives": True,
        "step0_transport": "cuda_nccl_p2p_unchanged",
        "context_transport": "trace_local_replay_unchanged",
        "calls_preserved": True,
    }
    _require(dict(contract) == expected, "host run-ahead transport contract changed")
    return True


def build_models(
    *,
    torch: Any,
    plan: Mapping[str, Any],
    bootstrap: Bootstrap,
    mode: str,
    rank: int,
    device: Any,
) -> ModelBundle:
    import torch.distributed as dist
    from safetensors.torch import load_file
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wanstreamer.fsdp import shard_model
    from wanstreamer.lora import inject_lora

    _require(mode in {"baseline", "pipeline"}, "unknown execution mode")
    config = WAN_CONFIGS["t2v-1.3B"]
    generator_ranks, score_ranks = _plan_role_ranks(plan, mode=mode)
    critic_ranks = _plan_critic_ranks(plan, mode=mode, score_ranks=score_ranks)
    # Every rank calls new_group in the same global order.
    generator_group = dist.new_group(ranks=list(generator_ranks))
    score_group = dist.new_group(ranks=list(score_ranks))
    pool_policy = str(plan.get("metadata", {}).get("pool_policy", ""))
    split_dmd_shadow = _split_dmd_shadow_enabled(plan, mode=mode)
    continuous_contract = _continuous_pipeline_contract(plan, mode=mode)
    critic_group = (
        dist.new_group(ranks=list(critic_ranks))
        if critic_ranks != score_ranks
        else score_group
    )
    runahead_host_group = (
        dist.new_group(ranks=list(range(WORLD_SIZE)), backend="gloo")
        if _generator_runahead_transport_enabled(plan, mode=mode)
        else None
    )
    critic_shadow_sync_group = None
    if continuous_contract is not None:
        pair_groups = [
            dist.new_group(ranks=[lane, lane + 8]) for lane in range(8)
        ]
        critic_shadow_sync_group = pair_groups[rank if rank < 8 else rank - 8]
    node_local_score = (
        mode == "pipeline"
        and pool_policy == "persistent_6g_10score_teacher_node_local_hierarchical"
    )
    score_node_groups: list[Any] = []
    score_leader_group = None
    if node_local_score:
        # Every world rank creates every group in this exact global order.
        score_node_groups = [
            dist.new_group(ranks=list(ranks))
            for ranks in PIPELINE_6G10S_SCORE_NODE_GROUPS
        ]
        score_leader_group = dist.new_group(
            ranks=list(PIPELINE_6G10S_SCORE_LEADERS)
        )
    owns_generator = rank in generator_ranks
    owns_score = rank in score_ranks
    owns_critic = rank in critic_ranks
    workload = dict(plan["workload"])
    metadata: dict[str, Any] = {
        "generator_ranks": list(generator_ranks),
        "score_ranks": list(score_ranks),
        "critic_ranks": list(critic_ranks),
        "student": None,
        "critic": None,
        "critic_shadow": None,
        "teacher": None,
    }
    student = critic = critic_shadow = teacher = None
    opt_g = opt_c = None
    generator_optimizer_parameter_names: dict[int, str] = {}
    critic_optimizer_parameter_names: dict[int, str] = {}
    lora_params: list[Any] = []
    shadow_lora_params: list[Any] = []
    critic_gradient_reducer = None

    if owns_generator:
        torch.manual_seed(0)
        student = _wan_model(WanModel, size="1.3B", config=config)
        student_root = Path(workload["generator"]["checkpoint"]).expanduser().resolve()
        _require(student_root == Path(bootstrap.value["student_base"]["root"]), "plan/bootstrap Student root mismatch")
        if bootstrap.value.get("student_state") is None:
            student.load_state_dict(_load_checkpoint(load_file, student_root), strict=True)
            state_receipt = {
                "kind": "base_weights",
                "sha256": bootstrap.value["student_base"]["sha256"],
                "load_policy": "base_checkpoint",
            }
        else:
            # The full Stage-1 state strictly covers every parameter, so reading
            # the base checkpoint first would be pure cold-start I/O and memory.
            state_receipt = _load_student_state(torch, student, bootstrap)
            state_receipt["load_policy"] = "complete_stage1_direct_mmap"
        fsdp_groups = 0
        phase_expanded = pool_policy == PHASE_EXPANDED_CRITIC16_POOL_POLICY
        if mode == "baseline":
            mesh = _make_mesh(torch, generator_ranks, generator_group)
            shard_model(student, mesh=mesh, reshard_after_forward=False)
            student.to(device)
            fsdp_groups = _configure_fsdp_reduction(student, group_size=len(generator_ranks))
        else:
            student.to(device)
        student.train()
        generator_options = dict(
            plan["training_backends"]["generator"].get("options", {})
        )
        optimizer_impl = str(generator_options.get("optimizer_impl", "foreach_adamw"))
        _require(
            optimizer_impl in {"foreach_adamw", "fused_adamw"},
            "unknown Generator optimizer implementation",
        )
        optimizer_kwargs = (
            {"fused": True} if optimizer_impl == "fused_adamw" else {"foreach": True}
        )
        generator_named_parameters = list(student.named_parameters())
        opt_g = torch.optim.AdamW(
            [parameter for _name, parameter in generator_named_parameters],
            lr=2e-6, betas=(0.0, 0.99),
            weight_decay=0.0, eps=1e-8, **optimizer_kwargs,
        )
        generator_optimizer_parameter_names = (
            _bind_trajectory_optimizer_parameter_names(
                named_parameters=generator_named_parameters,
                optimizer=opt_g,
            )
        )
        metadata["student"] = {
            "placement": (
                "world16_fsdp2_keep_unsharded"
                if mode == "baseline"
                else "gpu_resident_replica"
            ),
            "state": state_receipt,
            "fsdp_group_count": fsdp_groups,
            "optimizer_impl": optimizer_impl,
            "optimizer": materialize_adam_state(torch, opt_g, step=int(bootstrap.value["optimizer"]["generator_step"])),
        }
        gc.collect()

    if owns_critic:
        torch.manual_seed(0)
        # Match LiveWan's final BF16 value exactly while avoiding a transient
        # full FP32 model plus full BF16 conversion copy on every rank.
        critic = _wan_model(WanModel, size="1.3B", config=config).to(torch.bfloat16)
        critic_root = Path(workload["score"]["checkpoint"]).expanduser().resolve()
        _require(critic_root == Path(bootstrap.value["critic_base"]["root"]), "plan/bootstrap Critic root mismatch")
        critic.load_state_dict(_load_checkpoint(load_file, critic_root), strict=True)
        critic.eval().requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(bootstrap.value["critic_lora"]["seed"]))
            lora_params, lora_modules, lora_numel = inject_lora(critic, rank=LORA_RANK)
        initial_lora_sha256 = (
            _parameter_sequence_sha256(torch, lora_params)
            if split_dmd_shadow
            else None
        )
        phase_expanded = pool_policy == PHASE_EXPANDED_CRITIC16_POOL_POLICY
        critic_fsdp_groups = 0
        if phase_expanded:
            critic_mesh = _make_mesh(torch, critic_ranks, critic_group)
            shard_model(critic, mesh=critic_mesh, reshard_after_forward=False)
        critic.to(device)
        if phase_expanded:
            critic_fsdp_groups = _configure_fsdp_reduction(
                critic, group_size=len(critic_ranks)
            )
        opt_c = torch.optim.AdamW(
            lora_params, lr=4e-5, betas=(0.0, 0.99),
            weight_decay=0.0, eps=1e-8, foreach=True,
        )
        critic_optimizer_parameter_names = _bind_trajectory_optimizer_parameter_names(
            named_parameters=list(critic.named_parameters()),
            optimizer=opt_c,
            expected_parameter_ids={id(parameter) for parameter in lora_params},
        )
        critic_optimizer_receipt = materialize_adam_state(
            torch, opt_c, step=int(bootstrap.value["optimizer"]["critic_step"])
        )
        critic_materialized_units = 0
        if phase_expanded:
            critic_materialized_units = _materialize_fsdp_units(critic)
            torch.cuda.synchronize(device)
        metadata["critic"] = {
            "placement": (
                "world16_fsdp2_keep_unsharded"
                if phase_expanded
                else "gpu_resident_replica"
            ),
            "fsdp_group_count": critic_fsdp_groups,
            "explicit_pre_dmd_unshard_units": critic_materialized_units,
            "lora_rank": LORA_RANK,
            "lora_modules": int(lora_modules),
            "lora_numel": int(lora_numel),
            "base_sha256": bootstrap.value["critic_base"]["sha256"],
            "initial_lora_sha256": initial_lora_sha256,
            "initial_version": int(bootstrap.versions["critic"]),
            "optimizer": critic_optimizer_receipt,
        }
        if phase_expanded:
            metadata["critic_gradient_sync"] = {
                "policy": "fsdp2_reduce_scatter",
                "group_ranks": list(critic_ranks),
                "group_size": len(critic_ranks),
                "reduction": "SUM_THEN_DIVIDE_GROUP_SIZE",
            }
        gc.collect()

    if split_dmd_shadow and owns_generator:
        torch.manual_seed(0)
        critic_shadow = _wan_model(WanModel, size="1.3B", config=config).to(torch.bfloat16)
        critic_root = Path(workload["score"]["checkpoint"]).expanduser().resolve()
        _require(
            critic_root == Path(bootstrap.value["critic_base"]["root"]),
            "plan/bootstrap Critic-shadow root mismatch",
        )
        critic_shadow.load_state_dict(_load_checkpoint(load_file, critic_root), strict=True)
        critic_shadow.eval().requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(bootstrap.value["critic_lora"]["seed"]))
            shadow_lora_params, shadow_lora_modules, shadow_lora_numel = inject_lora(
                critic_shadow, rank=LORA_RANK
            )
        shadow_lora_sha256 = _parameter_sequence_sha256(torch, shadow_lora_params)
        critic_shadow.requires_grad_(False)
        _require(
            not any(parameter.requires_grad for parameter in critic_shadow.parameters()),
            "Critic shadow unexpectedly owns trainable parameters",
        )
        critic_shadow.to(device)
        metadata["critic_shadow"] = {
            "placement": "generator_rank_frozen_replica",
            "scope": "step0_fake_score_only",
            "base_sha256": bootstrap.value["critic_base"]["sha256"],
            "initial_lora_sha256": shadow_lora_sha256,
            "initial_version": int(bootstrap.versions["critic"]),
            "lora_rank": LORA_RANK,
            "lora_modules": int(shadow_lora_modules),
            "lora_numel": int(shadow_lora_numel),
            "trainable_parameter_count": 0,
            "optimizer": None,
        }
        gc.collect()

    if owns_score:
        torch.manual_seed(0)
        teacher = _wan_model(WanModel, size="14B", config=config).to(torch.bfloat16)
        teacher_root = Path(workload["teacher"]["checkpoint"]).expanduser().resolve()
        _require(teacher_root == Path(bootstrap.value["teacher_base"]["root"]), "plan/bootstrap Teacher root mismatch")
        teacher.load_state_dict(_load_checkpoint(load_file, teacher_root), strict=True)
        teacher.eval().requires_grad_(False)
        teacher_ranks = score_ranks
        teacher_group = score_group
        if node_local_score:
            group_index = next(
                index
                for index, ranks in enumerate(PIPELINE_6G10S_SCORE_NODE_GROUPS)
                if rank in ranks
            )
            teacher_ranks = PIPELINE_6G10S_SCORE_NODE_GROUPS[group_index]
            teacher_group = score_node_groups[group_index]
        teacher_mesh = _make_mesh(torch, teacher_ranks, teacher_group)
        shard_model(teacher, mesh=teacher_mesh, reshard_after_forward=True)
        teacher.to(device)
        fsdp_groups = _configure_fsdp_reduction(teacher, group_size=len(teacher_ranks))
        metadata["teacher"] = {
            "placement": f"fsdp2_group{len(teacher_ranks)}_reshard",
            "mesh_ranks": list(teacher_ranks),
            "fsdp_group_count": fsdp_groups,
        }
        gc.collect()

    if owns_critic and pool_policy in {
        ASYNC_CRITIC_POOL_POLICY,
        TRACE_BALANCED_POOL_POLICY,
    }:
        lora_ids = {id(parameter) for parameter in lora_params}
        named_lora = [
            (name, parameter)
            for name, parameter in critic.named_parameters()
            if id(parameter) in lora_ids
        ]
        _require(
            len(named_lora) == len(lora_params)
            and {id(parameter) for _name, parameter in named_lora} == lora_ids,
            "Critic async reducer lost the LoRA parameter inventory",
        )
        critic_gradient_reducer = GradReadyAsyncGroupReducer(
            torch,
            named_lora,
            group=critic_group,
            group_size=len(critic_ranks),
            bucket_bytes=32 * 1024 * 1024,
        )
        metadata["critic_gradient_sync"] = {
            "policy": "grad_ready_async_flat_all_reduce",
            "group_ranks": list(critic_ranks),
            "group_size": len(critic_ranks),
            "bucket_bytes_limit": 32 * 1024 * 1024,
            "reduction": "SUM_THEN_DIVIDE_GROUP_SIZE",
        }

    if continuous_contract is not None:
        metadata["continuous_pipeline"] = {
            "enabled": True,
            "cycles": int(continuous_contract["cycles"]),
            "lookahead_depth": int(continuous_contract["lookahead_depth"]),
            "critic_shadow_sync_group_ranks": [
                rank if rank < 8 else rank - 8,
                (rank if rank < 8 else rank - 8) + 8,
            ],
            "critic_shadow_sync_dtype": "torch.float32",
            "critic_shadow_sync_parameter_count": len(
                shadow_lora_params if owns_generator else lora_params
            ),
        }

    torch.cuda.synchronize(device)
    score_local_group = None
    score_local_ranks: tuple[int, ...] = ()
    if node_local_score and owns_score:
        group_index = next(
            index
            for index, ranks in enumerate(PIPELINE_6G10S_SCORE_NODE_GROUPS)
            if rank in ranks
        )
        score_local_group = score_node_groups[group_index]
        score_local_ranks = PIPELINE_6G10S_SCORE_NODE_GROUPS[group_index]
        metadata["critic_gradient_sync"] = {
            "policy": "hierarchical_node_local_sum_leader_sum_broadcast_divide",
            "local_ranks": list(score_local_ranks),
            "leader_ranks": list(PIPELINE_6G10S_SCORE_LEADERS),
            "global_divide_factor": len(score_ranks),
        }
    return ModelBundle(
        student=student,
        critic=critic,
        critic_shadow=critic_shadow,
        teacher=teacher,
        generator_optimizer=opt_g,
        critic_optimizer=opt_c,
        generator_optimizer_parameter_names=generator_optimizer_parameter_names,
        critic_optimizer_parameter_names=critic_optimizer_parameter_names,
        critic_lora_params=lora_params,
        critic_shadow_lora_params=shadow_lora_params,
        critic_gradient_reducer=critic_gradient_reducer,
        generator_group=generator_group if owns_generator else None,
        score_group=score_group if owns_score else None,
        critic_group=critic_group if owns_critic else None,
        score_local_group=score_local_group,
        score_leader_group=(
            score_leader_group
            if node_local_score and rank in PIPELINE_6G10S_SCORE_LEADERS
            else None
        ),
        runahead_host_group=runahead_host_group,
        critic_shadow_sync_group=critic_shadow_sync_group,
        score_local_ranks=score_local_ranks,
        score_leader_ranks=(
            PIPELINE_6G10S_SCORE_LEADERS if node_local_score else ()
        ),
        generator_group_size=len(generator_ranks),
        score_group_size=len(score_ranks),
        critic_group_size=len(critic_ranks),
        metadata=metadata,
    )


def _mean_replicated_gradients(
    torch: Any,
    parameters: Sequence[Any],
    *,
    group: Any,
    group_size: int,
    bucket_mib: int,
) -> dict[str, int]:
    import torch.distributed as dist
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    buckets: list[list[Any]] = []
    current: list[Any] = []
    current_bytes = 0
    limit = bucket_mib * 1024 * 1024
    for gradient in gradients:
        size = gradient.numel() * gradient.element_size()
        if current and current_bytes + size > limit:
            buckets.append(current)
            current, current_bytes = [], 0
        current.append(gradient)
        current_bytes += size
    if current:
        buckets.append(current)
    for bucket in buckets:
        dist.all_reduce_coalesced(bucket, op=dist.ReduceOp.SUM, group=group)
        torch._foreach_mul_(bucket, 1.0 / group_size)
    return {"gradient_tensors": len(gradients), "buckets": len(buckets)}


def _mean_replicated_gradients_hierarchical(
    torch: Any,
    parameters: Sequence[Any],
    *,
    local_group: Any,
    local_ranks: Sequence[int],
    leader_group: Any | None,
    leader_ranks: Sequence[int],
    rank: int,
    group_size: int,
    bucket_mib: int,
) -> dict[str, Any]:
    """Exact global mean without a large asymmetric Score10 collective.

    Each node first sums its resident Critic gradients.  The two leaders then
    sum those node totals, after which the result is broadcast inside each
    node-local Score group and divided by the unchanged global Score pool size.
    The reduction tree changes, but the sample mean and optimizer mathematics
    are identical to the flat Score10 SUM-then-divide contract.
    """

    import torch.distributed as dist

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    buckets: list[list[Any]] = []
    current: list[Any] = []
    current_bytes = 0
    limit = bucket_mib * 1024 * 1024
    for gradient in gradients:
        size = gradient.numel() * gradient.element_size()
        if current and current_bytes + size > limit:
            buckets.append(current)
            current, current_bytes = [], 0
        current.append(gradient)
        current_bytes += size
    if current:
        buckets.append(current)

    local_leader = int(local_ranks[0])
    leader_set = set(int(value) for value in leader_ranks)
    for bucket in buckets:
        flat = torch._utils._flatten_dense_tensors(bucket)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=local_group)
        if rank in leader_set:
            _require(leader_group is not None, "Score leader is missing its cross-node group")
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=leader_group)
        dist.broadcast(flat, src=local_leader, group=local_group)
        flat.mul_(1.0 / group_size)
        for gradient, synced in zip(
            bucket, torch._utils._unflatten_dense_tensors(flat, bucket)
        ):
            gradient.copy_(synced)
    return {
        "gradient_tensors": len(gradients),
        "buckets": len(buckets),
        "policy": "hierarchical_node_local_sum_leader_sum_broadcast_divide",
        "local_ranks": list(local_ranks),
        "leader_ranks": list(leader_ranks),
        "global_divide_factor": group_size,
    }


def _pipeline_slot_routes(
    generator_ranks: Sequence[int], score_ranks: Sequence[int]
) -> tuple[tuple[int, int, int], ...]:
    """Return the exact logical slot -> Generator -> Score routing table."""

    generator_by_slot = {
        slot: rank for rank, slots in balanced_slots(generator_ranks).items() for slot in slots
    }
    score_by_slot = {
        slot: rank for rank, slots in balanced_slots(score_ranks).items() for slot in slots
    }
    _require(
        set(generator_by_slot) == set(score_by_slot) == set(range(GLOBAL_BATCH)),
        "pipeline slot route is not a world16 bijection",
    )
    return tuple(
        (slot, generator_by_slot[slot], score_by_slot[slot])
        for slot in range(GLOBAL_BATCH)
    )


def _warm_communicators(
    torch: Any,
    *,
    bundle: ModelBundle,
    mode: str,
    rank: int,
    device: Any,
) -> dict[str, Any]:
    """Initialize Full5 communicators without touching workload state.

    The scalar exchanges run after construction but before the formal timer.
    Without the final rank-pair handshake, the pipeline's first timed latent
    send would also pay lazy cross-node NCCL communicator creation.
    """
    import torch.distributed as dist

    started = time.perf_counter()
    world = torch.ones(1, dtype=torch.int64, device=device)
    dist.all_reduce(world)
    _require(int(world.cpu()) == WORLD_SIZE, "default communicator warm-up failed")

    warmed_groups: list[str] = ["world"]
    if bundle.generator_group is not None:
        token = torch.ones(1, dtype=torch.int64, device=device)
        dist.all_reduce(token, group=bundle.generator_group)
        _require(
            int(token.cpu()) == bundle.generator_group_size,
            "Generator communicator warm-up failed",
        )
        warmed_groups.append("generator")
    if bundle.score_group is not None:
        token = torch.ones(1, dtype=torch.int64, device=device)
        dist.all_reduce(token, group=bundle.score_group)
        _require(
            int(token.cpu()) == bundle.score_group_size,
            "Score communicator warm-up failed",
        )
        warmed_groups.append("score")
    if (
        bundle.critic_group is not None
        and bundle.critic_group is not bundle.score_group
    ):
        token = torch.ones(1, dtype=torch.int64, device=device)
        dist.all_reduce(token, group=bundle.critic_group)
        _require(
            int(token.cpu()) == bundle.critic_group_size,
            "Critic communicator warm-up failed",
        )
        warmed_groups.append("critic")
    if bundle.score_local_group is not None:
        token = torch.ones(1, dtype=torch.int64, device=device)
        dist.all_reduce(token, group=bundle.score_local_group)
        _require(
            int(token.cpu()) == len(bundle.score_local_ranks),
            "node-local Score communicator warm-up failed",
        )
        warmed_groups.append("score_node_local")
    if rank in bundle.score_leader_ranks:
        _require(bundle.score_leader_group is not None, "Score leader group is missing")
        token = torch.ones(1, dtype=torch.int64, device=device)
        dist.all_reduce(token, group=bundle.score_leader_group)
        _require(
            int(token.cpu()) == len(bundle.score_leader_ranks),
            "Score leader communicator warm-up failed",
        )
        warmed_groups.append("score_leaders")
    if bundle.runahead_host_group is not None:
        # C19 uses a dedicated CPU/Gloo communicator only for the already
        # produced Critic-only latents.  Warm it before the Full5 timer so lazy
        # socket/process-group setup is not charged to the candidate.
        token = torch.ones(1, dtype=torch.int64, device="cpu")
        dist.all_reduce(token, group=bundle.runahead_host_group)
        _require(
            int(token.item()) == WORLD_SIZE,
            "run-ahead host communicator warm-up failed",
        )
        warmed_groups.append("runahead_host_gloo")
    if bundle.critic_shadow_sync_group is not None:
        token = torch.ones(1, dtype=torch.int64, device=device)
        dist.all_reduce(token, group=bundle.critic_shadow_sync_group)
        _require(
            int(token.cpu()) == 2,
            "Critic-shadow paired communicator warm-up failed",
        )
        warmed_groups.append("critic_shadow_pair")

    pair_peer = None
    routed_slots: list[int] = []
    if mode == "pipeline":
        generator_ranks = tuple(int(value) for value in bundle.metadata["generator_ranks"])
        score_ranks = tuple(int(value) for value in bundle.metadata["score_ranks"])
        if generator_ranks == PIPELINE_8G8S_LAYOUT.generator_ranks:
            pair_peer = rank + 8 if rank < 8 else rank - 8
            sent = torch.tensor([rank], dtype=torch.int64, device=device)
            received = torch.empty_like(sent)
            if rank < 8:
                dist.send(sent, dst=pair_peer)
                dist.recv(received, src=pair_peer)
            else:
                dist.recv(received, src=pair_peer)
                dist.send(sent, dst=pair_peer)
            _require(int(received.cpu()) == pair_peer, "pipeline P2P warm-up identity mismatch")
            warmed_groups.append("rank_pair_p2p")
        else:
            routes = _pipeline_slot_routes(generator_ranks, score_ranks)
            forward_ops: list[Any] = []
            reverse_ops: list[Any] = []
            received_by_slot: dict[int, Any] = {}
            acknowledgements: dict[int, Any] = {}
            retained: list[Any] = []
            for slot, generator_rank, score_rank in routes:
                if rank == generator_rank:
                    token = torch.tensor([slot, generator_rank, score_rank], dtype=torch.int64, device=device)
                    acknowledgement = torch.empty(1, dtype=torch.int64, device=device)
                    retained.append(token)
                    acknowledgements[slot] = acknowledgement
                    forward_ops.append(dist.P2POp(dist.isend, token, score_rank))
                    reverse_ops.append(dist.P2POp(dist.irecv, acknowledgement, score_rank))
                    routed_slots.append(slot)
                elif rank == score_rank:
                    token = torch.empty(3, dtype=torch.int64, device=device)
                    acknowledgement = torch.tensor([slot], dtype=torch.int64, device=device)
                    received_by_slot[slot] = token
                    retained.append(acknowledgement)
                    forward_ops.append(dist.P2POp(dist.irecv, token, generator_rank))
                    reverse_ops.append(dist.P2POp(dist.isend, acknowledgement, generator_rank))
                    routed_slots.append(slot)
            for request in dist.batch_isend_irecv(forward_ops):
                request.wait()
            for slot, value in received_by_slot.items():
                expected = next(row for row in routes if row[0] == slot)
                _require(
                    tuple(int(item) for item in value.cpu()) == expected,
                    "routed pipeline P2P warm-up identity mismatch",
                )
            for request in dist.batch_isend_irecv(reverse_ops):
                request.wait()
            for slot, value in acknowledgements.items():
                _require(int(value.cpu()) == slot, "routed pipeline P2P acknowledgement mismatch")
            warmed_groups.append("routed_batch_p2p")

    dist.barrier()
    torch.cuda.synchronize(device)
    return {
        "status": "PASS",
        "groups": warmed_groups,
        "pair_peer": pair_peer,
        "routed_slots": routed_slots,
        "wall_seconds_outside_full5": time.perf_counter() - started,
    }


def _sample_flow_t(torch: Any, *, seed: int, device: Any) -> Any:
    generator = torch.Generator(device=device).manual_seed(seed)
    u = torch.rand(1, device=device, generator=generator)
    value = 5.0 * u / (1.0 + 4.0 * u)
    return value.clamp(0.02, 0.98)


def _randn(torch: Any, shape: Sequence[int], *, seed: int, device: Any) -> Any:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(tuple(shape), device=device, dtype=torch.float32, generator=generator)


@dataclass
class PreparedRollout:
    row: OperationTraceRow
    streamer: Any
    recs: Any
    x0: Any
    base_offset: int
    new_blocks: int
    raw_context: Any
    context_lens: Any


class InputLoader:
    def __init__(self, *, torch: Any, trace: FormalTrace):
        self.torch = torch
        self.trace = trace
        self.prompts = torch.load(trace.prompts_path, map_location="cpu", weights_only=False)

    @property
    def negative(self) -> Any:
        return self.prompts["neg"][0].float().unsqueeze(0)

    def load(self, row: OperationTraceRow) -> tuple[Any, Any, int]:
        value = self.torch.load(row.sample_path, map_location="cpu", weights_only=False)
        latent = value["latents"].float()[:, : LATENT_SHAPE[2]]
        prompt_index = int(value["prompt_idx"])
        _require(
            prompt_index == row.dataset_index,
            f"trace/sample prompt identity mismatch for {row.operation_key}",
        )
        context = self.prompts["pos"][prompt_index].float()
        _require(tuple(latent.shape) == LATENT_SHAPE[1:], f"latent shape mismatch for {row.operation_key}")
        _require(tuple(context.shape) == TEXT_SHAPE[1:], f"text shape mismatch for {row.operation_key}")
        return latent, context, prompt_index

    def context_from_trace(self, row: OperationTraceRow, *, device: Any) -> Any:
        """Replay the exact prompt tensor without cross-node context P2P."""

        context = self.prompts["pos"][row.dataset_index].float()
        _require(
            tuple(context.shape) == TEXT_SHAPE[1:],
            f"trace-local text shape mismatch for {row.operation_key}",
        )
        return context.to(device, non_blocking=True).unsqueeze(0)


def _build_streamers(torch: Any, *, student: Any, count: int, device: Any):
    from wanstreamer.core import latent_geometry, make_rope_table
    from wanstreamer.stream import FewStepStreamer

    _, _, hp, wp = latent_geometry(640, 368)
    rope = make_rope_table(student, hp, wp, 278, device)
    return [
        FewStepStreamer(
            student,
            width=640,
            height=368,
            max_frames=278,
            device=device,
            dtype=torch.bfloat16,
            window_frames=None,
            time_scale=TIME_SCALE,
            cache_frames=22,
            block_frames=BLOCK_FRAMES,
            num_steps=STEPS_PER_BLOCK,
            rope=rope,
        )
        for _ in range(count)
    ]


def prepare_rollout(
    torch: Any,
    *,
    row: OperationTraceRow,
    streamer: Any,
    student: Any,
    loader: InputLoader,
    device: Any,
    elide_full_context_lens: bool = False,
) -> PreparedRollout:
    from distillschedule.runtime.livewan_rollout_cast_cache import model_rollout_cast_cache_scope
    from distillschedule.runtime.livewan_generator_rms_epilogue import model_runtime_scope as generator_rms_scope

    latent, context, _ = loader.load(row)
    z_teacher = latent.to(device, non_blocking=True).unsqueeze(0)
    raw_context = context.to(device, non_blocking=True).unsqueeze(0)
    _require(
        int(raw_context.shape[1]) == TEXT_SHAPE[1],
        "rollout prompt length differs from the immutable full-context contract",
    )
    context_lens = (
        None
        if elide_full_context_lens
        else torch.tensor([raw_context.shape[1]], device=device, dtype=torch.long)
    )
    with torch.no_grad(), model_rollout_cast_cache_scope(torch, student), generator_rms_scope(torch, student):
        student_context = student.text_embedding(raw_context)
        streamer.set_text(student_context, context_lens)
        streamer.rope_gap = row.rope_gap
        streamer.set_world(z_teacher[:, :, : row.world_frames])
        generator = torch.Generator(device=device).manual_seed(row.seeds["rollout"])
        blocks, recs = streamer.rollout_record(row.new_blocks, generator=generator)
    new = torch.cat([value.detach() for value in blocks], dim=2)
    x0 = torch.cat([z_teacher[:, :, : row.world_frames], new], dim=2)
    _require(tuple(x0.shape) == LATENT_SHAPE, f"rollout shape mismatch at {row.operation_key}")
    return PreparedRollout(
        row=row,
        streamer=streamer,
        recs=recs,
        x0=x0,
        base_offset=x0.shape[2] - row.new_blocks * BLOCK_FRAMES,
        new_blocks=row.new_blocks,
        raw_context=raw_context,
        context_lens=context_lens,
    )


@dataclass
class ScoreContext:
    rope: Any
    negative_teacher_context: Any


@dataclass
class ScoreSampleContext:
    teacher: Any | None
    critic: Any | None


def build_score_context(torch: Any, *, bundle: ModelBundle, loader: InputLoader, device: Any) -> ScoreContext:
    from wanstreamer.core import latent_geometry, make_rope_table
    critic_model = (
        bundle.critic if bundle.critic is not None else bundle.critic_shadow
    )
    _require(critic_model is not None, "Critic model bundle is incomplete")
    _, _, hp, wp = latent_geometry(640, 368)
    rope = make_rope_table(critic_model, hp, wp, LATENT_SHAPE[2] + 8, device)
    negative = None
    if bundle.teacher is not None:
        with torch.no_grad():
            negative = bundle.teacher.text_embedding(
                loader.negative.to(device).to(torch.bfloat16)
            )
    return ScoreContext(rope=rope, negative_teacher_context=negative)


def prepare_score_sample_context(
    torch: Any,
    *,
    bundle: ModelBundle,
    raw_context: Any,
    include_teacher: bool,
    critic_model: Any | None = None,
) -> ScoreSampleContext:
    """Project only the context tensors consumed by this logical step.

    The Teacher context belongs to the step-0 DMD objective.  Computing it in
    steps 1--4 is dead work and, because the Teacher is FSDP2, launches an
    otherwise-unused root-unit all-gather.  Preserve the original step-0 call
    order while omitting that projection from Critic-only steps.
    """

    with torch.no_grad():
        teacher_context = (
            bundle.teacher.text_embedding(raw_context.to(torch.bfloat16))
            if include_teacher
            else None
        )
        selected_critic = critic_model if critic_model is not None else bundle.critic
        critic_context = (
            selected_critic.text_embedding(raw_context.to(torch.bfloat16))
            if selected_critic is not None
            else None
        )
    return ScoreSampleContext(teacher=teacher_context, critic=critic_context)


@dataclass
class DMDFlowInputs:
    t: Any
    t_view: Any
    z_t: Any


def _dmd_flow_inputs(
    torch: Any, *, row: OperationTraceRow, x0: Any
) -> DMDFlowInputs:
    t = _sample_flow_t(torch, seed=row.seeds["generator_dmd_t"], device=x0.device)
    noise = _randn(
        torch,
        x0.shape,
        seed=row.seeds["generator_dmd_noise"],
        device=x0.device,
    )
    t_view = t
    while t_view.dim() < x0.dim():
        t_view = t_view.unsqueeze(-1)
    return DMDFlowInputs(t=t, t_view=t_view, z_t=(1.0 - t_view) * x0 + t_view * noise)


def _dmd_teacher_velocity(
    torch: Any,
    *,
    bundle: ModelBundle,
    score_context: ScoreContext,
    flow: DMDFlowInputs,
    sample_context: ScoreSampleContext,
    context_lens: Any,
) -> Any:
    from wanstreamer import dmd as D
    from wanstreamer.lora import lora_enabled

    _require(bundle.teacher is not None, "DMD Teacher model is missing")
    _require(sample_context.teacher is not None, "DMD requires Teacher context")
    _require(
        score_context.negative_teacher_context is not None,
        "DMD requires negative Teacher context",
    )
    with torch.no_grad(), lora_enabled(bundle.teacher, False):
        if context_lens is None:
            velocity = _cfg_velocity_full_context(
                torch,
                D,
                model=bundle.teacher,
                z=flow.z_t.to(torch.bfloat16),
                t=flow.t,
                rope=score_context.rope,
                context_positive=sample_context.teacher,
                context_negative=score_context.negative_teacher_context,
                dtype=torch.bfloat16,
                guidance=5.0,
                time_scale=TIME_SCALE,
            )
        else:
            velocity = D.cfg_velocity(
                bundle.teacher,
                flow.z_t.to(torch.bfloat16),
                flow.t,
                score_context.rope,
                sample_context.teacher,
                score_context.negative_teacher_context,
                context_lens,
                torch.bfloat16,
                5.0,
                TIME_SCALE,
            )
    return velocity.float()


def _dmd_fake_velocity(
    torch: Any,
    *,
    critic_model: Any,
    score_context: ScoreContext,
    flow: DMDFlowInputs,
    critic_context: Any,
    context_lens: Any,
) -> Any:
    from wanstreamer import dmd as D
    from wanstreamer.lora import lora_enabled

    _require(critic_model is not None, "DMD Fake-Score model is missing")
    _require(critic_context is not None, "DMD Fake-Score context is missing")
    with torch.no_grad(), lora_enabled(critic_model, True):
        velocity = D.bidirectional_velocity(
            critic_model,
            flow.z_t.to(torch.bfloat16),
            flow.t,
            score_context.rope,
            critic_context,
            context_lens,
            torch.bfloat16,
            TIME_SCALE,
        )
    return velocity.float()


def _combine_dmd_gradient(
    torch: Any,
    *,
    x0: Any,
    flow: DMDFlowInputs,
    v_real: Any,
    v_fake: Any,
) -> Any:
    with torch.no_grad():
        x0_real = (flow.z_t - flow.t_view * v_real).float()
        x0_fake = (flow.z_t - flow.t_view * v_fake).float()
        normalizer = (x0.float() - x0_real).abs().mean().clamp_min(1e-4)
        return torch.nan_to_num((x0_fake - x0_real) / normalizer)


def _cfg_velocity_full_context(
    torch: Any,
    D: Any,
    *,
    model: Any,
    z: Any,
    t: Any,
    rope: Any,
    context_positive: Any,
    context_negative: Any,
    dtype: Any,
    guidance: float,
    time_scale: float,
) -> Any:
    """Run LiveWan CFG with the exact no-padding context contract.

    This mirrors ``wanstreamer.dmd.cfg_velocity`` except that ``ctx_lens`` is
    deliberately ``None``.  Keep ``t`` unexpanded: the pinned helper passes
    the original B1 timestep into the B2 conditional/unconditional call and
    relies on broadcasting inside the model.
    """

    if not guidance or guidance == 1.0:
        return D.bidirectional_velocity(
            model, z, t, rope, context_positive, None, dtype, time_scale
        )
    velocity = D.bidirectional_velocity(
        model,
        torch.cat((z, z), dim=0),
        t,
        rope,
        torch.cat((context_positive, context_negative), dim=0),
        None,
        dtype,
        time_scale,
    )
    conditional, unconditional = velocity[0:1], velocity[1:2]
    return unconditional + guidance * (conditional - unconditional)


def dmd_gradient_for_sample(
    torch: Any,
    *,
    bundle: ModelBundle,
    score_context: ScoreContext,
    row: OperationTraceRow,
    x0: Any,
    sample_context: ScoreSampleContext,
    context_lens: Any,
) -> Any:
    flow = _dmd_flow_inputs(torch, row=row, x0=x0)
    v_real = _dmd_teacher_velocity(
        torch,
        bundle=bundle,
        score_context=score_context,
        flow=flow,
        sample_context=sample_context,
        context_lens=context_lens,
    )
    v_fake = _dmd_fake_velocity(
        torch,
        critic_model=bundle.critic,
        score_context=score_context,
        flow=flow,
        critic_context=sample_context.critic,
        context_lens=context_lens,
    )
    return _combine_dmd_gradient(
        torch, x0=x0, flow=flow, v_real=v_real, v_fake=v_fake
    )


def _batch_seeded_flow_inputs(
    torch: Any,
    *,
    rows: Sequence[OperationTraceRow],
    x0: Any,
    t_seed_key: str,
    noise_seed_key: str,
) -> tuple[Any, Any]:
    """Build one RNG-identical flow input per logical sample, then batch it.

    A single B2 generator would consume a different random stream from two B1
    calls.  The physical batching candidate is allowed to change model launch
    granularity, not random values, so every row keeps its own trace-qualified
    generator and the resulting tensors are concatenated afterwards.
    """

    _require(x0.shape[0] == len(rows), "batched latent/trace row count mismatch")
    times = [
        _sample_flow_t(torch, seed=row.seeds[t_seed_key], device=x0.device)
        for row in rows
    ]
    noises = [
        _randn(
            torch,
            x0[index : index + 1].shape,
            seed=row.seeds[noise_seed_key],
            device=x0.device,
        )
        for index, row in enumerate(rows)
    ]
    return torch.cat(times, dim=0), torch.cat(noises, dim=0)


def _per_sample_mse(torch: Any, prediction: Any, target: Any) -> Any:
    _require(prediction.shape == target.shape, "per-sample MSE shape mismatch")
    _require(prediction.dim() >= 2, "per-sample MSE needs a batch dimension")
    return (prediction.float() - target.float()).square().flatten(1).mean(1)


def dmd_gradient_for_batch(
    torch: Any,
    *,
    bundle: ModelBundle,
    score_context: ScoreContext,
    rows: Sequence[OperationTraceRow],
    x0: Any,
    sample_context: ScoreSampleContext,
    context_lens: Any,
) -> Any:
    """One exact physical Score call for multiple independent logical rows.

    RNG is constructed per trace row and DMD normalization remains per sample.
    Teacher CFG is physically B(2*N): all positive rows followed by all
    negative rows.  This is the batched analogue of N independent B2 CFG calls;
    it does not average samples together.
    """

    from wanstreamer import dmd as D
    from wanstreamer.lora import lora_enabled

    _require(sample_context.teacher is not None, "DMD requires Teacher context")
    batch = len(rows)
    _require(batch == x0.shape[0], "DMD B2 identity mismatch")
    if context_lens is not None:
        _require(batch == context_lens.numel(), "DMD B2 context-lens identity mismatch")
    _require(sample_context.teacher.shape[0] == batch, "Teacher context batch mismatch")
    _require(sample_context.critic.shape[0] == batch, "Critic context batch mismatch")
    t, noise = _batch_seeded_flow_inputs(
        torch,
        rows=rows,
        x0=x0,
        t_seed_key="generator_dmd_t",
        noise_seed_key="generator_dmd_noise",
    )
    t_view = t
    while t_view.dim() < x0.dim():
        t_view = t_view.unsqueeze(-1)
    z_t = (1.0 - t_view) * x0 + t_view * noise
    negative = score_context.negative_teacher_context
    _require(negative.shape[0] == 1, "negative Teacher context must be canonical B1")
    negative = negative.expand(batch, -1, -1)
    with torch.no_grad():
        with lora_enabled(bundle.teacher, False):
            teacher_input = torch.cat([z_t, z_t], dim=0).to(torch.bfloat16)
            teacher_context = torch.cat([sample_context.teacher, negative], dim=0)
            teacher_lens = (
                torch.cat([context_lens, context_lens], dim=0)
                if context_lens is not None
                else None
            )
            teacher_t = torch.cat([t, t], dim=0)
            teacher_velocity = D.bidirectional_velocity(
                bundle.teacher,
                teacher_input,
                teacher_t,
                score_context.rope,
                teacher_context,
                teacher_lens,
                torch.bfloat16,
                TIME_SCALE,
            )
            conditional = teacher_velocity[:batch]
            unconditional = teacher_velocity[batch:]
            v_real = (unconditional + 5.0 * (conditional - unconditional)).float()
        with lora_enabled(bundle.critic, True):
            v_fake = D.bidirectional_velocity(
                bundle.critic,
                z_t.to(torch.bfloat16),
                t,
                score_context.rope,
                sample_context.critic,
                context_lens,
                torch.bfloat16,
                TIME_SCALE,
            ).float()
        x0_real = (z_t - t_view * v_real).float()
        x0_fake = (z_t - t_view * v_fake).float()
        reduce_dims = tuple(range(1, x0.dim()))
        normalizer = (x0.float() - x0_real).abs().mean(
            dim=reduce_dims, keepdim=True
        ).clamp_min(1e-4)
        gradient = torch.nan_to_num((x0_fake - x0_real) / normalizer)
    return gradient


def critic_backward_for_sample(
    torch: Any,
    *,
    bundle: ModelBundle,
    score_context: ScoreContext,
    row: OperationTraceRow,
    x0: Any,
    sample_context: ScoreSampleContext,
    context_lens: Any,
    sample_weight: float,
    reducer_accumulation: tuple[int, int] | None = None,
) -> Any:
    from wanstreamer import dmd as D
    from wanstreamer.lora import lora_enabled

    t = _sample_flow_t(torch, seed=row.seeds["critic_t"], device=x0.device)
    noise = _randn(torch, x0.shape, seed=row.seeds["critic_noise"], device=x0.device)
    with lora_enabled(bundle.critic, True):
        loss = D.critic_loss(
            bundle.critic, x0.to(torch.bfloat16), t,
            score_context.rope, sample_context.critic, context_lens,
            torch.bfloat16, TIME_SCALE, grad_checkpoint=True, noise=noise,
        )
    scaled_loss = loss * sample_weight
    if bundle.critic_gradient_reducer is None:
        _require(
            reducer_accumulation is None,
            "Critic reducer accumulation requested without the registered reducer",
        )
        scaled_loss.backward()
    elif reducer_accumulation is not None:
        accumulation_index, accumulation_count = reducer_accumulation
        bundle.critic_gradient_reducer.backward_accumulate(
            scaled_loss,
            accumulation_index=accumulation_index,
            accumulation_count=accumulation_count,
        )
    else:
        bundle.critic_gradient_reducer.backward(scaled_loss)
    return loss.detach()


def critic_backward_for_batch(
    torch: Any,
    *,
    bundle: ModelBundle,
    score_context: ScoreContext,
    rows: Sequence[OperationTraceRow],
    x0: Any,
    sample_context: ScoreSampleContext,
    context_lens: Any,
    loss_scale: float = 1.0,
    reducer_accumulation: tuple[int, int] | None = None,
) -> tuple[Any, ...]:
    """Backpropagate a scaled local mean of exact per-sample Critic losses."""

    from wanstreamer import dmd as D
    from wanstreamer.data import add_noise
    from wanstreamer.lora import lora_enabled

    batch = len(rows)
    _require(batch == x0.shape[0], "Critic B2 identity mismatch")
    if context_lens is not None:
        _require(batch == context_lens.numel(), "Critic B2 context-lens identity mismatch")
    t, noise = _batch_seeded_flow_inputs(
        torch,
        rows=rows,
        x0=x0,
        t_seed_key="critic_t",
        noise_seed_key="critic_noise",
    )
    z_t, target, _ = add_noise(x0, t, noise)
    with lora_enabled(bundle.critic, True):
        velocity = D.bidirectional_velocity(
            bundle.critic,
            z_t,
            t,
            score_context.rope,
            sample_context.critic,
            context_lens,
            torch.bfloat16,
            TIME_SCALE,
            grad_checkpoint=True,
        )
    losses = _per_sample_mse(torch, velocity, target)
    scaled_loss = losses.mean().mul(loss_scale)
    if bundle.critic_gradient_reducer is None:
        _require(
            reducer_accumulation is None,
            "Critic B2 accumulation requested without the registered reducer",
        )
        scaled_loss.backward()
    elif reducer_accumulation is not None:
        accumulation_index, accumulation_count = reducer_accumulation
        bundle.critic_gradient_reducer.backward_accumulate(
            scaled_loss,
            accumulation_index=accumulation_index,
            accumulation_count=accumulation_count,
        )
    else:
        bundle.critic_gradient_reducer.backward(scaled_loss)
    return tuple(value.detach() for value in losses.unbind(0))


def _critic_b1_attribution_reducer_accumulation(wave: int) -> tuple[int, int]:
    """Return the exact two-backward/single-reduction C24-B1 boundary."""

    _require(wave in {0, 1}, "C24 Critic-B1 attribution wave must be 0 or 1")
    return wave, 2


def _physical_critic_backward_calls_for_step(
    *, critic_b2_only: bool, split_step1: bool
) -> int:
    """Return physical backward invocations for the branch that completed."""

    return 2 if split_step1 or not critic_b2_only else 1


def generator_backward_for_sample(
    torch: Any,
    *,
    prepared: PreparedRollout,
    dmd_gradient: Any,
    sample_weight: float,
) -> list[Any]:
    from wanstreamer import blockcausal as bc
    from wanstreamer import dmd as D

    prefix = bc.BufferPrefixKV(prepared.streamer.cache)
    generator = torch.Generator(device=prepared.x0.device).manual_seed(
        prepared.row.seeds["generator_grad_blocks"]
    )
    block_indices = torch.randperm(prepared.new_blocks, generator=generator, device=prepared.x0.device)[
        : min(GRAD_BLOCKS, prepared.new_blocks)
    ].tolist()
    losses: list[Any] = []
    for block_index in block_indices:
        record = prepared.recs[block_index][-1]
        _require(prepared.streamer.prefix_intact(record), "recorded rollout prefix was evicted")
        offset = prepared.base_offset + block_index * BLOCK_FRAMES
        x0_hat = prepared.streamer.recompute_step(record, prefix)
        loss = D.dmd_surrogate(
            x0_hat,
            dmd_gradient[:, :, offset : offset + BLOCK_FRAMES],
            scale=sample_weight / len(block_indices),
        )
        loss.backward()
        losses.append(loss.detach())
        del x0_hat, loss
    return losses


def _tensor_fingerprint(torch: Any, value: Any) -> dict[str, Any]:
    data = value.detach().float()
    return {
        "shape": list(data.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(data).all().cpu()),
        "mean": float(data.mean().cpu()),
        "norm": float(data.norm().cpu()),
        "max_abs": float(data.abs().max().cpu()),
    }


def _trajectory_model_named_values(
    model: Any | None, *, parameter_ids: set[int] | None = None
) -> list[tuple[str, Any, Any | None]]:
    if model is None:
        return []
    return [
        (_trajectory_livewan_state_name(name), parameter, None)
        for name, parameter in model.named_parameters()
        if (
            (parameter_ids is None and parameter.requires_grad)
            or (parameter_ids is not None and id(parameter) in parameter_ids)
        )
    ]


def _trajectory_livewan_state_name(name: str) -> str:
    """Map LiveWan's FSDP-only wrapper paths back to plain Wan names.

    ``wanstreamer.fsdp.attach_causal_modules`` re-registers ``blocks`` and
    ``head`` below ``CausalBlock.blk`` and ``CausalHead.head`` so composable
    FSDP hooks wrap the functions entered by the custom forward.  It does not
    create or replace parameters, but it changes their ``named_parameters()``
    paths.  State evidence must use the underlying Wan identity so FSDP and
    resident placements select the same deterministic tensor coordinates.
    """

    value = canonical_state_name(name)
    layer_prefix = "causal_layers."
    if value.startswith(layer_prefix):
        layer_and_name = value[len(layer_prefix) :]
        layer, separator, remainder = layer_and_name.partition(".blk.")
        if not separator or not layer.isdigit() or not remainder:
            raise ValueError(f"unsupported LiveWan CausalBlock parameter path: {value}")
        return f"blocks.{layer}.{remainder}"
    head_prefix = "causal_head.head."
    if value.startswith(head_prefix):
        remainder = value[len(head_prefix) :]
        if not remainder:
            raise ValueError(f"unsupported LiveWan CausalHead parameter path: {value}")
        return f"head.{remainder}"
    return value


def _bind_trajectory_optimizer_parameter_names(
    *,
    named_parameters: Sequence[tuple[str, Any]],
    optimizer: Any,
    expected_parameter_ids: set[int] | None = None,
) -> dict[int, str]:
    """Bind optimizer parameters to stable model names before FSDP execution.

    Composable FSDP2 temporarily exposes unsharded ``Parameter`` objects from
    ``module.named_parameters()`` while the optimizer continues to own the
    original sharded DTensor parameters.  Looking names up by object identity
    after a forward therefore fails even though the optimizer inventory is
    unchanged.  Capture the identity-to-name relation once, at optimizer
    construction, while both views are guaranteed to agree.

    The binding is diagnostics-only and does not mutate the model, optimizer,
    parameter groups, or training state.
    """

    names: dict[int, str] = {}
    canonical_ids: dict[str, int] = {}
    for raw_name, parameter in named_parameters:
        parameter_id = id(parameter)
        name = _trajectory_livewan_state_name(raw_name)
        previous_id = canonical_ids.get(name)
        if previous_id is not None and previous_id != parameter_id:
            raise AssertionError(
                f"trajectory model has duplicate canonical parameter name: {name}"
            )
        previous_name = names.get(parameter_id)
        if previous_name is not None and previous_name != name:
            raise AssertionError(
                "trajectory model parameter has multiple canonical names: "
                f"{previous_name} vs {name}"
            )
        names[parameter_id] = name
        canonical_ids[name] = parameter_id

    optimizer_ids: list[int] = []
    optimizer_id_set: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id in optimizer_id_set:
                raise AssertionError(
                    "trajectory optimizer parameter appears in multiple parameter groups"
                )
            optimizer_ids.append(parameter_id)
            optimizer_id_set.add(parameter_id)
            if parameter_id not in names:
                raise AssertionError(
                    "trajectory optimizer parameter has no model name at construction"
                )

    if expected_parameter_ids is not None and optimizer_id_set != expected_parameter_ids:
        missing = sorted(expected_parameter_ids - optimizer_id_set)
        unexpected = sorted(optimizer_id_set - expected_parameter_ids)
        raise AssertionError(
            "trajectory optimizer parameter inventory differs from the registered "
            f"trainable inventory: missing={len(missing)} unexpected={len(unexpected)}"
        )
    return {parameter_id: names[parameter_id] for parameter_id in optimizer_ids}


def _trajectory_optimizer_named_values(
    optimizer: Any | None,
    *,
    parameter_names: Mapping[int, str],
) -> list[tuple[str, Any, Any | None]]:
    if optimizer is None:
        _require(not parameter_names, "missing optimizer has a trajectory name binding")
        return []
    result: list[tuple[str, Any, Any | None]] = []
    observed_ids: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id in observed_ids:
                raise AssertionError(
                    "trajectory optimizer parameter appears in multiple parameter groups"
                )
            observed_ids.add(parameter_id)
            name = parameter_names.get(parameter_id)
            if name is None:
                raise AssertionError(
                    "trajectory optimizer parameter is absent from its construction-time "
                    "name binding"
                )
            state = optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    value = state[key]
                    layout_source = None if hasattr(value, "placements") else parameter
                    result.append((f"{name}/{key}", value, layout_source))
    if observed_ids != set(parameter_names):
        raise AssertionError(
            "trajectory optimizer parameter groups changed after name binding"
        )
    return result


def _trajectory_state_anchor(
    *,
    torch: Any,
    rank: int,
    cycle: int,
    output_dir: Path,
    bundle: ModelBundle,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Write one canonical numeric state sketch shared by the two role pools."""

    generator_ranks = tuple(int(value) for value in bundle.metadata["generator_ranks"])
    critic_ranks = tuple(int(value) for value in bundle.metadata["critic_ranks"])
    _require(bool(generator_ranks) and bool(critic_ranks), "trajectory role ranks are empty")
    generator_primary = min(generator_ranks)
    critic_primary = min(critic_ranks)
    critic_ids = {id(parameter) for parameter in bundle.critic_lora_params}
    components = {
        "generator_trainable": collect_numeric_state_sketch(
            torch=torch,
            named_values=_trajectory_model_named_values(bundle.student),
            samples_per_tensor=16,
            primary_rank=generator_primary,
        ),
        "generator_adam": collect_numeric_state_sketch(
            torch=torch,
            named_values=_trajectory_optimizer_named_values(
                bundle.generator_optimizer,
                parameter_names=bundle.generator_optimizer_parameter_names,
            ),
            samples_per_tensor=16,
            primary_rank=generator_primary,
        ),
        "critic_trainable": collect_numeric_state_sketch(
            torch=torch,
            named_values=_trajectory_model_named_values(
                bundle.critic, parameter_ids=critic_ids
            ),
            samples_per_tensor=16,
            primary_rank=critic_primary,
        ),
        "critic_adam": collect_numeric_state_sketch(
            torch=torch,
            named_values=_trajectory_optimizer_named_values(
                bundle.critic_optimizer,
                parameter_names=bundle.critic_optimizer_parameter_names,
            ),
            samples_per_tensor=16,
            primary_rank=critic_primary,
        ),
    }
    anchor = {
        "schema_version": "distillschedule.livewan_trajectory_state_anchor.v1",
        "cycle": cycle,
        "writer_rank": 0,
        "topology": {
            "generator_replicas": list(generator_ranks),
            "critic_replicas": list(critic_ranks),
            "representative_ranks": {
                "generator": generator_primary,
                "critic": critic_primary,
            },
        },
        "components": components,
        "comparison_scope": (
            "align each component by (tensor, flat_index), then compute cosine "
            "and normalized-L2 from value_fp32"
        ),
    }
    payload = json.dumps(anchor, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    anchor["anchor_sha256"] = digest
    path = output_dir / "trajectory" / "anchors" / f"cycle_{cycle:02d}_global.json"
    if rank == 0:
        atomic_json(path, anchor)
    torch.distributed.barrier()
    fingerprints = {
        name: hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for name, value in components.items()
    }
    fingerprints["anchor"] = digest
    return fingerprints, {"path": str(path), "sha256": digest}


def _trajectory_normalized_cycle_evidence(
    *,
    torch: Any,
    rank: int,
    cycle: int,
    trace: FormalTrace,
    local_losses: Sequence[Mapping[str, Any]],
    local_optimizer: Sequence[Mapping[str, Any]],
    local_call_census: Mapping[str, int],
) -> tuple[
    tuple[str, ...],
    dict[str, str],
    dict[str, int],
    dict[str, float | None],
    dict[str, float | None],
    bool,
]:
    """Normalize physical role-pool evidence to one logical world16 view.

    The sequential denominator executes one sample on every world rank.  C24
    executes two samples on each of eight Generator ranks and the matching two
    samples on each of eight Score ranks.  Trajectory comparison is defined in
    logical global-rank coordinates, so every physical rank first contributes
    its observed counters/losses and then receives the same global view.  Rank
    ``r`` writes only slot ``r``.  This makes sample/RNG/loss/call receipts
    directly comparable without pretending the two physical topologies match.

    Call counts are exact global totals, not a per-rank mean.  The immutable
    operation trace assigns rank-qualified history lengths, so the number of
    Student block evaluations legitimately differs between logical slots.  A
    global total therefore need not be divisible by ``trace.global_batch``;
    retaining the integer total also makes the baseline and C24 comparison
    lossless.
    """

    gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
    torch.distributed.all_gather_object(
        gathered,
        {
            "losses": [dict(row) for row in local_losses],
            "optimizer": [dict(row) for row in local_optimizer],
            "call_census": {str(key): int(value) for key, value in local_call_census.items()},
        },
    )
    _require(all(row is not None for row in gathered), "trajectory evidence gather is incomplete")

    loss_values: dict[tuple[str, str], list[float]] = {}
    optimizer_values: dict[str, list[float]] = {"generator": [], "critic": []}
    global_call_census: dict[str, int] = {}
    for contributor in gathered:
        assert contributor is not None
        for row in contributor["losses"]:
            key = (str(row["kind"]), str(row["operation_key"]))
            loss_values.setdefault(key, []).append(float(row["mean"]))
        for row in contributor["optimizer"]:
            role = str(row["role"])
            if role in optimizer_values:
                optimizer_values[role].append(float(row["clip_input_norm"]))
        for key, value in contributor["call_census"].items():
            global_call_census[str(key)] = global_call_census.get(str(key), 0) + int(value)

    exact_global_call_census = dict(sorted(global_call_census.items()))
    _require(bool(exact_global_call_census), "trajectory call census is empty")

    rows = tuple(
        trace.row(cycle * FULL5_STEPS + step, rank)
        for step in range(FULL5_STEPS)
    )
    generator_key = rows[0].operation_key
    generator_values = loss_values.get(("generator", generator_key), [])
    critic_values = [
        number
        for row in rows
        for number in loss_values.get(("critic", row.operation_key), [])
    ]
    _require(bool(generator_values), f"trajectory generator loss is missing for {generator_key}")
    _require(
        len(critic_values) == FULL5_STEPS,
        f"trajectory Critic loss inventory for logical rank {rank} is not five",
    )
    _require(
        bool(optimizer_values["generator"]) and bool(optimizer_values["critic"]),
        "trajectory gradient-norm inventory is incomplete",
    )

    losses: dict[str, float | None] = {
        "generator_mean": sum(generator_values) / len(generator_values),
        "critic_mean": sum(critic_values) / len(critic_values),
    }
    gradient_norms: dict[str, float | None] = {
        role + "_mean": sum(values) / len(values)
        for role, values in optimizer_values.items()
    }
    numeric = [
        *(float(value) for value in losses.values() if value is not None),
        *(float(value) for value in gradient_norms.values() if value is not None),
    ]
    finite = bool(numeric) and all(math.isfinite(value) for value in numeric)
    return (
        tuple(row.operation_key for row in rows),
        {
            row.operation_key: hashlib.sha256(
                json.dumps(
                    dict(row.seeds), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            for row in rows
        },
        exact_global_call_census,
        losses,
        gradient_norms,
        finite,
    )


def _append_trajectory_cycle_receipt(
    *,
    torch: Any,
    rank: int,
    cycle: int,
    trace: FormalTrace,
    seed: int,
    arm_id: str,
    execution_mode: str,
    implementation: Mapping[str, str],
    versions: Mapping[str, int],
    initial_versions: Mapping[str, int],
    local_losses: Sequence[Mapping[str, Any]],
    local_optimizer: Sequence[Mapping[str, Any]],
    local_call_census: Mapping[str, int],
    state_fingerprints: Mapping[str, str],
    writer: AppendOnlyTrajectoryWriter,
) -> tuple[bool, dict[str, int]]:
    """Append one post-cycle receipt after topology-independent normalization."""

    (
        sample_identities,
        rng_identities,
        call_census,
        losses,
        gradient_norms,
        finite,
    ) = _trajectory_normalized_cycle_evidence(
        torch=torch,
        rank=rank,
        cycle=cycle,
        trace=trace,
        local_losses=local_losses,
        local_optimizer=local_optimizer,
        local_call_census=local_call_census,
    )
    cycle_number = cycle + 1
    expected_versions = {
        "generator": int(initial_versions["generator"]) + cycle_number,
        "critic": int(initial_versions["critic"]) + CRITIC_UPDATES * cycle_number,
        "ema": int(initial_versions["ema"]),
    }
    writer.append(
        TrajectoryCycleReceipt(
            run_id=str(os.environ.get("DISTILLSCHEDULE_RUN_ID", "unknown")),
            workload="livewan_wan13b_teacher14b_sfdmd",
            variant=arm_id,
            rank=rank,
            world_size=WORLD_SIZE,
            seed=seed,
            cycle=cycle_number,
            expected_total_cycles=trace.cycles,
            logical_step=cycle * FULL5_STEPS,
            matched_contract={
                "trace_sha256": trace.sha256,
                "global_batch": str(trace.global_batch),
                "updates": "G1_C5_EMA0",
                "sample_policy": "80_SOURCE_ROWS_REUSED_FRESH_OPERATION_KEYED_RNG",
                "trajectory_schema": "trajectory_coarse_v1",
                "call_census_scope": "EXACT_GLOBAL_WORLD16_TOTALS",
            },
            implementation={
                "execution_mode": execution_mode,
                "trajectory_arm_id": arm_id,
                **{str(key): str(value) for key, value in implementation.items()},
            },
            sample_identities=sample_identities,
            rng_identities=rng_identities,
            versions=expected_versions,
            cycle_updates={"generator": 1, "critic": 5, "ema": 0},
            cumulative_updates={
                "generator": cycle_number,
                "critic": CRITIC_UPDATES * cycle_number,
                "ema": 0,
            },
            call_census=call_census,
            losses=losses,
            gradient_norms=gradient_norms,
            tensor_contracts={
                "rollout_latent": {
                    "shape": list(LATENT_SHAPE),
                    "dtype": "torch.float32",
                }
            },
            state_fingerprints=state_fingerprints,
            finite=finite,
            runtime_status="PASS" if finite else "NONFINITE",
        )
    )
    return finite, call_census


@dataclass
class RunEvidence:
    output_fingerprints: list[dict[str, Any]] = field(default_factory=list)
    output_tensors: dict[str, Any] = field(default_factory=dict)
    loss_fingerprints: list[dict[str, Any]] = field(default_factory=list)
    transport_bytes: int = 0
    gradient_sync: list[dict[str, Any]] = field(default_factory=list)
    optimizer_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    logical_score_calls: dict[str, int] = field(default_factory=dict)
    score_context_prefetch: dict[str, Any] = field(default_factory=dict)
    split_dmd_shadow: dict[str, Any] = field(default_factory=dict)
    generator_runahead_transport: dict[str, Any] = field(default_factory=dict)
    continuous_pipeline: dict[str, Any] = field(default_factory=dict)
    critic_shadow_hot_sync: dict[str, Any] = field(default_factory=dict)
    score_receive_waits: list[dict[str, Any]] = field(default_factory=list)
    generator_backpressure: list[dict[str, Any]] = field(default_factory=list)
    critic_checkpoint_rng: dict[str, Any] = field(default_factory=dict)
    physical_critic_backward_calls: int = 0

    def add_logical_score_calls(self, key: str, count: int) -> None:
        _require(count > 0, "logical Score call count must be positive")
        parts = key.split("/")
        _require(len(parts) == 3, "logical Score call key must include role/kind/replay")
        for logical_key in (parts[0], "/".join(parts[:2]), key):
            self.logical_score_calls[logical_key] = (
                self.logical_score_calls.get(logical_key, 0) + count
            )

    def add_physical_critic_backward_calls(self, count: int) -> None:
        """Record completed physical Critic backward invocations.

        This counter is intentionally independent of the logical Score-call
        inventory: a physical B2 backward represents two logical samples,
        whereas the attribution arm executes those samples as two physical B1
        backwards before the same single optimizer commit.
        """

        _require(count > 0, "physical Critic backward count must be positive")
        self.physical_critic_backward_calls += count


def _record_optimizer_diagnostics(
    torch: Any,
    evidence: RunEvidence,
    *,
    run_kind: str,
    role: str,
    step: int,
    parameters: Sequence[Any],
    clip_norm: Any,
    gradient_before_clip: Mapping[str, float | int] | None,
) -> None:
    trajectory_mode = run_kind == "trajectory"
    # Keep the historical correctness-only branch explicit.  Trajectory is an
    # additional low-volume diagnostic consumer; all other legacy run kinds
    # return at the same point as before.
    if run_kind != "correctness":
        if not trajectory_mode:
            return
    norm = clip_norm.full_tensor() if hasattr(clip_norm, "full_tensor") else clip_norm
    if trajectory_mode:
        evidence.optimizer_diagnostics.append(
            {
                "role": role,
                "step": step,
                "clip_input_norm": float(norm.detach().float().cpu()),
                "gradient_tensor_metrics": "NOT_COLLECTED_TRAJECTORY_LOW_OVERHEAD",
            }
        )
        return
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    evidence.optimizer_diagnostics.append(
        {
            "role": role,
            "step": step,
            "gradient_before_clip": gradient_before_clip,
            "gradient_after_clip": _state_metrics(torch, gradients),
            "clip_input_norm": float(norm.detach().float().cpu()),
        }
    )


def _write_rank_receipt(
    *,
    output_dir: Path,
    rank: int,
    value: Mapping[str, Any],
) -> Path:
    path = output_dir / "receipts" / f"rank_{rank:03d}.json"
    atomic_json(path, value)
    return path


def _score_physical_batch_size(plan: Mapping[str, Any], *, mode: str) -> int:
    value = int(plan.get("metadata", {}).get("score_physical_batch_size", 1))
    _require(value in {1, 2}, "unsupported Score physical batch size")
    if mode == "baseline":
        _require(value == 1, "sequential denominator must remain physical B1")
    if value == 2:
        _require(mode == "pipeline", "Score B2 is a pipeline-only candidate")
        _require(
            plan.get("metadata", {}).get("pool_policy")
            in {
                "persistent_8g_8score_teacher_fsdp8_score_b2",
                "persistent_6g_10score_teacher_fsdp10",
                "persistent_6g_10score_teacher_node_local_hierarchical",
            },
            "Score B2 plan policy identity mismatch",
        )
    return value


def _critic_physical_batch_size(plan: Mapping[str, Any], *, mode: str) -> int:
    value = int(plan.get("metadata", {}).get("critic_physical_batch_size", 1))
    _require(value in {1, 2}, "unsupported Critic physical batch size")
    if mode == "baseline":
        _require(value == 1, "sequential denominator must remain Critic B1")
    if value == 2:
        _require(
            plan.get("metadata", {}).get("pool_policy")
            in {
                "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2",
                "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_local_context",
                ASYNC_CRITIC_POOL_POLICY,
                TRACE_BALANCED_POOL_POLICY,
            },
            "Critic B2 is registered only for the run-ahead B1-DMD candidate",
        )
        _require(
            int(plan.get("metadata", {}).get("score_physical_batch_size", 1)) == 1,
            "Critic-only B2 must preserve the step0 DMD B1 path",
        )
    return value


def _full_context_lens_elision(plan: Mapping[str, Any], *, mode: str) -> bool:
    """Validate the exact full-text mask elision candidate."""

    metadata = plan.get("metadata", {})
    enabled = metadata.get("full_context_lens_elision", False)
    _require(isinstance(enabled, bool), "full-context lens elision must be boolean")
    if not enabled:
        return False
    _require(mode == "pipeline", "full-context lens elision is candidate-only")
    _require(
        metadata.get("pool_policy")
        in {TRACE_BALANCED_POOL_POLICY, PHASE_EXPANDED_CRITIC16_POOL_POLICY},
        "full-context lens elision requires the validated C10 substrate",
    )
    _require(
        metadata.get("context_length") == TEXT_SHAPE[1],
        "full-context lens length differs from the immutable prompt bundle",
    )
    _require(
        metadata.get("context_padding_policy") == "all_512_tokens_valid_no_padding",
        "full-context lens elision lacks its no-padding proof contract",
    )
    _require(
        metadata.get("attention_mask_equivalence")
        == "ctx_lens_512_equivalent_to_none_at_exact_tensor_length",
        "full-context attention-mask equivalence contract changed",
    )
    return True


def _score_context_prefetch_enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    """Validate the frozen-context head-bubble overlap candidate."""

    metadata = plan.get("metadata", {})
    enabled = metadata.get("score_context_prefetch", False)
    _require(isinstance(enabled, bool), "Score context prefetch must be boolean")
    if not enabled:
        return False
    _require(mode == "pipeline", "Score context prefetch is candidate-only")
    _require(
        metadata.get("pool_policy") == TRACE_BALANCED_POOL_POLICY,
        "Score context prefetch requires the validated C11 substrate",
    )
    _require(
        metadata.get("full_context_lens_elision") is True,
        "Score context prefetch requires the exact full-context contract",
    )
    expected = {
        "score_context_prefetch_scope": "all_10_lane_rows_before_first_latent_receive",
        "score_context_invariance": "frozen_teacher_and_critic_text_embeddings_across_full5",
        "score_context_calls_preserved": True,
        "score_context_prefetch_timing_position": "inside_full5_before_first_latent_receive",
        "score_context_prefetch_rows_per_lane": 10,
        "score_context_teacher_rows_per_lane": 2,
    }
    for key, value in expected.items():
        _require(metadata.get(key) == value, f"Score context prefetch contract mismatch: {key}")
    return True


def _critic_lora_scale_one_enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    """Validate the exact scale-one LoRA epilogue specialization."""

    metadata = plan.get("metadata", {})
    kernel = metadata.get("critic_lora_kernel")
    if kernel is None:
        return False
    _require(isinstance(kernel, Mapping), "Critic LoRA kernel contract must be a mapping")
    _require(mode == "pipeline", "Critic LoRA kernel is candidate-only")
    _require(
        metadata.get("pool_policy")
        in {TRACE_BALANCED_POOL_POLICY, PHASE_EXPANDED_CRITIC16_POOL_POLICY}
        and metadata.get("full_context_lens_elision") is True,
        "Critic LoRA kernel requires the validated C11 substrate",
    )
    expected = {
        "enabled": True,
        "scale": 1.0,
        "base_input_output_dtype": "bfloat16",
        "lora_parameter_compute_dtype": "float32",
        "source_forward_sha256": (
            "70d00d496aa2bd8ea06d64e6b04af873644eff52539396f74ec586a4cbbe97a3"
        ),
    }
    name = kernel.get("name")
    if name == "scale_one_identity_epilogue_v1":
        expected.update(
            name=name,
            math_change="remove_identity_multiply_only",
        )
    elif name == "scale_one_fused_cast_add_v1":
        expected.update(
            name=name,
            math_change=(
                "remove_identity_multiply_and_fuse_fp32_to_bf16_cast_with_bf16_add"
            ),
            fused_min_numel=8_000_000,
            triton_block_size=256,
        )
    elif name == "scale_one_autocast_bf16_input_elision_v1":
        expected.update(
            name=name,
            math_change=(
                "remove_identity_multiply_and_redundant_pre_autocast_fp32_input_cast"
            ),
            autocast_dtype="bfloat16",
            required_lora_output_dtype="bfloat16",
        )
    else:
        _require(False, "Unknown Critic LoRA kernel candidate")
    _require(dict(kernel) == expected, "Critic LoRA kernel contract changed")
    return True


def _score_modulation_kernel_enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    """Validate the exact Score-rank-only LiveWan modulation fusion."""

    metadata = plan.get("metadata", {})
    kernel = metadata.get("score_modulation_kernel")
    if kernel is None:
        return False
    _require(isinstance(kernel, Mapping), "Score modulation kernel must be a mapping")
    _require(mode == "pipeline", "Score modulation kernel is candidate-only")
    _require(
        metadata.get("pool_policy") == TRACE_BALANCED_POOL_POLICY
        and metadata.get("full_context_lens_elision") is True,
        "Score modulation kernel requires the validated C11 substrate",
    )
    expected = {
        "enabled": True,
        "name": "score_bf16_modulation_fusion_v1",
        "scope": "score_ranks_only_critic_and_teacher",
        "source_modulate_sha256": (
            "f244152360d40d6745e28bed4add081604312d04fe2f342842a32432f4bd20ad"
        ),
        "math_change": "fuse_exact_bf16_to_fp32_modulation_pointwise_chain",
        "normalized_shift_scale_dtype": "bfloat16",
        "output_dtype": "float32",
        "per_frame": False,
        "triton_block_size": 256,
    }
    _require(dict(kernel) == expected, "Score modulation kernel contract changed")
    return True


def _score_rmsnorm_kernel_enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    from distillschedule.runtime.livewan_score_rmsnorm import PLAN_CONTRACT

    value = plan.get("metadata", {}).get("score_rmsnorm_kernel")
    if value is None:
        return False
    _require(mode == "pipeline", "Score RMSNorm is pipeline-only")
    _require(value == PLAN_CONTRACT, "Score RMSNorm contract changed")
    _require(_score_modulation_kernel_enabled(plan, mode=mode), "Score RMSNorm requires C24")
    return True


def _generator_rollout_cast_cache_enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    from distillschedule.runtime.livewan_rollout_cast_cache import PLAN_CONTRACT

    metadata = plan.get("metadata", {})
    value = metadata.get("generator_rollout_cast_cache")
    if value is None:
        return False
    _require(mode == "pipeline" and dict(value) == PLAN_CONTRACT,
             "C28 Generator rollout cast-cache contract changed")
    _require(_score_rmsnorm_kernel_enabled(plan, mode=mode)
             and _generator_cross_kv_cache_enabled(plan, mode=mode)
             and _generator_attention_backend_enabled(plan, mode=mode)
             and metadata.get("score_prompt_staging") is None
             and metadata.get("critic_attention_checkpoint") is None
             and not metadata.get("score_context_prefetch"),
             "C28 requires unchanged C25 parent, no combined experiment")
    return True


def _generator_rms_epilogue_enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    from distillschedule.runtime.livewan_generator_rms_epilogue import PLAN_CONTRACT

    value = plan.get("metadata", {}).get("generator_rms_epilogue")
    if value is None:
        return False
    _require(mode == "pipeline" and dict(value) == PLAN_CONTRACT,
             "C29 Generator RMS epilogue contract changed")
    _require(_generator_rollout_cast_cache_enabled(plan, mode=mode),
             "C29 requires unchanged C28 parent")
    return True


def _score_prompt_staging_enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    from distillschedule.runtime.livewan_context_staging import PLAN_CONTRACT

    metadata = plan.get("metadata", {})
    value = metadata.get("score_prompt_staging")
    if value is None:
        return False
    _require(mode == "pipeline" and dict(value) == PLAN_CONTRACT,
             "C27 Score input staging contract changed")
    _require(_score_rmsnorm_kernel_enabled(plan, mode=mode)
             and _critic_physical_batch_size(plan, mode=mode) == 2
             and metadata.get("critic_attention_checkpoint") is None
             and metadata.get("score_context_prefetch", False) is False,
             "C27 requires C25 parent without SAC/projected context prefetch")
    return True


def _critic_attention_checkpoint_enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    from distillschedule.runtime.livewan_attention_checkpoint import PLAN_CONTRACT

    value = plan.get("metadata", {}).get("critic_attention_checkpoint")
    if value is None:
        return False
    _require(mode == "pipeline" and value == PLAN_CONTRACT, "C26 SAC contract changed")
    _require(_score_rmsnorm_kernel_enabled(plan, mode=mode), "C26 requires C25 parent")
    _require(_continuous_pipeline_contract(plan, mode=mode) is not None, "C26 requires continuous B16")
    _require(_critic_physical_batch_size(plan, mode=mode) == 2, "C26 Critic B2 changed")
    _require(not _critic_checkpoint_rng_elision_enabled(plan, mode=mode), "C26 cannot change checkpoint RNG")
    return True


def _score_rope_kernel_enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    """Validate the exact Score-rank-only LiveWan RoPE fusion."""

    metadata = plan.get("metadata", {})
    kernel = metadata.get("score_rope_kernel")
    if kernel is None:
        return False
    _require(isinstance(kernel, Mapping), "Score RoPE kernel must be a mapping")
    _require(mode == "pipeline", "Score RoPE kernel is candidate-only")
    _require(
        metadata.get("pool_policy") == TRACE_BALANCED_POOL_POLICY
        and metadata.get("full_context_lens_elision") is True,
        "Score RoPE kernel requires the validated C11 substrate",
    )
    expected = {
        "enabled": True,
        "name": "score_bf16_rope_fusion_v1",
        "scope": "score_ranks_only_critic_and_teacher",
        "source_apply_rope_sha256": (
            "8ab587e9f15cc969659609ec2f9cc79f4d4cf6fba468ae8c54267dc949224482"
        ),
        "math_change": "fuse_exact_fp32_complex_rope_and_immediate_bf16_conversion",
        "input_output_dtype": "bfloat16",
        "table_dtype": "complex64",
        "fp32_rounding_order": "mul_then_first_product_fma_rn",
        "triton_block_size": 512,
    }
    _require(dict(kernel) == expected, "Score RoPE kernel contract changed")
    return True


def _score_attention_backend_enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    """Validate the pinned Score-rank-only FlashAttention-2 backend."""

    metadata = plan.get("metadata", {})
    backend = metadata.get("score_attention_backend")
    if backend is None:
        return False
    _require(isinstance(backend, Mapping), "Score attention backend must be a mapping")
    _require(mode == "pipeline", "Score attention backend is candidate-only")
    _require(
        metadata.get("pool_policy") == TRACE_BALANCED_POOL_POLICY
        and metadata.get("full_context_lens_elision") is True
        and metadata.get("score_rope_kernel", {}).get("enabled") is True,
        "Score attention backend requires the validated C15 substrate",
    )
    expected = {
        "enabled": True,
        "name": "score_flash_attn2_unmasked_bf16_v1",
        "scope": "score_ranks_only_critic_and_teacher",
        "source_flash_attention_sha256": (
            "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
        ),
        "artifact_metadata_sha256": (
            "7749dd68b81a544c1973d68c30be5aa316ee1a718ae0228e0368c2cd04365516"
        ),
        "artifact_package_sha256": (
            "0c560f96b857c188c4a8297ff27d1299619957c3459a5e433e132329fbfdf066"
        ),
        "artifact_extension_sha256": (
            "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
        ),
        "math_change": (
            "replace_cudnn_sdpa_with_pinned_flash_attn2_under_fixed_parity_gate"
        ),
        "mask_contract": "full_length_unmasked_only",
        "input_output_dtype": "bfloat16",
        "head_dimension": 128,
    }
    _require(dict(backend) == expected, "Score attention backend contract changed")
    return True


def _generator_attention_backend_enabled(
    plan: Mapping[str, Any], *, mode: str
) -> bool:
    """Validate the C20 pinned Generator-rank FlashAttention-2 backend."""

    metadata = plan.get("metadata", {})
    backend = metadata.get("generator_attention_backend")
    if backend is None:
        return False
    _require(
        isinstance(backend, Mapping),
        "Generator attention backend must be a mapping",
    )
    _require(mode == "pipeline", "Generator attention backend is candidate-only")
    _require(
        metadata.get("pool_policy") == TRACE_BALANCED_POOL_POLICY
        and metadata.get("full_context_lens_elision") is True
        and metadata.get("generator_cross_attention_kv_cache", {}).get("enabled")
        is True
        and metadata.get("generator_runahead_transport", {}).get("enabled") is True,
        "Generator attention backend requires the validated C19 substrate",
    )
    expected = {
        "enabled": True,
        "name": "generator_flash_attn2_unmasked_exact_cast_v1",
        "scope": "generator_ranks_student_and_frozen_critic_shadow",
        "source_flash_attention_sha256": (
            "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
        ),
        "artifact_metadata_sha256": (
            "7749dd68b81a544c1973d68c30be5aa316ee1a718ae0228e0368c2cd04365516"
        ),
        "artifact_package_sha256": (
            "0c560f96b857c188c4a8297ff27d1299619957c3459a5e433e132329fbfdf066"
        ),
        "artifact_extension_sha256": (
            "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
        ),
        "math_change": (
            "replace_cudnn_sdpa_with_pinned_flash_attn2_under_fixed_parity_gate"
        ),
        "mask_contract": "full_length_unmasked_only",
        "raw_input_dtype_contract": [
            "bfloat16/bfloat16/bfloat16",
            "float32/float32/bfloat16",
        ],
        "kernel_compute_dtype": "bfloat16",
        "output_dtype": "preserve_query_dtype",
        "head_dimension": 128,
    }
    _require(dict(backend) == expected, "Generator attention backend contract changed")
    return True


def _generator_cross_kv_cache_enabled(
    plan: Mapping[str, Any], *, mode: str
) -> bool:
    """Validate exact no-grad text K/V reuse on Generator streamers."""

    metadata = plan.get("metadata", {})
    cache = metadata.get("generator_cross_attention_kv_cache")
    if cache is None:
        return False
    _require(isinstance(cache, Mapping), "Generator cross K/V cache must be a mapping")
    _require(mode == "pipeline", "Generator cross K/V cache is candidate-only")
    _require(
        metadata.get("pool_policy") == TRACE_BALANCED_POOL_POLICY
        and metadata.get("full_context_lens_elision") is True
        and metadata.get("score_attention_backend", {}).get("enabled") is True,
        "Generator cross K/V cache requires the validated C16 substrate",
    )
    expected = {
        "enabled": True,
        "name": "generator_nograd_text_kv_cache_v1",
        "scope": "generator_ranks_streamer_nograd_only",
        "source_set_text_sha256": (
            "def2452ea9528abb9af2aa166d78afdf9b8b7c5daa52c0fc87bfd63bc9832386"
        ),
        "source_cross_forward_sha256": (
            "0df9365afb37bb8a876d8c8659ca95d1caaffb89bf3e64918b782eb86b0a8253"
        ),
        "source_flash_attention_sha256": (
            "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
        ),
        "math_change": (
            "reuse_exact_cross_kv_projection_within_fixed_weight_no_grad_rollout"
        ),
        "context_contract": "b1_full_512_token_fp32_projected_context",
        "grad_contract": "grad_enabled_checkpoint_forward_and_replay_bypass_cache",
        "block_count": 30,
        "cache_bytes_per_streamer": 141_557_760,
        "max_streamers_per_generator_rank": 2,
    }
    _require(dict(cache) == expected, "Generator cross K/V cache contract changed")
    return True


def _prefetch_score_sample_contexts(
    torch: Any,
    *,
    trace: FormalTrace,
    loader: InputLoader,
    bundle: ModelBundle,
    lane_slots: Sequence[int],
    device: Any,
    run_kind: str,
) -> tuple[dict[tuple[int, int], ScoreSampleContext], dict[str, Any]]:
    """Project the ten immutable lane contexts inside the initial Score bubble.

    Only the frozen text-embedding modules participate.  The helper launches no
    synchronization and preserves the original two Teacher plus ten Critic
    projection calls; it merely moves them before the first latent receive.
    """

    _require(len(lane_slots) == 2, "Score context prefetch requires two lane slots")
    for role, model in (("teacher", bundle.teacher), ("critic", bundle.critic)):
        _require(model is not None, f"Score context prefetch is missing {role}")
        _require(
            all(not parameter.requires_grad for parameter in model.text_embedding.parameters()),
            f"Score context prefetch requires frozen {role} text embedding",
        )
        _require(
            not any(
                "dropout" in type(module).__name__.lower()
                for module in model.text_embedding.modules()
            ),
            f"Score context prefetch forbids stochastic {role} text embedding",
        )

    contexts: dict[tuple[int, int], ScoreSampleContext] = {}
    teacher_contexts = 0
    critic_contexts = 0
    resident_bytes = 0
    with _profile_range(
        torch,
        run_kind=run_kind,
        label="livewan/pipeline/score/context_prefetch_all",
    ):
        for step in range(FULL5_STEPS):
            rows = tuple(trace.row(step, slot) for slot in lane_slots)
            for wave, row in enumerate(rows):
                raw_context = loader.context_from_trace(row, device=device)
                sample_context = prepare_score_sample_context(
                    torch,
                    bundle=bundle,
                    raw_context=raw_context,
                    include_teacher=step == 0,
                )
                contexts[(step, wave)] = sample_context
                critic_contexts += 1
                resident_bytes += (
                    sample_context.critic.numel()
                    * sample_context.critic.element_size()
                )
                if sample_context.teacher is not None:
                    teacher_contexts += 1
                    resident_bytes += (
                        sample_context.teacher.numel()
                        * sample_context.teacher.element_size()
                    )
    _require(len(contexts) == 10, "Score context prefetch did not build ten lane rows")
    _require(critic_contexts == 10, "Score context prefetch Critic call count changed")
    _require(teacher_contexts == 2, "Score context prefetch Teacher call count changed")
    return contexts, {
        "enabled": True,
        "role": "score",
        "execution_status": "PASS",
        "logical_context_count": len(contexts),
        "critic_context_count": critic_contexts,
        "teacher_context_count": teacher_contexts,
        "gpu_resident_bytes": resident_bytes,
        "timing_position": "inside_full5_before_first_latent_receive",
        "calls_preserved": True,
    }


def _validate_plan_identity(plan: Mapping[str, Any], *, mode: str) -> None:
    _require(int(plan["hardware"]["world_size"]) == WORLD_SIZE, "plan is not world16")
    batch = int(plan["workload"]["shape"]["batch_size"])
    _require(batch in {GLOBAL_BATCH, 32}, "plan is not registered global-B16/B32")
    if batch == 32:
        _require(
            dict(plan.get("metadata", {}).get("continuous_pipeline", {})).get("name")
            == B32_CONTINUOUS_NAME,
            "global-B32 is registered only for the continuous systems diagnostic",
        )
        _pipeline_8x8_slot_groups(plan)
    _require(plan["metadata"]["execution_mode"] == mode, "plan execution mode mismatch")
    _require(plan["workload"]["metadata"]["livewan_commit"] == LIVEWAN_COMMIT, "plan LiveWan mismatch")
    _require(plan["workload"]["metadata"]["wan21_commit"] == WAN21_COMMIT, "plan Wan2.1 mismatch")
    _require(not plan["workload"]["branches"]["ema"], "EMA must be outside Full5")
    _score_physical_batch_size(plan, mode=mode)
    _critic_physical_batch_size(plan, mode=mode)
    _full_context_lens_elision(plan, mode=mode)
    _score_context_prefetch_enabled(plan, mode=mode)
    _critic_lora_scale_one_enabled(plan, mode=mode)
    _score_modulation_kernel_enabled(plan, mode=mode)
    _score_rmsnorm_kernel_enabled(plan, mode=mode)
    _critic_attention_checkpoint_enabled(plan, mode=mode)
    _score_prompt_staging_enabled(plan, mode=mode)
    _generator_rollout_cast_cache_enabled(plan, mode=mode)
    _generator_rms_epilogue_enabled(plan, mode=mode)
    _score_rope_kernel_enabled(plan, mode=mode)
    _score_attention_backend_enabled(plan, mode=mode)
    _generator_cross_kv_cache_enabled(plan, mode=mode)
    _generator_attention_backend_enabled(plan, mode=mode)
    _critic_checkpoint_rng_elision_enabled(plan, mode=mode)
    if (
        str(plan.get("metadata", {}).get("pool_policy", ""))
        == PHASE_EXPANDED_CRITIC16_POOL_POLICY
    ):
        groups = {
            str(row["name"]): tuple(int(rank) for rank in row["ranks"])
            for row in plan["rank_groups"]
        }
        _require(
            plan["training_backends"]["generator"]["kind"] == "native",
            "variable-count phase-expanded Generator must remain resident",
        )
        _require(
            plan["training_backends"]["score"]["kind"] == "fsdp",
            "phase-expanded Critic must use FSDP2",
        )
        _require(
            groups.get("critic") == tuple(range(WORLD_SIZE)),
            "phase-expanded Critic group must cover world16",
        )
        _require(
            plan["metadata"].get(
                "variable_rollout_call_counts_require_resident_generator"
            )
            is True,
            "phase-expanded variable rollout-count guard is missing",
        )


def _validate_only(
    *,
    plan: Mapping[str, Any],
    trace: FormalTrace,
    bootstrap: Bootstrap,
    source_receipt: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    _validate_plan_identity(plan, mode=mode)
    for component in ("student_base", "critic_base", "teacher_base"):
        for row in bootstrap.value[component]["files"]:
            path = Path(row["path"])
            _require(path.is_file() and path.stat().st_size == int(row["bytes"]), f"asset missing: {path}")
            _require(sha256_file(path) == row["sha256"], f"checkpoint SHA mismatch: {path}")
    for row in {item.sample_path: item for item in trace.rows}.values():
        path = Path(row.sample_path)
        _require(path.is_file(), f"trace sample is missing: {path}")
        if row.sample_sha256 is not None:
            _require(sha256_file(path) == row.sample_sha256, f"trace sample SHA mismatch: {path}")
    return {
        "status": "PASS",
        "stage": "validate_only",
        "mode": mode,
        "trace_sha256": trace.sha256,
        "bootstrap_sha256": bootstrap.sha256,
        "source_receipt_sha256": source_receipt["source_receipt_sha256"],
        "gpu_workload": "NOT_RUN",
    }


def run_worker(
    *,
    plan_path: str | Path,
    mode: str,
    run_kind: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    plan_path = Path(plan_path).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    critic_b1_attribution = _trajectory_critic_batch_ablation_enabled(
        plan, mode=mode
    )
    _require(
        not critic_b1_attribution or run_kind == "trajectory",
        "C24 Critic-B1 attribution is diagnostic trajectory-only",
    )
    trajectory_cycles = 0
    trajectory_seed = 0
    trajectory_arm_id = ""
    two_cycle_timing = run_kind == "two-cycle-timing"
    two_cycle_seed = 0
    two_cycle_arm_id = ""
    if run_kind == "trajectory":
        trajectory_cycles = int(os.environ.get("LIVEWAN_TRAJECTORY_CYCLES", "0"))
        trajectory_seed = int(os.environ.get("LIVEWAN_TRAJECTORY_SEED", "0"))
        trajectory_arm_id = os.environ.get("LIVEWAN_TRAJECTORY_ARM_ID", "")
        _require(trajectory_arm_id != "", "trajectory arm identity is missing")
        _apply_trajectory_contract(plan, cycles=trajectory_cycles, mode=mode)
    elif two_cycle_timing:
        two_cycle_seed = int(os.environ.get("LIVEWAN_TWO_CYCLE_SEED", "-1"))
        two_cycle_arm_id = os.environ.get("LIVEWAN_TWO_CYCLE_ARM_ID", "")
        _require(two_cycle_seed >= 0, "two-cycle timing seed is missing")
        _require(two_cycle_arm_id != "", "two-cycle timing arm identity is missing")
        _require(
            not os.environ.get("DISTILLSCHEDULE_NSYS_BIN")
            and not os.environ.get("DISTILLSCHEDULE_NCU_BIN"),
            "two-cycle native timing cannot run under a profiler",
        )
        _apply_two_cycle_timing_contract(plan, mode=mode)
    _validate_plan_identity(plan, mode=mode)
    continuous_contract = _continuous_pipeline_contract(plan, mode=mode)
    continuous_cycles = (
        int(continuous_contract["cycles"])
        if continuous_contract is not None
        else 1
    )
    trace_cycles = (
        trajectory_cycles
        if run_kind == "trajectory"
        else (TWO_CYCLE_TIMING_CYCLES if two_cycle_timing else continuous_cycles)
    )
    registered_batch = int(plan["workload"]["shape"]["batch_size"])
    trace = FormalTrace.load(
        plan["evidence"]["operation_trace"],
        expected_cycles=trace_cycles,
        expected_global_batch=registered_batch,
    )
    _validate_trace_balanced_pairing(plan, trace)
    bootstrap = Bootstrap.load(plan["evidence"]["checkpoint"], trace=trace)
    if run_kind == "trajectory" or two_cycle_timing:
        trace_payload = json.loads(trace.path.read_text(encoding="utf-8"))
        _require(
            int(trace_payload.get("seed", -1))
            == (two_cycle_seed if two_cycle_timing else trajectory_seed),
            "run seed differs from the immutable trace seed",
        )
        diagnostic = dict(bootstrap.value.get("continuous_diagnostic", {}))
        _require(
            int(diagnostic.get("cycles", -1)) == trace_cycles
            and diagnostic.get("source_reuse_across_cycles")
            is (trace_cycles > 1)
            and dict(diagnostic.get("expected_final_versions", {}))
            == {
                "generator": trace_cycles,
                "critic": trace_cycles * CRITIC_UPDATES,
                "ema": 0,
            },
            "bootstrap contract differs from the requested cycle count",
        )
    root = Path(repo_root).expanduser().resolve()
    source_receipt = validate_source_receipt(bootstrap, repo_root=root)
    two_cycle_gate = None
    if two_cycle_timing:
        two_cycle_gate = _load_two_cycle_timing_gate(
            path=os.environ.get("LIVEWAN_PAIR_CORRECTNESS_RECEIPT", ""),
            expected_sha256=os.environ.get("LIVEWAN_PAIR_CORRECTNESS_SHA256", ""),
            expected_head=os.environ.get("LIVEWAN_EXPECTED_HEAD", ""),
            mode=mode,
            pipeline_variant=os.environ.get("LIVEWAN_PIPELINE_VARIANT", ""),
            arm_id=two_cycle_arm_id,
            seed=two_cycle_seed,
            trace=trace,
            bootstrap=bootstrap,
            source_receipt=source_receipt,
        )
    configure_python_sources(repo_root=root, source_receipt=source_receipt)
    output_dir = Path(plan["evidence"]["output_dir"]).expanduser().resolve()
    if run_kind == "validate":
        result = _validate_only(
            plan=plan, trace=trace, bootstrap=bootstrap,
            source_receipt=source_receipt, mode=mode,
        )
        node_rank = int(os.environ.get("DISTILLSCHEDULE_NODE_RANK", "0"))
        result["node_rank"] = node_rank
        atomic_json(output_dir / f"validate_only.node{node_rank}.json", result)
        return result
    _require(
        run_kind in {"capacity", "correctness", "timing", "profile"}
        or run_kind in {"trajectory", "two-cycle-timing"},
        "invalid run kind",
    )

    import torch
    import torch.distributed as dist
    from wanstreamer import blockcausal as bc

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _require(world_size == WORLD_SIZE, "runtime is not world16")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    from distillschedule.runtime.livewan_cuda_setup import materialize_blas_handles

    # Shared hardware compatibility, before any model construction/timer.
    cuda_library_setup = materialize_blas_handles(torch, device)
    effective_plan_sha256 = _effective_plan_sha256(plan)
    if two_cycle_timing and local_rank == 0:
        node_rank = int(os.environ.get("DISTILLSCHEDULE_NODE_RANK", "-1"))
        _require(node_rank in {0, 1}, "two-cycle timing node rank is missing")
        atomic_json(
            output_dir / f"effective_plan.node{node_rank}.json",
            {**plan, "effective_plan_sha256": effective_plan_sha256},
        )
    critic_lora_kernel = {
        "enabled": False,
        "name": "native_livewan_lora",
        "execution_status": "DISABLED",
    }
    if _critic_lora_scale_one_enabled(plan, mode=mode):
        from distillschedule.runtime.livewan_lora_kernel import (
            install_scale_one_lora_epilogue,
        )

        kernel_name = plan["metadata"]["critic_lora_kernel"]["name"]
        critic_lora_kernel = install_scale_one_lora_epilogue(
            torch,
            fused_cast_add=(kernel_name == "scale_one_fused_cast_add_v1"),
            autocast_input_elision=(
                kernel_name == "scale_one_autocast_bf16_input_elision_v1"
            ),
            fused_min_numel=8_000_000,
        )
    score_modulation_kernel = {
        "enabled": False,
        "name": "native_livewan_modulation",
        "execution_status": "DISABLED",
    }
    if _score_modulation_kernel_enabled(plan, mode=mode):
        _, score_ranks = _plan_role_ranks(plan, mode=mode)
        if rank in score_ranks:
            from distillschedule.runtime.livewan_modulation_kernel import (
                install_score_modulation_kernel,
            )

            score_modulation_kernel = install_score_modulation_kernel(torch)
            score_modulation_kernel.update(
                role="score",
                execution_status="PASS",
            )
        else:
            score_modulation_kernel = {
                "enabled": True,
                "name": "score_bf16_modulation_fusion_v1",
                "source_modulate_sha256": (
                    "f244152360d40d6745e28bed4add081604312d04fe2f342842a32432f4bd20ad"
                ),
                "function_patched": False,
                "install_status": "NOT_APPLICABLE",
                "role": "generator",
                "execution_status": "NOT_APPLICABLE",
                "invocation_count": 0,
                "backward_invocation_count": 0,
                "element_count": 0,
                "batch_histogram": {},
                "dimension_histogram": {},
                "observed_contract": None,
            }
    score_rmsnorm_kernel = {"enabled": False, "execution_status": "DISABLED"}
    if _score_rmsnorm_kernel_enabled(plan, mode=mode):
        _, score_ranks = _plan_role_ranks(plan, mode=mode)
        if rank in score_ranks:
            from distillschedule.runtime.livewan_score_rmsnorm import install_frozen_score_rmsnorm

            score_rmsnorm_kernel = install_frozen_score_rmsnorm(torch)
        else:
            score_rmsnorm_kernel = {"enabled": True, "role": "generator",
                "execution_status": "NOT_APPLICABLE", "forward_calls": 0, "backward_calls": 0}
    score_rope_kernel = {
        "enabled": False,
        "name": "native_livewan_score_rope",
        "execution_status": "DISABLED",
    }
    if _score_rope_kernel_enabled(plan, mode=mode):
        _, score_ranks = _plan_role_ranks(plan, mode=mode)
        if rank in score_ranks:
            from distillschedule.runtime.livewan_rope_kernel import (
                install_score_rope_kernel,
            )

            score_rope_kernel = install_score_rope_kernel(torch)
            score_rope_kernel.update(
                role="score",
                execution_status="PASS",
            )
        else:
            score_rope_kernel = {
                "enabled": True,
                "name": "score_bf16_rope_fusion_v1",
                "source_apply_rope_sha256": (
                    "8ab587e9f15cc969659609ec2f9cc79f4d4cf6fba468ae8c54267dc949224482"
                ),
                "function_patched": False,
                "install_status": "NOT_APPLICABLE",
                "role": "generator",
                "execution_status": "NOT_APPLICABLE",
                "triton_block_size": 512,
                "fp32_rounding_order": "mul_then_first_product_fma_rn",
                "invocation_count": 0,
                "backward_invocation_count": 0,
                "element_count": 0,
                "backward_element_count": 0,
                "shape_histogram": {},
                "table_rank_histogram": {},
                "gradient_contiguous_copy_count": 0,
                "contract_failure_count": 0,
            }
    score_attention_backend = {
        "enabled": False,
        "name": "native_livewan_score_attention",
        "execution_status": "DISABLED",
    }
    if _score_attention_backend_enabled(plan, mode=mode):
        _, score_ranks = _plan_role_ranks(plan, mode=mode)
        if rank in score_ranks:
            from distillschedule.runtime.livewan_attention_backend import (
                install_score_flash_attn2,
            )

            score_attention_backend = install_score_flash_attn2(torch)
            score_attention_backend.update(
                role="score",
                execution_status="PASS",
            )
        else:
            score_attention_backend = {
                "enabled": True,
                "name": "score_flash_attn2_unmasked_bf16_v1",
                "source_flash_attention_sha256": (
                    "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
                ),
                "artifact_metadata_sha256": (
                    "7749dd68b81a544c1973d68c30be5aa316ee1a718ae0228e0368c2cd04365516"
                ),
                "artifact_package_sha256": (
                    "0c560f96b857c188c4a8297ff27d1299619957c3459a5e433e132329fbfdf066"
                ),
                "artifact_extension_sha256": (
                    "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
                ),
                "function_patched": False,
                "install_status": "NOT_APPLICABLE",
                "role": "generator",
                "execution_status": "NOT_APPLICABLE",
                "invocation_count": 0,
                "backward_invocation_count": 0,
                "element_count": 0,
                "backward_element_count": 0,
                "shape_histogram": {},
                "contract_failure_count": 0,
            }
    generator_cross_kv_cache = {
        "enabled": False,
        "name": "native_livewan_generator_cross_kv",
        "execution_status": "DISABLED",
    }
    if _generator_cross_kv_cache_enabled(plan, mode=mode):
        generator_ranks, _ = _plan_role_ranks(plan, mode=mode)
        if rank in generator_ranks:
            from distillschedule.runtime.livewan_cross_kv_cache import (
                install_generator_cross_kv_cache,
            )

            generator_cross_kv_cache = install_generator_cross_kv_cache(torch)
            generator_cross_kv_cache.update(
                role="generator",
                execution_status="PASS",
            )
        else:
            generator_cross_kv_cache = {
                "enabled": True,
                "name": "generator_nograd_text_kv_cache_v1",
                "source_set_text_sha256": (
                    "def2452ea9528abb9af2aa166d78afdf9b8b7c5daa52c0fc87bfd63bc9832386"
                ),
                "source_cross_forward_sha256": (
                    "0df9365afb37bb8a876d8c8659ca95d1caaffb89bf3e64918b782eb86b0a8253"
                ),
                "source_flash_attention_sha256": (
                    "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
                ),
                "function_patched": False,
                "install_status": "NOT_APPLICABLE",
                "role": "score",
                "execution_status": "NOT_APPLICABLE",
                "cache_build_context_count": 0,
                "cache_build_block_count": 0,
                "cache_build_bytes": 0,
                "cache_invalidation_count": 0,
                "cache_hit_count": 0,
                "grad_bypass_count": 0,
                "uncached_fallback_count": 0,
                "contract_failure_count": 0,
                "query_shape_histogram": {},
            }
    generator_attention_backend = {
        "enabled": False,
        "name": "native_livewan_generator_attention",
        "execution_status": "DISABLED",
    }
    if _generator_attention_backend_enabled(plan, mode=mode):
        generator_ranks, _ = _plan_role_ranks(plan, mode=mode)
        if rank in generator_ranks:
            # Install after the C18 K/V cache.  That cache source-pins the native
            # helper and preserves FP32 normalized keys; the backend reproduces
            # the helper's exact FP32->BF16 kernel casts.
            from distillschedule.runtime.livewan_attention_backend import (
                install_generator_flash_attn2,
            )

            generator_attention_backend = install_generator_flash_attn2(torch)
            generator_attention_backend.update(
                role="generator",
                execution_status="PASS",
            )
        else:
            generator_attention_backend = {
                "enabled": True,
                "name": "generator_flash_attn2_unmasked_exact_cast_v1",
                "source_flash_attention_sha256": (
                    "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
                ),
                "artifact_metadata_sha256": (
                    "7749dd68b81a544c1973d68c30be5aa316ee1a718ae0228e0368c2cd04365516"
                ),
                "artifact_package_sha256": (
                    "0c560f96b857c188c4a8297ff27d1299619957c3459a5e433e132329fbfdf066"
                ),
                "artifact_extension_sha256": (
                    "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
                ),
                "function_patched": False,
                "install_status": "NOT_APPLICABLE",
                "role": "score",
                "execution_status": "NOT_APPLICABLE",
                "invocation_count": 0,
                "backward_invocation_count": 0,
                "element_count": 0,
                "backward_element_count": 0,
                "shape_histogram": {},
                "raw_dtype_histogram": {},
                "input_cast_count": 0,
                "output_cast_count": 0,
                "contract_failure_count": 0,
            }
    before_oom = cgroup_oom_counts()
    torch.cuda.reset_peak_memory_stats(device)
    build_started = time.perf_counter()
    bundle = build_models(
        torch=torch, plan=plan, bootstrap=bootstrap, mode=mode, rank=rank, device=device,
    )
    bundle.metadata["cuda_library_setup"] = cuda_library_setup
    if _critic_lora_scale_one_enabled(plan, mode=mode):
        lora_model = (
            bundle.critic if bundle.critic is not None else bundle.critic_shadow
        )
        if lora_model is None:
            critic_lora_kernel.update(
                role="generator",
                execution_status="NOT_APPLICABLE",
                module_count=0,
                all_scales_one=True,
            )
        else:
            from wanstreamer.lora import LoRALinear

            lora_modules = [
                module
                for module in lora_model.modules()
                if isinstance(module, LoRALinear)
            ]
            _require(len(lora_modules) == 180, "Critic LoRA module count changed")
            _require(
                all(module.scale == 1.0 for module in lora_modules),
                "Critic LoRA scale-one specialization saw a non-unit scale",
            )
            critic_lora_kernel.update(
                role=(
                    "critic"
                    if plan.get("metadata", {}).get("pool_policy")
                    == PHASE_EXPANDED_CRITIC16_POOL_POLICY
                    else (
                        "generator_shadow"
                        if bundle.critic_shadow is not None
                        else "score"
                    )
                ),
                execution_status="PASS",
                module_count=len(lora_modules),
                all_scales_one=True,
            )
    rollout_cast_cache = None
    if _generator_rollout_cast_cache_enabled(plan, mode=mode) and rank < 8:
        from distillschedule.runtime.livewan_rollout_cast_cache import install as install_rollout_cast_cache

        rollout_cast_cache = install_rollout_cast_cache(torch, bundle.student)
    generator_rms_epilogue = None
    if _generator_rms_epilogue_enabled(plan, mode=mode) and rank < 8:
        from distillschedule.runtime.livewan_generator_rms_epilogue import install as install_generator_rms

        generator_rms_epilogue = install_generator_rms(torch, bundle.student)
    loader = InputLoader(torch=torch, trace=trace)
    pinned_score_context = None
    if _score_prompt_staging_enabled(plan, mode=mode) and rank >= 8:
        from distillschedule.runtime.livewan_context_staging import install as install_context_staging

        pinned_score_context = install_context_staging(
            torch, loader, allowed_indices={row.dataset_index for row in trace.rows},
        )
    generator_streamers = []
    if bundle.student is not None:
        generator_ranks = tuple(int(value) for value in bundle.metadata["generator_ranks"])
        streamer_count = (
            1
            if mode == "baseline"
            else len(balanced_slots(generator_ranks)[rank])
        )
        generator_streamers = _build_streamers(
            torch, student=bundle.student, count=streamer_count, device=device,
        )
    score_context = (
        build_score_context(torch, bundle=bundle, loader=loader, device=device)
        if bundle.critic is not None or bundle.critic_shadow is not None else None
    )
    bundle.metadata["communicator_warmup"] = _warm_communicators(
        torch, bundle=bundle, mode=mode, rank=rank, device=device,
    )
    runahead_host_transport = None
    runahead_host_enabled = _generator_runahead_transport_enabled(plan, mode=mode)
    if runahead_host_enabled and run_kind != "capacity":
        from distillschedule.runtime.livewan_host_runahead import (
            PinnedHostRunaheadTransport,
        )

        _require(
            bundle.runahead_host_group is not None,
            "C19 Gloo process group is missing",
        )
        runahead_host_transport = PinnedHostRunaheadTransport(
            torch=torch,
            dist=dist,
            group=bundle.runahead_host_group,
            rank=rank,
            peer=rank + 8 if rank < 8 else rank - 8,
            device=device,
            is_generator=rank < 8,
            shape=LATENT_SHAPE,
            dtype=torch.float32,
            cycles=continuous_cycles,
            cross_cycle_lookahead=continuous_cycles > 1,
            waves_per_step=int(
                plan["metadata"]["generator_runahead_transport"]["waves_per_step"]
            ),
        )
        bundle.metadata["generator_runahead_transport_setup"] = {
            "enabled": True,
            "name": "pinned_cpu_gloo_latent_v1",
            "buffer_count": len(runahead_host_transport.buffers),
            "pinned_buffer_count": sum(
                int(bool(value.is_pinned()))
                for value in runahead_host_transport.buffers.values()
            ),
            "payload_bytes": int(runahead_host_transport.payload_bytes),
            "setup_position": "after_communicator_warmup_before_full5_timer",
        }
    critic_shadow_hot_sync = None
    if continuous_contract is not None and run_kind != "capacity":
        from distillschedule.runtime.livewan_critic_shadow_sync import (
            PairedCriticShadowSync,
        )

        is_generator_rank = rank < 8
        parameters = (
            bundle.critic_shadow_lora_params
            if is_generator_rank
            else bundle.critic_lora_params
        )
        critic_shadow_hot_sync = PairedCriticShadowSync(
            torch=torch,
            dist=dist,
            group=bundle.critic_shadow_sync_group,
            rank=rank,
            peer=rank + 8 if is_generator_rank else rank - 8,
            score_rank=rank + 8 if is_generator_rank else rank,
            device=device,
            parameters=parameters,
            is_generator=is_generator_rank,
            run_kind=run_kind,
        )
        bundle.metadata["critic_shadow_hot_sync_setup"] = dict(
            critic_shadow_hot_sync.validate_pair_contract()
        )
    build_wall = time.perf_counter() - build_started
    setup_memory = memory_snapshot(torch, device)
    if run_kind == "capacity":
        torch.cuda.synchronize(device)
        dist.barrier()
        after_oom = cgroup_oom_counts()
        receipt = {
            "status": "PASS",
            "stage": "construction_capacity",
            "rank": rank,
            "local_rank": local_rank,
            "mode": mode,
            "run_kind": run_kind,
            "trace_sha256": trace.sha256,
            "bootstrap_sha256": bootstrap.sha256,
            "source_receipt_sha256": source_receipt["source_receipt_sha256"],
            "pool_policy": plan.get("metadata", {}).get(
                "pool_policy", "world16_fsdp2_sequential"
            ),
            "construction_wall_seconds": build_wall,
            "setup_memory": setup_memory,
            "peak_memory": memory_snapshot(torch, device),
            "host_memory": host_memory_snapshot(),
            "cgroup_oom_before": before_oom,
            "cgroup_oom_after": after_oom,
            "critic_lora_kernel": critic_lora_kernel,
            "score_modulation_kernel": score_modulation_kernel,
            "score_rmsnorm_kernel": score_rmsnorm_kernel,
            "score_rope_kernel": score_rope_kernel,
            "score_attention_backend": score_attention_backend,
            "generator_cross_attention_kv_cache": generator_cross_kv_cache,
            "generator_attention_backend": generator_attention_backend,
            "critic_checkpoint_rng": {
                "enabled": _critic_checkpoint_rng_elision_enabled(plan, mode=mode),
                "execution_status": "NOT_RUN_CAPACITY_ONLY",
            },
            "generator_runahead_transport": (
                runahead_host_transport.receipt()
                if runahead_host_transport is not None
                else {
                    "enabled": runahead_host_enabled,
                    "execution_status": (
                        "NOT_RUN_CAPACITY_ONLY" if runahead_host_enabled else "DISABLED"
                    ),
                }
            ),
            "build": bundle.metadata,
            "gpu_training_workload": "NOT_RUN",
        }
        _write_rank_receipt(output_dir=output_dir, rank=rank, value=receipt)
        dist.barrier()
        dist.destroy_process_group()
        return receipt
    versions = bootstrap.versions
    evidence = RunEvidence()
    critic_checkpoint_proxy = None
    critic_checkpoint_enabled = _critic_checkpoint_rng_elision_enabled(
        plan, mode=mode
    )
    if critic_checkpoint_enabled:
        _, score_ranks = _plan_role_ranks(plan, mode=mode)
        critic_ranks = _plan_critic_ranks(
            plan, mode=mode, score_ranks=score_ranks
        )
        if rank in critic_ranks:
            critic_checkpoint_proxy = _CheckpointRngElisionProxy(bc.ckpt)
            bc.ckpt = critic_checkpoint_proxy
            evidence.critic_checkpoint_rng = {
                "enabled": True,
                "name": "critic_checkpoint_rng_state_elision_v1",
                "role": "score",
                "execution_status": "PENDING",
            }
        else:
            evidence.critic_checkpoint_rng = {
                "enabled": True,
                "name": "critic_checkpoint_rng_state_elision_v1",
                "role": "generator",
                "execution_status": "NOT_APPLICABLE",
                "checkpoint_calls": 0,
            }
    else:
        evidence.critic_checkpoint_rng = {
            "enabled": False,
            "execution_status": "DISABLED",
            "checkpoint_calls": 0,
        }
    critic_attention_proxy = None
    critic_attention_receipt = {"enabled": False, "execution_status": "DISABLED"}
    if _critic_attention_checkpoint_enabled(plan, mode=mode):
        if rank in _plan_role_ranks(plan, mode=mode)[1]:
            from distillschedule.runtime.livewan_attention_backend import _load_flash_attn2
            from distillschedule.runtime.livewan_attention_checkpoint import CriticAttentionCheckpointProxy

            flash_module, _ = _load_flash_attn2()
            critic_attention_proxy = CriticAttentionCheckpointProxy(bc.ckpt, torch, flash_module)
            bc.ckpt = critic_attention_proxy
        else:
            critic_attention_receipt = {"enabled": True, "execution_status": "NOT_APPLICABLE", "checkpoint_calls": 0}
    inventory = CallInventory()
    trajectory_writer = (
        AppendOnlyTrajectoryWriter(
            output_dir / "trajectory" / f"rank_{rank:03d}.jsonl",
            run_id=str(os.environ.get("DISTILLSCHEDULE_RUN_ID", "unknown")),
            rank=rank,
        )
        if run_kind == "trajectory"
        else None
    )
    roles = {
        id(model): role
        for model, role in (
            (bundle.student, "student"),
            (bundle.critic, "critic"),
            (bundle.critic_shadow, "critic"),
            (bundle.teacher, "teacher"),
        )
        if model is not None
    }
    is_profile = run_kind == "profile"
    cycle_endpoint_records: list[dict[str, Any]] = []
    dist.barrier()
    if is_profile:
        # cudaProfilerApi capture state is process-scoped under the node-outer
        # Nsight session. Every torchrun child must arm its own CUDA context;
        # arming only local rank 0 can leave the other seven workers with NVTX
        # ranges but no CUDA activity in the exported SQLite.
        torch.cuda.cudart().cudaProfilerStart()
        # Make sure all eight workers on both nodes enter the workload after
        # every process-local profiler range has started.
        dist.barrier()
        # Nsight attaches to the eight child workers asynchronously after the
        # node-local rank calls cudaProfilerStart().  A world barrier proves
        # application readiness, but it does not prove that the profiler has
        # finished attaching to every child.  The previous diagnostic lost the
        # opening NVTX ranges from one rank even though its CUDA kernels were
        # present.  A bounded profile-only arm delay was validated with an
        # eight-rank canary and remains outside both the Full5 timer and outer
        # NVTX range.  Formal/correctness runs never execute this delay.
        time.sleep(10.0)
        dist.barrier()
    torch.cuda.synchronize(device)
    timing_start_event = torch.cuda.Event(enable_timing=True) if two_cycle_timing else None
    if timing_start_event is not None:
        timing_start_event.record(torch.cuda.current_stream(device))
    timing_start_ns = time.perf_counter_ns()
    start = time.perf_counter()
    with _profile_range(
        torch, run_kind=run_kind, label=f"livewan/full5/{mode}/rank_{rank}"
    ):
        with inventory.install(torch=torch, blockcausal=bc, roles=roles):
            if mode == "baseline":
                if run_kind == "trajectory":
                    _require(
                        trajectory_writer is not None and score_context is not None,
                        "baseline trajectory evidence configuration is incomplete",
                    )
                    _run_baseline_trajectory(
                        torch=torch,
                        rank=rank,
                        trace=trace,
                        loader=loader,
                        bundle=bundle,
                        streamers=generator_streamers,
                        score_context=score_context,
                        evidence=evidence,
                        versions=versions,
                        device=device,
                        inventory=inventory,
                        writer=trajectory_writer,
                        seed=trajectory_seed,
                        arm_id=trajectory_arm_id,
                        output_dir=output_dir,
                    )
                elif two_cycle_timing:
                    _require(score_context is not None, "baseline timing lacks Score context")
                    _run_baseline_two_cycle_timing(
                        torch=torch,
                        rank=rank,
                        trace=trace,
                        loader=loader,
                        bundle=bundle,
                        streamers=generator_streamers,
                        score_context=score_context,
                        evidence=evidence,
                        versions=versions,
                        device=device,
                        cycle_endpoint_records=cycle_endpoint_records,
                        inventory=inventory,
                    )
                else:
                    _run_baseline(
                        torch=torch, rank=rank, trace=trace, loader=loader, bundle=bundle,
                        streamers=generator_streamers, score_context=score_context,
                        run_kind=run_kind, evidence=evidence, versions=versions, device=device,
                    )
            elif continuous_contract is not None:
                _run_pipeline_8x8_continuous(
                    torch=torch,
                    rank=rank,
                    trace=trace,
                    loader=loader,
                    bundle=bundle,
                    streamers=generator_streamers,
                    score_context=score_context,
                    run_kind=run_kind,
                    evidence=evidence,
                    versions=versions,
                    device=device,
                    runahead_host_transport=runahead_host_transport,
                    critic_shadow_hot_sync=critic_shadow_hot_sync,
                    slot_pairs_by_lane=_pipeline_8x8_slot_groups(plan),
                    full_context_lens_elision=_full_context_lens_elision(
                        plan, mode=mode
                    ),
                    critic_step1_split_overlap=(
                        continuous_contract.get("name")
                        == "continuous_full5_b16_depth1_step1_b1_overlap_v1"
                    ),
                    critic_b1_attribution=critic_b1_attribution,
                    inventory=inventory,
                    trajectory_writer=trajectory_writer,
                    trajectory_seed=(
                        trajectory_seed if run_kind == "trajectory" else None
                    ),
                    trajectory_arm_id=(
                        trajectory_arm_id if run_kind == "trajectory" else None
                    ),
                    trajectory_output_dir=(
                        output_dir if run_kind == "trajectory" else None
                    ),
                    cycle_endpoint_records=(
                        cycle_endpoint_records if two_cycle_timing else None
                    ),
                )
            elif (
                plan.get("metadata", {}).get("pool_policy")
                == PHASE_EXPANDED_CRITIC16_POOL_POLICY
            ):
                _run_pipeline_phase_expanded_critic16(
                    torch=torch,
                    rank=rank,
                    trace=trace,
                    loader=loader,
                    bundle=bundle,
                    streamers=generator_streamers,
                    score_context=score_context,
                    run_kind=run_kind,
                    evidence=evidence,
                    versions=versions,
                    device=device,
                    slot_pairs_by_lane=_pipeline_8x8_slot_pairs(plan),
                )
            elif plan.get("metadata", {}).get("pool_policy") in {
                "persistent_8g_8score_teacher_fsdp8",
                "persistent_8g_8score_teacher_fsdp8_runahead",
                "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2",
                "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_local_context",
                ASYNC_CRITIC_POOL_POLICY,
                TRACE_BALANCED_POOL_POLICY,
            }:
                _run_pipeline_8x8(
                    torch=torch, rank=rank, trace=trace, loader=loader, bundle=bundle,
                    streamers=generator_streamers, score_context=score_context,
                    run_kind=run_kind, evidence=evidence, versions=versions, device=device,
                    rollout_runahead=(
                        plan.get("metadata", {}).get("pool_policy")
                        in {
                            "persistent_8g_8score_teacher_fsdp8_runahead",
                            "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2",
                            "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_local_context",
                            ASYNC_CRITIC_POOL_POLICY,
                            TRACE_BALANCED_POOL_POLICY,
                        }
                    ),
                    critic_b2_only=(
                        plan.get("metadata", {}).get("pool_policy")
                        in {
                            "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2",
                            "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_local_context",
                            ASYNC_CRITIC_POOL_POLICY,
                            TRACE_BALANCED_POOL_POLICY,
                        }
                    ),
                    score_context_from_trace=(
                        plan.get("metadata", {}).get("pool_policy")
                        in {
                            "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_local_context",
                            ASYNC_CRITIC_POOL_POLICY,
                            TRACE_BALANCED_POOL_POLICY,
                        }
                    ),
                    slot_pairs_by_lane=_pipeline_8x8_slot_pairs(plan),
                    full_context_lens_elision=_full_context_lens_elision(
                        plan, mode=mode
                    ),
                    score_context_prefetch=_score_context_prefetch_enabled(
                        plan, mode=mode
                    ),
                    split_dmd_shadow=_split_dmd_shadow_enabled(plan, mode=mode),
                    runahead_host_transport=runahead_host_transport,
                )
            elif plan.get("metadata", {}).get("pool_policy") == "persistent_8g_8score_teacher_fsdp8_score_b2":
                _run_pipeline_8x8_score_b2(
                    torch=torch, rank=rank, trace=trace, loader=loader, bundle=bundle,
                    streamers=generator_streamers, score_context=score_context,
                    run_kind=run_kind, evidence=evidence, versions=versions, device=device,
                )
            elif plan.get("metadata", {}).get("pool_policy") in {
                "persistent_6g_10score_teacher_fsdp10",
                "persistent_6g_10score_teacher_node_local_hierarchical",
            }:
                _run_pipeline_6g10score(
                    torch=torch, rank=rank, trace=trace, loader=loader, bundle=bundle,
                    streamers=generator_streamers, score_context=score_context,
                    run_kind=run_kind, evidence=evidence, versions=versions, device=device,
                )
            else:
                raise RuntimeError("unsupported LiveWan pipeline pool policy")
    torch.cuda.synchronize(device)
    dist.barrier()
    timing_stop_ns = time.perf_counter_ns()
    full5_wall = time.perf_counter() - start
    if is_profile:
        dist.barrier()
        torch.cuda.cudart().cudaProfilerStop()
    if _score_rmsnorm_kernel_enabled(plan, mode=mode):
        from distillschedule.runtime.livewan_score_rmsnorm import runtime_receipt_errors

        _require(not runtime_receipt_errors(score_rmsnorm_kernel, rank=rank),
                 "Score RMSNorm runtime contract/counters incomplete")
    local_final_versions = dict(versions)
    rollout_cast_cache_receipt = {"enabled": False, "execution_status": "DISABLED"}
    if _generator_rollout_cast_cache_enabled(plan, mode=mode):
        from distillschedule.runtime.livewan_rollout_cast_cache import runtime_receipt_errors as cast_scope_errors

        rollout_cast_cache_receipt = (
            rollout_cast_cache.receipt() if rollout_cast_cache is not None
            else {"enabled": True, "execution_status": "NOT_APPLICABLE"}
        )
        _require(not cast_scope_errors(rollout_cast_cache_receipt, rank=rank, cycles=continuous_cycles),
                 "C28 Generator rollout cast-cache lifecycle/count failed")
    generator_rms_epilogue_receipt = {"enabled": False, "execution_status": "DISABLED"}
    if _generator_rms_epilogue_enabled(plan, mode=mode):
        from distillschedule.runtime.livewan_generator_rms_epilogue import runtime_receipt_errors as rms_epilogue_errors

        generator_rms_epilogue_receipt = (
            generator_rms_epilogue.receipt() if generator_rms_epilogue is not None
            else {"enabled": True, "execution_status": "NOT_APPLICABLE"}
        )
        _require(not rms_epilogue_errors(generator_rms_epilogue_receipt, rank=rank, cycles=continuous_cycles),
                 "C29 Generator no-grad RMS epilogue scope/call contract failed")
    score_prompt_staging_receipt = {"enabled": False, "execution_status": "DISABLED"}
    if _score_prompt_staging_enabled(plan, mode=mode):
        from distillschedule.runtime.livewan_context_staging import SOURCE_SHA256, runtime_receipt_errors as staging_errors

        score_prompt_staging_receipt = (
            {**pinned_score_context.receipt(), "enabled": True, "execution_status": "PASS",
             "source_context_method_sha256": SOURCE_SHA256}
            if pinned_score_context is not None
            else {"enabled": True, "execution_status": "NOT_APPLICABLE"}
        )
        _require(not staging_errors(score_prompt_staging_receipt, rank=rank, cycles=continuous_cycles),
                 "C27 actual input staging census failed")
    version_tensor = torch.tensor(
        [versions["generator"], versions["critic"], versions["ema"]],
        device=device,
        dtype=torch.int64,
    )
    dist.all_reduce(version_tensor, op=dist.ReduceOp.MAX)
    versions.update(
        generator=int(version_tensor[0].cpu()),
        critic=int(version_tensor[1].cpu()),
        ema=int(version_tensor[2].cpu()),
    )
    after_oom = cgroup_oom_counts()

    two_cycle_timing_receipt = None
    if two_cycle_timing:
        _require(
            timing_start_event is not None
            and len(cycle_endpoint_records) == TWO_CYCLE_TIMING_CYCLES
            and timing_start_ns
            < int(cycle_endpoint_records[0]["host_completion_ns"])
            < int(cycle_endpoint_records[1]["host_completion_ns"])
            <= timing_stop_ns,
            "two-cycle timing endpoints are missing or out of order",
        )
        host_completion_offsets = [
            (int(value["host_completion_ns"]) - timing_start_ns) / 1e9
            for value in cycle_endpoint_records
        ]
        gpu_completion_offsets = [
            float(timing_start_event.elapsed_time(value["event"])) / 1000.0
            for value in cycle_endpoint_records
        ]
        _require(
            0.0 < gpu_completion_offsets[0] < gpu_completion_offsets[1],
            "two-cycle CUDA completion events are missing or out of order",
        )
        cycle_rows = []
        for index, value in enumerate(cycle_endpoint_records):
            previous_physical = (
                cycle_endpoint_records[index - 1]["physical_call_inventory"]
                if index else {}
            )
            previous_logical = (
                cycle_endpoint_records[index - 1]["logical_call_inventory"]
                if index else {}
            )
            cycle_rows.append(
                {
                    "cycle": index + 1,
                    "local_versions": value["local_versions"],
                    "physical_call_inventory_cumulative": value[
                        "physical_call_inventory"
                    ],
                    "logical_call_inventory_cumulative": value[
                        "logical_call_inventory"
                    ],
                    "physical_call_inventory_delta": {
                        key: int(count) - int(previous_physical.get(key, 0))
                        for key, count in value["physical_call_inventory"].items()
                        if int(count) - int(previous_physical.get(key, 0))
                    },
                    "logical_call_inventory_delta": {
                        key: int(count) - int(previous_logical.get(key, 0))
                        for key, count in value["logical_call_inventory"].items()
                        if int(count) - int(previous_logical.get(key, 0))
                    },
                }
            )
        two_cycle_timing_receipt = {
            "schema_version": "distillschedule.livewan_two_cycle_timing_rank.v1",
            "cycles_in_same_process": TWO_CYCLE_TIMING_CYCLES,
            "cycle_1": {
                "role": "WARMUP_DIAGNOSTIC",
                "completion_offset_seconds": gpu_completion_offsets[0],
                "host_enqueue_offset_seconds": host_completion_offsets[0],
                "local_endpoint_delta_seconds": gpu_completion_offsets[0],
                "claimable": False,
            },
            "cycle_2": {
                "role": "STEADY_STATE_CLAIMABLE",
                "completion_offset_seconds": gpu_completion_offsets[1],
                "host_enqueue_offset_seconds": host_completion_offsets[1],
                "local_endpoint_delta_seconds": (
                    gpu_completion_offsets[1] - gpu_completion_offsets[0]
                ),
                "claimable": True,
            },
            "final_sync_barrier_tail_seconds": (
                timing_stop_ns - int(cycle_endpoint_records[1]["host_completion_ns"])
            )
            / 1e9,
            "total_outer_wall_seconds": (timing_stop_ns - timing_start_ns) / 1e9,
            "endpoint_definition": (
                "rank terminal CUDA-event completion on the current compute stream; "
                "cycle2 is completion2-completion1; "
                "C24 cycle2 therefore includes inter-cycle lookahead and Critic-shadow "
                "hot-sync; no midpoint synchronize or barrier"
            ),
            "world_aggregation": (
                "baseline=max_all_rank_completion_offsets; "
                "pipeline=max_score_rank_completion_offsets"
            ),
            "profiled": False,
            "profiler_status": "NOT_RUN",
            "state_anchors": "NOT_COLLECTED_IN_TIMING_MODE",
            "evidence_extraction": "AFTER_BOTH_TIMED_CYCLE_ENDPOINTS",
            "per_cycle_contract": cycle_rows,
            "local_final_versions_before_world_max": local_final_versions,
            "effective_plan_sha256": effective_plan_sha256,
            "correctness_binding": two_cycle_gate,
        }

    output_tensor_artifact = None
    if run_kind == "correctness" and evidence.output_tensors:
        from safetensors.torch import save_file
        output_path = output_dir / "correctness" / f"rank_{rank:03d}_rollouts.safetensors"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(evidence.output_tensors, str(output_path))
        output_tensor_artifact = {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "tensor_count": len(evidence.output_tensors),
        }

    generator_metrics = None
    critic_metrics = None
    if run_kind != "trajectory" and bundle.student is not None:
        generator_metrics = _state_metrics(torch, list(bundle.student.parameters()))
    if run_kind != "trajectory" and bundle.critic is not None:
        # Critic is replicated in both paths.  One local metric is sufficient;
        # the summarizer checks equality across owners.
        critic_metrics = _state_metrics(torch, bundle.critic_lora_params)
    logical_call_inventory = dict(inventory.counts)
    logical_call_inventory.update(evidence.logical_score_calls)
    if critic_attention_proxy is not None:
        _require(bundle.critic is not None, "C26 Score checkpoint lacks Critic")
        critic_attention_receipt = critic_attention_proxy.receipt(
            expected_calls=int(inventory.counts.get("critic/grad/checkpoint", 0)) * len(bundle.critic.blocks))
    if critic_checkpoint_proxy is not None:
        _require(bundle.critic is not None, "Score checkpoint proxy lacks Critic")
        expected_checkpoint_calls = int(
            inventory.counts.get("critic/grad/checkpoint", 0)
        ) * len(bundle.critic.blocks)
        evidence.critic_checkpoint_rng = critic_checkpoint_proxy.receipt(
            expected_calls=expected_checkpoint_calls
        )
    slot_routing = None
    receipt_pool_policy = str(plan.get("metadata", {}).get("pool_policy", ""))
    if receipt_pool_policy in {
        TRACE_BALANCED_POOL_POLICY,
        PHASE_EXPANDED_CRITIC16_POOL_POLICY,
    }:
        slot_pairs = _pipeline_8x8_slot_groups(plan)
        lane = rank if rank < 8 else rank - 8
        slot_routing = {
            "policy": (
                "trace_balanced_phase_expanded_critic16_v1"
                if receipt_pool_policy == PHASE_EXPANDED_CRITIC16_POOL_POLICY
                else (
                    "trace_balanced_fixed_rank_groups_b32_v1"
                    if trace.global_batch == 32
                    else "trace_balanced_fixed_rank_pairs_v1"
                )
            ),
            "lane": lane,
            "logical_slots": list(slot_pairs[lane]),
            "peer_rank": rank + 8 if rank < 8 else rank - 8,
            "all_pairs": [list(pair) for pair in slot_pairs],
        }
        if receipt_pool_policy == PHASE_EXPANDED_CRITIC16_POOL_POLICY:
            slot_routing["critic_slot"] = int(
                slot_pairs[lane][1] if rank < 8 else slot_pairs[lane][0]
            )
    workload_cycles = int(trace.cycles)
    receipt = {
        "status": "PASS",
        "rank": rank,
        "local_rank": local_rank,
        "mode": mode,
        "run_kind": run_kind,
        "trace_sha256": trace.sha256,
        "bootstrap_sha256": bootstrap.sha256,
        "source_receipt_sha256": source_receipt["source_receipt_sha256"],
        "effective_plan_sha256": effective_plan_sha256,
        "logical_generator_updates": GENERATOR_UPDATES * workload_cycles,
        "logical_critic_updates": CRITIC_UPDATES * workload_cycles,
        "physical_generator_commits": (
            GENERATOR_UPDATES * workload_cycles if bundle.student is not None else 0
        ),
        "physical_critic_commits": (
            CRITIC_UPDATES * workload_cycles if bundle.critic is not None else 0
        ),
        "initial_versions": bootstrap.versions,
        "final_versions": versions,
        "call_inventory": inventory.counts,
        "physical_call_inventory": inventory.counts,
        "logical_call_inventory": logical_call_inventory,
        "score_physical_batch_size": _score_physical_batch_size(plan, mode=mode),
        "critic_physical_batch_size": _critic_physical_batch_size(plan, mode=mode),
        "pool_policy": plan.get("metadata", {}).get("pool_policy", "world16_fsdp2_sequential"),
        "full_context_lens_elision": {
            "enabled": _full_context_lens_elision(plan, mode=mode),
            "context_length": TEXT_SHAPE[1],
            "padding_policy": (
                "all_512_tokens_valid_no_padding"
                if _full_context_lens_elision(plan, mode=mode)
                else "explicit_context_lens"
            ),
            "attention_mask_equivalence": (
                "ctx_lens_512_equivalent_to_none_at_exact_tensor_length"
                if _full_context_lens_elision(plan, mode=mode)
                else "not_applied"
            ),
        },
        "score_context_prefetch": evidence.score_context_prefetch,
        "split_dmd_shadow": evidence.split_dmd_shadow,
        "critic_lora_kernel": critic_lora_kernel,
        "score_modulation_kernel": score_modulation_kernel,
        "score_rmsnorm_kernel": score_rmsnorm_kernel,
        "score_rope_kernel": score_rope_kernel,
        "score_attention_backend": score_attention_backend,
        "generator_cross_attention_kv_cache": generator_cross_kv_cache,
        "generator_attention_backend": generator_attention_backend,
        "generator_runahead_transport": evidence.generator_runahead_transport,
        "continuous_pipeline": evidence.continuous_pipeline,
        "critic_shadow_hot_sync": evidence.critic_shadow_hot_sync,
        "score_receive_waits": evidence.score_receive_waits,
        "generator_backpressure": evidence.generator_backpressure,
        "critic_checkpoint_rng": evidence.critic_checkpoint_rng,
        "critic_attention_checkpoint": critic_attention_receipt,
        "score_prompt_staging": score_prompt_staging_receipt,
        "generator_rollout_cast_cache": rollout_cast_cache_receipt,
        "generator_rms_epilogue": generator_rms_epilogue_receipt,
        "slot_routing": slot_routing,
        "full5_wall_seconds": full5_wall,
        "two_cycle_timing": two_cycle_timing_receipt,
        "profiled": False if two_cycle_timing else is_profile,
        "construction_wall_seconds": build_wall,
        "setup_memory": setup_memory,
        "peak_memory": memory_snapshot(torch, device),
        "host_memory": host_memory_snapshot(),
        "cgroup_oom_before": before_oom,
        "cgroup_oom_after": after_oom,
        "output_fingerprints": evidence.output_fingerprints,
        "output_tensor_artifact": output_tensor_artifact,
        "loss_fingerprints": evidence.loss_fingerprints,
        "transport_bytes": evidence.transport_bytes,
        "gradient_sync": evidence.gradient_sync,
        "optimizer_diagnostics": evidence.optimizer_diagnostics,
        "generator_state_metrics": generator_metrics,
        "critic_state_metrics": critic_metrics,
        "build": bundle.metadata,
        "ema": {"updates_in_full5": 0, "status": "NOT_RUN_SEPARATE_GATE"},
    }
    if run_kind == "trajectory":
        receipt["trajectory"] = {
            "schema": "trajectory_coarse_v1",
            "cycles": trajectory_cycles,
            "seed": trajectory_seed,
            "arm_id": trajectory_arm_id,
            "jsonl": str(trajectory_writer.path),
            "diagnostic_only": True,
            "formal_timing_eligible": False,
            "ordinary_numerical_drift_early_stop": False,
        }
    _write_rank_receipt(output_dir=output_dir, rank=rank, value=receipt)
    dist.barrier()
    if is_profile:
        # Nsight 2024.6.2 has now consumed cudaProfilerStop and every worker has
        # atomically closed its receipt.  With this large resident model, one
        # arbitrary worker can hang in NCCL/CUDA object destruction during
        # interpreter shutdown even after an explicit process-group teardown.
        # A profile-only immediate successful exit avoids that nondeterministic
        # destructor path.  It is after the capture range, timer, evidence, and
        # final barrier, and the OS still releases the CUDA context normally.
        os._exit(0)
    # Explicitly tear down NCCL before the interpreter starts destroying CUDA
    # objects in unprofiled runs.  This is deliberately after the Full5 timer,
    # evidence generation, and final world barrier, so it changes lifecycle
    # cleanup only and cannot alter the measured workload.
    dist.destroy_process_group()
    return receipt


def _record_output(torch: Any, evidence: RunEvidence, *, row: OperationTraceRow, x0: Any, run_kind: str) -> None:
    if run_kind == "correctness":
        evidence.output_fingerprints.append({"operation_key": row.operation_key, **_tensor_fingerprint(torch, x0)})
        evidence.output_tensors[row.operation_key] = x0.detach().to("cpu", torch.float32).contiguous()


def _run_baseline(
    *,
    torch: Any,
    rank: int,
    trace: FormalTrace,
    loader: InputLoader,
    bundle: ModelBundle,
    streamers: Sequence[Any],
    score_context: ScoreContext,
    run_kind: str,
    evidence: RunEvidence,
    versions: dict[str, int],
    device: Any,
    step_offset: int = 0,
) -> None:
    _require(bundle.student is not None and bundle.critic is not None and bundle.teacher is not None, "baseline needs all roles")
    for step in range(FULL5_STEPS):
        global_step = step_offset + step
        with _profile_range(
            torch, run_kind=run_kind, label=f"livewan/baseline/step_{step}"
        ):
            row = trace.row(global_step, rank)
            bundle.generator_optimizer.zero_grad(set_to_none=True)
            bundle.critic_optimizer.zero_grad(set_to_none=True)
            with _profile_range(
                torch, run_kind=run_kind,
                label=f"livewan/baseline/step_{step}/student_rollout",
            ):
                prepared = prepare_rollout(
                    torch, row=row, streamer=streamers[0], student=bundle.student,
                    loader=loader, device=device,
                )
                _record_output(torch, evidence, row=row, x0=prepared.x0, run_kind=run_kind)
            with _profile_range(
                torch, run_kind=run_kind,
                label=f"livewan/baseline/step_{step}/score_context",
            ):
                sample_context = prepare_score_sample_context(
                    torch,
                    bundle=bundle,
                    raw_context=prepared.raw_context,
                    include_teacher=step == 0,
                )
            if step == 0:
                with _profile_range(
                    torch, run_kind=run_kind,
                    label="livewan/baseline/step_0/dmd_score",
                ):
                    gradient = dmd_gradient_for_sample(
                        torch, bundle=bundle, score_context=score_context, row=row,
                        x0=prepared.x0, sample_context=sample_context,
                        context_lens=prepared.context_lens,
                    )
                    evidence.add_logical_score_calls("teacher/nograd/plain", 1)
                    evidence.add_logical_score_calls("critic/nograd/plain", 1)
                with _profile_range(
                    torch, run_kind=run_kind,
                    label="livewan/baseline/step_0/generator_backward",
                ):
                    losses = generator_backward_for_sample(
                        torch, prepared=prepared, dmd_gradient=gradient, sample_weight=1.0,
                    )
                if run_kind in {"correctness", "trajectory"}:
                    evidence.loss_fingerprints.extend(
                        {"operation_key": row.operation_key, "kind": "generator", **_tensor_fingerprint(torch, value)}
                        for value in losses
                    )
            with _profile_range(
                torch, run_kind=run_kind,
                label=f"livewan/baseline/step_{step}/critic_backward",
            ):
                critic_loss = critic_backward_for_sample(
                    torch, bundle=bundle, score_context=score_context, row=row,
                    x0=prepared.x0, sample_context=sample_context,
                    context_lens=prepared.context_lens, sample_weight=1.0,
                )
                evidence.add_logical_score_calls("critic/grad/checkpoint", 1)
            if run_kind in {"correctness", "trajectory"}:
                evidence.loss_fingerprints.append(
                    {"operation_key": row.operation_key, "kind": "critic", **_tensor_fingerprint(torch, critic_loss)}
                )
            if step == 0:
                with _profile_range(
                    torch, run_kind=run_kind,
                    label="livewan/baseline/step_0/generator_commit",
                ):
                    generator_parameters = list(bundle.student.parameters())
                    generator_pre_clip = (
                        _state_metrics(torch, [parameter.grad for parameter in generator_parameters])
                        if run_kind == "correctness" else None
                    )
                    generator_clip_norm = torch.nn.utils.clip_grad_norm_(generator_parameters, 1.0)
                    _record_optimizer_diagnostics(
                        torch, evidence, run_kind=run_kind, role="generator", step=global_step,
                        parameters=generator_parameters, clip_norm=generator_clip_norm,
                        gradient_before_clip=generator_pre_clip,
                    )
                    bundle.generator_optimizer.step()
                    versions["generator"] += 1
            with _profile_range(
                torch, run_kind=run_kind,
                label=f"livewan/baseline/step_{step}/critic_commit",
            ):
                sync = _mean_replicated_gradients(
                    torch, bundle.critic_lora_params, group=bundle.score_group,
                    group_size=bundle.score_group_size, bucket_mib=128,
                )
                evidence.gradient_sync.append({"step": global_step, "role": "critic", **sync})
                critic_pre_clip = (
                    _state_metrics(torch, [parameter.grad for parameter in bundle.critic_lora_params])
                    if run_kind == "correctness" else None
                )
                critic_clip_norm = torch.nn.utils.clip_grad_norm_(bundle.critic_lora_params, 1.0, foreach=True)
                _record_optimizer_diagnostics(
                    torch, evidence, run_kind=run_kind, role="critic", step=global_step,
                    parameters=bundle.critic_lora_params, clip_norm=critic_clip_norm,
                    gradient_before_clip=critic_pre_clip,
                )
                bundle.critic_optimizer.step()
                versions["critic"] += 1


def _run_baseline_trajectory(
    *,
    torch: Any,
    rank: int,
    trace: FormalTrace,
    loader: InputLoader,
    bundle: ModelBundle,
    streamers: Sequence[Any],
    score_context: ScoreContext,
    evidence: RunEvidence,
    versions: dict[str, int],
    device: Any,
    inventory: CallInventory,
    writer: AppendOnlyTrajectoryWriter,
    seed: int,
    arm_id: str,
    output_dir: Path,
) -> None:
    """Run sequential world16 in uninterrupted G1/C5 trajectory cycles."""

    _require(
        trace.cycles in TRAJECTORY_ALLOWED_CYCLES,
        "baseline trajectory cycle count is not registered",
    )
    initial_versions = dict(versions)
    reference_call_census: dict[str, int] | None = None
    for cycle in range(trace.cycles):
        loss_start = len(evidence.loss_fingerprints)
        optimizer_start = len(evidence.optimizer_diagnostics)
        inventory_before = dict(inventory.counts)
        logical_score_before = dict(evidence.logical_score_calls)
        _run_baseline(
            torch=torch,
            rank=rank,
            trace=trace,
            loader=loader,
            bundle=bundle,
            streamers=streamers,
            score_context=score_context,
            run_kind="trajectory",
            evidence=evidence,
            versions=versions,
            device=device,
            step_offset=cycle * FULL5_STEPS,
        )
        cycle_number = cycle + 1
        expected_versions = {
            "generator": int(initial_versions["generator"]) + cycle_number,
            "critic": int(initial_versions["critic"]) + CRITIC_UPDATES * cycle_number,
            "ema": int(initial_versions["ema"]),
        }
        _require(
            dict(versions) == expected_versions,
            "sequential trajectory physical commit inventory changed",
        )
        state_fingerprints: dict[str, str] = {}
        if cycle_number in TRAJECTORY_ANCHOR_CYCLES or cycle_number == trace.cycles:
            state_fingerprints, _state_anchor = _trajectory_state_anchor(
                torch=torch,
                rank=rank,
                cycle=cycle_number,
                output_dir=output_dir,
                bundle=bundle,
            )
        inventory_after = dict(inventory.counts)
        logical_score_after = dict(evidence.logical_score_calls)
        local_call_census = {
            **{
                f"application/{key}": int(value) - int(inventory_before.get(key, 0))
                for key, value in inventory_after.items()
                if key.startswith("student")
                and int(value) - int(inventory_before.get(key, 0))
            },
            **{
                f"logical_score/{key}": int(value)
                - int(logical_score_before.get(key, 0))
                for key, value in logical_score_after.items()
                if int(value) - int(logical_score_before.get(key, 0))
            },
        }
        finite, call_census = _append_trajectory_cycle_receipt(
            torch=torch,
            rank=rank,
            cycle=cycle,
            trace=trace,
            seed=seed,
            arm_id=arm_id,
            execution_mode="baseline",
            implementation={
                "pool_topology": "WORLD16_SEQUENTIAL_FSDP2",
                "physical_role": "all_roles",
                "critic_physical_batch_size": "1",
                "critic_physical_backward_calls": "5",
                "critic_optimizer_commits": "5",
            },
            versions=versions,
            initial_versions=initial_versions,
            local_losses=evidence.loss_fingerprints[loss_start:],
            local_optimizer=evidence.optimizer_diagnostics[optimizer_start:],
            local_call_census=local_call_census,
            state_fingerprints=state_fingerprints,
            writer=writer,
        )
        if reference_call_census is None:
            reference_call_census = dict(call_census)
        else:
            _require(
                call_census == reference_call_census,
                "sequential trajectory changed its per-cycle call census",
            )
        if not finite:
            raise FloatingPointError(
                f"non-finite LiveWan baseline trajectory evidence at cycle {cycle_number}"
            )


def _run_baseline_two_cycle_timing(
    *,
    torch: Any,
    rank: int,
    trace: FormalTrace,
    loader: InputLoader,
    bundle: ModelBundle,
    streamers: Sequence[Any],
    score_context: ScoreContext,
    evidence: RunEvidence,
    versions: dict[str, int],
    device: Any,
    cycle_endpoint_records: list[dict[str, Any]],
    inventory: CallInventory,
) -> None:
    """Run two uninterrupted sequential cycles with timestamp-only boundaries."""

    _require(
        trace.cycles == TWO_CYCLE_TIMING_CYCLES,
        "sequential timing trace must contain exactly two cycles",
    )
    for cycle in range(TWO_CYCLE_TIMING_CYCLES):
        _run_baseline(
            torch=torch,
            rank=rank,
            trace=trace,
            loader=loader,
            bundle=bundle,
            streamers=streamers,
            score_context=score_context,
            run_kind="two-cycle-timing",
            evidence=evidence,
            versions=versions,
            device=device,
            step_offset=cycle * FULL5_STEPS,
        )
        # A nonblocking event only: deliberately no host synchronize or
        # distributed collective at the cycle boundary.
        _record_two_cycle_endpoint(
            torch,
            records=cycle_endpoint_records,
            versions=versions,
            inventory=inventory,
            evidence=evidence,
        )


def _recv_pair(
    torch: Any, *, source: int, device: Any, receive_context: bool = True
) -> tuple[Any, Any | None]:
    import torch.distributed as dist
    x0 = torch.empty(LATENT_SHAPE, device=device, dtype=torch.float32)
    dist.recv(x0, src=source)
    context = None
    if receive_context:
        context = torch.empty(TEXT_SHAPE, device=device, dtype=torch.float32)
        dist.recv(context, src=source)
    return x0, context


def _send_pair(
    *,
    destination: int,
    prepared: PreparedRollout,
    asynchronous: bool,
    send_context: bool = True,
):
    import torch.distributed as dist
    if asynchronous:
        requests = [dist.isend(prepared.x0.contiguous(), dst=destination)]
        if send_context:
            requests.append(
                dist.isend(prepared.raw_context.contiguous(), dst=destination)
            )
        return tuple(requests)
    dist.send(prepared.x0.contiguous(), dst=destination)
    if send_context:
        dist.send(prepared.raw_context.contiguous(), dst=destination)
    return ()


def _exchange_routed_inputs(
    torch: Any,
    *,
    rank: int,
    device: Any,
    generator_ranks: Sequence[int],
    score_ranks: Sequence[int],
    prepared_by_slot: Mapping[int, PreparedRollout],
) -> dict[int, tuple[Any, Any]]:
    """Exchange all real B1 samples for one step in one collective P2P batch.

    Every world rank enters the first ``batch_isend_irecv`` on the default
    communicator.  This preserves NCCL's first-call requirement while allowing
    a non-paired 6G+10Score routing graph without dummy samples.
    """

    import torch.distributed as dist

    routes = _pipeline_slot_routes(generator_ranks, score_ranks)
    expected_generator_slots = set(balanced_slots(generator_ranks).get(rank, ()))
    expected_score_slots = set(balanced_slots(score_ranks).get(rank, ()))
    if rank in generator_ranks:
        _require(
            set(prepared_by_slot) == expected_generator_slots,
            "Generator routed-input slot inventory mismatch",
        )
    else:
        _require(not prepared_by_slot, "Score rank unexpectedly owns Generator inputs")

    operations: list[Any] = []
    received: dict[int, tuple[Any, Any]] = {}
    retained: list[Any] = []
    for slot, generator_rank, score_rank in routes:
        if rank == generator_rank:
            prepared = prepared_by_slot[slot]
            x0 = prepared.x0.contiguous()
            context = prepared.raw_context.contiguous()
            retained.extend((x0, context))
            operations.extend(
                (
                    dist.P2POp(dist.isend, x0, score_rank),
                    dist.P2POp(dist.isend, context, score_rank),
                )
            )
        elif rank == score_rank:
            x0 = torch.empty(LATENT_SHAPE, device=device, dtype=torch.float32)
            context = torch.empty(TEXT_SHAPE, device=device, dtype=torch.float32)
            received[slot] = (x0, context)
            operations.extend(
                (
                    dist.P2POp(dist.irecv, x0, generator_rank),
                    dist.P2POp(dist.irecv, context, generator_rank),
                )
            )
    _require(
        set(received) == expected_score_slots,
        "Score routed-input slot inventory mismatch",
    )
    for request in dist.batch_isend_irecv(operations):
        request.wait()
    return received


def _exchange_routed_gradients(
    torch: Any,
    *,
    rank: int,
    device: Any,
    generator_ranks: Sequence[int],
    score_ranks: Sequence[int],
    gradients_by_slot: Mapping[int, Any],
) -> dict[int, Any]:
    """Return one trace-qualified DMD latent gradient for every logical slot."""

    import torch.distributed as dist

    routes = _pipeline_slot_routes(generator_ranks, score_ranks)
    expected_generator_slots = set(balanced_slots(generator_ranks).get(rank, ()))
    expected_score_slots = set(balanced_slots(score_ranks).get(rank, ()))
    if rank in score_ranks:
        _require(
            set(gradients_by_slot) == expected_score_slots,
            "Score routed-gradient slot inventory mismatch",
        )
    else:
        _require(not gradients_by_slot, "Generator rank unexpectedly owns Score gradients")

    operations: list[Any] = []
    received: dict[int, Any] = {}
    retained: list[Any] = []
    for slot, generator_rank, score_rank in routes:
        if rank == score_rank:
            gradient = gradients_by_slot[slot].contiguous()
            retained.append(gradient)
            operations.append(dist.P2POp(dist.isend, gradient, generator_rank))
        elif rank == generator_rank:
            gradient = torch.empty(LATENT_SHAPE, device=device, dtype=torch.float32)
            received[slot] = gradient
            operations.append(dist.P2POp(dist.irecv, gradient, score_rank))
    _require(
        set(received) == expected_generator_slots,
        "Generator routed-gradient slot inventory mismatch",
    )
    for request in dist.batch_isend_irecv(operations):
        request.wait()
    return received


def _run_pipeline_8x8(
    *,
    torch: Any,
    rank: int,
    trace: FormalTrace,
    loader: InputLoader,
    bundle: ModelBundle,
    streamers: Sequence[Any],
    score_context: ScoreContext | None,
    run_kind: str,
    evidence: RunEvidence,
    versions: dict[str, int],
    device: Any,
    rollout_runahead: bool = False,
    critic_b2_only: bool = False,
    critic_b1_accumulate: bool = False,
    critic_step1_split_overlap: bool = False,
    score_context_from_trace: bool = False,
    slot_pairs_by_lane: Sequence[Sequence[int]] | None = None,
    full_context_lens_elision: bool = False,
    score_context_prefetch: bool = False,
    split_dmd_shadow: bool = False,
    runahead_host_transport: Any | None = None,
    step_offset: int = 0,
    cycle_index: int = 0,
    precomputed_first: PreparedRollout | None = None,
    critic_shadow_hot_sync: Any | None = None,
    finalize_host_transport: bool = True,
) -> None:
    import torch.distributed as dist

    is_generator = rank < 8
    peer = rank + 8 if is_generator else rank - 8
    lane = rank if is_generator else peer
    slot_pairs = validate_slot_pairs(
        slot_pairs_by_lane
        if slot_pairs_by_lane is not None
        else tuple((index, index + 8) for index in range(8))
    )
    lane_slots = slot_pairs[lane]
    pair_bytes = math.prod(LATENT_SHAPE) * 4
    if not score_context_from_trace:
        pair_bytes += math.prod(TEXT_SHAPE) * 4
    if is_generator:
        _require(bundle.student is not None and len(streamers) == 2, "pipeline G role is incomplete")
        if split_dmd_shadow:
            _require(
                bundle.critic_shadow is not None and score_context is not None,
                "split-DMD Generator shadow is incomplete",
            )
    else:
        _require(bundle.critic is not None and bundle.teacher is not None and score_context is not None, "pipeline Score role is incomplete")

    if not evidence.split_dmd_shadow:
        evidence.split_dmd_shadow = {
            "enabled": split_dmd_shadow,
            "role": "generator" if is_generator else "score",
            "execution_status": "PASS" if split_dmd_shadow else "DISABLED",
            "fake_velocity_calls": 0,
            "teacher_velocity_calls": 0,
            "received_teacher_velocities": 0,
            "received_final_gradients": 0,
        }
    else:
        _require(
            evidence.split_dmd_shadow.get("enabled") == split_dmd_shadow
            and evidence.split_dmd_shadow.get("role")
            == ("generator" if is_generator else "score"),
            "split-DMD evidence role changed across cycles",
        )
    evidence.generator_runahead_transport = (
        runahead_host_transport.receipt()
        if runahead_host_transport is not None
        else {"enabled": False, "execution_status": "DISABLED"}
    )
    if runahead_host_transport is not None:
        _require(
            rollout_runahead
            and score_context_from_trace
            and full_context_lens_elision,
            "host run-ahead transport requires the exact run-ahead/local-context graph",
        )
    if critic_step1_split_overlap and not is_generator:
        _require(
            critic_b2_only
            and rollout_runahead
            and runahead_host_transport is not None
            and bundle.critic_gradient_reducer is not None,
            "step1 split overlap requires C21 B2 run-ahead and its async reducer",
        )
    if critic_b1_accumulate and not is_generator:
        _require(
            not critic_b2_only
            and not critic_step1_split_overlap
            and rollout_runahead
            and runahead_host_transport is not None
            and bundle.critic_gradient_reducer is not None,
            "C24 Critic-B1 attribution requires two accumulated B1 backwards",
        )

    prefetched_contexts: dict[tuple[int, int], ScoreSampleContext] = {}
    if score_context_prefetch:
        _require(
            score_context_from_trace and full_context_lens_elision,
            "Score context prefetch requires trace-local full contexts",
        )
        if is_generator:
            evidence.score_context_prefetch = {
                "enabled": True,
                "role": "generator",
                "execution_status": "NOT_APPLICABLE",
                "logical_context_count": 0,
                "critic_context_count": 0,
                "teacher_context_count": 0,
                "gpu_resident_bytes": 0,
                "timing_position": "inside_full5_before_first_latent_receive",
                "calls_preserved": True,
            }
        else:
            prefetched_contexts, evidence.score_context_prefetch = (
                _prefetch_score_sample_contexts(
                    torch,
                    trace=trace,
                    loader=loader,
                    bundle=bundle,
                    lane_slots=lane_slots,
                    device=device,
                    run_kind=run_kind,
                )
            )
    else:
        evidence.score_context_prefetch = {
            "enabled": False,
            "role": "generator" if is_generator else "score",
            "execution_status": "DISABLED",
        }

    _require(step_offset == cycle_index * FULL5_STEPS, "cycle step offset changed")
    for step in range(FULL5_STEPS):
        global_step = step_offset + step
        if is_generator and rollout_runahead and step == 1:
            # Critic steps 1--4 do not feed a gradient back into Generator.
            # Keep exactly two persistent streamers and two pending P2P
            # payloads.  A streamer is reused only after its previous send has
            # completed, while the other streamer can produce the next real B1
            # sample during the preceding Critic backward/update.
            bundle.generator_optimizer.zero_grad(set_to_none=True)
            pending: list[tuple[PreparedRollout, tuple[Any, ...]] | None] = [
                None,
                None,
            ]
            staged_keys: list[tuple[int, int] | None] = [None, None]
            for ahead_step in range(1, FULL5_STEPS):
                ahead_global_step = step_offset + ahead_step
                ahead_rows = tuple(
                    trace.row(ahead_global_step, slot) for slot in lane_slots
                )
                for wave in (0, 1):
                    previous = pending[wave]
                    staged_key = staged_keys[wave]
                    if runahead_host_transport is not None and staged_key is not None:
                        with _profile_range(
                            torch,
                            run_kind=run_kind,
                            label=(
                                "livewan/pipeline_runahead/g/"
                                f"step_{ahead_step}/host_stage_wait_wave_{wave}"
                            ),
                        ):
                            runahead_host_transport.wait_staged(
                                step=staged_key[0], wave=staged_key[1]
                            )
                        staged_keys[wave] = None
                    elif previous is not None:
                        with _profile_range(
                            torch,
                            run_kind=run_kind,
                            label=(
                                "livewan/pipeline_runahead/g/"
                                f"step_{ahead_step}/reuse_wait_wave_{wave}"
                            ),
                        ):
                            for request in previous[1]:
                                request.wait()
                        pending[wave] = None
                    with _profile_range(
                        torch,
                        run_kind=run_kind,
                        label=(
                            "livewan/pipeline_runahead/g/"
                            f"step_{ahead_step}/rollout_wave_{wave}"
                        ),
                    ):
                        prepared = prepare_rollout(
                            torch,
                            row=ahead_rows[wave],
                            streamer=streamers[wave],
                            student=bundle.student,
                            loader=loader,
                            device=device,
                            elide_full_context_lens=full_context_lens_elision,
                        )
                    _record_output(
                        torch,
                        evidence,
                        row=ahead_rows[wave],
                        x0=prepared.x0,
                        run_kind=run_kind,
                    )
                    with _profile_range(
                        torch,
                        run_kind=run_kind,
                        label=(
                            "livewan/pipeline_runahead/g/"
                            f"step_{ahead_step}/"
                            f"{'host_stage' if runahead_host_transport is not None else 'isend'}_wave_{wave}"
                        ),
                    ):
                        if runahead_host_transport is not None:
                            runahead_host_transport.enqueue_send(
                                step=ahead_global_step, wave=wave, tensor=prepared.x0
                            )
                            works = ()
                        else:
                            works = _send_pair(
                                destination=peer,
                                prepared=prepared,
                                asynchronous=True,
                                send_context=not score_context_from_trace,
                            )
                    evidence.transport_bytes += pair_bytes
                    if runahead_host_transport is not None:
                        staged_keys[wave] = (ahead_global_step, wave)
                        pending[wave] = None
                    else:
                        # Retain payload tensors until NCCL reports completion.
                        pending[wave] = (prepared, tuple(works))
            if runahead_host_transport is not None:
                if finalize_host_transport:
                    with _profile_range(
                        torch,
                        run_kind=run_kind,
                        label="livewan/pipeline_runahead/g/final_host_send_wait",
                    ):
                        runahead_host_transport.finish()
                    evidence.generator_runahead_transport = (
                        runahead_host_transport.receipt()
                    )
            else:
                for wave, previous in enumerate(pending):
                    _require(previous is not None, "missing final run-ahead send")
                    with _profile_range(
                        torch,
                        run_kind=run_kind,
                        label=f"livewan/pipeline_runahead/g/final_wait_wave_{wave}",
                    ):
                        for request in previous[1]:
                            request.wait()
            break
        if is_generator:
            bundle.generator_optimizer.zero_grad(set_to_none=True)
            rows = tuple(trace.row(global_step, slot) for slot in lane_slots)
            if step == 0 and precomputed_first is not None:
                _require(
                    precomputed_first.row.operation_key == rows[0].operation_key,
                    "cross-cycle lookahead row differs from the next trace row",
                )
                _require(
                    runahead_host_transport is not None and cycle_index > 0,
                    "cross-cycle lookahead requires the continuous host transport",
                )
                first = precomputed_first
            else:
                with _profile_range(
                    torch, run_kind=run_kind,
                    label=f"livewan/pipeline/g/step_{step}/rollout_wave_0",
                ):
                    first = prepare_rollout(
                        torch, row=rows[0], streamer=streamers[0], student=bundle.student,
                        loader=loader, device=device,
                        elide_full_context_lens=full_context_lens_elision,
                    )
                _record_output(torch, evidence, row=rows[0], x0=first.x0, run_kind=run_kind)
                with _profile_range(
                    torch, run_kind=run_kind,
                    label=f"livewan/pipeline/g/step_{step}/send_wave_0",
                ):
                    _send_pair(
                        destination=peer,
                        prepared=first,
                        asynchronous=False,
                        send_context=not score_context_from_trace,
                    )
                evidence.transport_bytes += pair_bytes
            # C21-0 is deliberately depth-1: only next-cycle wave0 may be
            # produced before the previous C5 boundary.  Waiting for the exact
            # Critic-shadow version here prevents wave1 from silently becoming
            # a second cross-cycle lookahead sample.  Once the sync arrives,
            # wave1 can overlap the Score pool's Teacher work on wave0.
            if step == 0 and split_dmd_shadow and cycle_index > 0:
                _require(
                    critic_shadow_hot_sync is not None,
                    "continuous Generator lacks Critic-shadow hot sync",
                )
                sync_started = time.perf_counter()
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label=(
                        "livewan/continuous/"
                        f"cycle_{cycle_index}/critic_shadow_receive_C{cycle_index * 5}"
                    ),
                ):
                    critic_shadow_hot_sync.receive_and_load(
                        version=cycle_index * CRITIC_UPDATES
                    )
                evidence.generator_backpressure.append(
                    {
                        "cycle": cycle_index,
                        "kind": "critic_shadow_receive_wait_before_wave1",
                        "wall_seconds": time.perf_counter() - sync_started,
                    }
                )
            with _profile_range(
                torch, run_kind=run_kind,
                label=f"livewan/pipeline/g/step_{step}/rollout_wave_1",
            ):
                second = prepare_rollout(
                    torch, row=rows[1], streamer=streamers[1], student=bundle.student,
                    loader=loader, device=device,
                    elide_full_context_lens=full_context_lens_elision,
                )
            _record_output(torch, evidence, row=rows[1], x0=second.x0, run_kind=run_kind)
            if step == 0:
                if split_dmd_shadow:
                    with _profile_range(
                        torch,
                        run_kind=run_kind,
                        label="livewan/pipeline/g/step_0/fake_velocity_wave_0",
                    ):
                        shadow_context0 = prepare_score_sample_context(
                            torch,
                            bundle=bundle,
                            raw_context=first.raw_context,
                            include_teacher=False,
                            critic_model=bundle.critic_shadow,
                        )
                        flow0 = _dmd_flow_inputs(torch, row=rows[0], x0=first.x0)
                        fake0 = _dmd_fake_velocity(
                            torch,
                            critic_model=bundle.critic_shadow,
                            score_context=score_context,
                            flow=flow0,
                            critic_context=shadow_context0.critic,
                            context_lens=first.context_lens,
                        )
                    evidence.add_logical_score_calls("critic/nograd/plain", 1)
                    evidence.split_dmd_shadow["fake_velocity_calls"] += 1
                    real0 = torch.empty_like(first.x0)
                    with _profile_range(
                        torch,
                        run_kind=run_kind,
                        label="livewan/pipeline/g/step_0/recv_teacher_velocity_0",
                    ):
                        dist.recv(real0, src=peer)
                    evidence.transport_bytes += real0.numel() * real0.element_size()
                    evidence.split_dmd_shadow["received_teacher_velocities"] += 1
                    grad0 = _combine_dmd_gradient(
                        torch, x0=first.x0, flow=flow0, v_real=real0, v_fake=fake0
                    )
                    sends = _send_pair(
                        destination=peer,
                        prepared=second,
                        asynchronous=True,
                        send_context=not score_context_from_trace,
                    )
                    evidence.transport_bytes += pair_bytes
                    with _profile_range(
                        torch,
                        run_kind=run_kind,
                        label="livewan/pipeline/g/step_0/fake_velocity_wave_1",
                    ):
                        shadow_context1 = prepare_score_sample_context(
                            torch,
                            bundle=bundle,
                            raw_context=second.raw_context,
                            include_teacher=False,
                            critic_model=bundle.critic_shadow,
                        )
                        flow1 = _dmd_flow_inputs(torch, row=rows[1], x0=second.x0)
                        fake1 = _dmd_fake_velocity(
                            torch,
                            critic_model=bundle.critic_shadow,
                            score_context=score_context,
                            flow=flow1,
                            critic_context=shadow_context1.critic,
                            context_lens=second.context_lens,
                        )
                    evidence.add_logical_score_calls("critic/nograd/plain", 1)
                    evidence.split_dmd_shadow["fake_velocity_calls"] += 1
                else:
                    # The peers must enqueue matching default-group P2P calls.
                    grad0 = torch.empty_like(first.x0)
                    with _profile_range(
                        torch, run_kind=run_kind,
                        label="livewan/pipeline/g/step_0/recv_gradient_0",
                    ):
                        dist.recv(grad0, src=peer)
                    evidence.transport_bytes += grad0.numel() * grad0.element_size()
                    evidence.split_dmd_shadow["received_final_gradients"] += 1
                    sends = _send_pair(
                        destination=peer,
                        prepared=second,
                        asynchronous=True,
                        send_context=not score_context_from_trace,
                    )
                    evidence.transport_bytes += pair_bytes
                with _profile_range(
                    torch, run_kind=run_kind,
                    label="livewan/pipeline/g/step_0/backward_wave_0",
                ):
                    losses0 = generator_backward_for_sample(
                        torch, prepared=first, dmd_gradient=grad0, sample_weight=0.5,
                    )
                for request in sends:
                    request.wait()
                returned1 = torch.empty_like(second.x0)
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label=(
                        "livewan/pipeline/g/step_0/recv_teacher_velocity_1"
                        if split_dmd_shadow
                        else "livewan/pipeline/g/step_0/recv_gradient_1"
                    ),
                ):
                    dist.recv(returned1, src=peer)
                evidence.transport_bytes += returned1.numel() * returned1.element_size()
                if split_dmd_shadow:
                    evidence.split_dmd_shadow["received_teacher_velocities"] += 1
                    grad1 = _combine_dmd_gradient(
                        torch,
                        x0=second.x0,
                        flow=flow1,
                        v_real=returned1,
                        v_fake=fake1,
                    )
                else:
                    evidence.split_dmd_shadow["received_final_gradients"] += 1
                    grad1 = returned1
                with _profile_range(
                    torch, run_kind=run_kind,
                    label="livewan/pipeline/g/step_0/backward_wave_1",
                ):
                    losses1 = generator_backward_for_sample(
                        torch, prepared=second, dmd_gradient=grad1, sample_weight=0.5,
                    )
                if run_kind in {"correctness", "trajectory"}:
                    for row, losses in zip(rows, (losses0, losses1)):
                        evidence.loss_fingerprints.extend(
                            {"operation_key": row.operation_key, "kind": "generator", **_tensor_fingerprint(torch, value)}
                            for value in losses
                        )
                with _profile_range(
                    torch, run_kind=run_kind,
                    label="livewan/pipeline/g/step_0/generator_commit",
                ):
                    sync = _mean_replicated_gradients(
                        torch, list(bundle.student.parameters()), group=bundle.generator_group,
                        group_size=bundle.generator_group_size, bucket_mib=256,
                    )
                    evidence.gradient_sync.append({"step": global_step, "role": "generator", **sync})
                    generator_parameters = list(bundle.student.parameters())
                    generator_pre_clip = (
                        _state_metrics(torch, [parameter.grad for parameter in generator_parameters])
                        if run_kind == "correctness" else None
                    )
                    generator_clip_norm = torch.nn.utils.clip_grad_norm_(
                        generator_parameters, 1.0, foreach=True,
                    )
                    _record_optimizer_diagnostics(
                        torch, evidence, run_kind=run_kind, role="generator", step=global_step,
                        parameters=generator_parameters, clip_norm=generator_clip_norm,
                        gradient_before_clip=generator_pre_clip,
                    )
                    bundle.generator_optimizer.step()
                    versions["generator"] += 1
            else:
                sends = _send_pair(
                    destination=peer,
                    prepared=second,
                    asynchronous=True,
                    send_context=not score_context_from_trace,
                )
                evidence.transport_bytes += pair_bytes
                for request in sends:
                    request.wait()
        else:
            if bundle.critic_gradient_reducer is not None:
                bundle.critic_gradient_reducer.zero_grad()
            else:
                bundle.critic_optimizer.zero_grad(set_to_none=True)
            rows = tuple(trace.row(global_step, slot) for slot in lane_slots)
            receive_started = time.perf_counter()
            with _profile_range(
                torch, run_kind=run_kind,
                label=f"livewan/pipeline/score/step_{step}/recv_wave_0",
            ):
                if runahead_host_transport is not None and (
                    step > 0 or (cycle_index > 0 and step == 0)
                ):
                    first_x0 = runahead_host_transport.receive_to_device(
                        step=global_step, wave=0
                    )
                    first_context = None
                else:
                    first_x0, first_context = _recv_pair(
                        torch,
                        source=peer,
                        device=device,
                        receive_context=not score_context_from_trace,
                    )
            evidence.score_receive_waits.append(
                {
                    "cycle": cycle_index,
                    "step": step,
                    "global_step": global_step,
                    "wave": 0,
                    "wall_seconds": time.perf_counter() - receive_started,
                }
            )
            evidence.transport_bytes += pair_bytes
            if score_context_from_trace:
                if not score_context_prefetch:
                    with _profile_range(
                        torch,
                        run_kind=run_kind,
                        label=f"livewan/pipeline/score/step_{step}/local_context_0",
                    ):
                        first_context = loader.context_from_trace(rows[0], device=device)
            context_lens = (
                None
                if full_context_lens_elision
                else torch.tensor([TEXT_SHAPE[1]], device=device, dtype=torch.long)
            )
            if score_context_prefetch:
                first_sample_context = prefetched_contexts.pop((step, 0))
            else:
                _require(first_context is not None, "first Score context is missing")
                first_sample_context = prepare_score_sample_context(
                    torch,
                    bundle=bundle,
                    raw_context=first_context,
                    include_teacher=step == 0,
                )
            split_step1 = critic_step1_split_overlap and step == 1
            loss0 = None
            if split_step1:
                # Start the first B1 backward as soon as wave0 arrives.  The G
                # pool keeps producing/staging wave1 concurrently.  Both B1
                # contributions retain 0.5 weight and the reducer launches
                # collectives only on the second backward, so this remains one
                # exact B2-mean Critic commit.
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label=(
                        "livewan/continuous_c1/score/step_1/"
                        "critic_backward_wave_0_overlap"
                    ),
                ):
                    loss0 = critic_backward_for_sample(
                        torch,
                        bundle=bundle,
                        score_context=score_context,
                        row=rows[0],
                        x0=first_x0,
                        sample_context=first_sample_context,
                        context_lens=context_lens,
                        sample_weight=0.5,
                        reducer_accumulation=(0, 2),
                    )
                    evidence.add_logical_score_calls("critic/grad/checkpoint", 1)
            if step == 0:
                with _profile_range(
                    torch, run_kind=run_kind,
                    label=(
                        "livewan/pipeline/score/step_0/teacher_velocity_wave_0"
                        if split_dmd_shadow
                        else "livewan/pipeline/score/step_0/dmd_wave_0"
                    ),
                ):
                    if split_dmd_shadow:
                        flow0 = _dmd_flow_inputs(torch, row=rows[0], x0=first_x0)
                        returned0 = _dmd_teacher_velocity(
                            torch,
                            bundle=bundle,
                            score_context=score_context,
                            flow=flow0,
                            sample_context=first_sample_context,
                            context_lens=context_lens,
                        )
                        evidence.split_dmd_shadow["teacher_velocity_calls"] += 1
                    else:
                        returned0 = dmd_gradient_for_sample(
                            torch, bundle=bundle, score_context=score_context, row=rows[0],
                            x0=first_x0, sample_context=first_sample_context, context_lens=context_lens,
                        )
                    evidence.add_logical_score_calls("teacher/nograd/plain", 1)
                    if not split_dmd_shadow:
                        evidence.add_logical_score_calls("critic/nograd/plain", 1)
                    dist.send(returned0.contiguous(), dst=peer)
                evidence.transport_bytes += returned0.numel() * returned0.element_size()
            if cycle_index > 0 and step == 0:
                _require(
                    critic_shadow_hot_sync is not None,
                    "continuous Score lacks Critic-shadow hot sync",
                )
                critic_shadow_hot_sync.wait_source(
                    expected_version=cycle_index * CRITIC_UPDATES
                )
            receive_started = time.perf_counter()
            with _profile_range(
                torch, run_kind=run_kind,
                label=f"livewan/pipeline/score/step_{step}/recv_wave_1",
            ):
                if runahead_host_transport is not None and step > 0:
                    second_x0 = runahead_host_transport.receive_to_device(
                        step=global_step, wave=1
                    )
                    second_context = None
                else:
                    second_x0, second_context = _recv_pair(
                        torch,
                        source=peer,
                        device=device,
                        receive_context=not score_context_from_trace,
                    )
            evidence.score_receive_waits.append(
                {
                    "cycle": cycle_index,
                    "step": step,
                    "global_step": global_step,
                    "wave": 1,
                    "wall_seconds": time.perf_counter() - receive_started,
                }
            )
            evidence.transport_bytes += pair_bytes
            if score_context_from_trace:
                if not score_context_prefetch:
                    with _profile_range(
                        torch,
                        run_kind=run_kind,
                        label=f"livewan/pipeline/score/step_{step}/local_context_1",
                    ):
                        second_context = loader.context_from_trace(rows[1], device=device)
            if score_context_prefetch:
                second_sample_context = prefetched_contexts.pop((step, 1))
            else:
                _require(second_context is not None, "second Score context is missing")
                second_sample_context = prepare_score_sample_context(
                    torch,
                    bundle=bundle,
                    raw_context=second_context,
                    include_teacher=step == 0,
                )
            grad1_work = None
            grad1_send = None
            if step == 0:
                # DMD uses frozen Score/Teacher parameters and is independent
                # of the Critic gradient buffers.  Producing both returned
                # gradients before Critic backward therefore preserves the
                # original loss/backward order while removing a false cross-
                # pool dependency: G backward0 can now overlap Critic
                # backward0, instead of waiting for Critic0 before Score even
                # starts DMD1.  Keep the second send asynchronous so Score can
                # immediately enter Critic backward while G consumes grad0.
                with _profile_range(
                    torch, run_kind=run_kind,
                    label=(
                        "livewan/pipeline/score/step_0/teacher_velocity_wave_1"
                        if split_dmd_shadow
                        else "livewan/pipeline/score/step_0/dmd_wave_1"
                    ),
                ):
                    if split_dmd_shadow:
                        flow1 = _dmd_flow_inputs(torch, row=rows[1], x0=second_x0)
                        returned1 = _dmd_teacher_velocity(
                            torch,
                            bundle=bundle,
                            score_context=score_context,
                            flow=flow1,
                            sample_context=second_sample_context,
                            context_lens=context_lens,
                        )
                        evidence.split_dmd_shadow["teacher_velocity_calls"] += 1
                    else:
                        returned1 = dmd_gradient_for_sample(
                            torch, bundle=bundle, score_context=score_context, row=rows[1],
                            x0=second_x0, sample_context=second_sample_context, context_lens=context_lens,
                        )
                    evidence.add_logical_score_calls("teacher/nograd/plain", 1)
                    if not split_dmd_shadow:
                        evidence.add_logical_score_calls("critic/nograd/plain", 1)
                    grad1_send = returned1.contiguous()
                    grad1_work = dist.isend(grad1_send, dst=peer)
                evidence.transport_bytes += returned1.numel() * returned1.element_size()
            if split_step1:
                _require(loss0 is not None, "step1 wave0 Critic loss is missing")
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label=(
                        "livewan/continuous_c1/score/step_1/"
                        "critic_backward_wave_1"
                    ),
                ):
                    loss1 = critic_backward_for_sample(
                        torch,
                        bundle=bundle,
                        score_context=score_context,
                        row=rows[1],
                        x0=second_x0,
                        sample_context=second_sample_context,
                        context_lens=context_lens,
                        sample_weight=0.5,
                        reducer_accumulation=(1, 2),
                    )
                    evidence.add_logical_score_calls("critic/grad/checkpoint", 1)
            elif critic_b2_only:
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label=f"livewan/pipeline_runahead/score/step_{step}/critic_backward_b2",
                ):
                    batched_context = ScoreSampleContext(
                        teacher=(
                            torch.cat(
                                (
                                    first_sample_context.teacher,
                                    second_sample_context.teacher,
                                ),
                                dim=0,
                            )
                            if step == 0
                            else None
                        ),
                        critic=torch.cat(
                            (first_sample_context.critic, second_sample_context.critic),
                            dim=0,
                        ),
                    )
                    losses = critic_backward_for_batch(
                        torch,
                        bundle=bundle,
                        score_context=score_context,
                        rows=rows,
                        x0=torch.cat((first_x0, second_x0), dim=0),
                        sample_context=batched_context,
                        context_lens=(
                            None
                            if full_context_lens_elision
                            else torch.full(
                                (2,), TEXT_SHAPE[1], device=device, dtype=torch.long
                            )
                        ),
                        loss_scale=1.0,
                    )
                    loss0, loss1 = losses
                    evidence.add_logical_score_calls("critic/grad/checkpoint", 2)
            else:
                with _profile_range(
                    torch, run_kind=run_kind,
                    label=f"livewan/pipeline/score/step_{step}/critic_backward_wave_0",
                ):
                    loss0 = critic_backward_for_sample(
                        torch, bundle=bundle, score_context=score_context, row=rows[0],
                        x0=first_x0, sample_context=first_sample_context, context_lens=context_lens,
                        sample_weight=0.5,
                        reducer_accumulation=(
                            _critic_b1_attribution_reducer_accumulation(0)
                            if critic_b1_accumulate
                            else None
                        ),
                    )
                    evidence.add_logical_score_calls("critic/grad/checkpoint", 1)
                with _profile_range(
                    torch, run_kind=run_kind,
                    label=f"livewan/pipeline/score/step_{step}/critic_backward_wave_1",
                ):
                    loss1 = critic_backward_for_sample(
                        torch, bundle=bundle, score_context=score_context, row=rows[1],
                        x0=second_x0, sample_context=second_sample_context, context_lens=context_lens,
                        sample_weight=0.5,
                        reducer_accumulation=(
                            _critic_b1_attribution_reducer_accumulation(1)
                            if critic_b1_accumulate
                            else None
                        ),
                    )
                    evidence.add_logical_score_calls("critic/grad/checkpoint", 1)
            if run_kind == "trajectory":
                # Count completed physical backward invocations from the
                # branch that actually ran.  Do not infer this from logical
                # sample inventory: B2 contributes two logical calls but one
                # physical backward, while the B1 attribution arm contributes
                # two physical backwards before the same commit.
                evidence.add_physical_critic_backward_calls(
                    _physical_critic_backward_calls_for_step(
                        critic_b2_only=critic_b2_only,
                        split_step1=split_step1,
                    )
                )
            if grad1_work is not None:
                with _profile_range(
                    torch, run_kind=run_kind,
                    label="livewan/pipeline/score/step_0/send_gradient_1_wait",
                ):
                    grad1_work.wait()
                # Retain the contiguous send tensor until the asynchronous
                # work is complete; deleting it earlier would make the
                # transport lifetime implementation-dependent.
                del grad1_send
            if run_kind in {"correctness", "trajectory"}:
                for row, loss in zip(rows, (loss0, loss1)):
                    evidence.loss_fingerprints.append(
                        {"operation_key": row.operation_key, "kind": "critic", **_tensor_fingerprint(torch, loss)}
                    )
            with _profile_range(
                torch, run_kind=run_kind,
                label=f"livewan/pipeline/score/step_{step}/critic_commit",
            ):
                if bundle.critic_gradient_reducer is None:
                    sync = _mean_replicated_gradients(
                        torch, bundle.critic_lora_params, group=bundle.score_group,
                        group_size=bundle.score_group_size, bucket_mib=128,
                    )
                else:
                    sync = bundle.critic_gradient_reducer.finish()
                evidence.gradient_sync.append({"step": global_step, "role": "critic", **sync})
                critic_pre_clip = (
                    _state_metrics(torch, [parameter.grad for parameter in bundle.critic_lora_params])
                    if run_kind == "correctness" else None
                )
                critic_clip_norm = torch.nn.utils.clip_grad_norm_(
                    bundle.critic_lora_params, 1.0, foreach=True,
                )
                _record_optimizer_diagnostics(
                    torch, evidence, run_kind=run_kind, role="critic", step=global_step,
                    parameters=bundle.critic_lora_params, clip_norm=critic_clip_norm,
                    gradient_before_clip=critic_pre_clip,
                )
                bundle.critic_optimizer.step()
                versions["critic"] += 1
    if runahead_host_transport is not None and not is_generator and finalize_host_transport:
        runahead_host_transport.finish()
        evidence.generator_runahead_transport = runahead_host_transport.receipt()
    if score_context_prefetch and not is_generator:
        _require(
            not prefetched_contexts,
            "Score context prefetch left unconsumed lane contexts",
        )


def _run_pipeline_8x8_b32(
    *,
    torch: Any,
    rank: int,
    trace: FormalTrace,
    loader: InputLoader,
    bundle: ModelBundle,
    streamers: Sequence[Any],
    score_context: ScoreContext | None,
    run_kind: str,
    evidence: RunEvidence,
    versions: dict[str, int],
    device: Any,
    slot_groups_by_lane: Sequence[Sequence[int]],
    runahead_host_transport: Any,
    step_offset: int,
    cycle_index: int,
    precomputed_first: PreparedRollout | None,
    critic_shadow_hot_sync: Any,
) -> None:
    """Run one exact global-B32 cycle as two bounded B2 microbatches/lane.

    Four rank-qualified samples are consumed per fixed G/Score lane.  Only two
    persistent rollout streamers and one B2 Critic activation set are live at a
    time.  Generator gradients use four 0.25 contributions; Critic gradients
    use two B2 means scaled by 0.5 and are reduced only after the second B2.
    Both roles therefore retain one optimizer commit per logical update.
    """

    import torch.distributed as dist

    _require(trace.global_batch == 32, "B32 worker received a non-B32 trace")
    _require(
        runahead_host_transport is not None
        and critic_shadow_hot_sync is not None,
        "B32 continuous transports are incomplete",
    )
    _require(
        int(runahead_host_transport.waves_per_step) == 4,
        "B32 host transport does not own four waves",
    )
    is_generator = rank < 8
    peer = rank + 8 if is_generator else rank - 8
    lane = rank if is_generator else rank - 8
    lane_slots = validate_slot_groups(
        slot_groups_by_lane, batch=32, slots_per_lane=4
    )[lane]
    _require(len(streamers) == 2 if is_generator else not streamers, "B32 streamer residency changed")
    _require(step_offset == cycle_index * FULL5_STEPS, "B32 cycle offset changed")
    pair_bytes = math.prod(LATENT_SHAPE) * 4

    if not evidence.split_dmd_shadow:
        evidence.split_dmd_shadow = {
            "enabled": True,
            "role": "generator" if is_generator else "score",
            "execution_status": "PASS",
            "fake_velocity_calls": 0,
            "teacher_velocity_calls": 0,
            "received_teacher_velocities": 0,
            "received_final_gradients": 0,
        }
    evidence.generator_runahead_transport = runahead_host_transport.receipt()
    evidence.score_context_prefetch = {
        "enabled": False,
        "role": "generator" if is_generator else "score",
        "execution_status": "DISABLED",
    }

    def score_sample(row: OperationTraceRow, *, include_teacher: bool) -> ScoreSampleContext:
        raw_context = loader.context_from_trace(row, device=device)
        return prepare_score_sample_context(
            torch,
            bundle=bundle,
            raw_context=raw_context,
            include_teacher=include_teacher,
        )

    for step in range(FULL5_STEPS):
        global_step = step_offset + step
        rows = tuple(trace.row(global_step, slot) for slot in lane_slots)

        if is_generator and step == 1:
            # Produce all four B1 waves for Critic-only steps with two reusable
            # streamers.  Reuse waits only for the prior D2H staging event.
            bundle.generator_optimizer.zero_grad(set_to_none=True)
            staged_keys: list[tuple[int, int] | None] = [None, None]
            for ahead_step in range(1, FULL5_STEPS):
                ahead_global_step = step_offset + ahead_step
                ahead_rows = tuple(
                    trace.row(ahead_global_step, slot) for slot in lane_slots
                )
                for wave in range(4):
                    streamer_index = wave % 2
                    staged_key = staged_keys[streamer_index]
                    if staged_key is not None:
                        with _profile_range(
                            torch,
                            run_kind=run_kind,
                            label=(
                                "livewan/pipeline_b32/g/"
                                f"step_{ahead_step}/host_stage_wait_wave_{wave}"
                            ),
                        ):
                            runahead_host_transport.wait_staged(
                                step=staged_key[0], wave=staged_key[1]
                            )
                    with _profile_range(
                        torch,
                        run_kind=run_kind,
                        label=(
                            "livewan/pipeline_b32/g/"
                            f"step_{ahead_step}/rollout_wave_{wave}"
                        ),
                    ):
                        prepared = prepare_rollout(
                            torch,
                            row=ahead_rows[wave],
                            streamer=streamers[streamer_index],
                            student=bundle.student,
                            loader=loader,
                            device=device,
                            elide_full_context_lens=True,
                        )
                    _record_output(
                        torch,
                        evidence,
                        row=ahead_rows[wave],
                        x0=prepared.x0,
                        run_kind=run_kind,
                    )
                    runahead_host_transport.enqueue_send(
                        step=ahead_global_step, wave=wave, tensor=prepared.x0
                    )
                    evidence.transport_bytes += pair_bytes
                    staged_keys[streamer_index] = (ahead_global_step, wave)
            break

        if is_generator:
            _require(step == 0, "B32 Generator reached an unexpected logical step")
            _require(bundle.student is not None and bundle.critic_shadow is not None, "B32 G role is incomplete")
            _require(score_context is not None, "B32 G lacks Score context")
            bundle.generator_optimizer.zero_grad(set_to_none=True)
            generator_loss_rows: list[tuple[OperationTraceRow, list[Any]]] = []
            for microbatch in range(2):
                first_wave = microbatch * 2
                second_wave = first_wave + 1
                first_row, second_row = rows[first_wave], rows[second_wave]
                if first_wave == 0 and precomputed_first is not None:
                    _require(
                        precomputed_first.row.operation_key == first_row.operation_key,
                        "B32 cross-cycle lookahead row changed",
                    )
                    first = precomputed_first
                else:
                    with _profile_range(
                        torch,
                        run_kind=run_kind,
                        label=f"livewan/pipeline_b32/g/step_0/rollout_wave_{first_wave}",
                    ):
                        first = prepare_rollout(
                            torch,
                            row=first_row,
                            streamer=streamers[0],
                            student=bundle.student,
                            loader=loader,
                            device=device,
                            elide_full_context_lens=True,
                        )
                    _record_output(
                        torch, evidence, row=first_row, x0=first.x0, run_kind=run_kind
                    )
                    _send_pair(
                        destination=peer,
                        prepared=first,
                        asynchronous=False,
                        send_context=False,
                    )
                    evidence.transport_bytes += pair_bytes

                if first_wave == 0 and cycle_index > 0:
                    sync_started = time.perf_counter()
                    critic_shadow_hot_sync.receive_and_load(
                        version=cycle_index * CRITIC_UPDATES
                    )
                    evidence.generator_backpressure.append(
                        {
                            "cycle": cycle_index,
                            "kind": "critic_shadow_receive_wait_before_wave1",
                            "wall_seconds": time.perf_counter() - sync_started,
                        }
                    )

                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label=f"livewan/pipeline_b32/g/step_0/rollout_wave_{second_wave}",
                ):
                    second = prepare_rollout(
                        torch,
                        row=second_row,
                        streamer=streamers[1],
                        student=bundle.student,
                        loader=loader,
                        device=device,
                        elide_full_context_lens=True,
                    )
                _record_output(
                    torch, evidence, row=second_row, x0=second.x0, run_kind=run_kind
                )

                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label=f"livewan/pipeline_b32/g/step_0/fake_velocity_wave_{first_wave}",
                ):
                    shadow0 = prepare_score_sample_context(
                        torch,
                        bundle=bundle,
                        raw_context=first.raw_context,
                        include_teacher=False,
                        critic_model=bundle.critic_shadow,
                    )
                    flow0 = _dmd_flow_inputs(torch, row=first_row, x0=first.x0)
                    fake0 = _dmd_fake_velocity(
                        torch,
                        critic_model=bundle.critic_shadow,
                        score_context=score_context,
                        flow=flow0,
                        critic_context=shadow0.critic,
                        context_lens=first.context_lens,
                    )
                evidence.add_logical_score_calls("critic/nograd/plain", 1)
                evidence.split_dmd_shadow["fake_velocity_calls"] += 1
                real0 = torch.empty_like(first.x0)
                dist.recv(real0, src=peer)
                evidence.transport_bytes += real0.numel() * real0.element_size()
                evidence.split_dmd_shadow["received_teacher_velocities"] += 1
                grad0 = _combine_dmd_gradient(
                    torch, x0=first.x0, flow=flow0, v_real=real0, v_fake=fake0
                )

                sends = _send_pair(
                    destination=peer,
                    prepared=second,
                    asynchronous=True,
                    send_context=False,
                )
                evidence.transport_bytes += pair_bytes
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label=f"livewan/pipeline_b32/g/step_0/fake_velocity_wave_{second_wave}",
                ):
                    shadow1 = prepare_score_sample_context(
                        torch,
                        bundle=bundle,
                        raw_context=second.raw_context,
                        include_teacher=False,
                        critic_model=bundle.critic_shadow,
                    )
                    flow1 = _dmd_flow_inputs(torch, row=second_row, x0=second.x0)
                    fake1 = _dmd_fake_velocity(
                        torch,
                        critic_model=bundle.critic_shadow,
                        score_context=score_context,
                        flow=flow1,
                        critic_context=shadow1.critic,
                        context_lens=second.context_lens,
                    )
                evidence.add_logical_score_calls("critic/nograd/plain", 1)
                evidence.split_dmd_shadow["fake_velocity_calls"] += 1
                losses0 = generator_backward_for_sample(
                    torch, prepared=first, dmd_gradient=grad0, sample_weight=0.25
                )
                for request in sends:
                    request.wait()
                real1 = torch.empty_like(second.x0)
                dist.recv(real1, src=peer)
                evidence.transport_bytes += real1.numel() * real1.element_size()
                evidence.split_dmd_shadow["received_teacher_velocities"] += 1
                grad1 = _combine_dmd_gradient(
                    torch, x0=second.x0, flow=flow1, v_real=real1, v_fake=fake1
                )
                losses1 = generator_backward_for_sample(
                    torch, prepared=second, dmd_gradient=grad1, sample_weight=0.25
                )
                generator_loss_rows.extend(
                    ((first_row, losses0), (second_row, losses1))
                )
                del first, second, shadow0, shadow1, fake0, fake1, real0, real1, grad0, grad1

            if run_kind == "correctness":
                for row, losses in generator_loss_rows:
                    evidence.loss_fingerprints.extend(
                        {
                            "operation_key": row.operation_key,
                            "kind": "generator",
                            **_tensor_fingerprint(torch, value),
                        }
                        for value in losses
                    )
            with _profile_range(
                torch,
                run_kind=run_kind,
                label="livewan/pipeline_b32/g/step_0/generator_commit",
            ):
                generator_parameters = list(bundle.student.parameters())
                sync = _mean_replicated_gradients(
                    torch,
                    generator_parameters,
                    group=bundle.generator_group,
                    group_size=bundle.generator_group_size,
                    bucket_mib=256,
                )
                evidence.gradient_sync.append(
                    {"step": global_step, "role": "generator", **sync}
                )
                generator_pre_clip = (
                    _state_metrics(
                        torch,
                        [parameter.grad for parameter in generator_parameters],
                    )
                    if run_kind == "correctness"
                    else None
                )
                generator_clip_norm = torch.nn.utils.clip_grad_norm_(
                    generator_parameters, 1.0, foreach=True
                )
                _record_optimizer_diagnostics(
                    torch,
                    evidence,
                    run_kind=run_kind,
                    role="generator",
                    step=global_step,
                    parameters=generator_parameters,
                    clip_norm=generator_clip_norm,
                    gradient_before_clip=generator_pre_clip,
                )
                bundle.generator_optimizer.step()
                versions["generator"] += 1
            continue

        _require(bundle.critic is not None and bundle.teacher is not None, "B32 Score role is incomplete")
        _require(score_context is not None, "B32 Score context is missing")
        _require(bundle.critic_gradient_reducer is not None, "B32 Critic needs async accumulation")
        bundle.critic_gradient_reducer.zero_grad()
        critic_losses: list[tuple[OperationTraceRow, Any]] = []
        for microbatch in range(2):
            pair_rows = rows[microbatch * 2 : microbatch * 2 + 2]
            values: list[Any] = []
            contexts: list[ScoreSampleContext] = []
            teacher_send = None
            teacher_payload = None
            for local_wave, row in enumerate(pair_rows):
                wave = microbatch * 2 + local_wave
                receive_started = time.perf_counter()
                if runahead_host_transport is not None and (
                    step > 0 or (cycle_index > 0 and step == 0 and wave == 0)
                ):
                    x0 = runahead_host_transport.receive_to_device(
                        step=global_step, wave=wave
                    )
                else:
                    x0, _unused_context = _recv_pair(
                        torch, source=peer, device=device, receive_context=False
                    )
                evidence.score_receive_waits.append(
                    {
                        "cycle": cycle_index,
                        "step": step,
                        "global_step": global_step,
                        "wave": wave,
                        "wall_seconds": time.perf_counter() - receive_started,
                    }
                )
                evidence.transport_bytes += pair_bytes
                sample_context = score_sample(row, include_teacher=step == 0)
                values.append(x0)
                contexts.append(sample_context)
                if step == 0:
                    flow = _dmd_flow_inputs(torch, row=row, x0=x0)
                    teacher = _dmd_teacher_velocity(
                        torch,
                        bundle=bundle,
                        score_context=score_context,
                        flow=flow,
                        sample_context=sample_context,
                        context_lens=None,
                    )
                    evidence.add_logical_score_calls("teacher/nograd/plain", 1)
                    evidence.split_dmd_shadow["teacher_velocity_calls"] += 1
                    if local_wave == 0:
                        dist.send(teacher.contiguous(), dst=peer)
                    else:
                        teacher_payload = teacher.contiguous()
                        teacher_send = dist.isend(teacher_payload, dst=peer)
                    evidence.transport_bytes += teacher.numel() * teacher.element_size()
                    if microbatch == 0 and local_wave == 0 and cycle_index > 0:
                        critic_shadow_hot_sync.wait_source(
                            expected_version=cycle_index * CRITIC_UPDATES
                        )

            batched_context = ScoreSampleContext(
                teacher=(
                    torch.cat((contexts[0].teacher, contexts[1].teacher), dim=0)
                    if step == 0
                    else None
                ),
                critic=torch.cat((contexts[0].critic, contexts[1].critic), dim=0),
            )
            losses = critic_backward_for_batch(
                torch,
                bundle=bundle,
                score_context=score_context,
                rows=pair_rows,
                x0=torch.cat((values[0], values[1]), dim=0),
                sample_context=batched_context,
                context_lens=None,
                loss_scale=0.5,
                reducer_accumulation=(microbatch, 2),
            )
            evidence.add_logical_score_calls("critic/grad/checkpoint", 2)
            critic_losses.extend(zip(pair_rows, losses))
            if teacher_send is not None:
                teacher_send.wait()
                del teacher_payload
            del values, contexts, batched_context

        if run_kind == "correctness":
            for row, loss in critic_losses:
                evidence.loss_fingerprints.append(
                    {
                        "operation_key": row.operation_key,
                        "kind": "critic",
                        **_tensor_fingerprint(torch, loss),
                    }
                )
        with _profile_range(
            torch,
            run_kind=run_kind,
            label=f"livewan/pipeline_b32/score/step_{step}/critic_commit",
        ):
            sync = bundle.critic_gradient_reducer.finish()
            evidence.gradient_sync.append(
                {"step": global_step, "role": "critic", **sync}
            )
            critic_pre_clip = (
                _state_metrics(
                    torch,
                    [parameter.grad for parameter in bundle.critic_lora_params],
                )
                if run_kind == "correctness"
                else None
            )
            critic_clip_norm = torch.nn.utils.clip_grad_norm_(
                bundle.critic_lora_params, 1.0, foreach=True
            )
            _record_optimizer_diagnostics(
                torch,
                evidence,
                run_kind=run_kind,
                role="critic",
                step=global_step,
                parameters=bundle.critic_lora_params,
                clip_norm=critic_clip_norm,
                gradient_before_clip=critic_pre_clip,
            )
            bundle.critic_optimizer.step()
            versions["critic"] += 1


def _run_pipeline_8x8_continuous(
    *,
    torch: Any,
    rank: int,
    trace: FormalTrace,
    loader: InputLoader,
    bundle: ModelBundle,
    streamers: Sequence[Any],
    score_context: ScoreContext | None,
    run_kind: str,
    evidence: RunEvidence,
    versions: dict[str, int],
    device: Any,
    runahead_host_transport: Any,
    critic_shadow_hot_sync: Any,
    slot_pairs_by_lane: Sequence[Sequence[int]],
    full_context_lens_elision: bool,
    critic_step1_split_overlap: bool,
    critic_b1_attribution: bool,
    inventory: CallInventory,
    trajectory_writer: AppendOnlyTrajectoryWriter | None = None,
    trajectory_seed: int | None = None,
    trajectory_arm_id: str | None = None,
    trajectory_output_dir: Path | None = None,
    cycle_endpoint_records: list[dict[str, Any]] | None = None,
) -> None:
    """Run exact C24 cycles with one next-cycle rollout wave in flight."""

    trajectory_mode = run_kind == "trajectory"
    two_cycle_timing = run_kind == "two-cycle-timing"
    if trajectory_mode:
        _require(
            trace.cycles in TRAJECTORY_ALLOWED_CYCLES,
            "trajectory trace cycles must be one of 1, 2, 5, or 50",
        )
        _require(
            trajectory_writer is not None
            and trajectory_seed is not None
            and trajectory_arm_id is not None
            and trajectory_output_dir is not None,
            "trajectory evidence configuration is incomplete",
        )
    elif two_cycle_timing:
        _require(
            trace.cycles == TWO_CYCLE_TIMING_CYCLES
            and cycle_endpoint_records is not None,
            "C24 two-cycle timing trace/endpoint recorder is incomplete",
        )
    else:
        _require(trace.cycles == 3, "C21 continuous trace must contain three cycles")
    _require(
        runahead_host_transport is not None and critic_shadow_hot_sync is not None,
        "C21 continuous transports are incomplete",
    )
    is_generator = rank < 8
    lane = rank if is_generator else rank - 8
    lane_slots = validate_slot_groups(
        slot_pairs_by_lane,
        batch=trace.global_batch,
        slots_per_lane=trace.global_batch // 8,
    )[lane]
    batch32 = trace.global_batch == 32
    _require(
        not critic_b1_attribution or (trajectory_mode and not batch32),
        "C24 Critic-B1 attribution is restricted to trajectory B16",
    )
    latent_bytes = math.prod(LATENT_SHAPE) * 4
    started = time.perf_counter()
    cycle_events: list[dict[str, Any]] = []
    precomputed_first: PreparedRollout | None = None
    initial_versions = dict(versions)
    reference_call_census: dict[str, int] | None = None
    pending_application_calls: dict[str, int] = {}
    pending_logical_score_calls: dict[str, int] = {}

    for cycle in range(trace.cycles):
        cycle_started = time.perf_counter()
        if trajectory_mode:
            incoming_application_calls = pending_application_calls
            incoming_logical_score_calls = pending_logical_score_calls
            pending_application_calls = {}
            pending_logical_score_calls = {}
            loss_start = len(evidence.loss_fingerprints)
            optimizer_start = len(evidence.optimizer_diagnostics)
            inventory_before = dict(inventory.counts)
            logical_score_before = dict(evidence.logical_score_calls)
            physical_critic_backward_before = (
                evidence.physical_critic_backward_calls
            )
        with _profile_range(
            torch,
            run_kind=run_kind,
            label=f"livewan/continuous/cycle_{cycle}/rank_{rank}",
        ):
            if batch32:
                _require(
                    not critic_step1_split_overlap,
                    "B32 cannot select the rejected C21-C1 schedule",
                )
                _run_pipeline_8x8_b32(
                    torch=torch,
                    rank=rank,
                    trace=trace,
                    loader=loader,
                    bundle=bundle,
                    streamers=streamers,
                    score_context=score_context,
                    run_kind=run_kind,
                    evidence=evidence,
                    versions=versions,
                    device=device,
                    slot_groups_by_lane=slot_pairs_by_lane,
                    runahead_host_transport=runahead_host_transport,
                    step_offset=cycle * FULL5_STEPS,
                    cycle_index=cycle,
                    precomputed_first=precomputed_first,
                    critic_shadow_hot_sync=critic_shadow_hot_sync,
                )
            else:
                _run_pipeline_8x8(
                    torch=torch,
                    rank=rank,
                    trace=trace,
                    loader=loader,
                    bundle=bundle,
                    streamers=streamers,
                    score_context=score_context,
                    run_kind=run_kind,
                    evidence=evidence,
                    versions=versions,
                    device=device,
                    rollout_runahead=True,
                    critic_b2_only=not critic_b1_attribution,
                    critic_b1_accumulate=critic_b1_attribution,
                    critic_step1_split_overlap=critic_step1_split_overlap,
                    score_context_from_trace=True,
                    slot_pairs_by_lane=slot_pairs_by_lane,
                    full_context_lens_elision=full_context_lens_elision,
                    score_context_prefetch=False,
                    split_dmd_shadow=True,
                    runahead_host_transport=runahead_host_transport,
                    step_offset=cycle * FULL5_STEPS,
                    cycle_index=cycle,
                    precomputed_first=precomputed_first,
                    critic_shadow_hot_sync=critic_shadow_hot_sync,
                    finalize_host_transport=False,
                )
        role_returned = time.perf_counter()
        if two_cycle_timing:
            # The endpoint is immediately after this rank's terminal work for
            # the logical cycle.  For cycle 1 it is intentionally before the
            # next-cycle lookahead/hot-sync below, so the cycle-2 endpoint delta
            # includes all of that attributable cross-cycle work.  This is a
            # timestamp only: no midpoint synchronization or barrier is added.
            _record_two_cycle_endpoint(
                torch,
                records=cycle_endpoint_records,
                versions=versions,
                inventory=inventory,
                evidence=evidence,
            )
        if trajectory_mode:
            cycle_losses = list(evidence.loss_fingerprints[loss_start:])
            cycle_optimizer = list(evidence.optimizer_diagnostics[optimizer_start:])
            inventory_after = dict(inventory.counts)
            logical_score_after = dict(evidence.logical_score_calls)
        state_fingerprints: dict[str, str] = {}
        state_anchor = None
        cycle_number = cycle + 1
        if trajectory_mode and (
            cycle_number in TRAJECTORY_ANCHOR_CYCLES
            or cycle_number == trace.cycles
        ):
            state_fingerprints, state_anchor = _trajectory_state_anchor(
                torch=torch,
                rank=rank,
                cycle=cycle_number,
                output_dir=trajectory_output_dir,
                bundle=bundle,
            )
        event: dict[str, Any] = {
            "cycle": cycle,
            "role": "generator" if is_generator else "score",
            "cycle_local_wall_seconds": role_returned - cycle_started,
            "role_return_offset_seconds": role_returned - started,
            "generator_version": int(versions["generator"]),
            "critic_version": int(versions["critic"]),
        }
        precomputed_first = None
        if cycle + 1 < trace.cycles:
            next_global_step = (cycle + 1) * FULL5_STEPS
            if is_generator:
                # The final current-cycle run-ahead value in streamer0 must have
                # completed its D2H stage before the same persistent streamer is
                # reused for the next cycle's first logical sample.
                wait_started = time.perf_counter()
                runahead_host_transport.wait_staged(
                    step=cycle * FULL5_STEPS + 4,
                    wave=2 if batch32 else 0,
                )
                evidence.generator_backpressure.append(
                    {
                        "cycle": cycle,
                        "kind": "lookahead_streamer_reuse_wait",
                        "wall_seconds": time.perf_counter() - wait_started,
                    }
                )
                next_row = trace.row(next_global_step, lane_slots[0])
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label=(
                        "livewan/continuous/"
                        f"cycle_{cycle}/lookahead_next_cycle_wave0"
                    ),
                ):
                    precomputed_first = prepare_rollout(
                        torch,
                        row=next_row,
                        streamer=streamers[0],
                        student=bundle.student,
                        loader=loader,
                        device=device,
                        elide_full_context_lens=full_context_lens_elision,
                    )
                    _record_output(
                        torch,
                        evidence,
                        row=next_row,
                        x0=precomputed_first.x0,
                        run_kind=run_kind,
                    )
                    runahead_host_transport.enqueue_send(
                        step=next_global_step,
                        wave=0,
                        tensor=precomputed_first.x0,
                    )
                    evidence.transport_bytes += latent_bytes
                event["next_cycle_wave0_ready_offset_seconds"] = (
                    time.perf_counter() - started
                )
            else:
                expected_version = (cycle + 1) * CRITIC_UPDATES
                _require(
                    int(versions["critic"]) == expected_version,
                    "Score reached cross-cycle sync with the wrong Critic version",
                )
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label=(
                        "livewan/continuous/"
                        f"cycle_{cycle}/critic_shadow_source_C{expected_version}"
                    ),
                ):
                    critic_shadow_hot_sync.begin_source(version=expected_version)
                event["critic_shadow_sync_submit_offset_seconds"] = (
                    time.perf_counter() - started
                )
        if trajectory_mode:
            pending_application_calls = {
                key: int(value) - int(inventory_after.get(key, 0))
                for key, value in inventory.counts.items()
                if int(value) - int(inventory_after.get(key, 0))
            }
            pending_logical_score_calls = {
                key: int(value) - int(logical_score_after.get(key, 0))
                for key, value in evidence.logical_score_calls.items()
                if int(value) - int(logical_score_after.get(key, 0))
            }
        cycle_events.append(event)

        if trajectory_mode:
            local_call_census = {
                **{
                    f"application/{key}": int(value)
                    - int(inventory_before.get(key, 0))
                    + int(incoming_application_calls.get(key, 0))
                    for key, value in inventory_after.items()
                    if key.startswith("student")
                    and int(value) - int(inventory_before.get(key, 0))
                    + int(incoming_application_calls.get(key, 0))
                },
                **{
                    f"logical_score/{key}": int(value)
                    - int(logical_score_before.get(key, 0))
                    + int(incoming_logical_score_calls.get(key, 0))
                    for key, value in logical_score_after.items()
                    if int(value) - int(logical_score_before.get(key, 0))
                    + int(incoming_logical_score_calls.get(key, 0))
                },
            }
            expected_generator = int(initial_versions["generator"]) + cycle_number
            expected_critic = int(initial_versions["critic"]) + (
                CRITIC_UPDATES * cycle_number
            )
            expected_ema = int(initial_versions["ema"])
            if is_generator:
                _require(
                    int(versions["generator"]) == expected_generator
                    and int(versions["critic"]) == int(initial_versions["critic"])
                    and int(versions["ema"]) == expected_ema,
                    "Generator-pool physical commit inventory changed",
                )
            else:
                _require(
                    int(versions["generator"]) == int(initial_versions["generator"])
                    and int(versions["critic"]) == expected_critic
                    and int(versions["ema"]) == expected_ema,
                    "Score-pool physical commit inventory changed",
                )
            finite, call_census = _append_trajectory_cycle_receipt(
                torch=torch,
                rank=rank,
                cycle=cycle,
                trace=trace,
                seed=trajectory_seed,
                arm_id=trajectory_arm_id,
                execution_mode="pipeline",
                implementation={
                    "pool_topology": "8G_8SCORE",
                    "physical_role": "generator" if is_generator else "score",
                    "schedule": "C24_CONTINUOUS_B16_DEPTH1",
                    "critic_physical_batch_size": (
                        "1" if critic_b1_attribution else "2"
                    ),
                    "critic_physical_backward_calls": str(
                        evidence.physical_critic_backward_calls
                        - physical_critic_backward_before
                    ),
                    "critic_optimizer_commits": "0" if is_generator else "5",
                },
                versions=versions,
                initial_versions=initial_versions,
                local_losses=cycle_losses,
                local_optimizer=cycle_optimizer,
                local_call_census=local_call_census,
                state_fingerprints=state_fingerprints,
                writer=trajectory_writer,
            )
            if reference_call_census is None:
                reference_call_census = dict(call_census)
            else:
                _require(
                    call_census == reference_call_census,
                    "continuous trajectory changed its per-cycle call census",
                )
            event["trajectory_receipt"] = {
                "cycle": cycle_number,
                "finite": finite,
                "state_anchor": state_anchor,
            }
            if not finite:
                raise FloatingPointError(
                    f"non-finite LiveWan trajectory evidence at cycle {cycle_number}"
                )

    if trajectory_mode:
        _require(
            not pending_application_calls and not pending_logical_score_calls,
            "continuous execution left an unattributed lookahead call",
        )
    runahead_host_transport.finish()
    evidence.generator_runahead_transport = runahead_host_transport.receipt()
    critic_shadow_hot_sync.finish()
    evidence.critic_shadow_hot_sync = dict(critic_shadow_hot_sync.receipt())
    evidence.continuous_pipeline = {
        "enabled": True,
        "name": (
            TRAJECTORY_CONTINUOUS_NAME
            if trajectory_mode
            else (
                TWO_CYCLE_TIMING_CONTINUOUS_NAME
                if two_cycle_timing
                else (
                    B32_CONTINUOUS_NAME
                    if batch32
                    else (
                        "continuous_full5_b16_depth1_step1_b1_overlap_v1"
                        if critic_step1_split_overlap
                        else "continuous_full5_b16_depth1_v1"
                    )
                )
            )
        ),
        "cycles": trace.cycles,
        "cycle_length": FULL5_STEPS,
        "lookahead_depth": 1,
        "lookahead_scope": "next_cycle_step0_wave0_rollout_only",
        "critic_step1_physical_schedule": (
            "two_b2_backward_accumulations_single_commit"
            if batch32
            else (
                "wave0_b1_backward_then_wave1_b1_backward_single_commit"
                if critic_step1_split_overlap
                else (
                    "two_b1_backwards_accumulated_single_commit_every_step"
                    if critic_b1_attribution
                    else "single_b2_backward"
                )
            )
        ),
        "critic_physical_batch_size": 1 if critic_b1_attribution else 2,
        "critic_physical_backward_calls_per_step": (
            2 if critic_b1_attribution else 1
        ),
        "global_batch": trace.global_batch,
        "waves_per_step": len(lane_slots),
        "source_reuse_within_step": batch32,
        "training_quality_claimable": not batch32,
        "critic_step1_single_commit": True,
        "source_reuse_across_cycles": trace.source_reuse_across_cycles,
        "events": cycle_events,
        "role_total_wall_seconds": time.perf_counter() - started,
        "execution_status": "PASS",
    }


def _run_pipeline_phase_expanded_critic16(
    *,
    torch: Any,
    rank: int,
    trace: FormalTrace,
    loader: InputLoader,
    bundle: ModelBundle,
    streamers: Sequence[Any],
    score_context: ScoreContext | None,
    run_kind: str,
    evidence: RunEvidence,
    versions: dict[str, int],
    device: Any,
    slot_pairs_by_lane: Sequence[Sequence[int]],
) -> None:
    """8G+8Score DMD followed by five exact world16 Critic-B1 updates.

    Each fixed lane retains one of its two rollout latents on the Generator
    rank and sends the other to the paired Score rank.  Step 0 still sends both
    latents to Score for the unchanged Teacher/Fake DMD calls.  The Critic is a
    persistent identical replica on all sixteen ranks, so each update sees one
    logical B1 sample per rank and performs one world16 SUM-then-divide commit.
    """

    import torch.distributed as dist

    is_generator = rank < 8
    peer = rank + 8 if is_generator else rank - 8
    lane = rank if is_generator else peer
    lane_slots = validate_slot_pairs(slot_pairs_by_lane)[lane]
    remote_slot, generator_local_slot = lane_slots
    latent_bytes = math.prod(LATENT_SHAPE) * 4
    _require(
        bundle.critic is not None
        and bundle.critic_optimizer is not None
        and bundle.critic_gradient_reducer is None
        and bundle.critic_group_size == WORLD_SIZE
        and score_context is not None,
        "phase-expanded world16 Critic role is incomplete",
    )
    _require(
        bundle.metadata["critic"]["placement"]
        == "world16_fsdp2_keep_unsharded"
        and bundle.metadata["critic"]["explicit_pre_dmd_unshard_units"]
        == bundle.metadata["critic"]["fsdp_group_count"],
        "phase-expanded Critic is not fully materialized before asymmetric DMD",
    )
    if is_generator:
        _require(
            bundle.student is not None
            and bundle.generator_optimizer is not None
            and len(streamers) == 2,
            "phase-expanded Generator role is incomplete",
        )
        _require(
            bundle.metadata["student"]["placement"] == "gpu_resident_replica",
            "phase-expanded Student is not a resident replica",
        )
    else:
        _require(bundle.teacher is not None, "phase-expanded Score rank lacks Teacher")

    for step in range(FULL5_STEPS):
        rows = tuple(trace.row(step, slot) for slot in lane_slots)
        remote_row, generator_local_row = rows
        bundle.critic_optimizer.zero_grad(set_to_none=True)
        deferred_gradient_work = None
        deferred_gradient_tensor = None

        if is_generator:
            if step == 0:
                bundle.generator_optimizer.zero_grad(set_to_none=True)
            with _profile_range(
                torch,
                run_kind=run_kind,
                label=f"livewan/phase_expanded/g/step_{step}/rollout_remote",
            ):
                remote_prepared = prepare_rollout(
                    torch,
                    row=remote_row,
                    streamer=streamers[0],
                    student=bundle.student,
                    loader=loader,
                    device=device,
                    elide_full_context_lens=True,
                )
            _record_output(
                torch,
                evidence,
                row=remote_row,
                x0=remote_prepared.x0,
                run_kind=run_kind,
            )
            with _profile_range(
                torch,
                run_kind=run_kind,
                label=f"livewan/phase_expanded/g/step_{step}/send_remote",
            ):
                remote_send = dist.isend(remote_prepared.x0.contiguous(), dst=peer)
            evidence.transport_bytes += latent_bytes

            with _profile_range(
                torch,
                run_kind=run_kind,
                label=f"livewan/phase_expanded/g/step_{step}/rollout_local",
            ):
                local_prepared = prepare_rollout(
                    torch,
                    row=generator_local_row,
                    streamer=streamers[1],
                    student=bundle.student,
                    loader=loader,
                    device=device,
                    elide_full_context_lens=True,
                )
            _record_output(
                torch,
                evidence,
                row=generator_local_row,
                x0=local_prepared.x0,
                run_kind=run_kind,
            )
            remote_send.wait()

            if step == 0:
                grad_remote = torch.empty_like(remote_prepared.x0)
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label="livewan/phase_expanded/g/step_0/recv_gradient_remote",
                ):
                    dist.recv(grad_remote, src=peer)
                evidence.transport_bytes += grad_remote.numel() * grad_remote.element_size()
                local_send = dist.isend(local_prepared.x0.contiguous(), dst=peer)
                evidence.transport_bytes += latent_bytes
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label="livewan/phase_expanded/g/step_0/backward_remote",
                ):
                    remote_losses = generator_backward_for_sample(
                        torch,
                        prepared=remote_prepared,
                        dmd_gradient=grad_remote,
                        sample_weight=0.5,
                    )
                local_send.wait()
                grad_local = torch.empty_like(local_prepared.x0)
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label="livewan/phase_expanded/g/step_0/recv_gradient_local",
                ):
                    dist.recv(grad_local, src=peer)
                evidence.transport_bytes += grad_local.numel() * grad_local.element_size()
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label="livewan/phase_expanded/g/step_0/backward_local",
                ):
                    local_losses = generator_backward_for_sample(
                        torch,
                        prepared=local_prepared,
                        dmd_gradient=grad_local,
                        sample_weight=0.5,
                    )
                if run_kind == "correctness":
                    for row, losses in zip(
                        rows, (remote_losses, local_losses)
                    ):
                        evidence.loss_fingerprints.extend(
                            {
                                "operation_key": row.operation_key,
                                "kind": "generator",
                                **_tensor_fingerprint(torch, value),
                            }
                            for value in losses
                        )
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label="livewan/phase_expanded/g/step_0/generator_commit",
                ):
                    generator_parameters = list(bundle.student.parameters())
                    generator_sync = _mean_replicated_gradients(
                        torch,
                        generator_parameters,
                        group=bundle.generator_group,
                        group_size=bundle.generator_group_size,
                        bucket_mib=128,
                    )
                    generator_pre_clip = (
                        _state_metrics(
                            torch,
                            [
                                parameter.grad
                                for parameter in generator_parameters
                                if parameter.grad is not None
                            ],
                        )
                        if run_kind == "correctness"
                        else None
                    )
                    generator_clip_norm = torch.nn.utils.clip_grad_norm_(
                        generator_parameters, 1.0
                    )
                    _record_optimizer_diagnostics(
                        torch,
                        evidence,
                        run_kind=run_kind,
                        role="generator",
                        step=step,
                        parameters=generator_parameters,
                        clip_norm=generator_clip_norm,
                        gradient_before_clip=generator_pre_clip,
                    )
                    bundle.generator_optimizer.step()
                    evidence.gradient_sync.append(
                        {"step": step, "role": "generator", **generator_sync}
                    )
                    versions["generator"] += 1

            critic_row = generator_local_row
            critic_x0 = local_prepared.x0
            critic_raw_context = loader.context_from_trace(critic_row, device=device)
            critic_sample_context = prepare_score_sample_context(
                torch,
                bundle=bundle,
                raw_context=critic_raw_context,
                include_teacher=False,
            )
        else:
            with _profile_range(
                torch,
                run_kind=run_kind,
                label=f"livewan/phase_expanded/score/step_{step}/recv_remote",
            ):
                remote_x0 = torch.empty(
                    LATENT_SHAPE, device=device, dtype=torch.float32
                )
                dist.recv(remote_x0, src=peer)
            evidence.transport_bytes += latent_bytes
            remote_context = loader.context_from_trace(remote_row, device=device)
            remote_sample_context = prepare_score_sample_context(
                torch,
                bundle=bundle,
                raw_context=remote_context,
                include_teacher=step == 0,
            )
            if step == 0:
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label="livewan/phase_expanded/score/step_0/dmd_remote",
                ):
                    grad_remote = dmd_gradient_for_sample(
                        torch,
                        bundle=bundle,
                        score_context=score_context,
                        row=remote_row,
                        x0=remote_x0,
                        sample_context=remote_sample_context,
                        context_lens=None,
                    )
                    evidence.add_logical_score_calls("teacher/nograd/plain", 1)
                    evidence.add_logical_score_calls("critic/nograd/plain", 1)
                    dist.send(grad_remote.contiguous(), dst=peer)
                evidence.transport_bytes += grad_remote.numel() * grad_remote.element_size()
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label="livewan/phase_expanded/score/step_0/recv_local_for_dmd",
                ):
                    local_x0 = torch.empty(
                        LATENT_SHAPE, device=device, dtype=torch.float32
                    )
                    dist.recv(local_x0, src=peer)
                evidence.transport_bytes += latent_bytes
                local_context = loader.context_from_trace(
                    generator_local_row, device=device
                )
                local_sample_context = prepare_score_sample_context(
                    torch,
                    bundle=bundle,
                    raw_context=local_context,
                    include_teacher=True,
                )
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label="livewan/phase_expanded/score/step_0/dmd_local",
                ):
                    grad_local = dmd_gradient_for_sample(
                        torch,
                        bundle=bundle,
                        score_context=score_context,
                        row=generator_local_row,
                        x0=local_x0,
                        sample_context=local_sample_context,
                        context_lens=None,
                    )
                    evidence.add_logical_score_calls("teacher/nograd/plain", 1)
                    evidence.add_logical_score_calls("critic/nograd/plain", 1)
                    deferred_gradient_tensor = grad_local.contiguous()
                    deferred_gradient_work = dist.isend(
                        deferred_gradient_tensor, dst=peer
                    )
                evidence.transport_bytes += grad_local.numel() * grad_local.element_size()

            critic_row = remote_row
            critic_x0 = remote_x0
            critic_sample_context = ScoreSampleContext(
                teacher=None,
                critic=remote_sample_context.critic,
            )

        with _profile_range(
            torch,
            run_kind=run_kind,
            label=f"livewan/phase_expanded/critic/step_{step}/backward_b1",
        ):
            critic_loss = critic_backward_for_sample(
                torch,
                bundle=bundle,
                score_context=score_context,
                row=critic_row,
                x0=critic_x0,
                sample_context=critic_sample_context,
                context_lens=None,
                sample_weight=1.0,
            )
            evidence.add_logical_score_calls("critic/grad/checkpoint", 1)
        if deferred_gradient_work is not None:
            deferred_gradient_work.wait()
            del deferred_gradient_tensor
        if run_kind == "correctness":
            evidence.loss_fingerprints.append(
                {
                    "operation_key": critic_row.operation_key,
                    "kind": "critic",
                    **_tensor_fingerprint(torch, critic_loss),
                }
            )
        with _profile_range(
            torch,
            run_kind=run_kind,
            label=f"livewan/phase_expanded/critic/step_{step}/commit",
        ):
            evidence.gradient_sync.append(
                {
                    "step": step,
                    "role": "critic",
                    "backend": "FSDP2_REDUCE_SCATTER",
                    "group_size": WORLD_SIZE,
                    "reduction": "SUM_THEN_DIVIDE_GROUP_SIZE",
                    "overlap_with_backward": True,
                }
            )
            critic_pre_clip = (
                _state_metrics(
                    torch,
                    [
                        parameter.grad
                        for parameter in bundle.critic_lora_params
                        if parameter.grad is not None
                    ],
                )
                if run_kind == "correctness"
                else None
            )
            critic_clip_norm = torch.nn.utils.clip_grad_norm_(
                bundle.critic_lora_params, 1.0, foreach=False
            )
            _record_optimizer_diagnostics(
                torch,
                evidence,
                run_kind=run_kind,
                role="critic",
                step=step,
                parameters=bundle.critic_lora_params,
                clip_norm=critic_clip_norm,
                gradient_before_clip=critic_pre_clip,
            )
            bundle.critic_optimizer.step()
            versions["critic"] += 1


def _run_pipeline_8x8_score_b2(
    *,
    torch: Any,
    rank: int,
    trace: FormalTrace,
    loader: InputLoader,
    bundle: ModelBundle,
    streamers: Sequence[Any],
    score_context: ScoreContext | None,
    run_kind: str,
    evidence: RunEvidence,
    versions: dict[str, int],
    device: Any,
) -> None:
    """Persistent 8G+8Score with two logical B1 rows per physical Score call.

    The Generator still produces and records two independent trace-qualified B1
    samples.  Only the Score-side model launch granularity changes: Teacher CFG
    is one physical B4 call, Fake Score is one B2 call, and Critic training is
    one B2 call whose loss is the mean of the two original per-sample losses.
    """

    import torch.distributed as dist

    is_generator = rank < 8
    peer = rank + 8 if is_generator else rank - 8
    pair_bytes = (math.prod(LATENT_SHAPE) + math.prod(TEXT_SHAPE)) * 4
    if is_generator:
        _require(bundle.student is not None and len(streamers) == 2, "pipeline G role is incomplete")
    else:
        _require(
            bundle.critic is not None and bundle.teacher is not None and score_context is not None,
            "pipeline Score role is incomplete",
        )

    for step in range(FULL5_STEPS):
        if is_generator:
            bundle.generator_optimizer.zero_grad(set_to_none=True)
            rows = (trace.row(step, rank), trace.row(step, rank + 8))
            with _profile_range(
                torch,
                run_kind=run_kind,
                label=f"livewan/pipeline_score_b2/g/step_{step}/rollout_wave_0",
            ):
                first = prepare_rollout(
                    torch,
                    row=rows[0],
                    streamer=streamers[0],
                    student=bundle.student,
                    loader=loader,
                    device=device,
                )
            _record_output(torch, evidence, row=rows[0], x0=first.x0, run_kind=run_kind)
            with _profile_range(
                torch,
                run_kind=run_kind,
                label=f"livewan/pipeline_score_b2/g/step_{step}/send_wave_0",
            ):
                _send_pair(destination=peer, prepared=first, asynchronous=False)
            evidence.transport_bytes += pair_bytes

            with _profile_range(
                torch,
                run_kind=run_kind,
                label=f"livewan/pipeline_score_b2/g/step_{step}/rollout_wave_1",
            ):
                second = prepare_rollout(
                    torch,
                    row=rows[1],
                    streamer=streamers[1],
                    student=bundle.student,
                    loader=loader,
                    device=device,
                )
            _record_output(torch, evidence, row=rows[1], x0=second.x0, run_kind=run_kind)
            with _profile_range(
                torch,
                run_kind=run_kind,
                label=f"livewan/pipeline_score_b2/g/step_{step}/send_wave_1",
            ):
                _send_pair(destination=peer, prepared=second, asynchronous=False)
            evidence.transport_bytes += pair_bytes

            if step == 0:
                gradient_batch = torch.empty(
                    (2, *LATENT_SHAPE[1:]), device=device, dtype=torch.float32
                )
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label="livewan/pipeline_score_b2/g/step_0/recv_gradient_b2",
                ):
                    dist.recv(gradient_batch, src=peer)
                evidence.transport_bytes += gradient_batch.numel() * gradient_batch.element_size()
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label="livewan/pipeline_score_b2/g/step_0/backward_wave_0",
                ):
                    losses0 = generator_backward_for_sample(
                        torch,
                        prepared=first,
                        dmd_gradient=gradient_batch[0:1],
                        sample_weight=0.5,
                    )
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label="livewan/pipeline_score_b2/g/step_0/backward_wave_1",
                ):
                    losses1 = generator_backward_for_sample(
                        torch,
                        prepared=second,
                        dmd_gradient=gradient_batch[1:2],
                        sample_weight=0.5,
                    )
                if run_kind == "correctness":
                    for row, losses in zip(rows, (losses0, losses1)):
                        evidence.loss_fingerprints.extend(
                            {
                                "operation_key": row.operation_key,
                                "kind": "generator",
                                **_tensor_fingerprint(torch, value),
                            }
                            for value in losses
                        )
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label="livewan/pipeline_score_b2/g/step_0/generator_commit",
                ):
                    sync = _mean_replicated_gradients(
                        torch,
                        list(bundle.student.parameters()),
                        group=bundle.generator_group,
                        group_size=bundle.generator_group_size,
                        bucket_mib=256,
                    )
                    evidence.gradient_sync.append({"step": step, "role": "generator", **sync})
                    generator_parameters = list(bundle.student.parameters())
                    generator_pre_clip = (
                        _state_metrics(torch, [parameter.grad for parameter in generator_parameters])
                        if run_kind == "correctness"
                        else None
                    )
                    generator_clip_norm = torch.nn.utils.clip_grad_norm_(
                        generator_parameters, 1.0, foreach=True
                    )
                    _record_optimizer_diagnostics(
                        torch,
                        evidence,
                        run_kind=run_kind,
                        role="generator",
                        step=step,
                        parameters=generator_parameters,
                        clip_norm=generator_clip_norm,
                        gradient_before_clip=generator_pre_clip,
                    )
                    bundle.generator_optimizer.step()
                    versions["generator"] += 1
        else:
            bundle.critic_optimizer.zero_grad(set_to_none=True)
            rows = (trace.row(step, peer), trace.row(step, peer + 8))
            with _profile_range(
                torch,
                run_kind=run_kind,
                label=f"livewan/pipeline_score_b2/score/step_{step}/recv_wave_0",
            ):
                first_x0, first_context = _recv_pair(torch, source=peer, device=device)
            evidence.transport_bytes += pair_bytes
            with _profile_range(
                torch,
                run_kind=run_kind,
                label=f"livewan/pipeline_score_b2/score/step_{step}/recv_wave_1",
            ):
                second_x0, second_context = _recv_pair(torch, source=peer, device=device)
            evidence.transport_bytes += pair_bytes

            x0_batch = torch.cat([first_x0, second_x0], dim=0)
            raw_context_batch = torch.cat([first_context, second_context], dim=0)
            context_lens = torch.full(
                (2,), TEXT_SHAPE[1], device=device, dtype=torch.long
            )
            sample_context = prepare_score_sample_context(
                torch,
                bundle=bundle,
                raw_context=raw_context_batch,
                include_teacher=step == 0,
            )
            if step == 0:
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label="livewan/pipeline_score_b2/score/step_0/dmd_b2",
                ):
                    gradient_batch = dmd_gradient_for_batch(
                        torch,
                        bundle=bundle,
                        score_context=score_context,
                        rows=rows,
                        x0=x0_batch,
                        sample_context=sample_context,
                        context_lens=context_lens,
                    )
                    evidence.add_logical_score_calls("teacher/nograd/plain", 2)
                    evidence.add_logical_score_calls("critic/nograd/plain", 2)
                    dist.send(gradient_batch.contiguous(), dst=peer)
                evidence.transport_bytes += gradient_batch.numel() * gradient_batch.element_size()

            with _profile_range(
                torch,
                run_kind=run_kind,
                label=f"livewan/pipeline_score_b2/score/step_{step}/critic_backward_b2",
            ):
                losses = critic_backward_for_batch(
                    torch,
                    bundle=bundle,
                    score_context=score_context,
                    rows=rows,
                    x0=x0_batch,
                    sample_context=sample_context,
                    context_lens=context_lens,
                )
                evidence.add_logical_score_calls("critic/grad/checkpoint", 2)
            if run_kind == "correctness":
                for row, loss in zip(rows, losses):
                    evidence.loss_fingerprints.append(
                        {
                            "operation_key": row.operation_key,
                            "kind": "critic",
                            **_tensor_fingerprint(torch, loss),
                        }
                    )
            with _profile_range(
                torch,
                run_kind=run_kind,
                label=f"livewan/pipeline_score_b2/score/step_{step}/critic_commit",
            ):
                sync = _mean_replicated_gradients(
                    torch,
                    bundle.critic_lora_params,
                    group=bundle.score_group,
                    group_size=bundle.score_group_size,
                    bucket_mib=128,
                )
                evidence.gradient_sync.append({"step": step, "role": "critic", **sync})
                critic_pre_clip = (
                    _state_metrics(torch, [parameter.grad for parameter in bundle.critic_lora_params])
                    if run_kind == "correctness"
                    else None
                )
                critic_clip_norm = torch.nn.utils.clip_grad_norm_(
                    bundle.critic_lora_params, 1.0, foreach=True
                )
                _record_optimizer_diagnostics(
                    torch,
                    evidence,
                    run_kind=run_kind,
                    role="critic",
                    step=step,
                    parameters=bundle.critic_lora_params,
                    clip_norm=critic_clip_norm,
                    gradient_before_clip=critic_pre_clip,
                )
                bundle.critic_optimizer.step()
                versions["critic"] += 1


def _run_pipeline_6g10score(
    *,
    torch: Any,
    rank: int,
    trace: FormalTrace,
    loader: InputLoader,
    bundle: ModelBundle,
    streamers: Sequence[Any],
    score_context: ScoreContext | None,
    run_kind: str,
    evidence: RunEvidence,
    versions: dict[str, int],
    device: Any,
) -> None:
    """Persistent 6G+10Score exact global-B16 role pipeline.

    Logical slots are routed independently to both pools. Generator ranks own
    either two or three real B1 samples; Score ranks own one or two. Every Score
    rank executes exactly one physical model call per phase, so its FSDP10
    Teacher collective sequence remains identical even though local batch sizes
    differ. Loss scales compensate for the replicated-group gradient mean and
    recover the exact global sample mean without dummy work.
    """

    generator_ranks = PIPELINE_6G10S_LAYOUT.generator_ranks
    score_ranks = PIPELINE_6G10S_LAYOUT.score_ranks
    generator_slots = balanced_slots(generator_ranks).get(rank, ())
    score_slots = balanced_slots(score_ranks).get(rank, ())
    is_generator = rank in generator_ranks
    is_score = rank in score_ranks
    _require(is_generator != is_score, "6G+10Score role ownership is not exclusive")
    if is_generator:
        _require(
            bundle.student is not None and len(streamers) == len(generator_slots),
            "6G+10Score Generator role is incomplete",
        )
    else:
        _require(
            bundle.critic is not None
            and bundle.teacher is not None
            and score_context is not None,
            "6G+10Score role is incomplete",
        )

    pair_bytes = (math.prod(LATENT_SHAPE) + math.prod(TEXT_SHAPE)) * 4
    gradient_bytes = math.prod(LATENT_SHAPE) * 4
    for step in range(FULL5_STEPS):
        prepared_by_slot: dict[int, PreparedRollout] = {}
        if is_generator:
            bundle.generator_optimizer.zero_grad(set_to_none=True)
            for local_index, slot in enumerate(generator_slots):
                row = trace.row(step, slot)
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label=f"livewan/pipeline_6g10/g/step_{step}/rollout_slot_{slot}",
                ):
                    prepared = prepare_rollout(
                        torch,
                        row=row,
                        streamer=streamers[local_index],
                        student=bundle.student,
                        loader=loader,
                        device=device,
                    )
                prepared_by_slot[slot] = prepared
                _record_output(
                    torch, evidence, row=row, x0=prepared.x0, run_kind=run_kind
                )
        else:
            bundle.critic_optimizer.zero_grad(set_to_none=True)

        with _profile_range(
            torch,
            run_kind=run_kind,
            label=f"livewan/pipeline_6g10/step_{step}/exchange_inputs",
        ):
            received_by_slot = _exchange_routed_inputs(
                torch,
                rank=rank,
                device=device,
                generator_ranks=generator_ranks,
                score_ranks=score_ranks,
                prepared_by_slot=prepared_by_slot,
            )
        evidence.transport_bytes += pair_bytes * (
            len(generator_slots) if is_generator else len(score_slots)
        )

        rows: tuple[OperationTraceRow, ...] = ()
        x0_batch = raw_context_batch = context_lens = sample_context = None
        gradients_by_slot: dict[int, Any] = {}
        if is_score:
            rows = tuple(trace.row(step, slot) for slot in score_slots)
            x0_batch = torch.cat(
                [received_by_slot[slot][0] for slot in score_slots], dim=0
            )
            raw_context_batch = torch.cat(
                [received_by_slot[slot][1] for slot in score_slots], dim=0
            )
            context_lens = torch.full(
                (len(score_slots),),
                TEXT_SHAPE[1],
                device=device,
                dtype=torch.long,
            )
            sample_context = prepare_score_sample_context(
                torch,
                bundle=bundle,
                raw_context=raw_context_batch,
                include_teacher=step == 0,
            )
            if step == 0:
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label=(
                        "livewan/pipeline_6g10/score/step_0/"
                        f"dmd_b{len(score_slots)}"
                    ),
                ):
                    gradient_batch = dmd_gradient_for_batch(
                        torch,
                        bundle=bundle,
                        score_context=score_context,
                        rows=rows,
                        x0=x0_batch,
                        sample_context=sample_context,
                        context_lens=context_lens,
                    )
                gradients_by_slot = {
                    slot: gradient_batch[index : index + 1]
                    for index, slot in enumerate(score_slots)
                }
                evidence.add_logical_score_calls(
                    "teacher/nograd/plain", len(score_slots)
                )
                evidence.add_logical_score_calls(
                    "critic/nograd/plain", len(score_slots)
                )

        if step == 0:
            with _profile_range(
                torch,
                run_kind=run_kind,
                label="livewan/pipeline_6g10/step_0/exchange_gradients",
            ):
                received_gradients = _exchange_routed_gradients(
                    torch,
                    rank=rank,
                    device=device,
                    generator_ranks=generator_ranks,
                    score_ranks=score_ranks,
                    gradients_by_slot=gradients_by_slot,
                )
            evidence.transport_bytes += gradient_bytes * (
                len(generator_slots) if is_generator else len(score_slots)
            )
        else:
            received_gradients = {}

        # After the step-0 gradient exchange, Generator backward and Critic
        # backward are independent and execute concurrently on disjoint pools.
        if is_generator and step == 0:
            for slot in generator_slots:
                prepared = prepared_by_slot[slot]
                with _profile_range(
                    torch,
                    run_kind=run_kind,
                    label=f"livewan/pipeline_6g10/g/step_0/backward_slot_{slot}",
                ):
                    losses = generator_backward_for_sample(
                        torch,
                        prepared=prepared,
                        dmd_gradient=received_gradients[slot],
                        sample_weight=len(generator_ranks) / GLOBAL_BATCH,
                    )
                if run_kind == "correctness":
                    evidence.loss_fingerprints.extend(
                        {
                            "operation_key": prepared.row.operation_key,
                            "kind": "generator",
                            **_tensor_fingerprint(torch, value),
                        }
                        for value in losses
                    )
            with _profile_range(
                torch,
                run_kind=run_kind,
                label="livewan/pipeline_6g10/g/step_0/generator_commit",
            ):
                generator_parameters = list(bundle.student.parameters())
                sync = _mean_replicated_gradients(
                    torch,
                    generator_parameters,
                    group=bundle.generator_group,
                    group_size=bundle.generator_group_size,
                    bucket_mib=256,
                )
                evidence.gradient_sync.append(
                    {"step": step, "role": "generator", **sync}
                )
                generator_pre_clip = (
                    _state_metrics(
                        torch,
                        [
                            parameter.grad
                            for parameter in generator_parameters
                            if parameter.grad is not None
                        ],
                    )
                    if run_kind == "correctness"
                    else None
                )
                generator_clip_norm = torch.nn.utils.clip_grad_norm_(
                    generator_parameters, 1.0, foreach=True
                )
                _record_optimizer_diagnostics(
                    torch,
                    evidence,
                    run_kind=run_kind,
                    role="generator",
                    step=step,
                    parameters=generator_parameters,
                    clip_norm=generator_clip_norm,
                    gradient_before_clip=generator_pre_clip,
                )
                bundle.generator_optimizer.step()
                versions["generator"] += 1

        if is_score:
            local_batch = len(score_slots)
            with _profile_range(
                torch,
                run_kind=run_kind,
                label=(
                    f"livewan/pipeline_6g10/score/step_{step}/"
                    f"critic_backward_b{local_batch}"
                ),
            ):
                losses = critic_backward_for_batch(
                    torch,
                    bundle=bundle,
                    score_context=score_context,
                    rows=rows,
                    x0=x0_batch,
                    sample_context=sample_context,
                    context_lens=context_lens,
                    loss_scale=(len(score_ranks) * local_batch / GLOBAL_BATCH),
                )
                evidence.add_logical_score_calls(
                    "critic/grad/checkpoint", local_batch
                )
            if run_kind == "correctness":
                for row, loss in zip(rows, losses):
                    evidence.loss_fingerprints.append(
                        {
                            "operation_key": row.operation_key,
                            "kind": "critic",
                            **_tensor_fingerprint(torch, loss),
                        }
                    )
            with _profile_range(
                torch,
                run_kind=run_kind,
                label=f"livewan/pipeline_6g10/score/step_{step}/critic_commit",
            ):
                if bundle.score_local_group is not None:
                    sync = _mean_replicated_gradients_hierarchical(
                        torch,
                        bundle.critic_lora_params,
                        local_group=bundle.score_local_group,
                        local_ranks=bundle.score_local_ranks,
                        leader_group=bundle.score_leader_group,
                        leader_ranks=bundle.score_leader_ranks,
                        rank=rank,
                        group_size=bundle.score_group_size,
                        bucket_mib=128,
                    )
                else:
                    sync = _mean_replicated_gradients(
                        torch,
                        bundle.critic_lora_params,
                        group=bundle.score_group,
                        group_size=bundle.score_group_size,
                        bucket_mib=128,
                    )
                evidence.gradient_sync.append(
                    {"step": step, "role": "critic", **sync}
                )
                critic_pre_clip = (
                    _state_metrics(
                        torch,
                        [
                            parameter.grad
                            for parameter in bundle.critic_lora_params
                            if parameter.grad is not None
                        ],
                    )
                    if run_kind == "correctness"
                    else None
                )
                critic_clip_norm = torch.nn.utils.clip_grad_norm_(
                    bundle.critic_lora_params, 1.0, foreach=True
                )
                _record_optimizer_diagnostics(
                    torch,
                    evidence,
                    run_kind=run_kind,
                    role="critic",
                    step=step,
                    parameters=bundle.critic_lora_params,
                    clip_norm=critic_clip_norm,
                    gradient_before_clip=critic_pre_clip,
                )
                bundle.critic_optimizer.step()
                versions["critic"] += 1


__all__ = [
    "Bootstrap",
    "CallInventory",
    "FormalTrace",
    "OperationTraceRow",
    "run_worker",
]
