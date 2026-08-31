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
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

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


def _pipeline_8x8_slot_pairs(
    plan: Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    """Resolve the registered physical lane assignment without touching trace rows."""

    policy = str(plan.get("metadata", {}).get("pool_policy", ""))
    if policy == TRACE_BALANCED_POOL_POLICY:
        pairs = validate_slot_pairs(
            plan.get("metadata", {}).get("slot_pairs_by_lane", ())
        )
        _require(
            pairs == TRACE_BALANCED_8G8S_SLOT_PAIRS,
            "trace-balanced slot-pair policy differs from its preregistration",
        )
        return pairs
    return tuple((lane, lane + 8) for lane in range(8))


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
    def from_mapping(cls, value: Mapping[str, Any]) -> "OperationTraceRow":
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
        _require(row.operation_key == f"step={row.logical_step}/slot={row.logical_slot}", "operation key mismatch")
        _require(row.world_frames in WORLD_CHOICES, "world choice is outside source contract")
        _require(0 <= row.logical_step < FULL5_STEPS, "logical step outside Full5")
        _require(0 <= row.logical_slot < GLOBAL_BATCH, "logical slot outside global batch")
        _require(row.generator_update == (row.logical_step == 0), "G1/C5 trace predicate changed")
        _require(row.critic_update, "every Full5 step requires a Critic update")
        return row

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

    @classmethod
    def load(cls, path: str | Path) -> "FormalTrace":
        resolved = Path(path).expanduser().resolve()
        value = json.loads(resolved.read_text(encoding="utf-8"))
        _require(value.get("livewan_commit") == LIVEWAN_COMMIT, "trace LiveWan revision mismatch")
        _require(value.get("wan21_commit") == WAN21_COMMIT, "trace Wan2.1 revision mismatch")
        _require(int(value.get("world_size", -1)) == WORLD_SIZE, "trace is not world16")
        _require(int(value.get("steps", -1)) == FULL5_STEPS, "trace is not Full5")
        recorded = str(value.get("trace_sha256", ""))
        actual = hashlib.sha256(_canonical_without_sha(value, "trace_sha256")).hexdigest()
        _require(recorded == actual, "trace payload SHA mismatch")
        rows = tuple(OperationTraceRow.from_mapping(row) for row in value["rows"])
        pairs = {(row.logical_step, row.logical_slot) for row in rows}
        expected = {(step, slot) for step in range(FULL5_STEPS) for slot in range(GLOBAL_BATCH)}
        _require(len(rows) == FULL5_STEPS * GLOBAL_BATCH and pairs == expected, "trace is not a 5x16 bijection")
        prompts = dict(value["prompts"])
        prompts_path = Path(prompts["path"]).expanduser().resolve()
        _require(prompts_path.is_file(), "trace prompt bundle is missing")
        _require(sha256_file(prompts_path) == prompts["sha256"], "prompt bundle SHA mismatch")
        return cls(resolved, actual, prompts_path, str(prompts["sha256"]), rows)

    def row(self, step: int, slot: int) -> OperationTraceRow:
        return self.rows[step * GLOBAL_BATCH + slot]


def _validate_trace_balanced_pairing(
    plan: Mapping[str, Any], trace: FormalTrace
) -> None:
    """Fail closed if the registered routing no longer balances this trace."""

    if str(plan.get("metadata", {}).get("pool_policy", "")) != TRACE_BALANCED_POOL_POLICY:
        return
    pairs = _pipeline_8x8_slot_pairs(plan)
    current = tuple((lane, lane + 8) for lane in range(8))

    def step_max(candidate: Sequence[Sequence[int]]) -> list[int]:
        return [
            max(
                sum(trace.row(step, slot).new_blocks for slot in pair)
                for pair in candidate
            )
            for step in range(FULL5_STEPS)
        ]

    current_max = step_max(current)
    candidate_max = step_max(pairs)
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
    teacher: Any | None
    generator_optimizer: Any | None
    critic_optimizer: Any | None
    critic_lora_params: list[Any]
    critic_gradient_reducer: GradReadyAsyncGroupReducer | None
    generator_group: Any | None
    score_group: Any | None
    score_local_group: Any | None
    score_leader_group: Any | None
    score_local_ranks: tuple[int, ...]
    score_leader_ranks: tuple[int, ...]
    generator_group_size: int
    score_group_size: int
    metadata: dict[str, Any]


def _make_mesh(torch: Any, ranks: Sequence[int], group: Any) -> Any:
    from torch.distributed.device_mesh import DeviceMesh
    try:
        return DeviceMesh.from_group(group, "cuda", mesh_dim_names=("fsdp",))
    except (AttributeError, TypeError):
        return DeviceMesh("cuda", list(ranks), mesh_dim_names=("fsdp",))


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
        "persistent_8g_8score_teacher_fsdp8_score_b2": PIPELINE_8G8S_LAYOUT,
        "persistent_6g_10score_teacher_fsdp10": PIPELINE_6G10S_LAYOUT,
        "persistent_6g_10score_teacher_node_local_hierarchical": PIPELINE_6G10S_NODE_LOCAL_LAYOUT,
    }.get(policy)
    _require(expected is not None, "unknown pipeline pool policy")
    _require(generator_ranks == expected.generator_ranks, "Generator rank policy mismatch")
    _require(score_ranks == expected.score_ranks, "Score rank policy mismatch")
    return generator_ranks, score_ranks


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
    # Every rank calls new_group in the same global order.
    generator_group = dist.new_group(ranks=list(generator_ranks))
    score_group = dist.new_group(ranks=list(score_ranks))
    pool_policy = str(plan.get("metadata", {}).get("pool_policy", ""))
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
    workload = dict(plan["workload"])
    metadata: dict[str, Any] = {
        "generator_ranks": list(generator_ranks),
        "score_ranks": list(score_ranks),
        "student": None,
        "critic": None,
        "teacher": None,
    }
    student = critic = teacher = None
    opt_g = opt_c = None
    lora_params: list[Any] = []
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
        opt_g = torch.optim.AdamW(
            list(student.parameters()), lr=2e-6, betas=(0.0, 0.99),
            weight_decay=0.0, eps=1e-8, **optimizer_kwargs,
        )
        metadata["student"] = {
            "placement": "world16_fsdp2_keep_unsharded" if mode == "baseline" else "gpu_resident_replica",
            "state": state_receipt,
            "fsdp_group_count": fsdp_groups,
            "optimizer_impl": optimizer_impl,
            "optimizer": materialize_adam_state(torch, opt_g, step=int(bootstrap.value["optimizer"]["generator_step"])),
        }
        gc.collect()

    if owns_score:
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
        critic.to(device)
        opt_c = torch.optim.AdamW(
            lora_params, lr=4e-5, betas=(0.0, 0.99),
            weight_decay=0.0, eps=1e-8, foreach=True,
        )
        metadata["critic"] = {
            "placement": "gpu_resident_replica",
            "lora_rank": LORA_RANK,
            "lora_modules": int(lora_modules),
            "lora_numel": int(lora_numel),
            "optimizer": materialize_adam_state(torch, opt_c, step=int(bootstrap.value["optimizer"]["critic_step"])),
        }
        gc.collect()

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

        if pool_policy in {ASYNC_CRITIC_POOL_POLICY, TRACE_BALANCED_POOL_POLICY}:
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
                group=score_group,
                group_size=len(score_ranks),
                bucket_bytes=32 * 1024 * 1024,
            )
            metadata["critic_gradient_sync"] = {
                "policy": "grad_ready_async_flat_all_reduce",
                "group_ranks": list(score_ranks),
                "group_size": len(score_ranks),
                "bucket_bytes_limit": 32 * 1024 * 1024,
                "reduction": "SUM_THEN_DIVIDE_GROUP_SIZE",
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
        teacher=teacher,
        generator_optimizer=opt_g,
        critic_optimizer=opt_c,
        critic_lora_params=lora_params,
        critic_gradient_reducer=critic_gradient_reducer,
        generator_group=generator_group if owns_generator else None,
        score_group=score_group if owns_score else None,
        score_local_group=score_local_group,
        score_leader_group=(
            score_leader_group
            if node_local_score and rank in PIPELINE_6G10S_SCORE_LEADERS
            else None
        ),
        score_local_ranks=score_local_ranks,
        score_leader_ranks=(
            PIPELINE_6G10S_SCORE_LEADERS if node_local_score else ()
        ),
        generator_group_size=len(generator_ranks),
        score_group_size=len(score_ranks),
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
    with torch.no_grad():
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
    critic: Any


def build_score_context(torch: Any, *, bundle: ModelBundle, loader: InputLoader, device: Any) -> ScoreContext:
    from wanstreamer.core import latent_geometry, make_rope_table
    _require(bundle.teacher is not None and bundle.critic is not None, "Score model bundle is incomplete")
    _, _, hp, wp = latent_geometry(640, 368)
    rope = make_rope_table(bundle.critic, hp, wp, LATENT_SHAPE[2] + 8, device)
    with torch.no_grad():
        negative = bundle.teacher.text_embedding(loader.negative.to(device).to(torch.bfloat16))
    return ScoreContext(rope=rope, negative_teacher_context=negative)


def prepare_score_sample_context(
    torch: Any,
    *,
    bundle: ModelBundle,
    raw_context: Any,
    include_teacher: bool,
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
        critic_context = bundle.critic.text_embedding(raw_context.to(torch.bfloat16))
    return ScoreSampleContext(teacher=teacher_context, critic=critic_context)


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
    from wanstreamer import dmd as D
    from wanstreamer.lora import lora_enabled

    _require(sample_context.teacher is not None, "DMD requires Teacher context")

    t = _sample_flow_t(torch, seed=row.seeds["generator_dmd_t"], device=x0.device)
    noise = _randn(torch, x0.shape, seed=row.seeds["generator_dmd_noise"], device=x0.device)
    t_view = t
    while t_view.dim() < x0.dim():
        t_view = t_view.unsqueeze(-1)
    z_t = (1.0 - t_view) * x0 + t_view * noise
    with torch.no_grad():
        with lora_enabled(bundle.teacher, False):
            if context_lens is None:
                # Exact full-context analogue of LiveWan cfg_velocity().  The
                # immutable prompt bundle contains 512 valid tokens for every
                # row, so no padding mask is needed.  Avoiding a CUDA lens
                # tensor also avoids attention.py discovering that fact via
                # lens.min().item() in every Transformer block.
                v_real = _cfg_velocity_full_context(
                    torch,
                    D,
                    model=bundle.teacher,
                    z=z_t.to(torch.bfloat16),
                    t=t,
                    rope=score_context.rope,
                    context_positive=sample_context.teacher,
                    context_negative=score_context.negative_teacher_context,
                    dtype=torch.bfloat16,
                    guidance=5.0,
                    time_scale=TIME_SCALE,
                )
                v_real = v_real.float()
            else:
                v_real = D.cfg_velocity(
                    bundle.teacher, z_t.to(torch.bfloat16), t,
                    score_context.rope, sample_context.teacher, score_context.negative_teacher_context,
                    context_lens, torch.bfloat16, 5.0, TIME_SCALE,
                ).float()
        with lora_enabled(bundle.critic, True):
            v_fake = D.bidirectional_velocity(
                bundle.critic, z_t.to(torch.bfloat16), t,
                score_context.rope, sample_context.critic, context_lens,
                torch.bfloat16, TIME_SCALE,
            ).float()
        x0_real = (z_t - t_view * v_real).float()
        x0_fake = (z_t - t_view * v_fake).float()
        normalizer = (x0.float() - x0_real).abs().mean().clamp_min(1e-4)
        gradient = torch.nan_to_num((x0_fake - x0_real) / normalizer)
    return gradient


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
        scaled_loss.backward()
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
        scaled_loss.backward()
    else:
        bundle.critic_gradient_reducer.backward(scaled_loss)
    return tuple(value.detach() for value in losses.unbind(0))


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

    def add_logical_score_calls(self, key: str, count: int) -> None:
        _require(count > 0, "logical Score call count must be positive")
        parts = key.split("/")
        _require(len(parts) == 3, "logical Score call key must include role/kind/replay")
        for logical_key in (parts[0], "/".join(parts[:2]), key):
            self.logical_score_calls[logical_key] = (
                self.logical_score_calls.get(logical_key, 0) + count
            )


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
    if run_kind != "correctness":
        return
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    norm = clip_norm.full_tensor() if hasattr(clip_norm, "full_tensor") else clip_norm
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
        metadata.get("pool_policy") == TRACE_BALANCED_POOL_POLICY,
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
        metadata.get("pool_policy") == TRACE_BALANCED_POOL_POLICY
        and metadata.get("full_context_lens_elision") is True,
        "Critic LoRA kernel requires the validated C11 substrate",
    )
    expected = {
        "enabled": True,
        "name": "scale_one_identity_epilogue_v1",
        "scale": 1.0,
        "math_change": "remove_identity_multiply_only",
        "base_input_output_dtype": "bfloat16",
        "lora_parameter_compute_dtype": "float32",
        "source_forward_sha256": (
            "70d00d496aa2bd8ea06d64e6b04af873644eff52539396f74ec586a4cbbe97a3"
        ),
    }
    _require(dict(kernel) == expected, "Critic LoRA kernel contract changed")
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
    _require(int(plan["workload"]["shape"]["batch_size"]) == GLOBAL_BATCH, "plan is not global-B16")
    _require(plan["metadata"]["execution_mode"] == mode, "plan execution mode mismatch")
    _require(plan["workload"]["metadata"]["livewan_commit"] == LIVEWAN_COMMIT, "plan LiveWan mismatch")
    _require(plan["workload"]["metadata"]["wan21_commit"] == WAN21_COMMIT, "plan Wan2.1 mismatch")
    _require(not plan["workload"]["branches"]["ema"], "EMA must be outside Full5")
    _score_physical_batch_size(plan, mode=mode)
    _critic_physical_batch_size(plan, mode=mode)
    _full_context_lens_elision(plan, mode=mode)
    _score_context_prefetch_enabled(plan, mode=mode)
    _critic_lora_scale_one_enabled(plan, mode=mode)


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
    _validate_plan_identity(plan, mode=mode)
    trace = FormalTrace.load(plan["evidence"]["operation_trace"])
    _validate_trace_balanced_pairing(plan, trace)
    bootstrap = Bootstrap.load(plan["evidence"]["checkpoint"], trace=trace)
    root = Path(repo_root).expanduser().resolve()
    source_receipt = validate_source_receipt(bootstrap, repo_root=root)
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
    _require(run_kind in {"correctness", "timing", "profile"}, "invalid run kind")

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
    critic_lora_kernel = {
        "enabled": False,
        "name": "native_livewan_lora",
        "execution_status": "DISABLED",
    }
    if _critic_lora_scale_one_enabled(plan, mode=mode):
        from distillschedule.runtime.livewan_lora_kernel import (
            install_scale_one_lora_epilogue,
        )

        critic_lora_kernel = install_scale_one_lora_epilogue(torch)
    before_oom = cgroup_oom_counts()
    torch.cuda.reset_peak_memory_stats(device)
    build_started = time.perf_counter()
    bundle = build_models(
        torch=torch, plan=plan, bootstrap=bootstrap, mode=mode, rank=rank, device=device,
    )
    if _critic_lora_scale_one_enabled(plan, mode=mode):
        if bundle.critic is None:
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
                for module in bundle.critic.modules()
                if isinstance(module, LoRALinear)
            ]
            _require(len(lora_modules) == 180, "Critic LoRA module count changed")
            _require(
                all(module.scale == 1.0 for module in lora_modules),
                "Critic LoRA scale-one specialization saw a non-unit scale",
            )
            critic_lora_kernel.update(
                role="score",
                execution_status="PASS",
                module_count=len(lora_modules),
                all_scales_one=True,
            )
    loader = InputLoader(torch=torch, trace=trace)
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
        if bundle.critic is not None else None
    )
    bundle.metadata["communicator_warmup"] = _warm_communicators(
        torch, bundle=bundle, mode=mode, rank=rank, device=device,
    )
    build_wall = time.perf_counter() - build_started
    setup_memory = memory_snapshot(torch, device)
    versions = bootstrap.versions
    evidence = RunEvidence()
    inventory = CallInventory()
    roles = {
        id(model): role
        for model, role in (
            (bundle.student, "student"),
            (bundle.critic, "critic"),
            (bundle.teacher, "teacher"),
        )
        if model is not None
    }
    is_profile = run_kind == "profile"
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
    start = time.perf_counter()
    with _profile_range(
        torch, run_kind=run_kind, label=f"livewan/full5/{mode}/rank_{rank}"
    ):
        with inventory.install(torch=torch, blockcausal=bc, roles=roles):
            if mode == "baseline":
                _run_baseline(
                    torch=torch, rank=rank, trace=trace, loader=loader, bundle=bundle,
                    streamers=generator_streamers, score_context=score_context,
                    run_kind=run_kind, evidence=evidence, versions=versions, device=device,
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
    full5_wall = time.perf_counter() - start
    if is_profile:
        dist.barrier()
        torch.cuda.cudart().cudaProfilerStop()
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
    if bundle.student is not None:
        generator_metrics = _state_metrics(torch, list(bundle.student.parameters()))
    if bundle.critic is not None:
        # Critic is replicated in both paths.  One local metric is sufficient;
        # the summarizer checks equality across owners.
        critic_metrics = _state_metrics(torch, bundle.critic_lora_params)
    logical_call_inventory = dict(inventory.counts)
    logical_call_inventory.update(evidence.logical_score_calls)
    slot_routing = None
    if str(plan.get("metadata", {}).get("pool_policy", "")) == TRACE_BALANCED_POOL_POLICY:
        slot_pairs = _pipeline_8x8_slot_pairs(plan)
        lane = rank if rank < 8 else rank - 8
        slot_routing = {
            "policy": "trace_balanced_fixed_rank_pairs_v1",
            "lane": lane,
            "logical_slots": list(slot_pairs[lane]),
            "peer_rank": rank + 8 if rank < 8 else rank - 8,
            "all_pairs": [list(pair) for pair in slot_pairs],
        }
    receipt = {
        "status": "PASS",
        "rank": rank,
        "local_rank": local_rank,
        "mode": mode,
        "run_kind": run_kind,
        "trace_sha256": trace.sha256,
        "bootstrap_sha256": bootstrap.sha256,
        "source_receipt_sha256": source_receipt["source_receipt_sha256"],
        "logical_generator_updates": GENERATOR_UPDATES,
        "logical_critic_updates": CRITIC_UPDATES,
        "physical_generator_commits": GENERATOR_UPDATES if bundle.student is not None else 0,
        "physical_critic_commits": CRITIC_UPDATES if bundle.critic is not None else 0,
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
        "critic_lora_kernel": critic_lora_kernel,
        "slot_routing": slot_routing,
        "full5_wall_seconds": full5_wall,
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
) -> None:
    _require(bundle.student is not None and bundle.critic is not None and bundle.teacher is not None, "baseline needs all roles")
    for step in range(FULL5_STEPS):
        with _profile_range(
            torch, run_kind=run_kind, label=f"livewan/baseline/step_{step}"
        ):
            row = trace.row(step, rank)
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
                if run_kind == "correctness":
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
            if run_kind == "correctness":
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
                        torch, evidence, run_kind=run_kind, role="generator", step=step,
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
                evidence.gradient_sync.append({"step": step, "role": "critic", **sync})
                critic_pre_clip = (
                    _state_metrics(torch, [parameter.grad for parameter in bundle.critic_lora_params])
                    if run_kind == "correctness" else None
                )
                critic_clip_norm = torch.nn.utils.clip_grad_norm_(bundle.critic_lora_params, 1.0, foreach=True)
                _record_optimizer_diagnostics(
                    torch, evidence, run_kind=run_kind, role="critic", step=step,
                    parameters=bundle.critic_lora_params, clip_norm=critic_clip_norm,
                    gradient_before_clip=critic_pre_clip,
                )
                bundle.critic_optimizer.step()
                versions["critic"] += 1


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
    score_context_from_trace: bool = False,
    slot_pairs_by_lane: Sequence[Sequence[int]] | None = None,
    full_context_lens_elision: bool = False,
    score_context_prefetch: bool = False,
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
    else:
        _require(bundle.critic is not None and bundle.teacher is not None and score_context is not None, "pipeline Score role is incomplete")

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

    for step in range(FULL5_STEPS):
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
            for ahead_step in range(1, FULL5_STEPS):
                ahead_rows = tuple(
                    trace.row(ahead_step, slot) for slot in lane_slots
                )
                for wave in (0, 1):
                    previous = pending[wave]
                    if previous is not None:
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
                            f"step_{ahead_step}/isend_wave_{wave}"
                        ),
                    ):
                        works = _send_pair(
                            destination=peer,
                            prepared=prepared,
                            asynchronous=True,
                            send_context=not score_context_from_trace,
                        )
                    evidence.transport_bytes += pair_bytes
                    # Retain payload tensors until NCCL reports completion.
                    pending[wave] = (prepared, tuple(works))
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
            rows = tuple(trace.row(step, slot) for slot in lane_slots)
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
                # The rank-pair traffic shares the default NCCL process group.
                # Posting wave1 SEND before wave0-gradient RECV makes both peers
                # enqueue SEND as the next P2P operation (G sends wave1 while
                # Score sends grad0), so neither has a matching receive.  Keep
                # the useful G-wave1/Score-wave0 compute overlap, but receive
                # grad0 before posting wave1.  The wave1 transfer can still
                # overlap G backward0 with Score's receive/critic0 work.
                grad0 = torch.empty_like(first.x0)
                with _profile_range(
                    torch, run_kind=run_kind,
                    label="livewan/pipeline/g/step_0/recv_gradient_0",
                ):
                    dist.recv(grad0, src=peer)
                evidence.transport_bytes += grad0.numel() * grad0.element_size()
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
                grad1 = torch.empty_like(second.x0)
                with _profile_range(
                    torch, run_kind=run_kind,
                    label="livewan/pipeline/g/step_0/recv_gradient_1",
                ):
                    dist.recv(grad1, src=peer)
                evidence.transport_bytes += grad1.numel() * grad1.element_size()
                with _profile_range(
                    torch, run_kind=run_kind,
                    label="livewan/pipeline/g/step_0/backward_wave_1",
                ):
                    losses1 = generator_backward_for_sample(
                        torch, prepared=second, dmd_gradient=grad1, sample_weight=0.5,
                    )
                if run_kind == "correctness":
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
                    evidence.gradient_sync.append({"step": step, "role": "generator", **sync})
                    generator_parameters = list(bundle.student.parameters())
                    generator_pre_clip = (
                        _state_metrics(torch, [parameter.grad for parameter in generator_parameters])
                        if run_kind == "correctness" else None
                    )
                    generator_clip_norm = torch.nn.utils.clip_grad_norm_(
                        generator_parameters, 1.0, foreach=True,
                    )
                    _record_optimizer_diagnostics(
                        torch, evidence, run_kind=run_kind, role="generator", step=step,
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
            rows = tuple(trace.row(step, slot) for slot in lane_slots)
            with _profile_range(
                torch, run_kind=run_kind,
                label=f"livewan/pipeline/score/step_{step}/recv_wave_0",
            ):
                first_x0, first_context = _recv_pair(
                    torch,
                    source=peer,
                    device=device,
                    receive_context=not score_context_from_trace,
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
            if step == 0:
                with _profile_range(
                    torch, run_kind=run_kind,
                    label="livewan/pipeline/score/step_0/dmd_wave_0",
                ):
                    grad0 = dmd_gradient_for_sample(
                        torch, bundle=bundle, score_context=score_context, row=rows[0],
                        x0=first_x0, sample_context=first_sample_context, context_lens=context_lens,
                    )
                    evidence.add_logical_score_calls("teacher/nograd/plain", 1)
                    evidence.add_logical_score_calls("critic/nograd/plain", 1)
                    dist.send(grad0.contiguous(), dst=peer)
                evidence.transport_bytes += grad0.numel() * grad0.element_size()
            with _profile_range(
                torch, run_kind=run_kind,
                label=f"livewan/pipeline/score/step_{step}/recv_wave_1",
            ):
                second_x0, second_context = _recv_pair(
                    torch,
                    source=peer,
                    device=device,
                    receive_context=not score_context_from_trace,
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
                    label="livewan/pipeline/score/step_0/dmd_wave_1",
                ):
                    grad1 = dmd_gradient_for_sample(
                        torch, bundle=bundle, score_context=score_context, row=rows[1],
                        x0=second_x0, sample_context=second_sample_context, context_lens=context_lens,
                    )
                    evidence.add_logical_score_calls("teacher/nograd/plain", 1)
                    evidence.add_logical_score_calls("critic/nograd/plain", 1)
                    grad1_send = grad1.contiguous()
                    grad1_work = dist.isend(grad1_send, dst=peer)
                evidence.transport_bytes += grad1.numel() * grad1.element_size()
            if critic_b2_only:
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
                    )
                    evidence.add_logical_score_calls("critic/grad/checkpoint", 1)
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
            if run_kind == "correctness":
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
                evidence.gradient_sync.append({"step": step, "role": "critic", **sync})
                critic_pre_clip = (
                    _state_metrics(torch, [parameter.grad for parameter in bundle.critic_lora_params])
                    if run_kind == "correctness" else None
                )
                critic_clip_norm = torch.nn.utils.clip_grad_norm_(
                    bundle.critic_lora_params, 1.0, foreach=True,
                )
                _record_optimizer_diagnostics(
                    torch, evidence, run_kind=run_kind, role="critic", step=step,
                    parameters=bundle.critic_lora_params, clip_norm=critic_clip_norm,
                    gradient_before_clip=critic_pre_clip,
                )
                bundle.critic_optimizer.step()
                versions["critic"] += 1
    if score_context_prefetch and not is_generator:
        _require(
            not prefetched_contexts,
            "Score context prefetch left unconsumed lane contexts",
        )


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
