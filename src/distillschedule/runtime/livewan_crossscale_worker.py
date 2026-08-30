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

from distillschedule.workloads.livewan_crossscale import (
    CRITIC_UPDATES,
    FULL5_STEPS,
    GENERATOR_UPDATES,
    GLOBAL_BATCH,
    LIVEWAN_COMMIT,
    LORA_RANK,
    PIPELINE_8G8S_LAYOUT,
    WAN21_COMMIT,
    WORLD_SIZE,
    sha256_file,
)


DTYPE_NAME = "torch.bfloat16"
LATENT_SHAPE = (1, 16, 21, 46, 80)
TEXT_SHAPE = (1, 512, 4096)
WORLD_CHOICES = (3, 6, 9)
BLOCK_FRAMES = 3
STEPS_PER_BLOCK = 4
GRAD_BLOCKS = 3
TIME_SCALE = 1000.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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
            state["step"] = torch.tensor(float(step), dtype=torch.float32, device="cpu")
            state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            tensors += 1
            local_numel += int(local.numel())
    return {"tensor_count": tensors, "local_numel_each_moment": local_numel, "step": step}


def _state_metrics(torch: Any, parameters: Sequence[Any], *, group: Any | None) -> dict[str, float | int]:
    device = next(iter(parameters)).device
    values = torch.zeros(5, device=device, dtype=torch.float64)
    count = torch.zeros(1, device=device, dtype=torch.int64)
    for index, parameter in enumerate(parameters):
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        value = local.detach().double()
        values[0] += value.sum()
        values[1] += value.square().sum()
        values[2] = torch.maximum(values[2], value.abs().max())
        values[3] += value.sum() * float((index % 97) + 1)
        values[4] += value.square().sum() * float((index % 89) + 1)
        count += value.numel()
    if group is not None:
        import torch.distributed as dist
        dist.all_reduce(values[:2], group=group)
        dist.all_reduce(values[2], op=dist.ReduceOp.MAX, group=group)
        dist.all_reduce(values[3:], group=group)
        dist.all_reduce(count, group=group)
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
    generator_group: Any | None
    score_group: Any | None
    generator_group_size: int
    score_group_size: int
    metadata: dict[str, Any]


def _make_mesh(torch: Any, ranks: Sequence[int], group: Any) -> Any:
    from torch.distributed.device_mesh import DeviceMesh
    try:
        return DeviceMesh.from_group(group, "cuda", mesh_dim_names=("fsdp",))
    except (AttributeError, TypeError):
        return DeviceMesh("cuda", list(ranks), mesh_dim_names=("fsdp",))


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
    generator_ranks = tuple(range(WORLD_SIZE)) if mode == "baseline" else PIPELINE_8G8S_LAYOUT.generator_ranks
    score_ranks = tuple(range(WORLD_SIZE)) if mode == "baseline" else PIPELINE_8G8S_LAYOUT.score_ranks
    # Every rank calls new_group in the same global order.
    generator_group = dist.new_group(ranks=list(generator_ranks))
    score_group = dist.new_group(ranks=list(score_ranks))
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
        opt_g = torch.optim.AdamW(
            list(student.parameters()), lr=2e-6, betas=(0.0, 0.99),
            weight_decay=0.0, eps=1e-8, foreach=True,
        )
        metadata["student"] = {
            "placement": "world16_fsdp2_keep_unsharded" if mode == "baseline" else "gpu_resident_replica",
            "state": state_receipt,
            "fsdp_group_count": fsdp_groups,
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
        teacher_mesh = _make_mesh(torch, score_ranks, score_group)
        shard_model(teacher, mesh=teacher_mesh, reshard_after_forward=True)
        teacher.to(device)
        fsdp_groups = _configure_fsdp_reduction(teacher, group_size=len(score_ranks))
        metadata["teacher"] = {
            "placement": f"fsdp2_group{len(score_ranks)}_reshard",
            "fsdp_group_count": fsdp_groups,
        }
        gc.collect()

    torch.cuda.synchronize(device)
    return ModelBundle(
        student=student,
        critic=critic,
        teacher=teacher,
        generator_optimizer=opt_g,
        critic_optimizer=opt_c,
        critic_lora_params=lora_params,
        generator_group=generator_group if owns_generator else None,
        score_group=score_group if owns_score else None,
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

    pair_peer = None
    if mode == "pipeline":
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

    dist.barrier()
    torch.cuda.synchronize(device)
    return {
        "status": "PASS",
        "groups": warmed_groups,
        "pair_peer": pair_peer,
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
        context = self.prompts["pos"][prompt_index].float()
        _require(tuple(latent.shape) == LATENT_SHAPE[1:], f"latent shape mismatch for {row.operation_key}")
        _require(tuple(context.shape) == TEXT_SHAPE[1:], f"text shape mismatch for {row.operation_key}")
        return latent, context, prompt_index


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
) -> PreparedRollout:
    latent, context, _ = loader.load(row)
    z_teacher = latent.to(device, non_blocking=True).unsqueeze(0)
    raw_context = context.to(device, non_blocking=True).unsqueeze(0)
    context_lens = torch.tensor([raw_context.shape[1]], device=device, dtype=torch.long)
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
    teacher: Any
    critic: Any


def build_score_context(torch: Any, *, bundle: ModelBundle, loader: InputLoader, device: Any) -> ScoreContext:
    from wanstreamer.core import latent_geometry, make_rope_table
    _require(bundle.teacher is not None and bundle.critic is not None, "Score model bundle is incomplete")
    _, _, hp, wp = latent_geometry(640, 368)
    rope = make_rope_table(bundle.critic, hp, wp, LATENT_SHAPE[2] + 8, device)
    with torch.no_grad():
        negative = bundle.teacher.text_embedding(loader.negative.to(device).to(torch.bfloat16))
    return ScoreContext(rope=rope, negative_teacher_context=negative)


def prepare_score_sample_context(torch: Any, *, bundle: ModelBundle, raw_context: Any) -> ScoreSampleContext:
    with torch.no_grad():
        teacher_context = bundle.teacher.text_embedding(raw_context.to(torch.bfloat16))
        critic_context = bundle.critic.text_embedding(raw_context.to(torch.bfloat16))
    return ScoreSampleContext(teacher=teacher_context, critic=critic_context)


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

    t = _sample_flow_t(torch, seed=row.seeds["generator_dmd_t"], device=x0.device)
    noise = _randn(torch, x0.shape, seed=row.seeds["generator_dmd_noise"], device=x0.device)
    t_view = t
    while t_view.dim() < x0.dim():
        t_view = t_view.unsqueeze(-1)
    z_t = (1.0 - t_view) * x0 + t_view * noise
    with torch.no_grad():
        with lora_enabled(bundle.teacher, False):
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
    (loss * sample_weight).backward()
    return loss.detach()


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


def _write_rank_receipt(
    *,
    output_dir: Path,
    rank: int,
    value: Mapping[str, Any],
) -> Path:
    path = output_dir / "receipts" / f"rank_{rank:03d}.json"
    atomic_json(path, value)
    return path


def _validate_plan_identity(plan: Mapping[str, Any], *, mode: str) -> None:
    _require(int(plan["hardware"]["world_size"]) == WORLD_SIZE, "plan is not world16")
    _require(int(plan["workload"]["shape"]["batch_size"]) == GLOBAL_BATCH, "plan is not global-B16")
    _require(plan["metadata"]["execution_mode"] == mode, "plan execution mode mismatch")
    _require(plan["workload"]["metadata"]["livewan_commit"] == LIVEWAN_COMMIT, "plan LiveWan mismatch")
    _require(plan["workload"]["metadata"]["wan21_commit"] == WAN21_COMMIT, "plan Wan2.1 mismatch")
    _require(not plan["workload"]["branches"]["ema"], "EMA must be outside Full5")


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
    before_oom = cgroup_oom_counts()
    torch.cuda.reset_peak_memory_stats(device)
    build_started = time.perf_counter()
    bundle = build_models(
        torch=torch, plan=plan, bootstrap=bootstrap, mode=mode, rank=rank, device=device,
    )
    loader = InputLoader(torch=torch, trace=trace)
    generator_streamers = []
    if bundle.student is not None:
        generator_streamers = _build_streamers(
            torch, student=bundle.student, count=(1 if mode == "baseline" else 2), device=device,
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
    dist.barrier()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with inventory.install(torch=torch, blockcausal=bc, roles=roles):
        if mode == "baseline":
            _run_baseline(
                torch=torch, rank=rank, trace=trace, loader=loader, bundle=bundle,
                streamers=generator_streamers, score_context=score_context,
                run_kind=run_kind, evidence=evidence, versions=versions, device=device,
            )
        else:
            _run_pipeline_8x8(
                torch=torch, rank=rank, trace=trace, loader=loader, bundle=bundle,
                streamers=generator_streamers, score_context=score_context,
                run_kind=run_kind, evidence=evidence, versions=versions, device=device,
            )
    torch.cuda.synchronize(device)
    dist.barrier()
    full5_wall = time.perf_counter() - start
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
        group = bundle.generator_group if mode == "baseline" else None
        generator_metrics = _state_metrics(torch, list(bundle.student.parameters()), group=group)
    if bundle.critic is not None:
        # Critic is replicated in both paths.  One local metric is sufficient;
        # the summarizer checks equality across owners.
        critic_metrics = _state_metrics(torch, bundle.critic_lora_params, group=None)
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
        "generator_state_metrics": generator_metrics,
        "critic_state_metrics": critic_metrics,
        "build": bundle.metadata,
        "ema": {"updates_in_full5": 0, "status": "NOT_RUN_SEPARATE_GATE"},
    }
    _write_rank_receipt(output_dir=output_dir, rank=rank, value=receipt)
    dist.barrier()
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
        row = trace.row(step, rank)
        bundle.generator_optimizer.zero_grad(set_to_none=True)
        bundle.critic_optimizer.zero_grad(set_to_none=True)
        prepared = prepare_rollout(
            torch, row=row, streamer=streamers[0], student=bundle.student,
            loader=loader, device=device,
        )
        _record_output(torch, evidence, row=row, x0=prepared.x0, run_kind=run_kind)
        sample_context = prepare_score_sample_context(
            torch, bundle=bundle, raw_context=prepared.raw_context,
        )
        if step == 0:
            gradient = dmd_gradient_for_sample(
                torch, bundle=bundle, score_context=score_context, row=row,
                x0=prepared.x0, sample_context=sample_context,
                context_lens=prepared.context_lens,
            )
            losses = generator_backward_for_sample(
                torch, prepared=prepared, dmd_gradient=gradient, sample_weight=1.0,
            )
            if run_kind == "correctness":
                evidence.loss_fingerprints.extend(
                    {"operation_key": row.operation_key, "kind": "generator", **_tensor_fingerprint(torch, value)}
                    for value in losses
                )
        critic_loss = critic_backward_for_sample(
            torch, bundle=bundle, score_context=score_context, row=row,
            x0=prepared.x0, sample_context=sample_context,
            context_lens=prepared.context_lens, sample_weight=1.0,
        )
        if run_kind == "correctness":
            evidence.loss_fingerprints.append(
                {"operation_key": row.operation_key, "kind": "critic", **_tensor_fingerprint(torch, critic_loss)}
            )
        if step == 0:
            torch.nn.utils.clip_grad_norm_(list(bundle.student.parameters()), 1.0)
            bundle.generator_optimizer.step()
            versions["generator"] += 1
        sync = _mean_replicated_gradients(
            torch, bundle.critic_lora_params, group=bundle.score_group,
            group_size=bundle.score_group_size, bucket_mib=128,
        )
        evidence.gradient_sync.append({"step": step, "role": "critic", **sync})
        torch.nn.utils.clip_grad_norm_(bundle.critic_lora_params, 1.0, foreach=True)
        bundle.critic_optimizer.step()
        versions["critic"] += 1


def _recv_pair(torch: Any, *, source: int, device: Any) -> tuple[Any, Any]:
    import torch.distributed as dist
    x0 = torch.empty(LATENT_SHAPE, device=device, dtype=torch.float32)
    context = torch.empty(TEXT_SHAPE, device=device, dtype=torch.float32)
    dist.recv(x0, src=source)
    dist.recv(context, src=source)
    return x0, context


def _send_pair(*, destination: int, prepared: PreparedRollout, asynchronous: bool):
    import torch.distributed as dist
    if asynchronous:
        return (
            dist.isend(prepared.x0.contiguous(), dst=destination),
            dist.isend(prepared.raw_context.contiguous(), dst=destination),
        )
    dist.send(prepared.x0.contiguous(), dst=destination)
    dist.send(prepared.raw_context.contiguous(), dst=destination)
    return ()


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
) -> None:
    import torch.distributed as dist

    is_generator = rank < 8
    peer = rank + 8 if is_generator else rank - 8
    pair_bytes = (math.prod(LATENT_SHAPE) + math.prod(TEXT_SHAPE)) * 4
    if is_generator:
        _require(bundle.student is not None and len(streamers) == 2, "pipeline G role is incomplete")
    else:
        _require(bundle.critic is not None and bundle.teacher is not None and score_context is not None, "pipeline Score role is incomplete")

    for step in range(FULL5_STEPS):
        if is_generator:
            bundle.generator_optimizer.zero_grad(set_to_none=True)
            rows = (trace.row(step, rank), trace.row(step, rank + 8))
            first = prepare_rollout(
                torch, row=rows[0], streamer=streamers[0], student=bundle.student,
                loader=loader, device=device,
            )
            _record_output(torch, evidence, row=rows[0], x0=first.x0, run_kind=run_kind)
            _send_pair(destination=peer, prepared=first, asynchronous=False)
            evidence.transport_bytes += pair_bytes
            second = prepare_rollout(
                torch, row=rows[1], streamer=streamers[1], student=bundle.student,
                loader=loader, device=device,
            )
            _record_output(torch, evidence, row=rows[1], x0=second.x0, run_kind=run_kind)
            sends = _send_pair(destination=peer, prepared=second, asynchronous=True)
            evidence.transport_bytes += pair_bytes
            if step == 0:
                grad0 = torch.empty_like(first.x0)
                dist.recv(grad0, src=peer)
                evidence.transport_bytes += grad0.numel() * grad0.element_size()
                losses0 = generator_backward_for_sample(
                    torch, prepared=first, dmd_gradient=grad0, sample_weight=0.5,
                )
                for request in sends:
                    request.wait()
                grad1 = torch.empty_like(second.x0)
                dist.recv(grad1, src=peer)
                evidence.transport_bytes += grad1.numel() * grad1.element_size()
                losses1 = generator_backward_for_sample(
                    torch, prepared=second, dmd_gradient=grad1, sample_weight=0.5,
                )
                if run_kind == "correctness":
                    for row, losses in zip(rows, (losses0, losses1)):
                        evidence.loss_fingerprints.extend(
                            {"operation_key": row.operation_key, "kind": "generator", **_tensor_fingerprint(torch, value)}
                            for value in losses
                        )
                sync = _mean_replicated_gradients(
                    torch, list(bundle.student.parameters()), group=bundle.generator_group,
                    group_size=bundle.generator_group_size, bucket_mib=256,
                )
                evidence.gradient_sync.append({"step": step, "role": "generator", **sync})
                torch.nn.utils.clip_grad_norm_(list(bundle.student.parameters()), 1.0, foreach=True)
                bundle.generator_optimizer.step()
                versions["generator"] += 1
            else:
                for request in sends:
                    request.wait()
        else:
            bundle.critic_optimizer.zero_grad(set_to_none=True)
            rows = (trace.row(step, peer), trace.row(step, peer + 8))
            first_x0, first_context = _recv_pair(torch, source=peer, device=device)
            evidence.transport_bytes += pair_bytes
            context_lens = torch.tensor([TEXT_SHAPE[1]], device=device, dtype=torch.long)
            first_sample_context = prepare_score_sample_context(
                torch, bundle=bundle, raw_context=first_context,
            )
            if step == 0:
                grad0 = dmd_gradient_for_sample(
                    torch, bundle=bundle, score_context=score_context, row=rows[0],
                    x0=first_x0, sample_context=first_sample_context, context_lens=context_lens,
                )
                dist.send(grad0.contiguous(), dst=peer)
                evidence.transport_bytes += grad0.numel() * grad0.element_size()
            second_x0, second_context = _recv_pair(torch, source=peer, device=device)
            evidence.transport_bytes += pair_bytes
            second_sample_context = prepare_score_sample_context(
                torch, bundle=bundle, raw_context=second_context,
            )
            loss0 = critic_backward_for_sample(
                torch, bundle=bundle, score_context=score_context, row=rows[0],
                x0=first_x0, sample_context=first_sample_context, context_lens=context_lens,
                sample_weight=0.5,
            )
            if step == 0:
                grad1 = dmd_gradient_for_sample(
                    torch, bundle=bundle, score_context=score_context, row=rows[1],
                    x0=second_x0, sample_context=second_sample_context, context_lens=context_lens,
                )
                dist.send(grad1.contiguous(), dst=peer)
                evidence.transport_bytes += grad1.numel() * grad1.element_size()
            loss1 = critic_backward_for_sample(
                torch, bundle=bundle, score_context=score_context, row=rows[1],
                x0=second_x0, sample_context=second_sample_context, context_lens=context_lens,
                sample_weight=0.5,
            )
            if run_kind == "correctness":
                for row, loss in zip(rows, (loss0, loss1)):
                    evidence.loss_fingerprints.append(
                        {"operation_key": row.operation_key, "kind": "critic", **_tensor_fingerprint(torch, loss)}
                    )
            sync = _mean_replicated_gradients(
                torch, bundle.critic_lora_params, group=bundle.score_group,
                group_size=bundle.score_group_size, bucket_mib=128,
            )
            evidence.gradient_sync.append({"step": step, "role": "critic", **sync})
            torch.nn.utils.clip_grad_norm_(bundle.critic_lora_params, 1.0, foreach=True)
            bundle.critic_optimizer.step()
            versions["critic"] += 1


__all__ = [
    "Bootstrap",
    "CallInventory",
    "FormalTrace",
    "OperationTraceRow",
    "run_worker",
]
