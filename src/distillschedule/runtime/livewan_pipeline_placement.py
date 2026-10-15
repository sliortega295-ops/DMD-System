"""Opt-in H100 Teacher residency, independent of baseline placement/math.

Legacy plans are deliberately not mutated. A resident plan requires fresh
source-bound correctness evidence; a legacy C34 gate cannot authorize it.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .livewan_hardware import LiveWanHardware, livewan_hardware


FIELD = "pipeline_teacher_placement"
PLACEMENTS = ("legacy-sharded", "full-resident")


@dataclass(frozen=True)
class PipelineTeacherPlacement:
    name: str = "legacy-sharded"

    def __post_init__(self) -> None:
        if self.name not in PLACEMENTS:
            raise ValueError("unknown pipeline Teacher placement")

    @property
    def enabled(self) -> bool:
        return self.name == "full-resident"

    def receipt(self, hardware: LiveWanHardware | None = None) -> dict[str, Any]:
        hardware = hardware or livewan_hardware()
        if not self.enabled:
            return {}
        if not hardware.is_h100:
            raise ValueError("pipeline Teacher residency requires explicit h100-8-b16")
        return {
            "schema_version": "distillschedule.livewan_pipeline_teacher_placement.v1",
            "name": self.name,
            "hardware_profile": hardware.to_dict(),
            "teacher_fsdp": False,
            "teacher_parameter_offload": "off",
            "teacher_reshard_after_forward": False,
            "teacher_storage_dtype": "bfloat16",
            "teacher_trainable": False,
        }


def from_plan(plan: Mapping[str, Any], *, hardware: LiveWanHardware | None = None
              ) -> PipelineTeacherPlacement:
    value = plan.get("metadata", {}).get(FIELD, {})
    spec = PipelineTeacherPlacement(str(value.get("name", "legacy-sharded")))
    if value:
        if not spec.enabled or value != spec.receipt(hardware):
            raise ValueError("pipeline Teacher placement identity changed")
        if plan["metadata"].get("execution_mode") != "pipeline":
            raise ValueError("pipeline Teacher placement cannot change a baseline")
        if int(plan["workload"]["shape"]["batch_size"]) != 16:
            raise ValueError("pipeline Teacher placement requires global B16")
        backend = plan["training_backends"]["teacher"]
        if (backend.get("kind") != "native" or backend.get("parameter_offload") != "none"
                or backend.get("options", {}).get("reshard_after_forward") is not False):
            raise ValueError("pipeline Teacher backend differs from placement contract")
    return spec


def apply_contract(plan: dict[str, Any], *, mode: str, name: str
                   ) -> PipelineTeacherPlacement:
    spec = PipelineTeacherPlacement(name)
    if not spec.enabled:
        if plan.get("metadata", {}).get(FIELD):
            raise ValueError("pipeline Teacher plan/env placement mismatch")
        return spec
    if mode != "pipeline" or plan["metadata"].get("execution_mode") != "pipeline":
        raise ValueError("pipeline Teacher placement cannot change a baseline")
    # Check all admission conditions before mutating the caller's plan.
    receipt = spec.receipt()
    if int(plan["workload"]["shape"]["batch_size"]) != 16:
        raise ValueError("pipeline Teacher placement requires global B16")
    if plan.get("metadata", {}).get("baseline_placement"):
        raise ValueError("pipeline Teacher placement cannot combine baseline placement")
    if plan["training_backends"]["teacher"].get("parameter_offload", "none") != "none":
        raise ValueError("pipeline Teacher residency cannot enable parameter offload")
    plan["metadata"][FIELD] = receipt
    teacher = plan["training_backends"]["teacher"]
    teacher["kind"] = "native"
    teacher["parameter_offload"] = "none"
    teacher.setdefault("options", {})["reshard_after_forward"] = False
    return from_plan(plan)


def verify_teacher(spec: PipelineTeacherPlacement, model: Any, *, rank: int,
                   placement: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inspect actual parameters, outside timing; no tensor copies/collectives."""
    if not spec.enabled:
        return {}
    hardware = livewan_hardware()
    spec.receipt(hardware)
    if rank not in range(hardware.physical_world_size):
        raise ValueError("invalid physical rank")
    if rank in hardware.generator_ranks:
        if model is not None or placement is not None:
            raise ValueError("Generator rank unexpectedly owns Teacher")
        return {"status": "NOT_APPLICABLE", "rank": rank}
    if model is None:
        raise ValueError("Score rank is missing resident Teacher")
    parameters = list(model.parameters())
    if (not parameters or any(hasattr(p, "to_local") or p.requires_grad
            or p.device.type != "cuda" or str(p.dtype) != "torch.bfloat16"
            for p in parameters)):
        raise ValueError("Teacher must be frozen, non-DTensor, CUDA BF16")
    if (placement or {}).get("placement") != "gpu_resident_replica" or (
        placement.get("fsdp_group_count") != 0 or placement.get("mesh_ranks") != []
    ):
        raise ValueError("Teacher runtime still reports FSDP groups")
    return {"status": "PASS", "rank": rank, "registered_parameters": len(parameters),
            "logical_numel": sum(p.numel() for p in parameters),
            "local_storage_bytes": sum(p.numel() * p.element_size() for p in parameters),
            "storage_device": "cuda", "storage_dtype": "bfloat16", "fsdp_groups": 0}


def validate_gate_binding(value: Mapping[str, Any], *, name: str, mode: str,
                          hardware: LiveWanHardware | None = None) -> None:
    hardware = hardware or livewan_hardware()
    spec = PipelineTeacherPlacement(name)
    expected = spec.receipt(hardware)
    arm = value.get("arms", {}).get(mode, {})
    actual = arm.get(FIELD, {})
    if actual != expected:
        raise ValueError("timing gate pipeline Teacher placement mismatch")
    if spec.enabled:
        binding = arm.get("teacher_storage_binding", {})
        if not isinstance(binding.get("run_root"), str) or not binding["run_root"]:
            raise ValueError("timing gate is missing actual Teacher storage evidence")
        observed = audit_run(Path(binding["run_root"]), mode=mode, hardware=hardware)
        if observed != binding:
            raise ValueError("timing gate Teacher storage evidence changed")


def audit_run(run_root: Path, *, mode: str, hardware: LiveWanHardware
              ) -> dict[str, Any]:
    """Bind actual all-rank storage evidence to the timing admission gate."""
    plan_path = run_root / "effective_plan.node0.json"
    plan = json.loads(plan_path.read_text())
    spec = from_plan(plan, hardware=hardware)
    if not spec.enabled:
        return {}
    if mode != "pipeline":
        raise ValueError("resident pipeline evidence belongs to wrong mode")
    bindings = []
    sizes = set()
    for rank in range(hardware.physical_world_size):
        path = run_root / "receipts" / f"rank_{rank:03d}.json"
        raw = path.read_bytes()
        row = json.loads(raw)
        build = row.get("build", {})
        runtime = build.get(FIELD + "_runtime", {})
        expected_status = "PASS" if rank in hardware.score_ranks else "NOT_APPLICABLE"
        if (row.get("status") != "PASS" or row.get("rank") != rank
                or build.get(FIELD) != spec.receipt(hardware)
                or runtime.get("status") != expected_status or runtime.get("rank") != rank):
            raise ValueError(f"rank{rank}: pipeline Teacher storage evidence mismatch")
        if rank in hardware.score_ranks:
            teacher = build.get("teacher") or {}
            if (teacher.get("placement") != "gpu_resident_replica"
                    or teacher.get("fsdp_group_count") != 0 or teacher.get("mesh_ranks") != []
                    or runtime.get("storage_device") != "cuda"
                    or runtime.get("storage_dtype") != "bfloat16"
                    or runtime.get("fsdp_groups") != 0
                    or int(runtime.get("local_storage_bytes", 0)) <= 0):
                raise ValueError(f"rank{rank}: Teacher is not a full GPU replica")
            sizes.add((runtime.get("logical_numel"), runtime.get("local_storage_bytes")))
        elif build.get("teacher") is not None:
            raise ValueError(f"rank{rank}: unexpected Teacher ownership")
        bindings.append({"rank": rank, "path": str(path),
                         "sha256": hashlib.sha256(raw).hexdigest()})
    if len(sizes) != 1:
        raise ValueError("Teacher replica storage sizes differ")
    return {FIELD: spec.receipt(hardware), "run_root": str(run_root.resolve()),
            "rank_bindings": bindings,
            "effective_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest()}
