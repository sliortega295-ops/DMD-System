"""Explicit placement-only LiveWan baselines; legacy behavior stays unchanged.

The optional CPU staging is restricted to immutable Teacher parameters. It
does not wrap checkpoint/saved tensors, update parameters, or change samples.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PLACEMENTS = (
    "legacy-hybrid", "full-resident", "world16-fsdp",
    "world16-fsdp-student-retained",
)
TEACHER_OFFLOAD = ("off", "cpu-block")


@dataclass(frozen=True)
class BaselinePlacement:
    name: str = "legacy-hybrid"
    teacher_offload: str = "off"
    optimizer_device: str = "cuda"
    student_padding: str = "off"

    def __post_init__(self) -> None:
        if self.name not in PLACEMENTS or self.teacher_offload not in TEACHER_OFFLOAD:
            raise ValueError("unknown LiveWan baseline placement/offload")
        if self.teacher_offload != "off" and self.name != "full-resident":
            raise ValueError("Teacher CPU staging requires the full-resident arm")
        if self.optimizer_device not in {"cuda", "cpu"}:
            raise ValueError("unknown optimizer device")
        if self.optimizer_device == "cpu" and self.name != "full-resident":
            raise ValueError("CPU optimizer is restricted to the resident baseline")
        if self.student_padding not in {"off", "max-noop"}:
            raise ValueError("unknown Student padding policy")
        if self.student_padding != "off" and self.name != "world16-fsdp":
            raise ValueError("Student padding requires the strict world16 FSDP baseline")

    @property
    def student_fsdp(self) -> bool:
        return self.name != "full-resident"

    @property
    def student_reshard(self) -> bool:
        return self.name == "world16-fsdp"

    @property
    def critic_fsdp(self) -> bool:
        return self.name.startswith("world16-fsdp")

    @property
    def teacher_fsdp(self) -> bool:
        return self.name != "full-resident"

    def receipt(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "teacher_parameter_offload": self.teacher_offload,
            "student_fsdp": self.student_fsdp,
            "student_reshard_after_forward": self.student_reshard,
            "critic_fsdp": self.critic_fsdp,
            "critic_reshard_after_forward": self.critic_fsdp,
            "teacher_fsdp": self.teacher_fsdp,
            "teacher_reshard_after_forward": self.teacher_fsdp,
            "parameter_dtype_policy": "PRESERVE_ORIGINAL",
            "gradient_mean": "SUM_THEN_DIVIDE_16",
            "activation_offload": False,
            "optimizer_offload": self.optimizer_device == "cpu",
            "optimizer_device": self.optimizer_device,
            "student_rollout_padding": self.student_padding,
            "pipeline": False,
            "physical_batch_per_rank": 1,
            "all_model_weights_resident": self.name == "full-resident" and self.teacher_offload == "off",
            "pure_resident": (self.name == "full-resident" and self.teacher_offload == "off"
                              and self.optimizer_device == "cuda"),
        }


def from_plan(plan: Mapping[str, Any]) -> BaselinePlacement:
    value = plan.get("metadata", {}).get("baseline_placement", {})
    result = BaselinePlacement(
        str(value.get("name", "legacy-hybrid")),
        str(value.get("teacher_parameter_offload", "off")),
        str(value.get("optimizer_device", "cuda")),
        str(value.get("student_rollout_padding", "off")),
    )
    if value and value != result.receipt():
        raise ValueError("LiveWan baseline placement contract changed")
    return result


def apply_contract(plan: dict[str, Any], *, mode: str, name: str,
                   teacher_offload: str, optimizer_device: str = "cuda",
                   student_padding: str = "off") -> BaselinePlacement:
    result = BaselinePlacement(name, teacher_offload, optimizer_device, student_padding)
    if name == "legacy-hybrid":
        return result  # No default plan mutation, including its fingerprint.
    if mode != "baseline" or plan["metadata"]["execution_mode"] != "baseline":
        raise ValueError("baseline placement cannot change a pipeline plan")
    if int(plan["workload"]["shape"]["batch_size"]) != 16:
        raise ValueError("baseline placement requires unchanged global B16")
    plan["metadata"]["baseline_placement"] = result.receipt()
    for role, sharded, reshard in (
        ("generator", result.student_fsdp, result.student_reshard),
        ("score", result.critic_fsdp, result.critic_fsdp),
        ("teacher", result.teacher_fsdp, result.teacher_fsdp),
    ):
        backend = plan["training_backends"][role]
        backend["kind"] = "fsdp" if sharded else "native"
        backend.setdefault("options", {})["reshard_after_forward"] = reshard
        backend["parameter_offload"] = (
            "cpu" if role == "teacher" and teacher_offload == "cpu-block" else "none"
        )
    return result


def check_collective_trace(spec: BaselinePlacement, rows: Any) -> None:
    """FULL_SHARD cannot issue unequal forward AG inventories across ranks."""
    if not spec.student_reshard:
        return
    if spec.student_padding == "max-noop":
        if any(not 4 <= int(row.new_blocks) <= 6 for row in rows):
            raise RuntimeError("FAIL_FSDP_PADDING_TRACE")
        return
    steps: dict[int, set[int]] = {}
    for row in rows:
        steps.setdefault(int(row.logical_step), set()).add(int(row.new_blocks))
    mismatches = {step: sorted(counts) for step, counts in steps.items() if len(counts) != 1}
    if mismatches:
        raise RuntimeError(
            "FAIL_FSDP_COLLECTIVE_TRACE: rank-local Student rollout call counts "
            f"differ; cannot use reshard_after_forward=True unchanged: {mismatches}"
        )


def verify_models(spec: BaselinePlacement, bundle: Any) -> dict[str, Any]:
    """Check registered physical storage after construction, outside timing."""
    roles = {}
    for role, model, sharded in (("student", bundle.student, spec.student_fsdp),
                                ("critic", bundle.critic, spec.critic_fsdp),
                                ("teacher", bundle.teacher, spec.teacher_fsdp)):
        parameters = list(model.parameters())
        if not parameters or any(hasattr(p, "to_local") != sharded for p in parameters):
            raise ValueError(f"actual {role} DTensor placement differs from the baseline contract")
        locals_ = [p.to_local() if sharded else p for p in parameters]
        expected_device = "cpu" if role == "teacher" and spec.teacher_offload == "cpu-block" else "cuda"
        if any(p.device.type != expected_device for p in locals_):
            raise ValueError(f"actual {role} storage device differs from the baseline contract")
        roles[role] = {"sharded": sharded, "storage_device": expected_device,
            "registered_parameters": len(parameters),
            "logical_numel": sum(p.numel() for p in parameters),
            "local_storage_bytes": sum(p.numel()*p.element_size() for p in locals_)}
    if any(p.requires_grad for p in bundle.teacher.parameters()):
        raise ValueError("Teacher unexpectedly trainable")
    return {"status": "PASS", "roles": roles}


def validate_timing_gate(value: Mapping[str, Any], *, spec: BaselinePlacement,
                         expected_head: str, identity: Mapping[str, str] | None = None,
                         arm_id: str | None = None, seed: int | None = None) -> None:
    expected = {
        "schema_version": "distillschedule.livewan_baseline_placement_timing_gate.v1",
        "status": "PASS", "claimable_for_two_cycle_timing": True,
        "expected_head": expected_head, "cycles": 2,
        "final_versions": {"generator": 2, "critic": 10, "ema": 0},
        "baseline_placement": spec.receipt(),
        "numerical_policy": "REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW",
    }
    if identity is not None: expected["identity"] = dict(identity)
    if arm_id is not None: expected["arm_id"] = arm_id
    if seed is not None: expected["seed"] = seed
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"placement timing gate changed {key}")
    for key in ("correctness_summary", "comparison"):
        source = value[key]
        path = Path(source["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"placement timing gate evidence changed {key}")
        body = json.loads(path.read_text())
        if key == "correctness_summary":
            if (body.get("status") != "PASS" or body.get("head") != expected_head
                or body.get("rank_cycle_receipts") != 32 or body.get("cycles") != 2
                or body.get("final_versions") != expected["final_versions"]):
                raise ValueError("placement correctness run is incomplete")
            for rank in range(16):
                receipt = json.loads((path.parent / "receipts" / f"rank_{rank:03d}.json").read_text())
                build = receipt.get("build", {})
                if (receipt.get("status") != "PASS" or receipt.get("rank") != rank
                    or receipt.get("final_versions") != expected["final_versions"]
                    or build.get("baseline_placement") != spec.receipt()
                    or build.get("baseline_placement_runtime", {}).get("status") != "PASS"):
                    raise ValueError(f"placement correctness rank{rank} changed")
                if spec.optimizer_device == "cpu":
                    for role, expected_steps in (("student", 2), ("critic", 10)):
                        optimizer = build.get(role, {}).get("cpu_optimizer", {})
                        if (optimizer.get("state_device") != "cpu"
                            or optimizer.get("steps") != expected_steps
                            or optimizer.get("gradient_d2h_bytes", 0) <= 0
                            or optimizer.get("updated_parameter_h2d_bytes", 0) <= 0):
                            raise ValueError(f"rank{rank} CPU optimizer was not verified")
                if spec.student_padding == "max-noop":
                    padding = receipt.get("student_collective_padding", {})
                    records = padding.get("records", [])
                    if len(records) != 10 or any(r["aligned_nograd_forward_calls"] !=
                        1+5*r["maximum_new_blocks"] for r in records):
                        raise ValueError(f"rank{rank} collective padding was not verified")
                if spec.teacher_offload == "cpu-block":
                    staging = build.get("teacher", {}).get("cpu_staging", {})
                    if (staging.get("active_units_at_receipt") != 0
                        or staging.get("max_active_units") != 1
                        or staging.get("h2d_parameter_bytes", 0) <= 0
                        or staging.get("d2h_parameter_writeback_bytes") != 0):
                        raise ValueError(f"rank{rank} frozen Teacher staging was not verified")
        elif (body.get("complete") is not True
              or body.get("contract", {}).get("status") != "PASS"
              or body.get("numeric_eligibility", {}).get("status") != "PASS"
              or body.get("numeric_eligibility", {}).get("violations") != []):
            raise ValueError("placement numerical/semantic comparison is not PASS")


class FrozenTeacherCPUStaging:
    """Keep immutable CPU weights; upload one entered causal unit at a time.

    Uses the same entered CausalBlock/CausalHead wrappers as the existing FSDP
    helper, not hooks on original blocks that blockcausal.forward never calls.
    CPU weights are pinned once. No D2H writeback is needed for frozen weights.
    There is deliberately no speculative prefetch or compute optimization.
    """

    def __init__(self, torch: Any, model: Any, *, device: Any,
                 attach_causal_modules: Any, pin_memory: bool = True) -> None:
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("Teacher CPU staging requires frozen parameters")
        if any(p.device.type != "cpu" or hasattr(p, "to_local") for p in model.parameters()):
            raise ValueError("Teacher CPU staging requires non-sharded CPU parameters")
        attach_causal_modules(model)
        self.torch, self.device = torch, device
        self.units = list(model.causal_layers) + [model.causal_head,
            model.patch_embedding, model.text_embedding,
            model.time_embedding, model.time_projection]
        self.records: dict[int, list[tuple[Any, Any]]] = {}
        self.handles: list[Any] = []
        self.active: set[int] = set()
        self.h2d_bytes = self.h2d_events = self.unit_calls = 0
        self.max_active_units = 0
        self.pinned_bytes = 0
        seen: set[int] = set()
        for unit in self.units:
            records = []
            for value in list(unit.parameters()) + list(unit.buffers()):
                if id(value) in seen:
                    raise ValueError("CPU staging units share a parameter/buffer")
                seen.add(id(value))
                backing = value.detach()
                if pin_memory:
                    backing = backing.pin_memory()
                value.data = backing
                self.pinned_bytes += backing.numel() * backing.element_size() if backing.is_pinned() else 0
                records.append((value, backing))
            self.records[id(unit)] = records
            self.handles.append(unit.register_forward_pre_hook(self._enter))
            self.handles.append(unit.register_forward_hook(self._leave, always_call=True))
        if {id(p) for p in model.parameters()} - seen:
            raise ValueError("Teacher CPU staging left unhandled parameters")

    def _enter(self, unit: Any, _args: Any) -> None:
        key = id(unit)
        if self.active:
            raise RuntimeError("nested/reentrant Teacher staging is not supported")
        self.active.add(key)
        self.max_active_units = max(self.max_active_units, len(self.active))
        self.unit_calls += 1
        try:
            for value, backing in self.records[key]:
                value.data = backing.to(self.device, non_blocking=backing.is_pinned())
                self.h2d_bytes += backing.numel() * backing.element_size()
                self.h2d_events += 1
        except Exception:
            self._restore(key)
            raise

    def _restore(self, key: int) -> None:
        for value, backing in self.records[key]:
            value.data = backing
        self.active.discard(key)

    def _leave(self, unit: Any, _args: Any, _output: Any) -> None:
        self._restore(id(unit))

    def receipt(self) -> dict[str, Any]:
        return {
            "enabled": True, "name": "frozen_teacher_cpu_block_staging_v1",
            "units": len(self.units), "unit_calls": self.unit_calls,
            "active_units_at_receipt": len(self.active),
            "max_active_units": self.max_active_units,
            "h2d_parameter_bytes": self.h2d_bytes,
            "h2d_parameter_tensor_events": self.h2d_events,
            "d2h_parameter_writeback_bytes": 0,
            "pinned_cpu_bytes": self.pinned_bytes,
            "trainable_parameters_offloaded": 0,
            "activation_offload": False,
            "weight_values_changed": False,
            "transfers_inside_timing": True,
            "teacher_construction_wave_size_per_node": 2,
        }
