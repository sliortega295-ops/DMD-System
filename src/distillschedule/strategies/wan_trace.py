"""Reuse original trace tensor loaders with a new, task-bound admission gate.

A different task needs a newly captured native trace. No cropping, relabeling of
old tensors, RNG synthesis, or truncation of a Full5 bundle is performed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from distillschedule.core.contracts import require
from .wan_task import OriginalWanTask, TRACE_KEY


def validate_rank_controls(task, controls):
    """Validate every rank before model construction; payloads load on their rank."""
    from runtime.static_world16_trace import _generator_keys, _critic_keys
    from runtime.operation_trace import validate_official_rank_stage_schedule_semantics
    require(set(controls) == set(range(task.world_size)), "all rank controls required", path="wan.trace")
    for rank, trace in controls.items():
        c = trace.manifest.get("contract", {})
        require(c.get(TRACE_KEY) == task.to_dict(), "trace is not bound to this task; recapture, do not relabel", path="wan.trace")
        expected = {"resolution": [task.video_height, task.video_width],
                    "stage_spatial_shapes": [list(s) for s in task.stage_spatial_shapes],
                    "full_latent_shape": list(task.latent_shape), "dtype": "bfloat16",
                    "dfake_gen_update_ratio": task.fake_updates}
        require(all(c.get(k) == v for k, v in expected.items()), "trace shape/precision/frequency mismatch", path="wan.trace")
        operations = trace.manifest.get("operations", [])
        keys = [o["key"] for o in operations]
        require(len(keys) == len(set(keys)), "duplicate operation identity", path="wan.trace")
        branch_keys = {k for k in keys if k.startswith("step/") and k.endswith("/branch")}
        require(branch_keys == {f"step/{s}/branch" for s in task.source_steps},
                "trace contains another cycle or update frequency", path="wan.trace")
        for step in task.source_steps:
            branch = trace.operation(f"step/{step}/branch")["control"]
            wanted = {"global_step": step, "dfake_gen_update_ratio": task.fake_updates,
                      "train_generator": step == task.first_step, "train_critic": True,
                      "gan": False, "reward": False, "vae_decode": False}
            require(all(branch.get(k) == v for k, v in wanted.items()), "native branch mismatch", path="wan.trace")
            trace.operation(f"step/{step}/critic/train")
            phases = [("critic", _critic_keys(step, gan=False))]
            if step == task.first_step:
                trace.operation(f"step/{step}/generator/dmd")
                phases.append(("generator", _generator_keys(step, gan=False)))
            for phase, tensor_keys in phases:
                descriptors = trace.manifest.get("tensors", {})
                require(set(tensor_keys.values()) <= set(descriptors), "missing operation tensors", path="wan.trace")
                for name in ("entry_noise", "rollout_noise", "critic_noise" if phase == "critic" else "dmd_noise"):
                    d = descriptors[tensor_keys[name]]
                    require(d.get("shape") == list(task.latent_shape) and d.get("dtype") == "bfloat16",
                            "trace latent tensor shape/dtype mismatch", path="wan.trace")
                control = trace.operation(f"step/{step}/{phase}/rollout")["control"]
                if task.stage_schedule_policy == 'benchmark_v4_strict_global':
                    require(control.get('stage_schedule_policy') == task.stage_schedule_policy
                            and type(control.get('all_history_zero')) is bool,
                            'explicit benchmark branch scope/predicate missing', path='wan.trace')
                schedule = control.get("pyramid_stage_call_sequence", [])
                require(len(schedule) == 3 and {s.get("stage") for s in schedule} == {0, 1, 2},
                        "three-stage native schedule required", path="wan.trace")
                for item in schedule:
                    stage = item["stage"]
                    require(item.get("latent_shape") == [*task.latent_shape[:3], *task.stage_spatial_shapes[stage]],
                            "stage geometry mismatch", path="wan.trace")
                    require(type(item.get("call_count")) is int and 1 <= item["call_count"] <= task.stage2_inference_steps[stage],
                            "stage call count exceeds declared denoising horizon", path="wan.trace")
                    if task.stage_schedule_policy == 'benchmark_v4_strict_global':
                        expected_budget = task.stage2_inference_steps[stage] * (2 if control['all_history_zero'] else 1)
                        require(item.get('denoising_step_budget') == expected_budget,
                                'history-dependent native timestep budget changed', path='wan.trace')
                    # Native simulator horizon/exit is independently checked across all ranks below.
    checked = validate_official_rank_stage_schedule_semantics(controls)
    from runtime.static_world16_trace import StaticWorld16Trace
    for operation in [f"step/{task.first_step}/generator/rollout",
                      *(f"step/{s}/critic/rollout" for s in task.source_steps)]:
        signatures = [StaticWorld16Trace._physical_signature(t.operation(operation)) for t in controls.values()]
        require(all(s == signatures[0] for s in signatures), "rank physical geometry diverges", path="wan.trace")
    return checked


def load_original_trace(task: OriginalWanTask, root, *, expected_hash: str):
    from runtime.static_world16_trace import StaticWorld16Trace
    from runtime.operation_trace import OperationTrace, load_operation_trace_bundle_manifest, resolve_rank_trace_root
    root = Path(root).resolve()
    bundle = load_operation_trace_bundle_manifest(root, verify=True)
    require(bundle is not None and bundle["bundle_hash"] == expected_hash
            and len(bundle["ranks"]) == task.world_size, "trace bundle identity/ranks changed", path="wan.trace")
    controls = {}
    for rank in range(task.world_size):
        trace_root = resolve_rank_trace_root(root, rank).resolve()
        require(root in trace_root.parents, "rank trace escapes bundle", path="wan.trace")
        controls[rank] = OperationTrace(root=trace_root,
            manifest=json.loads((trace_root / "trace.json").read_text()), tensors={})
    legacy = all(TRACE_KEY not in t.manifest.get("contract", {}) for t in controls.values())
    if legacy:
        require(task == OriginalWanTask(), "legacy Full5 trace is valid only for its exact original task", path="wan.trace")
        return StaticWorld16Trace(root, variant="no-gan")
    validation = validate_rank_controls(task, controls)
    # Same original loader/entry types/math, different explicit admission contract.
    obj = StaticWorld16Trace.__new__(StaticWorld16Trace)
    obj.root, obj.variant, obj.gan = root, "no-gan", False
    obj.logical_steps, obj.bundle_hash = task.source_steps, expected_hash
    obj._control, obj._loaded, obj.rank_control_validation = controls, {}, validation
    obj.binding_hash = hashlib.sha256(json.dumps({"task": task.to_dict(), "bundle_hash": expected_hash},
                          sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return obj
