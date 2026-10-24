"""Reuse original trace tensor loaders with a new, task-bound admission gate.

A different task needs a newly captured native trace. No cropping, relabeling of
old tensors, RNG synthesis, or truncation of a Full5 bundle is performed here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from distillschedule.core.contracts import require
from .wan_task import OriginalWanTask, TRACE_KEY, NATIVE_HISTORY_POLICY

NATIVE_HISTORY_SEMANTICS = 'official_helios_gt_history_rank_local_amplified_v1'


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
                if task.stage_schedule_policy in ('benchmark_v4_strict_global', NATIVE_HISTORY_POLICY):
                    require(control.get('stage_schedule_policy') == task.stage_schedule_policy
                            and type(control.get('all_history_zero')) is bool,
                            'explicit history branch scope/predicate missing', path='wan.trace')
                schedule = control.get("pyramid_stage_call_sequence", [])
                require(len(schedule) == 3 and {s.get("stage") for s in schedule} == {0, 1, 2},
                        "three-stage native schedule required", path="wan.trace")
                for item in schedule:
                    stage = item["stage"]
                    require(item.get("latent_shape") == [*task.latent_shape[:3], *task.stage_spatial_shapes[stage]],
                            "stage geometry mismatch", path="wan.trace")
                    budget = task.stage2_inference_steps[stage] * (2 if
                        task.stage_schedule_policy == NATIVE_HISTORY_POLICY and control['all_history_zero'] else 1)
                    require(type(item.get("call_count")) is int and 1 <= item["call_count"] <= budget,
                            "stage call count exceeds declared denoising horizon", path="wan.trace")
                    if task.stage_schedule_policy in ('benchmark_v4_strict_global', NATIVE_HISTORY_POLICY):
                        expected_budget = task.stage2_inference_steps[stage] * (2 if control['all_history_zero'] else 1)
                        require(item.get('denoising_step_budget') == expected_budget,
                                'history-dependent native timestep budget changed', path='wan.trace')
                    # Native simulator horizon/exit is independently checked across all ranks below.
    if task.stage_schedule_policy == NATIVE_HISTORY_POLICY:
        return _validate_native_history_schedules(task, controls)
    checked = validate_official_rank_stage_schedule_semantics(controls)
    from runtime.static_world16_trace import StaticWorld16Trace
    for operation in [f"step/{task.first_step}/generator/rollout",
                      *(f"step/{s}/critic/rollout" for s in task.source_steps)]:
        signatures = [StaticWorld16Trace._physical_signature(t.operation(operation)) for t in controls.values()]
        require(all(s == signatures[0] for s in signatures), "rank physical geometry diverges", path="wan.trace")
    return checked


def _validate_native_history_schedules(task, controls):
    """Resident-only contract: common initial exits, native local exceptions.

    Do not weaken the legacy FSDP validator or invent padded calls. Every rank
    keeps its own exact physical work and validates it after execution.
    """
    operations, exceptions, parity = [], 0, True
    for key in [f'step/{task.first_step}/generator/rollout',
                *(f'step/{s}/critic/rollout' for s in task.source_steps)]:
        rows, initial = [], None
        for rank in range(task.world_size):
            control = controls[rank].operation(key)['control']
            require(control.get('rank') == rank and type(control.get('rank')) is int
                    and control.get('stage_schedule_scope') == 'native_rank0_initial_with_rank_local_amplified',
                    'native history schedule rank/scope mismatch', path='wan.trace')
            exits = control.get('initial_synchronized_pyramid_stage_exit')
            require(isinstance(exits, list) and len(exits) == 3 and all(
                type(e) is int and 0 <= e < h for e, h in zip(exits, task.stage2_inference_steps)),
                'invalid initial synchronized exits', path='wan.trace')
            require(initial is None or exits == initial, 'initial exits must still agree across ranks', path='wan.trace')
            initial = exits
            sequence = control['pyramid_stage_call_sequence']
            require([s['stage'] for s in sequence] == [0, 1, 2], 'stage order changed', path='wan.trace')
            counts, executed = [], []
            for stage, item in enumerate(sequence):
                exit_index, calls, times = item.get('denoising_exit'), item['call_count'], item.get('timestep')
                require(type(exit_index) is int and exit_index >= 0 and calls == exit_index+1,
                        'stage calls must equal native exit plus one', path='wan.trace')
                require(isinstance(times, list) and len(times) == calls
                        and all(type(t) in (float, int) and math.isfinite(t) for t in times),
                        'native stage timesteps/calls mismatch', path='wan.trace')
                require(control['all_history_zero'] or exit_index == initial[stage],
                        'normal history must use synchronized exits', path='wan.trace')
                counts.append(calls); executed.append(exit_index)
            require(type(control.get('call_count')) is int and control['call_count'] == sum(counts),
                    'native total call count mismatch', path='wan.trace')
            exceptions += int(control['all_history_zero'])
            rows.append(dict(rank=rank, all_history_zero=control['all_history_zero'],
                executed_exit=executed, per_stage_call_counts=counts,
                logical_call_count=sum(counts), physical_call_count=sum(counts)))
        parity = parity and all(r['per_stage_call_counts'] == rows[0]['per_stage_call_counts'] for r in rows)
        operations.append(dict(operation=key, rank_count=task.world_size,
                               initial_synchronized_exit=initial, ranks=rows))
    return dict(status='PASS', semantics=NATIVE_HISTORY_SEMANTICS, rank_count=task.world_size,
        cross_rank_call_count_parity=parity, local_exception_entry_count=exceptions,
        operation_count=len(operations), global_entry_count=task.world_size*len(operations), operations=operations,
        execution_requirement='RESIDENT_FORWARD_WITHOUT_PARAMETER_COLLECTIVES')


def expected_rank_rollout_inventory(trace, *, operation, rank=None):
    from distillschedule.runtime.fsdp2_p0 import expected_trace_rollout_call_inventory
    checked = trace.rank_control_validation
    if checked.get('semantics') != NATIVE_HISTORY_SEMANTICS:
        return expected_trace_rollout_call_inventory(checked, operation=operation, expected_rank_count=16)
    require(checked.get('status') == 'PASS' and checked.get('rank_count') == 16
            and type(rank) is int and 0 <= rank < 16,
            'validated native history census requires the executing rank', path='wan.trace')
    matches = [row for row in checked['operations'] if row['operation'] == operation]
    require(len(matches) == 1 and len(matches[0]['ranks']) == 16,
            'missing/duplicate native rank census', path='wan.trace')
    row = matches[0]['ranks'][rank]
    require(row['rank'] == rank and len(row['per_stage_call_counts']) == 3
            and all(type(c) is int and c > 0 for c in row['per_stage_call_counts'])
            and row['logical_call_count'] == row['physical_call_count']
            == sum(row['per_stage_call_counts']), 'invalid native rank census', path='wan.trace')
    stages = dict(enumerate(row['per_stage_call_counts']))
    return dict(call_count=row['physical_call_count'], stage_call_count=stages,
                logical_call_count=row['logical_call_count'], logical_stage_call_count=dict(stages), dummy_call_count=0)


def validate_loaded_history(task, loaded):
    """Validate the native predicate from actual CPU payloads before CUDA copies."""
    if task.stage_schedule_policy != NATIVE_HISTORY_POLICY:
        return
    for step, phase in [(task.first_step, 'generator'), *((s, 'critic') for s in task.source_steps)]:
        prefix = f'step/{step}/{phase}'
        tensors = [loaded.tensor(prefix+'/'+k) for k in ('history_short', 'history_mid', 'history_long')]
        require(all(t.device.type == 'cpu' for t in tensors),
                'history predicate admission requires CPU payloads before CUDA transfer', path='wan.trace')
        zero = all(bool(t.sum() == 0) for t in tensors)
        require(zero == loaded.operation(prefix+'/rollout')['control']['all_history_zero'],
                'history payload disagrees with native amplified predicate', path='wan.trace')


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
