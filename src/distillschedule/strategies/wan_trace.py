"""Task-qualified trace validation; old Full5 manifests are not relabeled."""
from __future__ import annotations
import json
from pathlib import Path
from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint
from .wan_spec import WanTask, PyramidRolloutSpec

TASK_KEY = "parameterized_wan_task"


def rollout_spec_for(trace) -> PyramidRolloutSpec:
    contract = trace.manifest.get("contract", {})
    if TASK_KEY not in contract:
        return PyramidRolloutSpec()  # byte-for-byte legacy argument values
    task = WanTask.from_dict(contract[TASK_KEY])
    expected = {"resolution": list(task.resolution),
                "full_latent_shape": list(task.rollout_spec.latent_shape),
                "stage_spatial_shapes": task.rollout_spec.spatial_shapes,
                "dtype": "bfloat16", "dfake_gen_update_ratio": len(task.source_steps)}
    require(all(contract.get(k) == v for k,v in expected.items()),
            "trace shape/frequency differs from its parameterized contract", path="wan.trace")
    return task.rollout_spec


def validate_rank_manifest(task: WanTask, manifest: dict, *, rank: int) -> None:
    require(manifest.get("contract", {}).get(TASK_KEY) == task.to_dict(),
            "trace must be recorded for this exact task, not patched from Full5", path="wan.trace")
    from types import SimpleNamespace
    spec = rollout_spec_for(SimpleNamespace(manifest=manifest))
    rows = manifest.get("operations", [])
    require(isinstance(rows,list) and all(isinstance(r,dict) and isinstance(r.get("key"),str) for r in rows),
            "invalid operation manifest", path="wan.trace")
    ops = {r["key"]: r for r in rows}
    require(len(ops)==len(rows), "duplicate trace operation", path="wan.trace")
    first=task.source_steps[0]
    for step in task.source_steps:
        key=f"step/{step}/branch"
        expected=dict(global_step=step,dfake_gen_update_ratio=len(task.source_steps),
                      train_generator=step==first,train_critic=True,gan=False,reward=False,vae_decode=False)
        require(key in ops and all(ops[key].get("control",{}).get(k)==v for k,v in expected.items()),
                f"rank{rank} step{step} branch/update contract mismatch", path="wan.trace")
        for role in (("generator","critic") if step==first else ("critic",)):
            prefix=f"step/{step}/{role}"
            require(prefix+("/dmd" if role=="generator" else "/train") in ops,
                    "missing training operation", path="wan.trace")
            control=ops.get(prefix+"/rollout",{}).get("control",{})
            stages=control.get("pyramid_stage_call_sequence",[])
            require(len(stages)==3,"missing three-stage rollout schedule",path="wan.trace")
            for i,row in enumerate(stages):
                shape=[1,16,spec.latent_shape[2],*spec.spatial_shapes[i]]
                require(row.get("stage")==i and row.get("latent_shape")==shape,
                        "trace stage geometry mismatch",path="wan.trace")
                count=row.get("call_count")
                require(type(count) is int and 1<=count<=2*spec.stage_steps[i]
                        and count==row.get("denoising_exit",-2)+1,
                        "trace exit exceeds declared stage budget",path="wan.trace")
            require(control.get("call_count")==sum(s["call_count"] for s in stages),
                    "trace total call count mismatch",path="wan.trace")


def load_parameterized_trace(root: str | Path, task: WanTask, *, expected_hash: str):
    # Import model-facing trace classes only after metadata-only planning.
    from runtime.static_world16_trace import StaticWorld16Trace
    from runtime.operation_trace import (OperationTrace, load_operation_trace_bundle_manifest,
        resolve_rank_trace_root, validate_official_rank_stage_schedule_semantics)

    class TaskTrace(StaticWorld16Trace):
        def __init__(self):
            self.root=Path(root).resolve()
            self.variant,self.gan="no-gan",False
            self.logical_steps=task.source_steps
            self.rollout_spec=task.rollout_spec
            bundle=load_operation_trace_bundle_manifest(self.root,verify=True)
            require(bundle is not None and len(bundle.get("ranks",[]))==task.world_size,
                    "exact world16 trace bundle required",path="wan.trace")
            self.bundle_hash=str(bundle["bundle_hash"])
            require(self.bundle_hash==expected_hash,"trace bundle identity changed",path="wan.trace")
            self._control,self._loaded={},{}
            for rank in range(task.world_size):
                path=resolve_rank_trace_root(self.root,rank)
                manifest=json.loads((path/"trace.json").read_text())
                validate_rank_manifest(task,manifest,rank=rank)
                self._control[rank]=OperationTrace(path,manifest,{})
            self.rank_control_validation=validate_official_rank_stage_schedule_semantics(self._control)
            self.binding_hash=fingerprint(dict(task=task.digest,bundle=self.bundle_hash))

        def _tensors(self, trace, keys, device):
            values=super()._tensors(trace,keys,device)
            for name in ("entry_noise","rollout_noise","dmd_noise","critic_noise"):
                if name in values:
                    require(tuple(values[name].shape)==task.rollout_spec.latent_shape,
                            f"actual {name} shape differs from task",path="wan.trace")
            return values
    return TaskTrace()
