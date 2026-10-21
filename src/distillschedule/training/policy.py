"""Execute the existing planner's measured selection across training cycles.

The initial plan retains its exact source/state/input admission. Subsequent
cycles bind the admitted schedule to new data and increasing logical versions;
they do not pretend the evolving model is the original calibration snapshot.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

from .config import digest


def cycle_structure(workload):
    from distillschedule.open_workload.torch_backend import code_signature, signature
    import torch

    def shapes(value):
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        if isinstance(value, dict):
            return {key: shapes(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [shapes(item) for item in value]
        return value

    program = workload.program.to_dict()
    program["initial_versions"] = dict.fromkeys(program["initial_versions"], 0)
    for call in program["calls"]:
        # A cycle index is the only explicitly relocatable Call attribute.
        if "cycle" in call["attributes"]:
            call["attributes"]["cycle"] = 0
    modules = {key: {"class": code_signature(type(model)),
                     "parameters": {name: {**shapes(p), "requires_grad": p.requires_grad}
                                    for name, p in model.named_parameters()}}
               for key, model in workload.state.items() if isinstance(model, torch.nn.Module)}
    return digest({"program": program, "modules": modules, "feeds": shapes(dict(workload.feeds)),
                   "bindings": {key: code_signature(fn) for key, fn in workload.bindings.items()},
                   "immutable_modules": workload.immutable_modules})


class CyclePolicy:
    def __init__(self, execution):
        self.execution = dict(execution)
        self.bound = None

    def restore(self, value):
        if not isinstance(value, dict) or value.get("schema_version") != "distillschedule.cycle_policy.v1":
            raise ValueError("checkpoint has no admitted cycle policy")
        self.bound = deepcopy(value)

    def run(self, workload, output):
        from distillschedule.open_workload.torch_backend import Session, candidates
        from distillschedule.open_workload.planning import source_identity
        current_structure = cycle_structure(workload)
        if self.bound is None:
            mode = self.execution.get("mode", "reference")
            if mode in ("auto", "plan"):
                from distillschedule.open_workload.planning import calibrate, plan, run
                if mode == "auto":
                    limit = self.execution.get("memory_limit_bytes")
                    if type(limit) is not int or limit <= 0:
                        raise ValueError("auto training requires explicit memory_limit_bytes")
                    root = Path(output) / "planning"
                    root.mkdir(parents=True, exist_ok=False)
                    calibrate(workload, root / "calibration",
                              max_candidates=self.execution.get("max_candidates", 4),
                              samples=self.execution.get("samples", 1),
                              max_program_runs=self.execution.get("max_program_runs", 16),
                              profile_hardware=False)
                    document = plan(root / "calibration/calibration.json", memory_limit_bytes=limit,
                                    max_states=self.execution.get("max_states", 50000))
                    plan_path = root / "selected-plan.json"
                    plan_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
                else:
                    plan_path = Path(self.execution["plan"])
                    document = json.loads(plan_path.read_text())
                report, session, outputs = run(workload, plan_path, Path(output) / "plan-admission")
                if report["reference_gate"]["status"] != "PASS":
                    raise ValueError("selected plan numerical admission needs review; training did not commit")
                self.bound = {"schema_version": "distillschedule.cycle_policy.v1",
                              "mode": mode, "source": source_identity(), "structure": current_structure,
                              "implementation": deepcopy(document["selected"]["implementation"]),
                              "order": list(document["selected"]["order"]),
                              "plan_sha256": document["plan_sha256"],
                              "memory_limit_bytes": document["memory_limit_bytes"],
                              "initial_reference_gate": report["reference_gate"]}
                return session, outputs
            default = {"backend": "torch-callable-resident-v1", "share_immutable": False,
                       "checkpoint_bindings": []}
            implementation = self.execution.get("implementation", default) if mode == "manual" else default
            session = Session(workload, implementation, copy_state=False)
            order = self.execution.get("order")
            outputs = session.run(order)
            self.bound = {"schema_version": "distillschedule.cycle_policy.v1", "mode": mode,
                          "source": source_identity(), "structure": current_structure,
                          "implementation": deepcopy(implementation),
                          "order": list(order or (*[c.id for c in workload.program.calls], "__open_finish__")),
                          "initial_reference_gate": {"status": "NOT_RUN", "reason": "explicit user strategy"}}
            return session, outputs
        if self.bound["source"] != source_identity() or self.bound["structure"] != current_structure:
            raise ValueError("training policy source, workload, model precision or input structure changed")
        if self.bound["mode"] != self.execution.get("mode", "reference"):
            raise ValueError("checkpoint execution policy changed")
        implementation = self.bound["implementation"]
        if implementation not in candidates(workload, limit=128)["candidates"]:
            raise ValueError("selected implementation no longer satisfies runtime capabilities")
        session = Session(workload, implementation, copy_state=False)
        if session.device.type == "cuda" and "memory_limit_bytes" in self.bound:
            from distillschedule.open_workload.hardware_probe import plan_memory_limit
            plan_memory_limit(session.device, distributed=False, consensus=lambda value: [value],
                              requested=self.bound["memory_limit_bytes"], reserve_bytes=0)
        return session, session.run(self.bound["order"])
