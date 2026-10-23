#!/usr/bin/env python3
"""CPU-only native worker import and resolvable global-call signature gate.

This detects cross-branch API omissions before model construction. It does not
execute model functions or claim to validate calls on runtime-created objects.
Run in the existing training environment; no additional packages are required.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts/distillschedule")]


def resolve_global(node, namespace):
    if isinstance(node, ast.Name):
        return namespace.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = resolve_global(node.value, namespace)
        # Do not evaluate arbitrary object properties while checking source.
        if isinstance(owner, types.ModuleType):
            return vars(owner).get(node.attr)
    return None


def check_module(module):
    path = Path(inspect.getsourcefile(module))
    failures, checked = [], 0
    tree = ast.parse(path.read_text())
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
            continue
        target = resolve_global(call.func, vars(module))
        if not callable(target):
            continue
        try:
            signature = inspect.signature(target)
        except (ValueError, TypeError):
            continue
        checked += 1
        try:
            signature.bind(*([None] * len(call.args)), **{k.arg: None for k in call.keywords})
        except TypeError as error:
            failures.append({"file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "line": call.lineno,
                             "call": ast.unparse(call.func), "signature": str(signature), "error": str(error)})
    return checked, failures


def main():
    import torch
    assert not torch.cuda.is_initialized(), "API check must start without a CUDA context"
    names = ("probe_wan13b_trajectory_world16", "probe_wan13b_two_cycle_timing_world16",
             "summarize_wan13b_trajectory_world16", "summarize_wan13b_two_cycle_timing_world16",
             "compare_wan13b_trajectory_world16")
    modules, failures, checked = {}, [], 0
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception as error:
            failures.append({"module": name, "import_error": repr(error)})
            continue
        modules[name] = module
        for attr in ("fair", "base", "p0"):
            child = getattr(module, attr, None)
            if isinstance(child, types.ModuleType):
                modules[child.__name__] = child
    for module in modules.values():
        count, errors = check_module(module)
        checked += count
        failures.extend(errors)
    assert not torch.cuda.is_initialized(), "import initialized CUDA unexpectedly"
    print(json.dumps({"status": "FAIL_API" if failures else "PASS_CPU_API",
                      "checked_global_calls": checked, "modules": sorted(modules), "failures": failures,
                      "runtime_object_calls": "NOT_CHECKED", "gpu_model_workload": "NOT_RUN"}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
