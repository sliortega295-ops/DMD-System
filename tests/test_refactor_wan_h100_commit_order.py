from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from distillschedule.runtime.hardware_profile import hardware_profile, MicrobatchCommitEngine

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py"


def test_two_logical_rows_share_parameter_versions_and_one_commit():
    events = []

    class Engine:
        def __init__(self, version, role):
            self.global_steps = version
            self.role = role
            self.module = object()
            self.gradients = []

        def backward(self, value):
            self.gradients.append(value)

        def step(self):
            events.append(("commit", self.role, self.global_steps))
            self.global_steps += 1

    g = Engine(2, "G")
    c = Engine(10, "C")

    def generator(**kw):
        events.append(("generator", kw["rank"], kw["generator_engine"].global_steps))
        kw["generator_engine"].backward(2.0)
        kw["generator_engine"].step()
        return {"source_rank": kw["rank"]}

    def rollout(**kw):
        events.append(("rollout", kw["rank"], kw["generator_engine"].global_steps))
        kw["entry_outputs"].update({s: object() for s in kw["steps"]})
        return {s: object() for s in kw["steps"]}, [], {}

    def critic(**kw):
        events.append(("critic", kw["step"], kw["rank"], kw["critic_engine"].global_steps))
        kw["critic_engine"].backward(2.0)
        kw["critic_engine"].step()
        return {"logical_step": kw["step"]}

    class EMA:
        count = 0

        def step(self):
            self.count += 1
            events.append(("ema", c.global_steps))

    fn = next(n for n in ast.parse(WORKER.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "_run_workload_cycle")
    namespace = {
        "Any": object, "torch": SimpleNamespace(device=object, Tensor=object),
        "HARDWARE": hardware_profile("h100-8-b16"), "SOURCE_STEPS": tuple(range(10, 15)),
        "MicrobatchCommitEngine": MicrobatchCommitEngine,
        "p0": SimpleNamespace(_run_nogan_generator_update_p0=generator,
                              _precompute_critic_rollouts=rollout,
                              _run_nogan_critic_precomputed=critic),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(WORKER), "exec"), namespace)
    ema = EMA()
    rows = namespace["_run_workload_cycle"](
        rank=3, device=None, generator_engine=g, critic_engine=c, ema=ema,
        trace=None, negative=None, diagnostics=None, policy=None,
        clean_runtime=False, runtime_cache=None, deferred_finite_checks=[],
    )
    assert [r["logical_slot"] for r in rows["logical_samples"]] == [3, 11]
    assert [e for e in events if e[0] == "generator"] == [("generator", 3, 2), ("generator", 11, 2)]
    assert [e for e in events if e[0] == "rollout"] == [("rollout", 3, 3), ("rollout", 11, 3)]
    for step in range(10, 15):
        assert [e for e in events if e[0] == "critic" and e[1] == step] == [
            ("critic", step, 3, step), ("critic", step, 11, step)]
    assert len([e for e in events if e[0] == "commit" and e[1] == "G"]) == 1
    assert len([e for e in events if e[0] == "commit" and e[1] == "C"]) == 5
    assert g.gradients == [1.0, 1.0] and c.gradients == [1.0] * 10
    assert (g.global_steps, c.global_steps, ema.count) == (3, 15, 5)
