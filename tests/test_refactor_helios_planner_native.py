"""CPU execution evidence only; these fixtures do not establish GPU parity."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_hardened_score_wrapper_forwards_explicit_finite_policy():
    path = ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py"
    function = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_profiled_score_forward")
    calls = []
    namespace = dict(Any=Any, torch=NS(Tensor=object),
                     _nvtx_call=lambda _label, fn, *a, **kw: fn(*a, **kw),
                     _original_score_forward=lambda *a, **kw: calls.append(kw))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    for finite in (True, False):
        namespace["_profiled_score_forward"](object(), {"hidden_states": NS(shape=(1,))},
            condition=object(), adapter_enabled=False, validate_finite=finite)
        assert calls[-1]["validate_finite"] is finite


def _implementation(**overrides):
    from distillschedule.search.helios_implementation import HeliosImplementation
    return HeliosImplementation(**dict(variant="no-gan", mesh="world16-fsdp",
        chunk_tokens=512, rollout_window=5, persistent_unshard=False, **overrides))


@pytest.mark.parametrize("variant,mesh,chunk,persistent,window", [
    ("no-gan", "world16-fsdp", 512, False, 1),
    ("no-gan", "node-hsdp", 2048, False, 5),
    ("gan", "world16-fsdp", 512, False, 3),
    ("gan", "world16-fsdp", 2048, True, 5),
])
def test_generated_identity_profile_and_mode_are_explicit(variant, mesh, chunk, persistent, window):
    from distillschedule.search.helios_implementation import HeliosImplementation, current_implementation
    from distillschedule.search.contracts import semantic_contract
    from distillschedule.workloads.best_known.registry import load_profile
    impl = HeliosImplementation(variant, mesh, chunk, window, persistent)
    assert HeliosImplementation.parse(impl.id) == impl
    assert HeliosImplementation.from_assignment(impl.assignment) == impl
    env = impl.environment()
    assert env["DISTILLSCHEDULE_HELIOS_PLANNER_MODE"] == "off"
    assert json.loads(env["DISTILLSCHEDULE_HELIOS_PLANNER_ASSIGNMENT"]) == impl.assignment
    env["DISTILLSCHEDULE_HELIOS_PLANNER_MODE"] = "native"
    assert current_implementation(env) == impl
    baseline = load_profile(impl.workload)
    before = json.dumps(baseline, sort_keys=True)
    lowered = impl.lower_profile()
    assert semantic_contract(lowered) == semantic_contract(baseline)
    assert json.dumps(load_profile(impl.workload), sort_keys=True) == before
    assert lowered["runtime"]["arguments"][7] == mesh
    assert lowered["contract"]["continuous_cycles"] == 2
    assert lowered["contract"]["final_versions"] == (
        dict(generator=202, critic=1010, ema=802) if variant == "gan"
        else dict(generator=4, critic=20, ema=20))
    assert lowered["memory"] == baseline["memory"]
    for key, value in baseline["runtime"]["environment"].items():
        if key not in {"DISTILLSCHEDULE_HELIOS_MODULATION_CHUNK_TOKENS",
                       "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_PERSISTENT_UNSHARD"}:
            assert env[key] == value


@pytest.mark.parametrize("changes", [
    {"variant": "other"}, {"mesh": "resident"}, {"chunk_tokens": True},
    {"chunk_tokens": 1024}, {"rollout_window": 0}, {"rollout_window": 6},
    {"rollout_window": True}, {"persistent_unshard": 1}, {"persistent_unshard": True},
    {"variant": "gan", "mesh": "node-hsdp"},
    {"variant": "gan", "persistent_unshard": True, "rollout_window": 4},
])
def test_invalid_implementation_rejects(changes):
    from distillschedule.search.helios_implementation import HeliosImplementation
    args = dict(variant="no-gan", mesh="world16-fsdp", chunk_tokens=512,
                rollout_window=5, persistent_unshard=False)
    with pytest.raises(ValueError):
        HeliosImplementation(**(args | changes))


def test_default_off_and_unbound_or_unknown_generated_controls_reject():
    from distillschedule.search.helios_implementation import current_implementation, HeliosImplementation
    assert current_implementation({}) is None
    assert current_implementation({"DISTILLSCHEDULE_HELIOS_PLANNER_MODE": "off"}) is None
    impl = _implementation()
    for env in ({"DISTILLSCHEDULE_HELIOS_PLANNER_MODE": "native"}, impl.environment(),
                {"DISTILLSCHEDULE_SEARCH_CANDIDATE_ID": "joint-helios2-future"},
                impl.environment() | {"DISTILLSCHEDULE_HELIOS_PLANNER_MODE": "native",
                                      "DISTILLSCHEDULE_HELIOS_PLANNER_ASSIGNMENT": "{}"}):
        with pytest.raises(ValueError):
            current_implementation(env)
    for assignment in (impl.assignment | {"attention": "sdpa"},
                       impl.assignment | {"rollout_window": "05"},
                       {k: v for k, v in impl.assignment.items() if k != "mesh"}):
        with pytest.raises(ValueError):
            HeliosImplementation.from_assignment(assignment)


def test_worker_hook_is_before_original_timer_and_uses_hardened_module():
    source = (ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py").read_text()
    assert source.index("run_two_cycles(") < source.index("started_perf = time.perf_counter()")
    assert "p0=sys.modules[__name__]" in source
    hardened = (ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py").read_text()
    assert "impl.StaticWorld16Trace = RetainedP0StaticWorld16Trace" in hardened


def _fixture(monkeypatch, *, variant="no-gan", window=3, persistence=False, mode="native"):
    import math
    from contextlib import nullcontext
    from distillschedule.search.helios_implementation import HeliosImplementation
    impl = HeliosImplementation(variant, "world16-fsdp", 512, window, persistence)
    env = impl.environment() | {"DISTILLSCHEDULE_HELIOS_PLANNER_MODE": mode}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    events = []
    class Tensor:
        dtype = "torch.bfloat16"
        device = "cpu"
        requires_grad = False
        grad_fn = None
        def __init__(self, value=1., shape=()):
            self.value, self.shape = value, shape
            self.data = self
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def all(self): return self
        def mean(self): return Tensor(self.value)
        def item(self): return self.value
        def numel(self): return math.prod(self.shape)
        def element_size(self): return 2
        def stride(self): return tuple(1 for _ in self.shape)
        def fill_(self, value): self.value = value
    class Model:
        def __init__(self):
            self.weight = Tensor()
            self.adapters_enabled = True
            self._forward_pre_hooks, self._forward_hooks = {}, {}
        def modules(self): return (self,)
        def named_parameters(self): return (("weight", self.weight),)
        def register_forward_pre_hook(self, hook, *, with_kwargs):
            key = len(self._forward_pre_hooks)
            self._forward_pre_hooks[key] = hook
            return NS(remove=lambda: self._forward_pre_hooks.pop(key))
        def register_forward_hook(self, hook, *, with_kwargs):
            key = len(self._forward_hooks)
            self._forward_hooks[key] = hook
            return NS(remove=lambda: self._forward_hooks.pop(key))
        def enable_adapters(self):
            self.adapters_enabled = True
        def disable_adapters(self):
            self.adapters_enabled = False
        def __call__(self, **kwargs):
            for hook in self._forward_pre_hooks.values(): hook(self, (), kwargs)
            output = (Tensor(self.weight.value, kwargs["hidden_states"].shape),)
            for hook in self._forward_hooks.values(): hook(self, (), kwargs, output)
            return output
    class Engine:
        def __init__(self, step):
            self.module, self.global_steps = Model(), step
            self.optimizer = NS(state={}, step=lambda: None, zero_grad=lambda **kw: None)
        def backward(self, loss):
            pass
        def step(self):
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_steps += 1
    initial = dict(generator=2, critic=10, ema=800) if variant == "gan" else dict(generator=2, critic=10, ema=10)
    g, c = Engine(initial["generator"]), Engine(initial["critic"])
    ema = NS(optimization_step=initial["ema"], last_transfer_bytes=0)
    def ema_step():
        events.append(("ema", c.global_steps))
        ema.optimization_step += 1
        return 0.5
    ema.step = ema_step
    steps = tuple(range(1000, 1005)) if variant == "gan" else tuple(range(10, 15))
    hidden = Tensor(shape=(1, 16, 9, 48, 80))
    def rollout_record(step, *, generator=False):
        n = 3 if generator else 3 + steps.index(step)
        return dict(call_count=n, stage_call_count={0: 1, 1: 1, 2: n-2},
                    logical_call_count=n, logical_stage_call_count={0: 1, 1: 1, 2: n-2},
                    dummy_call_count=0, output_shape=list(hidden.shape), output_dtype=str(hidden.dtype),
                    output_requires_grad=generator, calls=[{} for _ in range(n)],
                    logical_steps=[step], batch_size=1, member_trace_identities=[f"r0-s{step}"],
                    trace_identity_sha256=f"r0-s{step}")
    def entry(step, generator=False):
        record = rollout_record(step, generator=generator)
        operation = f"step/{step}/{'generator' if generator else 'critic'}/rollout"
        control = dict(call_count=record["call_count"], pyramid_stage_call_sequence=[
            dict(stage=s, call_count=n) for s,n in record["stage_call_count"].items()])
        optrace = NS(trace_hash="optrace0", operation=lambda _key: {"control": control})
        member = NS(logical_step=step, source_trace_rank=0, identity_sha256=f"r0-s{step}",
                    operation_trace_hash="optrace0", rollout_operation=operation)
        if generator:
            return NS(logical_step=step, source_id=0, identity_sha256=f"r0-s{step}",
                      identity={"source_rank": 0}, operation_trace=optrace, rollout_operation=operation,
                      tensors={"entry_noise": hidden})
        return NS(pool_entry=member, operation_trace=optrace, tensors={"entry_noise": hidden})
    class Trace:
        logical_steps = steps
        bundle_hash, binding_hash = "bundle", "binding"
        def __init__(self):
            self.cached = {}
            self.loads, self.hits = 0, 0
        def load_generator_entry(self, *, rank, device):
            return entry(steps[0], True)
        def load_critic_entry(self, *, rank, step, device):
            if step in self.cached:
                self.hits += 1
                return self.cached.pop(step)
            self.loads += 1
            self.cached[step] = entry(step)
            return self.cached[step]
        def receipt(self):
            return dict(critic_entries_still_cached=len(self.cached), critic_entry_h2d_misses=self.loads,
                        critic_entry_reuse_hits=self.hits)
    trace = Trace()
    def gupdate(**kw):
        kw["trace"].load_generator_entry(rank=0, device="cpu")
        events.append(("g", g.global_steps))
        g.module.enable_adapters()
        for _ in range(3):
            value = g.module(hidden_states=hidden)[0]
        with nullcontext():
            for index, teacher in enumerate([False, True, True, False] if variant == "gan" else [False, True, True]):
                # Official GAN compute_kl_grad starts in existing adapter state.
                if variant != "gan" or index > 0:
                    (c.module.disable_adapters if teacher else c.module.enable_adapters)()
                c.module(hidden_states=hidden, gan_mode=variant == "gan" and index == 3)
        g.backward(value.float().mean()); g.step()
        return dict(logical_step=steps[0], source_rank=0, rollout=rollout_record(steps[0], generator=True))
    def precompute(**kw):
        selected = kw["steps"]
        events.append(("r", selected, g.global_steps))
        g.module.enable_adapters()
        outputs, records = {}, []
        for step in selected:
            kw["trace"].load_critic_entry(rank=0, step=step, device="cpu")
            with nullcontext():
                for _ in range(rollout_record(step)["call_count"]):
                    outputs[step] = g.module(hidden_states=hidden)[0]
            records.append(rollout_record(step))
        return outputs, records, dict(enabled=persistence, restored=True,
            rollout_batch_count=len(selected), set_no_reshard_count=41 if persistence else 0,
            explicit_reshard_count=41 if persistence else 0, unsharded_after_exit_count=0)
    def critic(**kw):
        step = kw["step"]
        kw["trace"].load_critic_entry(rank=0, step=step, device="cpu")
        events.append(("c", step, c.global_steps))
        c.module.enable_adapters()
        for index in range(2 if variant == "gan" else 1):
            # Logical B1; official GAN-D's fake/real/R1 packed physical B3.
            packed = variant == "gan" and index == 1
            value = c.module(hidden_states=Tensor(shape=(3,16,9,24,40)) if packed else kw["x0"],
                             gan_mode=packed)[0]
            c.backward(value.float().mean())
        c.step()  # Exactly one, including GAN's two-backward official boundary.
        return dict(logical_step=step, source_rank=0, critic_optimizer_committed=True)
    class CUDA:
        def synchronize(self, *_): events.append(("fence",))
        def reset_peak_memory_stats(self, *_): events.append(("reset",))
        def memory_allocated(self, *_): return 100
        def memory_reserved(self, *_): return 200
        def max_memory_allocated(self, *_): return 150
        def max_memory_reserved(self, *_): return 250
        def mem_get_info(self, *_): return 1000, 2000
    cuda = CUDA()
    torch_view = NS(cuda=cuda, is_tensor=lambda v: isinstance(v, Tensor), is_grad_enabled=lambda: False,
                    isfinite=lambda v: Tensor(math.isfinite(v.value)),
                    stack=lambda values: NS(cpu=lambda: NS(tolist=lambda: [v.value for v in values])), Tensor=Tensor)
    p0 = NS(torch=torch_view, dist=NS(barrier=lambda: events.append(("barrier",))),
            base=NS(_run_gan_generator_update=gupdate), _run_nogan_generator_update_p0=gupdate,
            _precompute_critic_rollouts=precompute, _run_nogan_critic_precomputed=critic,
            _run_gan_critic_precomputed=critic)
    policy = NS(mesh_policy=impl.mesh, rollout_batch_size=1, pack_real_score_pair=False,
                rollout_backend="native-fsdp2-batch", selective_flash_checkpoint_stride=2,
                modulation_chunk_tokens=impl.chunk_tokens, generator_nograd_persistent_unshard=persistence,
                elide_official_generator_free_memory=variant == "gan",
                elide_critic_scalar_logs=variant == "gan")
    args = NS(variant=variant, timing_mode=True, profile_mode=False, profile_diagnostics=False,
              ema_device="cuda", attention_backend="local-fa2", critic_only_step=None,
              generator_residual_storage="cuda_out_of_place", critic_residual_storage="cuda_out_of_place",
              operation_source=False)
    result = dict(rank=0, world_size=16, status="RUNNING", initial={})
    return NS(impl=impl, events=events, g=g, c=c, ema=ema, trace=trace, result=result,
              kwargs=dict(p0=p0, args=args, policy=policy, result=result, output=Path("unused"),
                  rank=0, device="cpu", logical=NS(component=lambda key: {"optimizer_step": initial[key]},
                      manifest={"systems_only_control": dict(classification="systems_only_gan_bootstrap",
                          logical_schedule_versions=dict(generator=200, critic=1000, ema=800),
                          logical_control_window=list(range(1000,1005)), official_checkpoint_1000=False)}),
                  trace=trace, negative=hidden, cfg=NS(), generator_engine=g, critic_engine=c, ema=ema,
                  diagnostics=NS(add_bytes=lambda *a, **kw: None), generator_free_memory={}))


@pytest.mark.parametrize("variant,window,persistence", [
    ("no-gan", 1, False), ("no-gan", 3, False), ("no-gan", 5, False),
    ("gan", 1, False), ("gan", 5, True),
])
def test_two_cycles_execute_native_helpers_order_versions_and_actual_census(monkeypatch, variant, window, persistence):
    from distillschedule.runtime.helios_planner_cycle import run_two_cycles
    f = _fixture(monkeypatch, variant=variant, window=window, persistence=persistence)
    run_two_cycles(**f.kwargs)
    assert f.result["status"] == f.result["semantics_status"] == "PASS"
    assert f.result["numerical_status"] == "NOT_RUN"
    assert f.result["profiled"] is False
    assert len(f.result["source_head"]) == 40
    assert f.result["runtime_identity"]["python_executable"] == sys.executable
    assert f.result["source_cohort_policy"] == "fixed_source_cohort_replay"
    assert len(f.result["cycles"]) == 2
    assert [row["measured"] for row in f.result["cycles"]] == [False, True]
    assert f.result["cycles"][1]["initial_versions"] == f.result["cycles"][0]["final_versions"]
    assert f.result["final_versions"] == (dict(generator=202, critic=1010, ema=802) if variant == "gan"
                                         else dict(generator=4, critic=20, ema=20))
    assert f.g.global_steps == 4 and f.c.global_steps == 20
    assert f.result["initial_physical_versions"] == dict(generator=2, critic=10, ema=800 if variant == "gan" else 10)
    assert f.result["final_physical_versions"] == dict(generator=4, critic=20, ema=802 if variant == "gan" else 20)
    assert f.result["final"]["generator_physical_step"] == 4
    assert f.result["final"]["critic_physical_step"] == 20
    assert f.trace.loads == f.trace.hits == 10 and not f.trace.cached
    for cycle in f.result["cycles"]:
        actions = cycle["actions"]
        assert sum(a["counts"]["generator_optimizer"] for a in actions) == 1
        assert sum(a["counts"]["critic_optimizer"] for a in actions) == 5
        assert sum(a["counts"]["critic_backward"] for a in actions) == (10 if variant == "gan" else 5)
        assert cycle["peak_live_rollouts"] == window
        names = [a["action_id"] for a in actions]
        if persistence:
            assert names[:3] == ["g", "r_all", "c0"]
            assert len([e for e in f.events if e[0] == "r"]) == 2
            assert actions[1]["persistence"]["explicit_reshard_count"] == 41
        else:
            assert names.index("c0") == window + 1
        assert names[names.index("c0")+1] == "e0"
        assert len([n for n in names if n.startswith("e")]) == (1 if variant == "gan" else 5)
        assert all(a["calibration_seconds"] is None for a in actions)
        assert all(a["retained_after"]["count"] <= window for a in actions)
        assert cycle["actions"][-1]["retained_after"]["count"] == 0
    assert len([e for e in f.events if e[0] == "fence"]) == 4


def test_gan_w5_off_uses_one_composite_scope_just_like_on(monkeypatch):
    from distillschedule.runtime.helios_planner_cycle import run_two_cycles
    f = _fixture(monkeypatch, variant="gan", window=5, persistence=False, mode="calibration")
    run_two_cycles(**f.kwargs)
    for cycle in f.result["cycles"]:
        assert cycle["selected_order"][:3] == ["g", "r_all", "c0"]
        assert cycle["actions"][1]["persistence"]["enabled"] is False
    assert len([e for e in f.events if e[0] == "r"]) == 2


@pytest.mark.parametrize("field,value", [("generator", 200), ("critic", 1000)])
def test_gan_logical_label_cannot_replace_physical_optimizer_step(monkeypatch, field, value):
    from distillschedule.runtime.helios_planner_cycle import run_two_cycles
    f = _fixture(monkeypatch, variant="gan", window=5)
    setattr(f.g if field == "generator" else f.c, "global_steps", value)
    with pytest.raises(ValueError, match="physical"):
        run_two_cycles(**f.kwargs)
    assert not f.events


@pytest.mark.parametrize("change", [
    {"logical_schedule_versions": {"generator": 201, "critic": 1000, "ema": 800}},
    {"logical_control_window": list(range(10, 15))},
    {"classification": "official"}, {"official_checkpoint_1000": True},
])
def test_gan_loaded_systems_control_mapping_is_fail_closed(monkeypatch, change):
    from distillschedule.runtime.helios_planner_cycle import run_two_cycles
    f = _fixture(monkeypatch, variant="gan", window=5)
    f.kwargs["logical"].manifest["systems_only_control"].update(change)
    with pytest.raises(ValueError, match="logical version mapping"):
        run_two_cycles(**f.kwargs)
    assert (f.g.global_steps, f.c.global_steps, f.ema.optimization_step) == (2, 10, 800)
    assert not f.events


@pytest.mark.parametrize("field,value", [("selective_flash_checkpoint_stride", 0),
                                         ("rollout_backend", "external")])
def test_effective_nonsearched_policy_must_match_canonical(monkeypatch, field, value):
    from distillschedule.runtime.helios_planner_cycle import run_two_cycles
    f = _fixture(monkeypatch)
    setattr(f.kwargs["policy"], field, value)
    with pytest.raises(ValueError, match="effective|backend|checkpoint"):
        run_two_cycles(**f.kwargs)
    assert not any(e[0] == "g" for e in f.events)


def test_rollout_member_identity_mismatch_cannot_pass_with_same_call_counts(monkeypatch):
    from distillschedule.runtime.helios_planner_cycle import run_two_cycles
    f = _fixture(monkeypatch)
    original = f.kwargs["p0"]._precompute_critic_rollouts
    def changed(**kw):
        outputs, records, persistence = original(**kw)
        records[0]["member_trace_identities"] = ["foreign-rank-or-step"]
        return outputs, records, persistence
    f.kwargs["p0"]._precompute_critic_rollouts = changed
    with pytest.raises(ValueError, match="identity"):
        run_two_cycles(**f.kwargs)


def test_correctness_probes_are_outside_both_fenced_cycle_boundaries(monkeypatch):
    from distillschedule.runtime import helios_planner_cycle as runtime
    f = _fixture(monkeypatch, mode="correctness")
    def sketch(*args):
        f.events.append(("state_probe",))
        return {"scope": "fixture_only"}
    monkeypatch.setattr(runtime, "_correctness_sketch", sketch)
    runtime.run_two_cycles(**f.kwargs)
    positions = [i for i,e in enumerate(f.events) if e[0] == "state_probe"]
    fences = [i for i,e in enumerate(f.events) if e[0] == "fence"]
    assert positions[0] < fences[0] < fences[1] < positions[1] < fences[2] < fences[3] < positions[2]
    assert f.result["numerical_status"] == "NOT_RUN"  # sampled states are not paired parity


def test_calibration_fences_every_action_and_keeps_setup_outside(monkeypatch):
    from distillschedule.runtime.helios_planner_cycle import run_two_cycles
    f = _fixture(monkeypatch, mode="calibration")
    run_two_cycles(**f.kwargs)
    actions = [a for c in f.result["cycles"] for a in c["actions"]]
    assert all(a["calibration_seconds"] > 0 for a in actions)
    assert all(a["memory_before"]["driver_free_bytes"] == 1000 for a in actions)
    assert len([e for e in f.events if e[0] == "fence"]) == 4 + 2*len(actions)
    assert f.events.index(("reset",)) < next(i for i,e in enumerate(f.events) if e[0] == "g")


def test_native_counter_mismatch_fails_and_wrappers_restore(monkeypatch):
    from distillschedule.runtime.helios_planner_cycle import run_two_cycles
    f = _fixture(monkeypatch)
    original = f.kwargs["p0"]._run_nogan_critic_precomputed
    old_backward, old_step = f.c.backward, f.c.step
    def double_commit(**kw):
        row = original(**kw)
        f.c.step()
        return row
    f.kwargs["p0"]._run_nogan_critic_precomputed = double_commit
    with pytest.raises(ValueError, match="commit|count|version"):
        run_two_cycles(**f.kwargs)
    assert f.c.backward == old_backward and f.c.step == old_step
    assert not f.c.module._forward_pre_hooks


def test_native_finite_gate_is_after_cycle_boundary(monkeypatch):
    from distillschedule.runtime.helios_planner_cycle import run_two_cycles
    f = _fixture(monkeypatch)
    f.g.module.weight.data.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        run_two_cycles(**f.kwargs)
    assert len([e for e in f.events if e[0] == "fence"]) == 2


def test_finite_gpu_scans_also_stay_outside_native_timer(monkeypatch):
    from distillschedule.runtime.helios_planner_cycle import run_two_cycles
    f = _fixture(monkeypatch)
    original = f.kwargs['p0'].torch.isfinite
    def scan(value):
        f.events.append(('finite_scan',))
        return original(value)
    f.kwargs['p0'].torch.isfinite = scan
    run_two_cycles(**f.kwargs)
    fences = [i for i,e in enumerate(f.events) if e[0] == 'fence']
    scans = [i for i,e in enumerate(f.events) if e[0] == 'finite_scan']
    assert len(fences) == 4 and len(scans) == 12
    assert all(fences[1] < i < fences[2] for i in scans[:6])
    assert all(fences[3] < i for i in scans[6:])
