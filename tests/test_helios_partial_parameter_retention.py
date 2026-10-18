"""CPU-only real scope/dispatcher tests; fake FSDP is not a GPU qualification."""
import ast
import copy
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from distillschedule.runtime.fsdp2_nograd_persistence import generator_nograd_persistent_unshard_scope
from distillschedule.runtime.fsdp2_partial_nograd_persistence import partial_generator_nograd_scope, validate_receipt
from distillschedule.search.helios_implementation import HeliosImplementation, MODE_ENV, current_implementation
from distillschedule.search.helios_model_adapter import safe_incumbent, implementations


def incumbent():
    return replace(safe_incumbent(), retained_critic_blocks=(36, 37, 38, 39), retention_policy="tail4.v1")


class FSDP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.group = NS(_reshard_after_forward=True, is_unsharded=False)
        self.mutations = []

    def _get_fsdp_state(self):
        return NS(_fsdp_param_group=self.group)

    def set_reshard_after_forward(self, value, *, recurse):
        assert recurse is False
        self.mutations.append(value)
        self.group._reshard_after_forward = value

    def reshard(self):
        self.group.is_unsharded = False


class Block(FSDP):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.03125]))

    def forward(self, x):
        self.group.is_unsharded = True
        x = x + self.weight
        if self.group._reshard_after_forward:
            self.reshard()
        return x


class Model(FSDP):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(Block() for _ in range(40))

    def forward(self, x):
        self.group.is_unsharded = True
        for block in self.blocks:
            x = block(x)
        self.reshard()
        return x


def scope(model, version=lambda: 3):
    return partial_generator_nograd_scope(model, retained_block_count=24,
        parameter_version_reader=version, module_type=FSDP, memory_snapshot=lambda: {"CPU_ONLY": 0})


def released(model):
    assert all(m.group._reshard_after_forward and not m.group.is_unsharded for m in model.modules() if isinstance(m, FSDP))
    assert not model._forward_pre_hooks


def run_scope(model):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    expected = model(torch.ones(1)).detach()
    with scope(model) as receipt, torch.no_grad():
        for _ in range(5):
            assert torch.equal(model(torch.ones(1)), expected)
            receipt["rollout_batch_count"] += 1
            assert sum(b.group.is_unsharded for b in model.blocks) == 24
            assert not model.group.is_unsharded
    released(model)
    validate_receipt(receipt, expected_calls=5)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    return receipt


def test_real_scope_exact_cpu_math_storage_rng_and_exception_safe_lifetimes():
    model = Model()
    for _ in range(2):
        receipt = run_scope(model)
        assert receipt["partial_parameter_retention"]["generator_version_before"] == 3
    assert all(not b.mutations for b in model.blocks[24:]) and not model.mutations
    assert all(b.mutations == [False, True, False, True] for b in model.blocks[:24])


@pytest.mark.parametrize("failure", ["grad", "version", "body", "version_at_exit"])
def test_actual_failures_release_and_restore(failure):
    model = Model()
    version = [3]
    with pytest.raises((AssertionError, RuntimeError)):
        with scope(model, lambda: version[0]) as receipt:
            if failure == "grad":
                model(torch.ones(1))
            else:
                with torch.no_grad():
                    model(torch.ones(1))
                    receipt["rollout_batch_count"] = 5
                    if failure.startswith("version"):
                        version[0] = 4
                        if failure == "version": model(torch.ones(1))
                    else:
                        raise RuntimeError("injected original forward failure")
    released(model)


@pytest.mark.parametrize("change", ["missing_block", "unsharded", "policy", "aliased_group"])
def test_entry_invariants_fail_before_any_mutation(change):
    model = Model()
    if change == "missing_block": del model.blocks[-1]
    if change == "unsharded": model.blocks[0].group.is_unsharded = True
    if change == "policy": model.blocks[0].group._reshard_after_forward = False
    if change == "aliased_group": model.blocks[1].group = model.blocks[0].group
    with pytest.raises(AssertionError):
        with scope(model): pytest.fail("should reject before entering")
    assert all(not m.mutations for m in model.modules() if isinstance(m, FSDP))


@pytest.mark.parametrize("field,value", [("actual_generator_forward_count", 4), ("weights_merged", True),
    ("total_unsharded_after_exit_count", 1), ("set_no_reshard_count", 40),
    ("generator_version_after", 4), ("grad_enabled_failures", 1), ("restored", False),
    ("explicit_reshard_count", 0), ("parameter_dtype_changed", True)])
def test_consumer_rejects_actual_census_drift(field, value):
    receipt = run_scope(Model())
    receipt["partial_parameter_retention"][field] = value
    with pytest.raises(AssertionError): validate_receipt(receipt, expected_calls=5)


def test_manual_identity_binding_lowering_and_no_automatic_enumeration():
    legacy = incumbent()
    for count in (0, 24):
        impl = replace(legacy, partial_generator_parameter_blocks=count)
        assert impl.id.endswith(f"-pg{count}-v1")
        assert HeliosImplementation.parse(impl.id) == impl
        assert HeliosImplementation.from_assignment(impl.assignment) == impl
        assert impl.lower_profile()["contract"] == legacy.lower_profile()["contract"]
        env = impl.environment(); env[MODE_ENV] = "native"
        assert current_implementation(env) == impl
        env[MODE_ENV] = "calibration"
        with pytest.raises(ValueError, match="no search calibration"): current_implementation(env)
    assert not any(i.partial_generator_parameter_blocks is not None for i in implementations(legacy.workload))
    assert legacy.partial_generator_parameter_blocks is None
    assert "partial_generator_parameter_blocks" not in legacy.assignment
    assert not legacy.id.endswith("-pg0-v1")


@pytest.mark.parametrize("updates", [{"mesh": "world16-fsdp"}, {"rollout_window": 1}, {"chunk_tokens": 2048},
    {"partial_generator_parameter_blocks": 16}, {"partial_generator_parameter_blocks": True},
    {"retained_critic_blocks": (38, 39), "retention_policy": "bounded2.v1"}])
def test_manual_scope_cannot_silently_widen_existing_contract(updates):
    with pytest.raises(ValueError): replace(incumbent(), **({"partial_generator_parameter_blocks": 24} | updates))
    with pytest.raises(ValueError): replace(incumbent(), persistent_unshard=True)


def test_original_disabled_scope_is_unchanged_and_full_partial_mixing_rejects():
    with generator_nograd_persistent_unshard_scope(object(), enabled=False) as receipt:
        assert receipt["restored"] and "partial_parameter_retention" not in receipt
    with pytest.raises(ValueError):
        with generator_nograd_persistent_unshard_scope(object(), enabled=True, retained_block_count=24): pass


def test_native_dispatcher_calls_real_scope_for_five_b1s(monkeypatch):
    """Execute the production dispatcher AST -> real writer scope -> consumer."""
    path = Path(__file__).resolve().parents[1] / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_precompute_critic_rollouts")
    engine = NS(module=Model(), global_steps=3)
    def actual_scope(*a, **kw):
        return generator_nograd_persistent_unshard_scope(*a, **kw, module_type=FSDP, memory_snapshot=lambda: {"CPU_ONLY": 0})
    def run(**kw):
        with torch.no_grad():
            return [kw["model"](torch.ones(1)) for _ in kw["loaded_entries"]], {"CPU_ONLY": True}
    namespace = dict(torch=NS(cuda=NS(**{k: lambda *_: 0 for k in (
        "memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved")})),
        generator_nograd_persistent_unshard_scope=actual_scope, run_generator_trace_rollout_batch=run,
        chunked=lambda s, _: [(v,) for v in s], PACKED_SCORE_MEAN_REL_TOLERANCE=1e-5)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])), str(path), "exec"), namespace)
    args = dict(rank=0, steps=(10, 11, 12, 13, 14), device="CPU_SYNTHETIC_TEST_ONLY", generator_engine=engine,
        trace=NS(load_critic_entry=lambda **kw: NS(pool_entry=NS(logical_step=kw["step"]))),
        policy=NS(rollout_batch_size=1, generator_nograd_persistent_unshard=False,
                  generator_nograd_persistent_unshard_flush_allocator_cache=True,
                  generator_nograd_cross_kv_cache=False, generator_nograd_static_context_cache=False,
                  elide_explicit_rollout_sync=False), collect_diagnostics=False,
        diagnostics=NS(scope=lambda *_: nullcontext(), add_tensor_tree=lambda *a, **kw: None),
        rollout_parity_check=False, partial_generator_parameter_blocks=24)
    outputs, records, receipt = namespace[node.name](**args)
    assert len(outputs) == len(records) == 5
    validate_receipt(receipt, expected_calls=5)
    released(engine.module)
    assert not torch.cuda.is_initialized()


def test_manual_control_child_share_exact_five_b1_scope_and_original_call_order():
    from distillschedule.runtime.helios_planner_cycle import _cycle_graph
    legacy = incumbent()
    control = replace(legacy, partial_generator_parameter_blocks=0)
    child = replace(legacy, partial_generator_parameter_blocks=24)
    steps = (10, 11, 12, 13, 14)
    versions = dict(generator=2, critic=10, ema=10)
    old_graph, old_order = _cycle_graph(legacy, steps, versions)
    cg, co = _cycle_graph(control, steps, versions)
    ng, no = _cycle_graph(child, steps, versions)
    assert cg.to_dict() == ng.to_dict() and co == no
    assert list(co) == ["g", "r_all", "c0", "e0", "c1", "e1", "c2", "e2", "c3", "e3", "c4", "e4"]
    expand = lambda order: [y for x in order for y in ([f"r{i}" for i in range(5)] if x == "r_all" else [x])]
    assert expand(co) == list(old_order)
    assert cg.metadata["final_versions"] == old_graph.metadata["final_versions"]
    assert "r_all" not in old_order
