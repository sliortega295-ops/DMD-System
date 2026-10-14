"""Pure geometry lowering and native alias restoration, no GPU computation."""
import ast
from contextlib import ExitStack
import inspect
from pathlib import Path
import textwrap

import pytest

from distillschedule.runtime.wan13b_parameter_grid import WanParameterGridSpec
from distillschedule.runtime.wan13b_grid_geometry import _adapt_rollout, bind_grid_rollout_geometry

ROOT = Path(__file__).resolve().parents[1]


class _Shape:
    shape = [1, 16, 9, 56, 104]


def _audit(**kwargs):
    return kwargs


def _sample(value, detailed=False):
    if detailed:
        audit = _audit(base_height=48, base_width=80)
    else:
        audit = _audit(base_height=48, base_width=80)
    if list(value.shape) != [1, 16, 9, 48, 80]:
        raise AssertionError('shape contract')
    return audit, value


def test_shape_lowering_executes_and_preserves_tensor_identity():
    adapted = _adapt_rollout(_sample, (56, 104))
    x = _Shape()
    for detailed in (False, True):
        audit, returned = adapted(x, detailed=detailed)
        assert audit == dict(base_height=56, base_width=104)
        assert returned is x
    assert adapted.grid_geometry_receipt['substitutions'] == dict(
        base_height=2, base_width=2, output_shapes=1)
    with pytest.raises(AssertionError):
        _sample(x)


@pytest.mark.parametrize('resolution', [(448, 832), (704, 1280)])
def test_actual_rollout_functions_and_all_loaded_aliases_restore(resolution):
    from runtime import gst_pool_helios, static_fsdp2_p0_helios, critic_pool_helios
    names = ((gst_pool_helios, 'run_generator_update_trace_rollout'),
             (static_fsdp2_p0_helios, 'run_generator_trace_rollout_batch'),
             (critic_pool_helios, 'run_generator_trace_rollout'))
    before = [getattr(m, name) for m, name in names]
    spec = WanParameterGridSpec(*resolution, fake_updates=1, pyramid_budget=(1, 1, 1))
    with ExitStack() as stack:
        receipt = bind_grid_rollout_geometry(stack, ROOT, spec)
        assert receipt['status'] == 'PASS_EXPLICIT_GRID_GEOMETRY_BINDING'
        assert len(receipt['endpoints']) == 3
        for (module, name), original, r in zip(names, before, receipt['endpoints'], strict=True):
            adapted = getattr(module, name)
            assert adapted is not original
            assert inspect.signature(adapted) == inspect.signature(original)
            assert r['substitutions']['base_height'] == r['substitutions']['base_width'] == 2
    assert all(getattr(m, name) is f for (m, name), f in zip(names, before, strict=True))


def test_canonical_geometry_is_not_rewritten():
    spec = WanParameterGridSpec(384, 640, 5, (2, 2, 2))
    with ExitStack() as stack:
        assert bind_grid_rollout_geometry(stack, ROOT, spec)['endpoints'] == []


def test_modified_native_source_fails_closed(monkeypatch):
    source = textwrap.dedent(inspect.getsource(_sample)).replace('base_width=80', 'base_width=81')
    monkeypatch.setattr(inspect, 'getsource', lambda _: source)
    with pytest.raises(ValueError, match='source changed'):
        _adapt_rollout(_sample, (56, 104))


def test_worker_binding_is_opt_in_and_outside_timed_cycle():
    for name in ('probe_wan13b_parameter_grid_c2.py', 'probe_wan13b_static_3d_world16.py'):
        tree = ast.parse((ROOT / 'scripts/distillschedule' / name).read_text())
        assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == 'bind_grid_rollout_geometry' for n in ast.walk(tree))


def test_worker_graph_gate_uses_the_fresh_workload_not_old_dimensions():
    path = ROOT / 'scripts/distillschedule/probe_wan13b_parameter_grid_c2.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'check_grid_graphs')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    spec = WanParameterGridSpec(448, 832, 1, (2, 2, 2))
    gen = dict(enabled=True, weights_merged=False, active=False, model_call_count=8,
               graph_replay_model_call_count=8, rollout_entry_count=2,
               entries=[dict(version=3+i, logical_steps=[30]) for i in range(2)],
               cache_entries=[dict(latent_spatial_shape=list(s)) for s in spec.spatial_shapes])
    build = dict(generator=dict(metadata=dict(generator_rollout_cuda_graph=gen)),
                 critic=dict(metadata={}))
    arm = dict(generator_rollout_cuda_graph=True, critic_grad_cuda_graph=False)
    gate = namespace['check_grid_graphs']
    assert gate(build, arm, spec=spec, inventory=[{}, dict(inventory=dict(call_count=4))])
    gen['cache_entries'] = [dict(latent_spatial_shape=[48, 80])]
    with pytest.raises(ValueError, match='pyramid shape mismatch'):
        gate(build, arm, spec=spec, inventory=[{}, dict(inventory=dict(call_count=4))])
