"""Cheap static integration contracts; no native targetGPU timing claim."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/distillschedule/probe_wan13b_static_3d_world16.py'


def test_canonical_defaults_are_not_replaced_by_the_grid():
    tree = ast.parse(SOURCE.read_text())
    rows = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument']
    flags = {ast.literal_eval(n.args[0]): {k.arg: ast.literal_eval(k.value)
             for k in n.keywords if k.arg in ('choices', 'default')} for n in rows
             if n.args and isinstance(n.args[0], ast.Constant)}
    assert flags['--workload']['default'] == 'wan13b'
    assert flags['--pipeline-memory-policy']['default'] == 'gpipe8'
    assert 'wan13b-grid' in flags['--workload']['choices']


def test_grid_controls_do_not_truncate_a_canonical_F5_trace():
    source = SOURCE.read_text()
    assert 'trace=WanParameterGridTrace(args.trace_bundle,workload=grid_spec)' in source
    assert 'trace=StaticWorld16Trace(args.trace_bundle,variant=' in source
    assert 'for step in logical_steps:' in source
    assert 'for step in range(10,15):' not in source
    assert "row.update(grid_workload=grid_spec.receipt()" in source
    assert "sample_count=8*(1+len(logical_steps))" in source


def test_grid_source_hashes_and_original_commit_boundary_are_required():
    source = SOURCE.read_text()
    for field in ('expected_grid_workload_hash', 'expected_grid_trace_admission_hash',
                  'workload_fingerprint', 'bundle_hash'):
        assert field in source
    assert 'critic_optimizer_updates=len(logical_steps)' in source
    assert 'ema_updates=len(logical_steps)' in source
    assert 'where=\'generator optimizer\';generator.step()' in source
