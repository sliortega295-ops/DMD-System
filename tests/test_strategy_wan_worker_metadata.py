"""CPU-only control-plane collective regression; training NCCL remains untouched."""
from types import SimpleNamespace as NS
import inspect
from distillschedule.strategies import wan_worker


def test_metadata_gather_uses_explicit_cpu_group_not_tree_nccl():
    calls=[];group=object()
    def gather(output,value,*,group):
        calls.append((group,value));output[:]=[value]*16
    distributed=NS(all_gather_object=gather,get_backend=lambda g:'gloo' if g is group else 'nccl')
    rows=wan_worker.gather_metadata(distributed,{'identity':'fixture'},group)
    assert rows==[{'identity':'fixture'}]*16
    assert calls==[(group,{'identity':'fixture'})]


def test_metadata_gather_rejects_default_or_non_cpu_group_before_collective():
    import pytest
    def gather(*a,**kw):raise AssertionError('must not launch wrong collective')
    distributed=NS(all_gather_object=gather,get_backend=lambda g:'nccl')
    for group in (None,object()):
        with pytest.raises(ValueError,match='explicit Gloo'):
            wan_worker.gather_metadata(distributed,{},group)


def test_worker_routes_both_metadata_checks_and_cleans_group_outside_timer():
    source=inspect.getsource(wan_worker.run_worker)
    assert "new_group(backend='gloo'" in source
    assert source.index("new_group(backend='gloo'")<source.index('OriginalWanSession(')
    # Asset/cohort agreements remain unconditional in their owning path;
    # environment admission is an additional opt-in restart check.
    assert source.count('gather_metadata(')==3
    assert source.index('admit_run_controls(')<source.index('environments = gather_metadata(')
    assert 'dist.all_gather_object(' not in source
    assert 'destroy_process_groups(dist, metadata_group)' in source
    assert 'all_gather' not in inspect.getsource(wan_worker.execute_cycles)


def test_world_teardown_attempted_even_if_metadata_teardown_fails():
    import pytest
    calls=[];group=object()
    def destroy(*args):
        calls.append(args)
        if args:raise RuntimeError('metadata teardown failure')
    with pytest.raises(RuntimeError,match='metadata teardown failure'):
        wan_worker.destroy_process_groups(NS(destroy_process_group=destroy),group)
    assert calls==[(group,),()]


def test_restart_agreement_rejects_mismatched_intervals_before_branching():
    import pytest
    def gather(output, value, *, group):
        output[:] = [value] * 16
        output[8] = {**value, 'controls': {**value['controls'], 'checkpoint_every': 0}}
    group = object()
    dist = NS(all_gather_object=gather, get_backend=lambda g: 'gloo')
    with pytest.raises(ValueError, match='rank.*control'):
        wan_worker.admit_run_controls(dist, group, mode='train', cycles=4,
                                     checkpoint_every=1, resume_from=None)


def test_restart_agreement_compares_checkpoint_content_not_local_alias(tmp_path):
    from distillschedule.workloads.best_known.registry import file_sha256
    import pytest
    path = tmp_path / 'checkpoint'
    path.mkdir()
    (path / 'manifest.json').write_text('{"schema_version":"fixture-only"}')
    sha = file_sha256(path / 'manifest.json')
    (path / '_SUCCESS').write_text(sha + '\n')
    alias = tmp_path / 'alias'
    alias.symlink_to(path, target_is_directory=True)
    values = []
    def gather(output, value, *, group):
        values.append(value)
        output[:] = [value] * 16
    dist = NS(all_gather_object=gather, get_backend=lambda g: 'gloo')
    for root in (path, alias):
        wan_worker.admit_run_controls(dist, object(), mode='train', cycles=4,
                                     checkpoint_every=1, resume_from=root)
    assert values[0] == values[1]
    assert values[0]['controls']['resume_manifest_sha256'] == sha
    (path / '_SUCCESS').write_text('incorrect')
    with pytest.raises(ValueError, match='checkpoint.*complete'):
        wan_worker.admit_run_controls(dist, object(), mode='train', cycles=4,
                                     checkpoint_every=1, resume_from=path)


def test_completed_build_census_keeps_fresh_and_resumed_counts_separate():
    """Execute the real metadata-only branch; this is not GPU qualification."""
    import ast
    source = ast.parse(inspect.getsource(wan_worker.run_worker))
    branches = [node for node in ast.walk(source) if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name) and node.test.id == 'start_cycle']
    assert len(branches) == 1
    code = compile(ast.fix_missing_locations(ast.Module(body=branches, type_ignores=[])),
                   '<actual-native-build-census>', 'exec')
    for resumed, expected in ((0, 4), (1, 3), (3, 1)):
        calls = []
        strategy, build = object(), object()
        def validate(selected, observed, **kwargs):
            calls.append((selected, observed, kwargs))
            return {'status': 'CPU_TEST_ONLY'}
        receipt = {}
        exec(code, {'start_cycle': resumed, 'cycles': 4, 'strategy': strategy,
                    'session': NS(build=build), 'receipt': receipt, 'validate_build': validate})
        assert calls == [(strategy, build, {'executed': True, 'cycles': expected})]
        assert receipt['backend_effects'] == {'status': 'CPU_TEST_ONLY'}
