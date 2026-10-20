"""CPU-only control-plane collective regression; training NCCL remains untouched."""
from datetime import timedelta
from types import SimpleNamespace as NS
import ast,inspect
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
    assert source.count('gather_metadata(')==2
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
