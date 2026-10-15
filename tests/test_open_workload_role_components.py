"""Real native instrumentation, independent math and measured P2P; CPU evidence."""
from datetime import timedelta
from pathlib import Path
import json
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.role_program import RoleTask, PoolLayout, AllocatedWorld
from distillschedule.open_workload.role_transport import RoleFunctions, RolePoolExecutor
from distillschedule.open_workload.role_profiling import ComponentRecorder, profile_links
from test_open_workload_role_transport import (
    make_model, make_opt, sample, generate, guidance, score_train, independent_reference,
)


def test_saved_accounting_does_not_keep_extra_values_or_skip_version_check():
    model=torch.nn.Linear(3,4)
    x=torch.ones(2,3,requires_grad=True)
    recorder=ComponentRecorder('cpu',True)
    with recorder.saved(model,0),recorder.span('forward',0):
        y=model(x).square()
    assert recorder.saved_bytes['0']>=x.untyped_storage().nbytes()
    with torch.no_grad():x.add_(1)
    with pytest.raises(ValueError,match='mutated'):
        y.sum().backward()


def _worker(rank,world,rdzv,out):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,
                            world_size=world,timeout=timedelta(seconds=60))
    engine=None
    try:
        task=RoleTask(4,(2,4),score_updates=1)
        layout=PoolLayout(tuple(range(world//2)),tuple(range(world//2,world)),1)
        hw=AllocatedWorld((2**26,)*world)
        bindings=RoleFunctions(make_model,make_opt,sample,generate,guidance,score_train)
        links=profile_links(task,layout,'cpu',samples=2)
        assert all(len(t)==2 and min(t)>0 for t in links['samples_ns'].values())
        assert len(links['samples_ns'])==world
        engine=RolePoolExecutor(task,layout,hw,bindings,device='cpu',profile_components=True)
        receipt,model,opt=engine.run()
        reference=independent_reference(task)[engine.role]
        assert all(torch.allclose(v.detach(),reference[k],atol=1e-7,rtol=1e-6)
                   for k,v in model.state_dict().items())
        component=receipt['ranks'][rank]['components']
        assert component['enabled'] is True
        assert all(row['duration_ns']>0 for row in component['records'])
        actions=[r['action'] for r in component['records']]
        if engine.role=='generator':
            assert actions.count('forward')==4//len(layout.generator)
            assert actions.count('backward')==actions.count('forward')
            assert actions[-1]=='generator_commit'
            assert all(b>0 for b in component['saved_bytes_by_microbatch'].values())
        else:
            assert actions.count('score')==4//len(layout.score)
            assert actions[-1]=='score_updates'
        Path(out,f'{rank}.json').write_text(json.dumps(receipt))
    finally:
        if engine is not None:engine.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('world',[2,4])
def test_real_native_component_profile_and_directional_links(tmp_path,world):
    mp.spawn(_worker,args=(world,str(tmp_path/'rdzv'),str(tmp_path)),nprocs=world,join=True)
    assert all((tmp_path/f'{r}.json').is_file() for r in range(world))
