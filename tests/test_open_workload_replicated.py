"""Real two-process Gloo tests; no CUDA/GPU performance is inferred."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.replica_contract import ReplicaSemantics
from distillschedule.open_workload.replicated import inventory, make_session, identity
from distillschedule.open_workload.torch_backend import Session, candidates, compare
from test_open_workload_execution import build


def worker(rank, filename, output, count, width, length):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+filename,rank=rank,world_size=2,timeout=timedelta(seconds=35))
    try:
        w=build(n=count,width=width,length=length)
        w.feeds={'x':w.feeds['x']+rank*.2}
        w.replica_semantics=ReplicaSemantics({'forward':'learner'},tuple(f'b{i}' for i in range(count)),length)
        initial=w.identity
        rows=identity(w)
        assert rows['world_size']==2
        # An independent one-process GLOBAL-batch reference is more than comparing
        # two distributed implementations which might share a normalization bug.
        global_w=build(n=count,width=width,length=length)
        global_w.feeds={'x':torch.cat([global_w.feeds['x'],global_w.feeds['x']+.2])}
        ref=Session(global_w,candidates(global_w,limit=1)['candidates'][0]);out=ref.run()
        global_state=ref.result_state(out)['state']
        reference=None;effects=[]
        for impl in inventory(w,limit=12)['candidates']:
            s=make_session(w,impl);value=s.run(measure=True)
            state=s.result_state(value)
            assert compare(global_state,state['state'],rtol=2e-5,atol=2e-7)['status']=='PASS'
            if reference is None: reference=state
            assert compare(reference,state,rtol=2e-5,atol=2e-7)['status']=='PASS'
            assert s.effects['gradient_collectives']>0 and s.effects['replica_state_consensus']
            if impl['local']['share_immutable']:
                a=s.state['learner'];a=a.module if hasattr(a,'module') else a
                assert a.base.weight is s.state['guide'].base.weight
            effects.append(s.effects)
            del s,value
        assert w.identity==initial
        Path(output,f'rank{rank}.json').write_text(json.dumps({'effects':effects,'gpu_execution':'NOT_RUN'}))
    finally: dist.destroy_process_group()


@pytest.mark.parametrize('count,width,length',[(1,4,3),(2,8,5)])
def test_unknown_model_distributed_variants_match_global_batch(tmp_path,count,width,length):
    mp.spawn(worker,args=(str(tmp_path/'rdzv'),str(tmp_path),count,width,length),nprocs=2,join=True)
    rows=[json.loads((tmp_path/f'rank{r}.json').read_text()) for r in range(2)]
    assert all(len(r['effects'])==12 for r in rows)
    assert any(e['recompute_calls'] for e in rows[0]['effects'])
    assert any(e['reduction']=='ddp_overlap' for e in rows[0]['effects'])
