"""CPU contract proof only; real world16 source/capacity/numerics still required."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
from distillschedule.core.config import load_plan
from distillschedule.runtime import livewan_block_retention as feature
from distillschedule.runtime import livewan_crossscale_worker as worker
from distillschedule.runtime import livewan_replay_ablation as replay
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def plans(monkeypatch):
    for key in ('LIVEWAN_STUDENT_CHECKPOINT','LIVEWAN_CRITIC_CHECKPOINT','LIVEWAN_TEACHER_CHECKPOINT','LIVEWAN_OPERATION_TRACE','LIVEWAN_SYSTEMS_BOOTSTRAP','DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key,'/unread/'+key)
    path=ROOT/'configs/distillschedule/plans'
    return tuple(load_plan(path/f).plan.to_dict() for f in (replay.PARENT_FILE,feature.PLAN_FILE))


def test_only_checkpoint_policy_changes_real_lowering_and_installation(plans,monkeypatch):
    from distillschedule.runtime import livewan_score_fa3 as fa3
    a,b=plans
    assert not feature.enabled(a,mode='pipeline') and feature.enabled(b,mode='pipeline')
    assert feature.parent_plan(b,mode='pipeline')==a
    assert not b['metadata'].get('critic_ffn_down_checkpoint')
    assert not replay.enabled(b,mode='pipeline')
    replay.validate_effective_policy(b)
    for lower in (lambda p:worker._apply_trajectory_contract(p,cycles=2,mode='pipeline'),lambda p:worker._apply_two_cycle_timing_contract(p,mode='pipeline')):
        x,y=copy.deepcopy(a),copy.deepcopy(b);lower(x);lower(y)
        worker._validate_plan_identity(x,mode='pipeline');worker._validate_plan_identity(y,mode='pipeline')
        assert feature.parent_plan(y,mode='pipeline')==x
        assert worker._effective_plan_sha256(x)!=worker._effective_plan_sha256(y)
    sentinel=object();blocks=[object() for _ in range(30)];calls=[]
    monkeypatch.setattr(fa3,'load_flash_attn3',lambda:(sentinel,None))
    monkeypatch.setattr(feature,'CriticBlockRetentionProxy',lambda c,t,f,*,blocks:calls.append((c,t,f,blocks)) or sentinel)
    assert worker._build_critic_fa3_checkpoint_proxy(b,'checkpoint','torch',blocks=blocks) is sentinel
    assert calls==[('checkpoint','torch',sentinel,blocks)]
    with pytest.raises(RuntimeError):worker._build_critic_fa3_checkpoint_proxy(b,'checkpoint','torch')


@pytest.mark.parametrize('change',[
    lambda p:p['metadata']['critic_block_retention'].__setitem__('blocks',[25,26,27,28,29]),
    lambda p:p['metadata']['critic_block_retention'].__setitem__('cpu_offload',True),
    lambda p:p['metadata'].pop('critic_block_retention'),
    lambda p:p['metadata'].__setitem__('generator_rollout_cast_cache',None),
    lambda p:p['metadata'].__setitem__('critic_attention_checkpoint',replay.RECOMPUTE_CONTRACT),
    lambda p:p['workload']['shape'].__setitem__('batch_size',8),
    lambda p:p.__setitem__('id',replay.PLAN_ID),
])
def test_fail_closed_parent_or_child_drift(plans,change):
    _,p=copy.deepcopy(plans);change(p)
    with pytest.raises(RuntimeError):worker._validate_plan_identity(p,mode='pipeline')


def test_canonical_stays_strict_and_no_mislabelled_child(plans):
    a,b=plans
    with pytest.raises(RuntimeError):feature.enabled(b,mode='baseline')
    for field in ('generator_rollout_cast_cache','critic_attention_checkpoint','generator_cross_attention_kv_cache'):
        p=copy.deepcopy(a);p['metadata'].pop(field,None)
        with pytest.raises(RuntimeError):worker._validate_plan_identity(p,mode='pipeline')
    p=copy.deepcopy(a);p['metadata']['critic_block_retention']=copy.deepcopy(feature.CONTRACT)
    with pytest.raises(RuntimeError):worker._validate_plan_identity(p,mode='pipeline')


@pytest.mark.parametrize('pinned_census',[False,True])
def test_real_original_module_hooks_checkpoint_autograd_rng_adam(plans,monkeypatch,record_property,pinned_census):
    torch=pytest.importorskip('torch')
    import torch.utils.checkpoint as checkpoint
    from distillschedule.runtime import livewan_score_fa3 as fa3
    assert not torch.cuda.is_initialized()
    if pinned_census and not torch.__version__.startswith('2.10.'):
        pytest.skip('Production SAC replay counters require actual Torch2.10, not local2.14')
    # CPU operator substitution changes only the GPU-specific discovery/metadata.
    # The production factory, proxy, real SAC, real checkpoint and autograd execute.
    @torch.library.custom_op('livewan_retention_cpu::attention_'+str(pinned_census).lower(),mutates_args=())
    def attention(x:torch.Tensor)->torch.Tensor:return x.sin()
    @attention.register_fake
    def _(x):return torch.empty_like(x)
    def setup(ctx,inputs,output):ctx.save_for_backward(inputs[0])
    def backward(ctx,grad):return grad*ctx.saved_tensors[0].cos()
    attention.register_autograd(backward,setup_context=setup)
    op=getattr(torch.ops.livewan_retention_cpu,'attention_'+str(pinned_census).lower()).default
    monkeypatch.setattr(fa3,'discover_flash_forward',lambda _:(op,{'interface_sha256':fa3.INTERFACE_SHA256,'operator':fa3.OP_NAME.replace('::','.')+'.default'}))
    monkeypatch.setattr(fa3,'attention_metadata',lambda t,args,kw:(((2,19320,12,128),'CPU_FIXTURE'),)*3)
    monkeypatch.setattr(fa3,'load_flash_attn3',lambda:(object(),None))
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__();self.w=torch.nn.Parameter(torch.randn(6,6)*0.01)
        def forward(self,x):
            return x+0.1*(attention(x)@self.w)*torch.rand_like(x)
    def call_original_module(block,x):return block(x)
    results=[]
    for plan in plans:
        torch.manual_seed(5301)
        blocks=torch.nn.ModuleList([Block()for _ in range(30)])
        hooks=[0]*30
        for i,block in enumerate(blocks):block.register_forward_pre_hook(lambda m,args,i=i:hooks.__setitem__(i,hooks[i]+1))
        proxy=worker._build_critic_fa3_checkpoint_proxy(plan,checkpoint,torch,blocks=blocks)
        opt=torch.optim.AdamW(blocks.parameters(),lr=0.001)
        x=torch.randn(2,6,requires_grad=True)
        for block in blocks:x=proxy.checkpoint(call_original_module,block,x,use_reentrant=False)
        out=x.detach().clone();x.square().sum().backward();rng=torch.get_rng_state().clone()
        grads=[p.grad.clone()for p in blocks.parameters()];opt.step()
        states=[p.detach().clone()for p in blocks.parameters()]
        moments=[(opt.state[p]['exp_avg'].clone(),opt.state[p]['exp_avg_sq'].clone())for p in blocks.parameters()]
        receipt=proxy.receipt(expected_calls=30) if pinned_census else proxy.contexts.receipt()
        results.append((out,grads,rng,states,moments,receipt,hooks,proxy))
    a,b=results
    assert torch.equal(a[0],b[0]) and torch.equal(a[2],b[2])
    assert all(torch.equal(x,y)for x,y in zip(a[1],b[1]))
    assert all(torch.equal(x,y)for x,y in zip(a[3],b[3]))
    assert all(torch.equal(x,y)for p,q in zip(a[4],b[4])for x,y in zip(p,q))
    assert a[5]['checkpoint_calls']==30 and b[5]['checkpoint_calls']==26
    assert sum(b[7].direct_calls)==sum(b[7].backward_calls)==4
    assert b[6][26:]==[1]*4 and min(a[6][26:])>=2
    if pinned_census:
        assert b[5]['block_retention']['direct_retained_calls']==4
        assert b[5]['block_retention']['retained_backward_calls']==4
        bad=copy.deepcopy(b[5]);bad['block_retention']['retained_backward_calls']=0
        assert feature.runtime_receipt_errors(bad,rank=8,expected_calls=30)
    record_property('forward_gradient_rng_parameters_adam_equal',True)
    record_property('actual_control_checkpoint_calls',30)
    record_property('actual_candidate_checkpoint_calls',26)
    record_property('actual_candidate_direct_calls',4)
    record_property('CUDA_initialized',torch.cuda.is_initialized())
    assert not torch.cuda.is_initialized()


def test_transport_and_timing_node_gate_rejects_wrong_candidate(tmp_path):
    from test_refactor_livewan_replay_ablation import _header_fixture,_node_timing_header
    path=tmp_path/'SYNTHETIC_TEST_ONLY_gate.json';gate=_header_fixture(True)
    path.write_text(json.dumps(gate))
    assert _node_timing_header(path,variant=feature.VARIANT,mode='pipeline').returncode!=0
    gate['arms']['candidate']['pipeline_variant']=feature.VARIANT;path.write_text(json.dumps(gate))
    assert _node_timing_header(path,variant=feature.VARIANT,mode='pipeline').returncode==0
    assert _node_timing_header(path,variant=replay.VARIANT,mode='pipeline').returncode!=0
    p=ROOT/'scripts/distillschedule/probe_livewan_crossscale_transport.py'
    spec=importlib.util.spec_from_file_location('retention_transport',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.build_parser().parse_args(['--output-dir',str(tmp_path),'--pipeline-variant',feature.VARIANT]).pipeline_variant==feature.VARIANT
