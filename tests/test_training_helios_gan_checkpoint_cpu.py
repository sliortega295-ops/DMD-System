"""W3 CPU component evidence only: never CUDA/topology/FA3-kernel admission."""
from contextlib import ExitStack
from copy import deepcopy
import ast
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

from test_training_helios_gan_native_history import public_config, task
from helios_reference_fixtures import frozen_assets as frozen_assets, original_logical as original_logical

PRESET = 'helios_gan.native_history_fa3_down40_checkpoint_cpu.v1'


def config(tmp_path):
    old = public_config(tmp_path)
    raw = old.to_dict()
    raw['execution'].update(preset=PRESET, capacity_policy=str(tmp_path/'capacity.json'))
    return type(old).from_dict(raw, source=old.source)


def source_functions(*names):
    """Execute unmodified original class/function AST, not replacement math."""
    from distillschedule.workloads.best_known.registry import repository_root
    path = repository_root()/'third_party/helios_compat/source/helios/utils/utils_helios_post.py'
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    namespace = dict(torch=torch, F=torch.nn.functional, free_memory=lambda:None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0), *nodes], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4,4,dtype=torch.float64)*.1)
        # Metadata-only frozen down owner required by REAL NativeDownSAC.bind.
        # Never executed or presented as native capacity evidence.
        self.ffn = torch.nn.Module(); self.ffn.net = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity(), torch.nn.Module()])
        self.ffn.net[2].base_layer = torch.nn.Linear(13824,5120,device='meta')
        self.ffn.net[2].base_layer.requires_grad_(False)
        self.calls = 0
    def forward(self, hidden):
        self.calls += 1
        return hidden + torch.nn.functional.dropout(torch.sin(hidden @ self.weight), .1, training=True)*.1


class UnusedLoRA(torch.nn.Module):
    def forward(self,x):
        return x


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(TinyBlock() for _ in range(40))
        self.gradient_checkpointing = True
        self.gan_heads=torch.nn.ModuleDict({k:torch.nn.Linear(4,1,dtype=torch.float64) for k in ('5','15','25','35')})
        self.gan_final_head=torch.nn.Linear(4,1,dtype=torch.float64)
        self.unused_lora=UnusedLoRA()
        # These original auxiliary boundaries precede the 40 block checkpoints.
        # Identity keeps this component fixture small; calls below come from
        # the pinned production process_input_hidden_states method itself.
        for name in ('patch_embedding','patch_short','patch_mid','patch_long'):
            self.add_module(name,torch.nn.Identity())
    def forward(self, *, hidden_states, gan_mode=False, generator_hidden=False, **kwargs):
        hidden_states=original_patch_checkpoints(self,hidden_states)
        x = hidden_states if generator_hidden else hidden_states.reshape(1,-1,1).expand(-1,-1,4).contiguous()
        for block in self.blocks:
            x = self._gradient_checkpointing_func(block,x) if torch.is_grad_enabled() else block(x)
        if generator_hidden:return x
        logits=[self._gradient_checkpointing_func(head,x) for head in (*self.gan_heads.values(),self.gan_final_head)]
        return x, torch.cat(logits,dim=2).mean(dim=1)


def original_patch_checkpoints(owner,hidden):
    """Execute original dispatch expressions, not a mirrored checkpoint list."""
    from distillschedule.workloads.best_known.registry import repository_root
    path=repository_root()/'third_party/helios_compat/source/helios/modules/transformer_helios.py'
    cls=next(n for n in ast.parse(path.read_text()).body
        if isinstance(n,ast.ClassDef) and n.name=='HeliosTransformer3DModel')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)
        and n.name=='process_input_hidden_states')
    expressions=[n for n in ast.walk(method)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='gradient_checkpointing_method' and len(n.args)==2
        and isinstance(n.args[0],ast.Attribute) and isinstance(n.args[1],ast.Name)]
    # Use the actual tensor branch, not NaViT's separate per-item .to() branch.
    calls={n.args[0].attr:n for n in expressions}
    assert len(expressions)==len(calls)==4
    assert set(calls)=={'patch_embedding','patch_short','patch_mid','patch_long'}
    dispatch=next(n for n in cls.body if isinstance(n,ast.FunctionDef)
        and n.name=='gradient_checkpointing_method')
    namespace={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[dispatch],type_ignores=[])),str(path),'exec'),namespace)
    environment=dict(self=owner,latents=hidden,latents_history_short=hidden,
        latents_history_mid=hidden,latents_history_long=hidden)
    # Bind the original method on a separate proxy; never mutate owner state.
    from types import SimpleNamespace,MethodType
    proxy=SimpleNamespace(gradient_checkpointing=owner.gradient_checkpointing,
        _gradient_checkpointing_func=owner._gradient_checkpointing_func,
        **{name:getattr(owner,name) for name in calls})
    proxy.gradient_checkpointing_method=MethodType(namespace['gradient_checkpointing_method'],proxy)
    environment['self']=proxy
    for call in sorted(calls.values(),key=lambda n:n.lineno):
        result=eval(compile(ast.Expression(body=call),str(path),'eval'),environment)
        assert result.shape==hidden.shape
    return result


@pytest.fixture
def bound_original(monkeypatch):
    """Actual configure + NativeDownStreamSAC.bind with explicit CPU gate substitutions.

    Only pinned CUDA-version/source admission is replaced, because this is CPU
    Torch2.14. The original bind, checkpoint factories and closures execute.
    No substitution is usable as a production CUDA/source admission receipt.
    """
    import torch.utils.checkpoint as checkpoint
    from distillschedule.runtime import fsdp2_static as fsdp
    from distillschedule.runtime.helios_gan_manual.stream import NativeDownStreamSAC
    from distillschedule.runtime.helios_gan_manual.down_contract import FA3
    functions = source_functions('_critic_loss', 'Gan_D_Loss_With_Cached_Grad', 'cal_gan_loss')
    module = ModuleType('helios.utils.utils_helios_post'); module.__dict__.update(functions)
    monkeypatch.setitem(sys.modules,'helios.utils.utils_helios_post',module)
    generator,critic = NS(module=TinyModel(),global_steps=2),NS(module=TinyModel(),global_steps=10)
    from distillschedule.runtime.helios_lora_epilogue import install_generator_nograd_lora_fused_epilogue
    for role,engine in (('generator',generator),('critic',critic)):
        install_generator_nograd_lora_fused_epilogue(engine.module,enabled=True,role=role,
            linear_type=UnusedLoRA,variant_kwarg_keys=(),fused_epilogue=lambda base,update,scale:base+update*scale,
            all_role_scale1=True,checkpoint_replay_scope=True)
    generator.backward=lambda loss:loss.backward()
    critic.backward=lambda loss:loss.backward()
    critic.step=lambda:None
    fsdp.configure_fsdp2_checkpointing(generator.module)
    fsdp.configure_fsdp2_checkpointing(critic.module,selective_flash_checkpoint_stride=2)
    captured = generator.module._gradient_checkpointing_func
    core=NativeDownStreamSAC(torch=torch,rank=0)
    monkeypatch.setattr(core,'validate_sources',lambda *a:None)
    monkeypatch.setattr(torch,'__version__','2.10.0+cu128')
    monkeypatch.setattr(fsdp,'SELECTIVE_FLASH_ATTN_OP',FA3)
    p0=NS(_run_gan_critic_precomputed=lambda **kwargs:None)
    with ExitStack() as stack:
        core.bind(stack,p0=p0,native=NS(),fsdp=fsdp,checkpoint=checkpoint,
                  generator=generator,critic=critic,ema=NS(optimization_step=800))
        core.cohort=0  # Explicit CPU fresh-cohort fixture, not qualify_rows admission.
        yield NS(generator=generator,critic=critic,core=core,fsdp=fsdp,checkpoint=checkpoint,
                 captured=captured,functions=functions)
    core.restored()


def test_old_capacity_and_original_callable_refusal_are_characterized(bound_original):
    from distillschedule.runtime.helios_checkpoint_inputs_cpu import _original_checkpoint
    b=bound_original
    assert b.captured.func is not b.fsdp.fsdp2_metadata_tolerant_checkpoint
    assert b.core.current is None and not b.core.stores
    with pytest.raises(ValueError,match='exact original checkpoint callable'):
        _original_checkpoint(b.generator.module)
    # R322's exact named-owner optimistic rank-zero base + G minimum + C40.
    base=31_127_740_544+1_788_898_568+3*(312_860_672+558_667_648)
    old_minimum=base+4_521_984_000+1_867_776_000
    assert old_minimum > 38.89385986328125*2**30
    assert old_minimum-int(38.89385986328125*2**30)==159_020_040


def test_new_policy_is_explicit_and_default_w3_still_rejects_capacity(tmp_path):
    from distillschedule.training import native, helios_gan_checkpoint_cpu as policy
    conf=config(tmp_path)
    assert native._options(conf,'W3')=='manual'
    assert policy.capacity_template()['rank_pageable_limit_bytes']==7_593_984_000
    assert policy.critic_boundary_contract()['hidden_input_bytes']==1_867_776_000
    assert native.describe_native(conf)['capabilities']['checkpoint_input_cpu']['gpu_capacity']=='NOT_RUN'
    raw=conf.to_dict();raw['execution']['preset']='helios_gan.fa3_down40_nomerge_stream.v1'
    with pytest.raises(ValueError):native._options(type(conf).from_dict(raw,source=conf.source),'W3')
    raw=conf.to_dict();raw['data']['input_kind']='original_helios_gan_cohort_stream.v1'
    with pytest.raises(ValueError):native._options(type(conf).from_dict(raw,source=conf.source),'W3')


def test_owned_composition_authenticates_captured_g_and_true_selective_c(bound_original):
    from distillschedule.runtime.helios_gan_checkpoint_cpu import authenticate_callables
    b=bound_original
    result=authenticate_callables(b.core)
    assert result['generator'] is b.captured
    assert result['critic'] is b.critic.module._gradient_checkpointing_func
    assert result['original'] is b.captured.func
    assert b.core.current is None and not b.core.stores


@pytest.fixture
def tiny_w3(monkeypatch):
    from distillschedule.training import helios_gan_checkpoint_cpu as policy
    spec=dict(schema_version='CPU_SYNTHETIC_TEST_ONLY',shapes=[[1,3,4],[1,5,4],[1,7,4]],
        dtype='torch.float64',checkpoint_blocks=40,max_connected_grad_stages=3)
    c={**policy.critic_boundary_contract(), 'latent_shape':[1,1,1,1,5],
        'hidden_shape':[1,5,4],'dtype':'torch.float64','selective_FA3_results':20}
    monkeypatch.setattr(policy,'native_contract',lambda:deepcopy(spec))
    monkeypatch.setattr(policy,'critic_boundary_contract',lambda:deepcopy(c))
    # EXPLICIT_CPU_LIBRARY_LAYOUT_SUBSTITUTION_ONLY: native2.10 requires empty
    # FIFO queues + actual replay callbacks, unchanged. Torch2.14 instead records
    # indexed _CONSUMED/_RECOMPUTE sentinels and never calls replay policy_fn.
    import torch.utils.checkpoint as cp
    if hasattr(cp,'_CONSUMED'):
        from distillschedule.runtime import helios_gan_checkpoint_cpu as runtime
        def cpu214_drained(stores,delta,expected):
            return delta==[expected,0] and all(type(q) is dict and all(
                item is cp._CONSUMED or item is cp._RECOMPUTE for item in q.values())
                for store in stores for q in store.values())
        monkeypatch.setattr(runtime,'_selective_drained',cpu214_drained)
    return policy


def run_action(b,exits,registration=None,engine=None):
    from contextlib import nullcontext
    g,c=b.generator.module,b.critic.module
    wrapper_before=[root._distillschedule_helios_nograd_lora_epilogue['checkpoint_scope_execution_count'] for root in (g,c)]
    optimizer=engine.optimizer if engine is not None else torch.optim.AdamW([p for p in g.parameters() if p.device.type=='cpu'],lr=.001)
    optimizer.zero_grad(set_to_none=True)
    x=torch.randn(1,3,4,dtype=torch.float64,requires_grad=True)
    with registration if registration is not None else nullcontext():
        for stage,tokens in enumerate((3,5,7)):
            x=torch.nn.functional.interpolate(x.transpose(1,2),size=tokens,mode='linear',align_corners=False).transpose(1,2).contiguous()
            for _ in range(exits[stage]):
                with torch.no_grad():
                    x=g(hidden_states=x,generator_hidden=True)
            x=g(hidden_states=x,generator_hidden=True)
        x0=torch.nn.functional.adaptive_avg_pool1d(x.transpose(1,2),5).mean(dim=1).reshape(1,1,1,1,5)
        leaf=x0.detach().requires_grad_(True)
        loss=b.functions['Gan_D_Loss_With_Cached_Grad'].apply(leaf,c,None,None,None,None,None,None,None,None,None,-1)
        grad=torch.autograd.grad(loss,leaf)[0].detach()
        surrogate=(x0*grad).sum()
        (engine or b.generator).backward(surrogate)
        grads={n:None if p.grad is None else p.grad.clone() for n,p in g.named_parameters() if p.device.type=='cpu'}
        engine.step() if engine is not None else optimizer.step()
    return dict(output=x0.detach(),loss=loss.detach(),grads=grads,
        parameters={n:p.detach().clone() for n,p in g.named_parameters() if p.device.type=='cpu'},
        adam=[{k:v.clone() if torch.is_tensor(v) else v for k,v in state.items()} for state in optimizer.state.values()],
        rng=torch.get_rng_state(),g_calls=[block.calls for block in g.blocks],c_calls=[block.calls for block in c.blocks],
        replay_wrappers=tuple(root._distillschedule_helios_nograd_lora_epilogue['checkpoint_scope_execution_count']-before
            for root,before in zip((g,c),wrapper_before)),c_grads=[p.grad for p in c.parameters()])


@pytest.mark.parametrize('exits,restores',[((0,1,1),40),((0,1,0),80),((0,0,0),120)])
def test_real_checkpoint_owned_sac_cached_gan_grad_and_all_g_state_exact(bound_original,tiny_w3,monkeypatch,exits,restores):
    from distillschedule.runtime.helios_gan_checkpoint_cpu import GAction
    from test_helios_step_owner_cpu import equal
    b=bound_original
    # Actual public SAC engine saving sin on CPU, NOT a FA3 kernel substitute.
    monkeypatch.setattr(b.fsdp,'SELECTIVE_FLASH_ATTN_OP','aten.sin.default')
    state=deepcopy({n:p.detach().clone() for n,p in b.generator.module.named_parameters() if p.device.type=='cpu'})
    rng=torch.get_rng_state().clone()
    expected=run_action(b,exits)
    with torch.no_grad():
        for n,p in b.generator.module.named_parameters():
            if p.device.type=='cpu':p.copy_(state[n])
    for root in (b.generator.module,b.critic.module):
        for block in root.blocks:block.calls=0
    torch.set_rng_state(rng)
    identity=dict(source='CPU_COMPONENT_ONLY',task_sha256=task().digest,generator_version=2,
                  critic_version=10,cohort_index=0,rank=0)
    registration=GAction(b.core,capacity=tiny_w3.capacity_template(),identity=identity,device='cpu')
    actual=run_action(b,exits,registration)
    assert equal(actual,expected)
    stats=registration.session.receipt()['stats']
    assert stats['cpu_copy_count']==stats['release_count']==120
    assert stats['restore_count']==stats['restored_release_count']==restores
    assert stats['unused_release_count']==120-restores
    assert stats['live_cpu_bytes']==stats['live_leases']==stats['cancel_count']==0
    assert registration.boundary.receipt()['original_callable_restored'] is True
    assert registration.boundary.counts['patch_calls']==4
    assert registration.boundary.patches==tiny_w3.critic_boundary_contract()['patch_modules']
    assert registration.boundary.counts['drained_before_g_backward'] is True


@pytest.mark.parametrize('mutation',['missing','duplicate','order','foreign','all_missing'])
def test_original_patch_checkpoint_owners_and_cleanup_are_required(bound_original,tiny_w3,monkeypatch,mutation):
    from distillschedule.runtime.helios_gan_checkpoint_cpu import GAction
    b=bound_original
    monkeypatch.setattr(b.fsdp,'SELECTIVE_FLASH_ATTN_OP','aten.sin.default')
    original=original_patch_checkpoints
    def invalid(owner,hidden):
        if owner is not b.critic.module:return original(owner,hidden)
        names=list(tiny_w3.critic_boundary_contract()['patch_modules'])
        if mutation=='missing':names.remove('patch_short')
        elif mutation=='duplicate':names.insert(1,'patch_embedding')
        elif mutation=='order':names[0],names[1]=names[1],names[0]
        elif mutation=='all_missing':names=[]
        if mutation=='foreign':owner._gradient_checkpointing_func(torch.nn.Identity(),hidden)
        for name in names:owner._gradient_checkpointing_func(getattr(owner,name),hidden)
        return hidden
    monkeypatch.setattr(sys.modules[__name__],'original_patch_checkpoints',invalid)
    identity=dict(source='CPU_COMPONENT_ONLY',task_sha256=task().digest,generator_version=2,
                  critic_version=10,cohort_index=0,rank=0)
    registration=GAction(b.core,capacity=tiny_w3.capacity_template(),identity=identity,device='cpu')
    with pytest.raises(ValueError,match='W3 .*checkpoint'):
        run_action(b,(0,1,1),registration)
    stats=registration.session.receipt()['stats']
    assert stats['live_cpu_bytes']==stats['live_leases']==0
    assert registration.boundary.receipt()['original_callable_restored'] is True
    assert b.generator.module._gradient_checkpointing_func is b.captured


@pytest.mark.parametrize('mutation',['source','wrapped_dispatch','wrapped_g','wrong_replay','ambiguous_partial','c_selector','stale_c_block','crossed_role','reused_graph'])
def test_actual_composition_rejects_wrong_source_callable_role_and_stale_owner(bound_original,monkeypatch,mutation):
    from functools import wraps,partial
    from distillschedule.runtime import helios_gan_checkpoint_cpu as runtime
    b=bound_original
    dispatch_before=b.fsdp.fsdp2_metadata_tolerant_checkpoint
    def wrapped(original):
        @wraps(original)
        def call(*a,**kw):return original(*a,**kw)
        return call
    if mutation=='source':monkeypatch.setattr(runtime,'_SOURCE',{**runtime._SOURCE,next(iter(runtime._SOURCE)):'0'*64})
    elif mutation=='wrapped_dispatch':b.fsdp.fsdp2_metadata_tolerant_checkpoint=wrapped(b.fsdp.fsdp2_metadata_tolerant_checkpoint)
    elif mutation=='wrapped_g':b.generator.module._gradient_checkpointing_func=wrapped(b.captured)
    elif mutation=='wrong_replay':b.generator.module._distillschedule_lora_checkpoint_function_wrapper=lambda f:f
    elif mutation=='ambiguous_partial':b.generator.module._gradient_checkpointing_func=partial(b.captured,operation_source=True)
    elif mutation=='c_selector':b.critic.module._gradient_checkpointing_func=wrapped(b.critic.module._gradient_checkpointing_func)
    elif mutation=='stale_c_block':b.critic.module.blocks[0]=TinyBlock()
    elif mutation=='crossed_role':b.core.current=dict(role='critic')
    else:b.core.stores['stale']='graph'
    try:
        with pytest.raises(ValueError):runtime.authenticate_callables(b.core)
    finally:
        b.core.current=None;b.core.stores.clear()
        if mutation=='wrapped_dispatch':b.fsdp.fsdp2_metadata_tolerant_checkpoint=dispatch_before


@pytest.mark.parametrize('mutation',['dtype','shape','capacity','nested','retained_graph','extra_C_forward','wrong_C_role','unbackwarded_C','crossed_window','version','inplace_alias'])
def test_component_error_cleanup_refuses_without_leases_or_callable_residue(bound_original,tiny_w3,monkeypatch,mutation):
    from distillschedule.runtime.helios_gan_checkpoint_cpu import GAction
    b=bound_original
    monkeypatch.setattr(b.fsdp,'SELECTIVE_FLASH_ATTN_OP','aten.sin.default')
    identity=dict(source='CPU_COMPONENT_ONLY',task_sha256=task().digest,generator_version=2,critic_version=10,cohort_index=0,rank=0)
    registration=GAction(b.core,capacity=tiny_w3.capacity_template(),identity=identity,device='cpu')
    original_c=b.critic.module._gradient_checkpointing_func
    with pytest.raises((ValueError,RuntimeError)):
        with registration:
            x=torch.randn(1,3,4,dtype=torch.float64,requires_grad=True)
            if mutation=='dtype':x=x.float()
            elif mutation=='shape':x=x[:,:2]
            elif mutation=='capacity':registration.session.budget.limit=0
            if mutation=='crossed_window':b.core.current={'scope':'wrong'}
            elif mutation=='version':b.critic.global_steps+=1
            if mutation=='crossed_window':b.generator.backward(x.sum())
            elif mutation=='version':
                b.critic.module(hidden_states=torch.randn(1,1,1,1,5,dtype=torch.float64,requires_grad=True),gan_mode=True)
            else:
                y=b.generator.module._gradient_checkpointing_func(b.generator.module.blocks[0],x)
                if mutation=='nested':
                    with registration:pass
                elif mutation=='inplace_alias':
                    x.detach().add_(1);y.sum().backward()
                elif mutation in ('extra_C_forward','wrong_C_role','unbackwarded_C'):
                    c_input=torch.randn(1,1,1,1,5,dtype=torch.float64,requires_grad=True)
                    b.critic.module(hidden_states=c_input,gan_mode=mutation!='wrong_C_role')
                    if mutation=='extra_C_forward':b.critic.module(hidden_states=c_input,gan_mode=True)
                    else:b.generator.backward(y.sum())
                # retained_graph intentionally exits before backward.
    b.core.current=None
    assert b.generator.module._gradient_checkpointing_func is b.captured
    assert b.critic.module._gradient_checkpointing_func is original_c
    receipt=registration.session.receipt()
    assert receipt['stats']['live_cpu_bytes']==receipt['stats']['live_leases']==0
    assert receipt['original_callable_restored'] is True


def binding_for(tmp_path):
    from distillschedule.training import helios_gan_checkpoint_cpu as policy
    from distillschedule.runtime.helios_gan_native_history import descriptor
    cap=policy.capacity_template();(tmp_path/'capacity.json').write_text(__import__('json').dumps(cap))
    identity=dict(schema_version=policy.IDENTITY,source_head='a'*40,source_fingerprint='b'*64,
        stream_sha256='c'*64,checkpoint_input_capacity=cap,gan_native_history=descriptor(),
        execution_policy=policy.descriptor(),verified_immutable_assets={})
    return dict(config=config(tmp_path),identity=identity,gan_stream=True,owns_process_group=True,metadata_group=object(),
        implementation=policy.implementation(),effective_configuration=policy.descriptor(),
        stream=dict(stream_sha256='c'*64,identity=dict(source_fingerprint='b'*64)))


@pytest.mark.parametrize('mutation',['capacity','source','version','w2_schema','old_w3_schema','stream','rank','unbounded','layout'])
def test_explicit_policy_capacity_startup_identity_and_old_checkpoint_fallback_refuse(tmp_path,monkeypatch,mutation):
    from distillschedule.training import helios_gan_checkpoint_cpu as policy,helios_gan_training as training,helios_lifecycle as lifecycle,host_memory
    b=binding_for(tmp_path)
    observed=[]
    monkeypatch.setattr(host_memory,'admit_host_memory',lambda n,r:observed.append((n,r)) or {'scope':'CPU_COMPONENT_ONLY'})
    with lifecycle.original_training_scope(b):
        policy.admit_startup(b,rank=0)
        assert policy.require_binding() is b
        assert observed==[(8*7_593_984_000,512*2**30)]
        assert training._checkpoint_policy(b['identity'])['schema_version']==policy.CHECKPOINT
        if mutation=='capacity':b['identity']['checkpoint_input_capacity']['rank_pageable_limit_bytes']+=1
        elif mutation=='source':b['identity']['source_head']='0'*40
        elif mutation=='version':b['identity']['execution_policy']['schema_version']='old'
        elif mutation=='w2_schema':b['identity']['schema_version']='distillschedule.helios_checkpoint_hidden_cpu_identity.v1'
        elif mutation=='old_w3_schema':b['identity']['schema_version']='distillschedule.helios_gan_native_history_training_identity.v1'
        elif mutation=='stream':b['stream']['stream_sha256']='e'*64
        elif mutation=='rank':b['checkpoint_input_startup']['rank']=16
        elif mutation=='layout':b['identity']['native_owner_layout']=dict(topology={'mesh_dim_names':['replicate','shard']},trainables_sha256='d'*64)
        else:b['identity']['checkpoint_input_capacity']['storage']='unbounded'
        with pytest.raises(ValueError):policy.require_binding()
        if mutation in ('w2_schema','old_w3_schema'):
            with pytest.raises(ValueError):training._checkpoint_policy(b['identity'])


def test_captured_replay_pointer_alone_cannot_authenticate_a_new_fake_wrapper(bound_original):
    from functools import partial
    from distillschedule.runtime.helios_gan_checkpoint_cpu import authenticate_callables
    b=bound_original
    fake=lambda fn:fn
    b.generator.module._distillschedule_lora_checkpoint_function_wrapper=fake
    b.generator.module._gradient_checkpointing_func=partial(b.captured.func,function_wrapper=fake)
    with pytest.raises(ValueError,match='replay'):
        authenticate_callables(b.core)


def original_parser():
    """Actual two original parser bodies; do not import the optional model stack."""
    import argparse
    from distillschedule.workloads.best_known.registry import repository_root
    from distillschedule.profiling.fsdp2_operation_source import DEFAULT_TARGET_BYTES
    root=repository_root()
    namespace=dict(argparse=argparse,Path=Path,DEFAULT_TARGET_BYTES=DEFAULT_TARGET_BYTES)
    for relative,name in (('probe_static_fsdp2_full5_world16.py','parser'),('probe_static_fsdp2_p0_world16.py','_parser')):
        path=root/'scripts/distillschedule'/relative
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),namespace)
        if name=='parser':namespace['base']=NS(parser=namespace['parser'])
    return namespace['_parser']()


def test_public_metadata_actual_original_parser_worker_identity_and_cpu_first_capability(tmp_path,frozen_assets,original_logical,monkeypatch):
    """Metadata producer + actual public worker, explicit CPU construction boundary.

    No metadata shard is read as tensor/GPU evidence. The isolated source-dirty
    observation and host admission are explicitly substituted for this CPU test.
    """
    from contextlib import nullcontext
    from dataclasses import fields
    from test_training_helios_gan_native_history import native_metadata_stream
    import test_training_helios_gan_stream as fixtures
    from distillschedule.training import native,helios_gan_training as training,helios_gan_inputs as inputs
    from distillschedule.training import helios_gan_checkpoint_cpu as policy,helios_worker as worker,helios_lifecycle as lifecycle,host_memory
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    assets=fixtures.gan_assets(tmp_path,frozen_assets,original_logical)
    old=fixtures.metadata_stream
    monkeypatch.setattr(fixtures,'metadata_stream',lambda root,cycles=2:old(root,cycles,config_sha=assets['assets']['gan_config']['sha256']))
    doc=native_metadata_stream(tmp_path/'inputs')
    conf=config(tmp_path)
    raw=conf.to_dict();raw['data']['prepared']=str(tmp_path/'inputs/stream.json');raw['execution'].update(task=str(tmp_path/'task.json'),assets=str(tmp_path/'assets.json'))
    conf=type(conf).from_dict(raw,source=conf.source)
    (tmp_path/'task.json').write_text(json.dumps(inputs.REQUEST.to_dict()))
    (tmp_path/'assets.json').write_text(json.dumps(assets))
    (tmp_path/'capacity.json').write_text(json.dumps(policy.capacity_template()))
    source=native._source(conf);source['identity']['dirty']=False
    monkeypatch.setattr(native,'_source',lambda _:deepcopy(source))
    workload,observed,paths,admitted=native._admit(conf)
    assert workload=='W3' and observed==source and admitted['selected']==policy.descriptor()
    assert admitted['stream']==doc and admitted['checkpoint_input_capacity']==policy.capacity_template()
    parsed=original_parser().parse_args(admitted['argv']);parsed._planner_gan_manual_bundle=True
    assert parsed.variant=='gan' and parsed.mesh_policy=='world16-fsdp' and parsed.selective_flash_checkpoint_stride==2
    assert all(getattr(parsed,k) is v for k,v in training.effective_flags().items())
    physical=StaticFSDP2P0Policy(**{f.name:getattr(parsed,f.name) for f in fields(StaticFSDP2P0Policy)})
    report=native._report(conf,workload,observed,paths,admitted)
    assert report['checkpoint_input_capacity']==policy.capacity_template()
    monkeypatch.setattr(training.distributed,'world16_reference_process_group',lambda label:nullcontext(dict(owns_process_group=True,metadata_group=object())))
    monkeypatch.setattr(lifecycle,'admit_original_output',lambda **kw:None)
    observed_host=[]
    monkeypatch.setattr(host_memory,'admit_host_memory',lambda n,r:observed_host.append((n,r)) or {'scope':'CPU_COMPONENT_ONLY_NOT_RESERVATION'})
    monkeypatch.setenv('RANK','0')
    def main():
        b=lifecycle.current_training();identity=b['identity']
        assert identity['schema_version']==policy.IDENTITY and identity['checkpoint_input_capacity']==policy.capacity_template()
        assert training.bind_effective(args=parsed,policy=physical,native_policy=task().native_policy)==policy.descriptor()
        identity['verified_immutable_assets']={};b['effective_configuration']=policy.descriptor()
        policy.admit_startup(b,rank=0)
        lifecycle._bind_native_history_construction(policy=physical)
        assert lifecycle.native_history_cpu_first_construction(policy=physical)
        assert training._checkpoint_policy(identity)['schema_version']==policy.CHECKPOINT
        return 23
    monkeypatch.setattr(worker,'_backend',lambda:NS(main=main))
    assert training.run_training(conf,admit=lambda:(workload,observed,paths,admitted))==23
    assert observed_host==[(60_751_872_000,512*2**30)] and lifecycle.current_training() is None


@pytest.mark.parametrize('preloaded_cuda_namespace',[False,True])
def test_cold_public_discovery_uses_no_torch_or_source_bound_costs(tmp_path,preloaded_cuda_namespace):
    import os,subprocess
    root=Path(__file__).resolve().parents[1]
    script='''
import sys
from types import ModuleType
if sys.argv[2]=='1':sys.modules.setdefault('cuda',ModuleType('cuda'))
before=set(sys.modules)
from pathlib import Path
from distillschedule.training.config import TrainingConfig
from distillschedule.training.native import describe_native
from distillschedule.search.helios_gan_bundle import HeliosGANBundle
def forbid(*a,**kw):raise AssertionError('source costs forbidden in metadata')
HeliosGANBundle.source_identity=forbid
p=Path(sys.argv[1])/'recipes/helios_gan/training.native-history-checkpoint-cpu.json'
c=TrainingConfig.from_dict(__import__('json').loads(p.read_text()),source=p)
assert describe_native(c)['capabilities']['checkpoint_input_cpu']['gpu_capacity']=='NOT_RUN'
assert not any(n=='torch' or n.startswith(('torch.','triton','cuda.','diffusers','transformers')) for n in sys.modules)
assert 'cuda' not in set(sys.modules)-before
'''
    result=subprocess.run([sys.executable,'-B','-c',script,str(root),'1' if preloaded_cuda_namespace else '0'],env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


def native_receipt_row(exits=(0,1,1),zero=True):
    """Only validator metadata, not observed CUDA receipts or capacity evidence."""
    from test_training_helios_checkpoint_inputs_cpu import native_suffix_row
    from distillschedule.training import helios_gan_checkpoint_cpu as policy
    from distillschedule.runtime import helios_gan_native_history as history
    from distillschedule.workloads.best_known.registry import fingerprint
    first=max([0]+[s for s in (1,2) if exits[s]>0])
    restored_bytes=40*sum(a*b*c*2 for a,b,c in policy.native_contract()['shapes'][first:])
    row=native_suffix_row(exits,40*(3-first),restored_bytes)
    phase=row['native_history_phase'];cp=phase['checkpoint_input_cpu']
    phase.update(schema_version=history.SCHEMA,execution_requirement=history.REQUIREMENT,task_sha256=task().digest)
    cp['identity'].update(task_sha256=task().digest,cohort_index=0,critic_version=10)
    cp.update(schema_version=policy.SESSION_SCHEMA,policy=policy.PRESET_ID,capacity=policy.capacity_template(),native_contract=policy.native_contract())
    digest=fingerprint(cp['identity']);cp['identity_sha256']=digest
    for record in cp['boundaries']:record.update(**cp['identity'],identity_sha256=digest)
    entry=row['trace_entries'][0]
    entry.update(native_history_semantics=history.SEMANTICS,native_task_sha256=task().digest,all_history_zero=zero)
    for s in entry['stage_schedule']:s['denoising_step_budget']=4 if zero else 2
    row['physical_versions_before']['critic']=10
    required=policy.critic_boundary_contract()['hidden_input_bytes']+policy.GPU_RESERVE
    phase['gan_g_checkpoint_boundary']=dict(schema_version='distillschedule.helios_gan_g_c_boundary.v1',
        identity=deepcopy(cp['identity']),contract=policy.critic_boundary_contract(),counts=dict(
            forwards=1,patch_calls=4,block_calls=40,head_calls=5,selective_contexts=20,input_derivatives=1,g_backwards=1,drained_before_g_backward=True),
        live_selective_stores=0,cpu_copies=0,original_callable_restored=True,capacity_observations=[dict(
            free_bytes=required,total_bytes=40*2**30,required_subset_plus_floor=required,
            scope='KNOWN_C_CHECKPOINT_SUBSET_ONLY_NOT_TOTAL_FIT_PROOF_NOT_RESERVATION')])
    return row


@pytest.mark.parametrize('zero',[False,True])
@pytest.mark.parametrize('exits',[(0,1,1),(0,1,0),(0,0,0)])
def test_w3_receipt_all_trace_suffixes_and_history_horizons_are_independent_of_w2(zero,exits):
    from distillschedule.runtime.helios_gan_native_history import validate_action
    validate_action(native_receipt_row(exits,zero))


@pytest.mark.parametrize('mutation',['wrong_source','wrong_rank','wrong_version','wrong_dtype','wrong_shape','unused_restore','missing_restore',
    'missing_patch','extra_patch','wrong_patch_contract','missing_head','undrained_C','extra_C_copy','wrong_C_role','missing_C_floor','wrong_C_version','w2_schema','old_policy','unreleased'])
def test_w3_collector_refuses_each_wrong_boundary_and_original_policy_fallback(mutation):
    from distillschedule.runtime.helios_gan_native_history import validate_action
    row=native_receipt_row();phase=row['native_history_phase'];cp=phase['checkpoint_input_cpu'];c=phase['gan_g_checkpoint_boundary']
    validate_action(row)
    if mutation=='wrong_source':cp['identity']['source']='x'*40
    elif mutation=='wrong_rank':cp['identity']['rank']=1
    elif mutation=='wrong_version':row['physical_versions_before']['generator']=3
    elif mutation=='wrong_dtype':cp['boundaries'][80]['dtype']='torch.float32'
    elif mutation=='wrong_shape':cp['boundaries'][80]['shape']=[1,4560,5120]
    elif mutation=='unused_restore':cp['boundaries'][0]['restore_count']=1
    elif mutation=='missing_restore':cp['boundaries'][80]['restore_count']=0
    elif mutation=='missing_head':c['counts']['head_calls']=4
    elif mutation=='missing_patch':c['counts']['patch_calls']=3
    elif mutation=='extra_patch':c['counts']['patch_calls']=5
    elif mutation=='wrong_patch_contract':c['contract']['patch_modules'].reverse()
    elif mutation=='undrained_C':c['live_selective_stores']=1
    elif mutation=='extra_C_copy':c['cpu_copies']=1
    elif mutation=='wrong_C_role':c['contract']['role']='critic_update'
    elif mutation=='missing_C_floor':c['capacity_observations'][0]['free_bytes']-=1
    elif mutation=='wrong_C_version':row['physical_versions_before']['critic']=15
    elif mutation=='w2_schema':cp['schema_version']='distillschedule.helios_g_checkpoint_hidden_cpu_execution.v1'
    elif mutation=='old_policy':cp['policy']='helios_gan.fa3_down40_nomerge_stream.v1'
    else:cp['stats']['live_cpu_bytes']=1
    with pytest.raises(ValueError):validate_action(row)


def test_native2_10_fifo_drain_gate_does_not_broaden_for_local2_14():
    from distillschedule.runtime.helios_gan_checkpoint_cpu import _selective_drained
    assert _selective_drained([{'FA3':[]}],[40,40],40)
    assert not _selective_drained([{'FA3':{0:None}}],[40,0],40)
    assert not _selective_drained([{'FA3':[object()]}],[40,40],40)


def test_full_joint_five_head_named_state_adam_rng_resume_and_all16_content_collector(tmp_path,monkeypatch):
    """CPU state + 16 file-inventory fixtures, NOT 16 initialized native ranks.

    Actual save/restore uses one physical CPU rank. The all16 collector fixture
    is explicitly constructed from those genuine small payload bytes solely to
    exercise complete inventory, per-file hash, source and policy rejection.
    """
    import hashlib,shutil
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    from distillschedule.training import helios_gan_training as training
    from distillschedule.training import checkpoint as cp
    from distillschedule.training.native_state import engine_state
    from test_helios_step_owner_cpu import equal
    class Model(torch.nn.Module):
        def __init__(self,critic=False):
            super().__init__();self.weight=torch.nn.Parameter(torch.randn(2,2))
            if critic:
                self.gan_heads=torch.nn.ModuleDict({k:torch.nn.Linear(2,1) for k in ('5','15','25','35')})
                self.gan_final_head=torch.nn.Linear(2,1)
    def state():
        engines=[]
        for critic,step in ((False,3),(True,15)):
            model=Model(critic);opt=torch.optim.AdamW(model.parameters(),lr=.001)
            sum(p.square().sum() for p in model.parameters()).backward();opt.step();opt.zero_grad(set_to_none=True)
            lr=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.);lr.last_epoch=step
            engines.append(FSDP2EngineAdapter(model,opt,lr,physical_step=step,global_batch=16,dp_world_size=1))
        # Same explicit CPU EMA serialization fixture used by native-state tests;
        # this does not simulate or claim the original EMA update recurrence.
        names,parameters=zip(*engines[0].module.named_parameters())
        ema=NS(_names=list(names),_parameters=list(parameters),_shadows=[p.detach().clone() for p in parameters],
            decay=.99,min_decay=0.,update_after_step=750,device='cpu',optimization_step=801,cur_decay_value=.8)
        return engine_state(dict(zip(('generator','critic'),engines)),ema=ema)
    identity=binding_for(tmp_path)['identity']
    original=state();snapshot=cp.pack_state(original);saved_rng=cp.rng_state()
    path=cp.save_checkpoint(tmp_path/'checkpoint',original,identity=identity,cycle=1,cursor=1,
        versions=training.expected_versions(1),policy=training._checkpoint_policy(identity))
    restored=state();cp.restore_checkpoint(path,restored,identity=identity)
    assert equal(snapshot,cp.pack_state(restored)) and equal(saved_rng,cp.rng_state())
    names=restored['critic.weights'].state_dict()['tensors']
    assert all(any(name.startswith(prefix) for name in names) for prefix in
        ('gan_heads.5.','gan_heads.15.','gan_heads.25.','gan_heads.35.','gan_final_head.'))
    assert not any('lease' in key or 'checkpoint_input' in key for key in snapshot)
    manifest=cp.read_manifest(path,identity=identity)
    training.validate_progress(manifest,dict(stream={'identity':{},'cohorts':[{},{}]},checkpoint_policy=training._checkpoint_policy(identity)))
    for bad in ({**identity,'schema_version':'distillschedule.helios_checkpoint_hidden_cpu_identity.v1'},
                {**identity,'checkpoint_input_capacity':{'unbounded':True}},{**identity,'source_head':'f'*40}):
        with pytest.raises(ValueError):cp.read_manifest(path,identity=bad)
    # ALL16_CONTENT_SHARD_FIXTURE_ONLY; no production writer/group substitution.
    inventory=tmp_path/'all16-content-fixture';shutil.copytree(path,inventory)
    payload=(inventory/'state.pt').read_bytes();(inventory/'state.pt').unlink()
    manifest['topology']={'world_size':16,'scope':'ALL16_CONTENT_SHARD_FIXTURE_ONLY'}
    manifest['files']={}
    for rank in range(16):
        name=f'rank-{rank:05d}.pt';(inventory/name).write_bytes(payload)
        manifest['files'][name]=hashlib.sha256(payload).hexdigest()
    raw=json.dumps(manifest,sort_keys=True).encode();(inventory/'manifest.json').write_bytes(raw)
    (inventory/'_SUCCESS').write_text(hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(cp.distributed,'rank_world',lambda:(0,16))
    monkeypatch.setattr(cp.distributed,'topology',lambda:deepcopy(manifest['topology']))
    assert cp.read_checkpoint(inventory,identity=identity)['versions']==dict(generator=3,critic=15,ema=801)
    last=inventory/'rank-00015.pt';last.write_bytes(payload+b'corrupt')
    with pytest.raises(ValueError,match='shard missing or hash changed'):cp.read_checkpoint(inventory,identity=identity)
    last.unlink()
    with pytest.raises(ValueError,match='shard missing or hash changed'):cp.read_manifest(inventory,identity=identity)


@pytest.mark.parametrize('exits,restores',[((0,1,1),40),((0,1,0),80),((0,0,0),120)])
def test_actual_w3_phase_branch_installs_after_factory_and_releases_before_adam(
        bound_original,tiny_w3,tmp_path,monkeypatch,exits,restores):
    """CPU_SYNTHETIC_PHYSICAL_SUBSTITUTION_ONLY, not FS16/CUDA admission.

    Reuse the established phase protocol's Unit type solely for physical
    materialization. Actual W3 factory/branch, lease session, original
    checkpoint/LoRA/SAC/cached-GAN and real Adam still execute together.
    """
    import torch.distributed.fsdp as fsdp
    from test_runtime_fsdp2_variable_forward import Unit,BlockUnit
    from distillschedule.runtime import helios_gan_native_history as history,helios_gan_checkpoint_cpu as runtime
    from distillschedule.training import helios_lifecycle as lifecycle,host_memory
    from test_helios_step_owner_cpu import equal
    b=bound_original
    monkeypatch.setattr(b.fsdp,'SELECTIVE_FLASH_ATTN_OP','aten.sin.default')
    class Root(TinyModel,Unit):pass
    class Block(TinyBlock,BlockUnit):pass
    root=b.generator.module;root.__class__=Root
    events=[];Unit.__init__(root,events,'root')
    for i,block in enumerate(root.blocks):
        block.__class__=Block;BlockUnit.__init__(block,events,str(i))
    upstream=ModuleType('helios.modules.transformer_helios');upstream.HeliosTransformerBlock=BlockUnit
    monkeypatch.setitem(sys.modules,'helios.modules.transformer_helios',upstream)
    monkeypatch.setattr(fsdp,'FSDPModule',Unit)
    snapshot={n:p.detach().clone() for n,p in root.named_parameters() if p.device.type=='cpu'}
    rng=torch.get_rng_state().clone();expected=run_action(b,exits)
    with torch.no_grad():
        for name,p in root.named_parameters():
            if p.device.type=='cpu':p.copy_(snapshot[name])
    for owner in (root,b.critic.module):
        for block in owner.blocks:block.calls=0
    torch.set_rng_state(rng)
    b.generator.optimizer=torch.optim.AdamW([p for p in root.parameters() if p.device.type=='cpu'],lr=.001)
    def step():
        assert all(not u.group.is_unsharded for u in (root,*root.blocks))
        b.generator.optimizer.step();b.generator.global_steps+=1
    b.generator.step=step
    def physical_only(engine,*,task):
        assert engine is b.generator and all(not u.group.is_unsharded for u in (root,*root.blocks))
        return dict(fsdp_units=41,checkpoint_blocks=40,critic_fsdp_units=46,
            topology='CPU_SYNTHETIC_PHYSICAL_SUBSTITUTION_ONLY',callables=runtime.authenticate_callables(b.core))
    # Only this physical admission is substituted, not generator_scope or the
    # phase owner/checkpoint validators/callables/math/replay/lease engine.
    monkeypatch.setattr(runtime,'admit_engine',physical_only)
    monkeypatch.setattr(host_memory,'admit_host_memory',lambda *args:{'scope':'CPU_COMPONENT_ONLY'})
    binding=binding_for(tmp_path);binding['gan_core']=b.core
    with lifecycle.original_training_scope(binding):
        tiny_w3.admit_startup(binding,rank=0)
        with history.generator_phase(b.generator,task=task(),with_grad=True,action_id='g') as (view,receipt):
            assert all(u.group.is_unsharded for u in (root,*root.blocks))
            actual=run_action(b,exits,engine=view)
        assert equal(actual,expected)
    assert receipt['unshard_count']==receipt['reshard_count']==41 and receipt['restored'] is True
    assert receipt['backward_count']==receipt['commit_count']==1
    assert receipt['checkpoint_input_cpu']['stats']['cpu_copy_count']==120
    assert receipt['checkpoint_input_cpu']['stats']['restore_count']==restores
    assert receipt['checkpoint_input_cpu']['stats']['unused_release_count']==120-restores
    assert receipt['gan_g_checkpoint_boundary']['counts']['drained_before_g_backward'] is True
    assert binding['checkpoint_input_executions'][0]['checkpoint_input_cpu']==receipt['checkpoint_input_cpu']
