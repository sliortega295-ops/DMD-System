"""CPU_SYNTHETIC_TEST_ONLY; no native HSDP, target-2.10, capacity or quality claim."""
from copy import deepcopy
import gc
import json
import weakref
from types import SimpleNamespace as NS

import pytest
import torch

from test_training_helios_native_safe_stream import native_safe_config
from test_training_helios_native_history import metadata_stream as metadata_stream
from test_training_helios_native_history import phase_protocol as phase_protocol
from test_training_helios_native_history import task

PRESET = 'helios_nogan.native_history_safe_ed8_checkpoint_cpu.v1'


def candidate_config(tmp_path):
    raw = native_safe_config(tmp_path).to_dict()
    raw['execution'].update(preset=PRESET, capacity_policy=str(tmp_path/'capacity.json'))
    return type(native_safe_config(tmp_path)).from_dict(raw, source=tmp_path/'candidate.json')


def test_public_candidate_is_explicit_separate_and_old_presets_still_reject_capacity(tmp_path):
    from distillschedule.training import native
    conf = candidate_config(tmp_path)
    assert native._options(conf, 'W2') == 'manual'
    from distillschedule.training.native_presets import describe_preset, preset_implementation
    from distillschedule.training.helios_native_safe_stream import implementation
    assert preset_implementation('W2', PRESET) == implementation()
    described = describe_preset('W2', PRESET)
    assert described['gpu_execution'] == 'NOT_RUN'
    assert 'checkpoint_hidden_input_cpu' in described['mechanisms']
    for old in ('helios_nogan.native_history_safe_ed8.v1', 'helios_nogan.safe_stream_ed8.v1'):
        raw = conf.to_dict(); raw['execution']['preset'] = old
        with pytest.raises(ValueError): native._options(type(conf).from_dict(raw, source=conf.source), 'W2')
    for change in ({'mode':'reference'}, {'plan':'old'}, {'capacity_policy':None}):
        raw = conf.to_dict(); raw['execution'].update(change)
        with pytest.raises(ValueError): native._options(type(conf).from_dict(raw, source=conf.source), 'W2')


def test_native_shape_contract_is_exact_three_original_stages_and_partitioned_node_bound():
    from distillschedule.training import helios_checkpoint_cpu as policy
    spec = policy.native_contract()
    assert spec['shapes'] == [[1,2940,5120], [1,4560,5120], [1,11040,5120]]
    assert spec['dtype'] == 'torch.bfloat16' and spec['checkpoint_blocks'] == 40
    assert spec['max_connected_grad_stages'] == 3
    cap = policy.capacity_template('pageable')
    assert cap['rank_pageable_limit_bytes'] == 7_593_984_000
    assert cap['node_pageable_limit_bytes'] == 8*7_593_984_000
    assert cap['rank_pinned_limit_bytes'] == cap['node_pinned_limit_bytes'] == 0
    for field in cap:
        if field.endswith('_bytes'):
            bad=deepcopy(cap); bad[field] += 1
            with pytest.raises(ValueError): policy.validate_capacity(bad)
    for patch in ({'schema_version':'old'}, {'storage':'unbounded'}, {'unknown':1}):
        with pytest.raises(ValueError): policy.validate_capacity({**cap, **patch})


@pytest.fixture
def tiny_contract(monkeypatch):
    from distillschedule.training import helios_checkpoint_cpu as policy
    spec = dict(schema_version='CPU_SYNTHETIC_TEST_ONLY', shapes=[[1,3,4],[1,5,4],[1,7,4]],
        dtype='torch.float64', checkpoint_blocks=2, max_connected_grad_stages=3)
    monkeypatch.setattr(policy, 'native_contract', lambda:deepcopy(spec))
    return spec


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.randn(4,4,dtype=torch.float64)); self.calls=0
    def forward(self, hidden, shared):
        self.calls += 1
        return torch.nn.functional.dropout(torch.sin(hidden @ self.weight + shared), .2, training=True)


def session(tiny_contract):
    from distillschedule.training import helios_checkpoint_cpu as policy
    from distillschedule.runtime.helios_checkpoint_inputs_cpu import GCheckpointInputSession
    from distillschedule.runtime.fsdp2_static import fsdp2_metadata_tolerant_checkpoint
    owner=torch.nn.Module(); owner.blocks=torch.nn.ModuleList([TinyBlock() for _ in range(2)])
    owner.gradient_checkpointing=True; owner._gradient_checkpointing_func=fsdp2_metadata_tolerant_checkpoint
    value=GCheckpointInputSession(owner=owner, blocks=tuple(owner.blocks),
        capacity=policy.capacity_template('pageable'), identity=dict(source='CPU_SYNTHETIC_TEST_ONLY',
        task_sha256='a'*64, generator_version=2), device='cpu')
    return owner,value


def train(owner, scope=None, optimizer=None):
    from contextlib import nullcontext
    optimizer=optimizer or torch.optim.AdamW(owner.parameters(),lr=.001)
    optimizer.zero_grad(set_to_none=True)
    inputs=[torch.randn(*shape,dtype=torch.float64,requires_grad=True) for shape in ((1,3,4),(1,5,4),(1,7,4))]
    shared=torch.randn(1,1,4,dtype=torch.float64,requires_grad=True)
    with scope if scope is not None else nullcontext():
        outputs=[]
        for x in inputs:
            for block in owner.blocks:x=owner._gradient_checkpointing_func(block,x,shared)
            outputs.append(x)
        loss=sum(x.square().sum() for x in outputs); loss.backward()
        grads={name:p.grad.clone() for name,p in owner.named_parameters()}
        input_grads=[x.grad.clone() for x in inputs]+[shared.grad.clone()]
        optimizer.step()
    return dict(outputs=[x.detach().clone() for x in outputs], grads=grads, input_grads=input_grads,
        parameters={name:p.detach().clone() for name,p in owner.named_parameters()},
        adam={name:{k:v.clone() if torch.is_tensor(v) else v for k,v in optimizer.state[p].items()}
              for name,p in owner.named_parameters()}, rng=torch.get_rng_state(), calls=[b.calls for b in owner.blocks])


def test_complete_checkpoint_rebuild_restores_named_adam_rng_and_new_lease_owner(tiny_contract,tmp_path):
    from test_training_end_to_end import equal
    from distillschedule.training import helios_checkpoint_cpu as policy, helios_lifecycle as lifecycle
    from distillschedule.training.checkpoint import save_checkpoint,restore_checkpoint
    torch.manual_seed(19);control,_=session(tiny_contract)
    torch.manual_seed(19);candidate,first=session(tiny_contract)
    co=torch.optim.AdamW(control.parameters(),lr=.001)
    ca=torch.optim.AdamW(candidate.parameters(),lr=.001)
    torch.manual_seed(72);train(control,optimizer=co);expected=train(control,optimizer=co)
    torch.manual_seed(72);train(candidate,first,ca)
    ca.zero_grad(set_to_none=True)  # Original complete-cycle save boundary.
    identity=dict(schema_version=policy.IDENTITY,native_history=policy.history_descriptor(),
        execution_policy=policy.descriptor(),checkpoint_input_capacity=policy.capacity_template('pageable'),
        source='CPU_SYNTHETIC_TEST_ONLY')
    path=save_checkpoint(tmp_path/'checkpoint',dict(model=candidate,optimizer=ca),identity=identity,
        cycle=1,cursor=1,versions=lifecycle.expected_versions(1),policy=lifecycle._checkpoint_policy(identity,optimized=True))
    rebuilt,new_session=session(tiny_contract)
    restored_optimizer=torch.optim.AdamW(rebuilt.parameters(),lr=.001)
    restore_checkpoint(path,dict(model=rebuilt,optimizer=restored_optimizer),identity=identity)
    actual=train(rebuilt,new_session,restored_optimizer)
    expected.pop('calls');actual.pop('calls')
    assert equal(actual,expected)
    assert new_session is not first and new_session.receipt()['stats']['cpu_copy_count']==6
    assert first.receipt()['stats']['live_leases']==new_session.receipt()['stats']['live_leases']==0
    for field in ('source','schema_version','checkpoint_input_capacity'):
        wrong=deepcopy(identity);wrong[field]='foreign'
        before=deepcopy(rebuilt.state_dict());rng=torch.get_rng_state().clone()
        with pytest.raises(ValueError):restore_checkpoint(path,dict(model=rebuilt,optimizer=restored_optimizer),identity=wrong)
        assert equal(before,rebuilt.state_dict()) and torch.equal(rng,torch.get_rng_state())


def test_candidate_identity_native_state_restore_and_startup_fingerprint_are_bound(tmp_path, monkeypatch):
    """CPU native-state/identity seam only, not an all-rank HSDP checkpoint."""
    from test_training_native_state import _engine, _update
    from test_training_end_to_end import equal
    from distillschedule.training import helios_checkpoint_cpu as policy, helios_lifecycle as lifecycle, host_memory
    from distillschedule.training.native_state import engine_state
    from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint, pack_state, rng_state
    from distillschedule.workloads.best_known.registry import fingerprint
    capacity = policy.capacity_template()
    (tmp_path/'capacity.json').write_text(json.dumps(capacity))
    identity = dict(schema_version=policy.IDENTITY, native_history=policy.history_descriptor(),
        execution_policy=policy.descriptor(), checkpoint_input_capacity=capacity,
        source_head='CPU_SYNTHETIC_TEST_ONLY', source_fingerprint='b'*64, stream_sha256='c'*64,
        verified_immutable_assets={})
    binding = dict(config=candidate_config(tmp_path), identity=identity, implementation=policy.implementation(),
        effective_configuration=policy.descriptor(), owns_process_group=True, metadata_group=object())
    monkeypatch.setattr(host_memory, 'admit_host_memory', lambda *args:dict(scope='CPU_SYNTHETIC_TEST_ONLY'))
    with lifecycle.original_training_scope(binding):
        startup = policy.admit_startup(binding, rank=0)
        assert policy.require_binding() is binding
        # Native owner layout is bound after construction; its addition must
        # not invalidate the already-checked preconstruction fingerprint.
        identity['native_owner_layout'] = dict(topology=policy.descriptor()['topology'], trainables_sha256='d'*64)
        assert policy.require_binding() is binding
        saved_fingerprint = startup['identity_sha256']
        startup['identity_sha256'] = 'foreign'
        with pytest.raises(ValueError, match='preconstruction'):policy.require_binding()
        startup['identity_sha256'] = saved_fingerprint
        assert saved_fingerprint == fingerprint({k:v for k,v in identity.items() if k != 'native_owner_layout'})
        checkpoint_policy = lifecycle._checkpoint_policy(identity, optimized=True)
    engine = _engine();_update(engine)
    state = engine_state({'generator':engine})
    path = save_checkpoint(tmp_path/'native-state', state, identity=identity, cycle=1, cursor=1,
        versions=dict(generator=1), policy=checkpoint_policy)
    assert (path/'_SUCCESS').is_file()
    expected_state = pack_state(state);expected_rng = rng_state()
    rebuilt = _engine();ids = [id(p) for p in rebuilt.module.parameters()]
    restored_state = engine_state({'generator':rebuilt})
    restore_checkpoint(path, restored_state, identity=identity)
    assert ids == [id(p) for p in rebuilt.module.parameters()]
    assert equal(pack_state(restored_state), expected_state) and equal(rng_state(), expected_rng)
    assert not any('lease' in name or 'checkpoint_input' in name for name in restored_state)
    for field in ('schema_version','checkpoint_input_capacity','source_fingerprint','stream_sha256','native_owner_layout'):
        wrong = deepcopy(identity);wrong[field] = 'foreign'
        before = pack_state(restored_state);before_rng = rng_state()
        with pytest.raises(ValueError):restore_checkpoint(path, restored_state, identity=wrong)
        assert equal(pack_state(restored_state), before) and equal(rng_state(), before_rng)


def test_outer_saved_hooks_and_recomputation_do_not_capture_extra_intermediates(tiny_contract):
    owner,value=session(tiny_contract);outer=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda x:outer.append(tuple(x.shape)) or x.detach(),lambda x:x):
        actual=train(owner,value)
    assert actual['calls']==[6,6]
    assert value.receipt()['stats']['cpu_copy_count']==6
    assert value.receipt()['stats']['restore_count']==6
    assert value.receipt()['stats']['live_leases']==0


def test_failed_host_copy_releases_reservation_without_holding_cpu_storage(tiny_contract,monkeypatch):
    owner,value=session(tiny_contract)
    original=torch.empty_like
    def fail(*args,**kwargs):
        if kwargs.get('device')=='cpu':raise RuntimeError('synthetic CPU allocation refusal')
        return original(*args,**kwargs)
    monkeypatch.setattr(torch,'empty_like',fail)
    with pytest.raises(RuntimeError,match='allocation refusal'):
        with value:
            owner._gradient_checkpointing_func(owner.blocks[0],
                torch.ones(1,3,4,dtype=torch.float64,requires_grad=True),torch.zeros(1,1,4,dtype=torch.float64))
    stats=value.receipt()['stats']
    assert stats['failed_pack_count']==1 and stats['cpu_copy_count']==0
    assert stats['live_cpu_bytes']==stats['live_leases']==stats['release_count']==0


def test_real_nonreentrant_outputs_all_gradients_named_adam_rng_and_calls_are_exact(tiny_contract):
    from test_training_end_to_end import equal
    torch.manual_seed(19);control,_=session(tiny_contract)
    torch.manual_seed(19);candidate,offload=session(tiny_contract)
    torch.manual_seed(72);expected=train(control)
    torch.manual_seed(72);actual=train(candidate,offload)
    assert equal(actual,expected)
    row=offload.receipt(); stats=row['stats']
    assert row['closed'] and stats['checkpoint_calls']==stats['cpu_copy_count']==stats['restore_count']==6
    assert stats['release_count']==6 and stats['live_cpu_bytes']==stats['live_leases']==0
    assert stats['peak_live_cpu_bytes']==2*(3+5+7)*4*8
    assert len(row['boundaries'])==6 and all(r['generator_version']==2 for r in row['boundaries'])
    assert all(r['dtype']=='torch.float64' and r['source']=='CPU_SYNTHETIC_TEST_ONLY' for r in row['boundaries'])


def test_no_hidden_or_full_args_strong_reference_survives_forward_and_shared_not_duplicated(tiny_contract):
    owner,value=session(tiny_contract)
    with value:
        x=torch.randn(1,3,4,dtype=torch.float64,requires_grad=True)*2
        shared=torch.randn(1,1,4,dtype=torch.float64)
        ref=weakref.ref(x); original_version=x._version
        y=owner._gradient_checkpointing_func(owner.blocks[0],x,shared)
        assert x._version==original_version
        del x;gc.collect()
        assert ref() is None, 'pack/unpack closures must not keep original checkpoint input alive'
        assert value.receipt()['stats']['cpu_copy_count']==1
        y.sum().backward()
    assert value.receipt()['stats']['live_cpu_bytes']==0


def test_alias_inplace_error_is_preserved_even_after_original_object_dies(tiny_contract):
    owner,value=session(tiny_contract)
    with pytest.raises(RuntimeError,match='modified.*inplace'):
        with value:
            x=torch.randn(1,3,4,dtype=torch.float64,requires_grad=True)*2
            alias=x.detach(); ref=weakref.ref(x)
            y=owner._gradient_checkpointing_func(owner.blocks[0],x,torch.zeros(1,1,4,dtype=torch.float64))
            del x;gc.collect(); assert ref() is None
            alias.add_(1);y.sum().backward()
    assert value.receipt()['stats']['live_cpu_bytes']==0
    assert value.receipt()['stats']['cancel_count']==1


@pytest.mark.parametrize('offload', [False, True])
@pytest.mark.parametrize('mutation', ['direct', 'detached_alias_after_gc'])
def test_shared_checkpoint_input_mutation_still_refuses_without_a_cpu_copy(tiny_contract, offload, mutation):
    from contextlib import nullcontext
    owner, value = session(tiny_contract)
    with pytest.raises(RuntimeError, match='modified.*inplace'):
        with value if offload else nullcontext():
            hidden = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
            shared = torch.randn(1, 1, 4, dtype=torch.float64)
            alias = shared.detach()
            ref = weakref.ref(shared)
            output = owner._gradient_checkpointing_func(owner.blocks[0], hidden, shared)
            if mutation == 'detached_alias_after_gc':
                del shared
                gc.collect()
                # The original SavedVariable may itself retain its Python
                # wrapper; the hook must not need that wrapper to check aliases.
                if offload:
                    assert ref() is None
                alias.add_(1)
            else:
                shared.add_(1)
            output.sum().backward()
    if offload:
        assert value.receipt()['stats']['cpu_copy_count'] == 1
        assert value.receipt()['stats']['live_cpu_bytes'] == 0


def test_repeated_unpack_recompute_and_error_cleanup_do_not_release_host_early(tiny_contract):
    owner,value=session(tiny_contract)
    with value:
        x=torch.randn(1,3,4,dtype=torch.float64,requires_grad=True)
        y=owner._gradient_checkpointing_func(owner.blocks[0],x,torch.zeros(1,1,4,dtype=torch.float64))
        y.sum().backward(retain_graph=True)
        assert value.receipt()['stats']['live_leases']==1
        y.sum().backward()
    assert value.receipt()['stats']['restore_count']==2
    assert value.receipt()['stats']['release_count']==1
    boundary=value.receipt()['boundaries'][0]
    assert boundary['restore_count']==2 and boundary['release_count']==1 and boundary['release_outcome']=='restored'
    assert value.receipt()['stats']['restored_release_count']==1
    owner,value=session(tiny_contract)
    with pytest.raises(RuntimeError,match='user forward failed'):
        with value:
            y=owner._gradient_checkpointing_func(owner.blocks[0],x,torch.zeros(1,1,4,dtype=torch.float64))
            raise RuntimeError('user forward failed')
    assert value.receipt()['stats']['live_cpu_bytes']==0
    with pytest.raises(RuntimeError,match='cancelled|closed'): y.sum().backward()


@pytest.mark.parametrize('mutation',['shape','dtype','duplicate_stage','capacity','callable','nested','parameter'])
def test_fail_closed_runtime_boundaries_and_capacity(tiny_contract,mutation):
    owner,value=session(tiny_contract)
    x=torch.randn(1,3,4,dtype=torch.float64,requires_grad=True)
    if mutation=='shape':x=x[:,:2]
    if mutation=='dtype':x=x.float()
    if mutation=='capacity':value.budget.limit=1
    if mutation=='parameter':owner.register_parameter('hidden_parameter',torch.nn.Parameter(x));x=owner.hidden_parameter
    if mutation=='callable':owner._gradient_checkpointing_func=lambda *args:None
    with pytest.raises((ValueError,RuntimeError)):
        with value:
            if mutation=='nested':
                with value:pass
            else:
                y=owner._gradient_checkpointing_func(owner.blocks[0],x,torch.zeros(1,1,4,dtype=x.dtype))
                if mutation=='duplicate_stage':owner._gradient_checkpointing_func(owner.blocks[0],x,torch.zeros(1,1,4,dtype=x.dtype))
                y.sum().backward()
    assert value.receipt()['stats']['live_cpu_bytes']==0


def test_real_producer_metadata_worker_lowering_identity_and_preconstruction_capacity(metadata_stream,tmp_path,monkeypatch):
    from distillschedule.training import helios_checkpoint_cpu as policy, helios_worker as worker, helios_lifecycle as lifecycle
    from distillschedule.training.helios_reference_assets import SCHEMA as ASSETS
    from distillschedule.search.helios_request import OriginalHeliosRequest
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    root,stream=metadata_stream;conf=candidate_config(tmp_path)
    (tmp_path/'capacity.json').write_text(json.dumps(policy.capacity_template('pageable')))
    raw=conf.to_dict();raw['data']['prepared']=str(root/'stream.json');conf=type(conf).from_dict(raw,source=conf.source)
    assets=dict(schema_version=ASSETS,assets={name:dict(path=f'/assets/{name}') for name in (
        'generator_model','score_model','generator_precast','score_precast','logical_checkpoint','negative_prompt')})
    path=tmp_path/'assets.json';path.write_text(json.dumps(assets))
    monkeypatch.setattr(worker,'reference_assets',lambda value:value)
    monkeypatch.setattr(worker,'verify_reference_pins',lambda value:dict(scope='CPU_SYNTHETIC_TEST_ONLY'))
    monkeypatch.setattr(worker,'load_reference_bootstrap',lambda *a,**kw:(NS(manifest_hash='a'*64),{}))
    from distillschedule.training import native
    (tmp_path/'task.json').write_text(json.dumps(OriginalHeliosRequest().to_dict()))
    source=dict(identity=dict(head='CPU_SYNTHETIC_TEST_ONLY',dirty=False))
    monkeypatch.setattr(native,'_source',lambda config:source)
    workload,observed,paths,admitted=native._admit(conf)
    assert workload=='W2' and observed==source
    assert native._report(conf,workload,observed,paths,admitted)['checkpoint_input_capacity']==policy.capacity_template('pageable')
    assert admitted['stream']==stream and admitted['selected']==policy.descriptor()
    assert all(('--' if enabled else '--no-')+name.replace('_','-') in admitted['argv']
        for name,enabled in policy.effective_flags().items())
    assert admitted['argv'][admitted['argv'].index('--mesh-policy')+1]=='node-hsdp'
    for key,val in dict(WORLD_SIZE='16',LOCAL_WORLD_SIZE='8',RANK='0',LOCAL_RANK='0').items():monkeypatch.setenv(key,val)
    monkeypatch.setattr(policy,'validate_environment',lambda **kw:None)
    monkeypatch.setattr(lifecycle,'admit_original_output',lambda **kw:None)
    observations=[]
    from distillschedule.training import host_memory
    monkeypatch.setattr(host_memory,'admit_host_memory',lambda needed,reserve: observations.append((needed,reserve)) or
        dict(required_bytes=needed,reserve_bytes=reserve,scope='CPU_SYNTHETIC_TEST_ONLY'))
    def original_main():
        binding=lifecycle.current_training();identity=binding['identity']
        assert identity['schema_version']==policy.IDENTITY and identity['checkpoint_input_capacity']==policy.capacity_template('pageable')
        identity['verified_immutable_assets']={}
        binding['effective_configuration']=policy.descriptor()
        policy.admit_startup(binding,rank=0)
        lifecycle._bind_native_history_construction(policy=StaticFSDP2P0Policy(mesh_policy='node-hsdp'))
        assert lifecycle.native_history_cpu_first_construction(policy=StaticFSDP2P0Policy(mesh_policy='node-hsdp'))
        assert lifecycle._checkpoint_policy(identity,optimized=True)['schema_version']==policy.CHECKPOINT
        old=deepcopy(identity);old['schema_version']='distillschedule.helios_native_safe_stream_identity.v1'
        with pytest.raises(ValueError):lifecycle._checkpoint_policy(old,optimized=True)
        return 17
    assert worker._run_admitted(conf,admitted,dict(identity=dict(head='CPU_SYNTHETIC_TEST_ONLY',dirty=False)),
        backend=NS(main=original_main),optimized=True,
        ownership=dict(owns_process_group=True,metadata_group=object()))==17
    assert observations==[(8*7_593_984_000,512*2**30)]


def test_startup_capacity_overflow_refuses_before_construction_capability(tmp_path,monkeypatch):
    from distillschedule.training import helios_checkpoint_cpu as policy,host_memory
    conf=candidate_config(tmp_path)
    capacity=policy.capacity_template('pageable')
    (tmp_path/'capacity.json').write_text(json.dumps(capacity))
    binding=dict(config=conf,owns_process_group=True,metadata_group=object(),identity=dict(schema_version=policy.IDENTITY,native_history=policy.history_descriptor(),
        execution_policy=policy.descriptor(),checkpoint_input_capacity=capacity,verified_immutable_assets={}))
    def refuse(*args):raise ValueError('synthetic host/cgroup reserve refusal')
    monkeypatch.setattr(host_memory,'admit_host_memory',refuse)
    with pytest.raises(ValueError,match='reserve refusal'):policy.admit_startup(binding,rank=0)
    assert 'checkpoint_input_startup' not in binding and '_native_history_construction' not in binding
    altered=deepcopy(capacity);altered['rank_pageable_limit_bytes']-=1
    (tmp_path/'capacity.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='capacity declaration'):policy.admit_startup(binding,rank=0)


def test_production_factory_precedes_real_cpu_fsdp_slot_substitution(tmp_path, monkeypatch):
    """CPU_PHYSICAL_SUBSTITUTION_ONLY, not native CUDA/2x8 admission.

    The real factory and strict owner/callable checks are never replaced.
    Native topology rejects this CPU owner first. Only the fixture's expected
    topology is then substituted, so the unchanged CUDA-only factory refusal
    can witness ordering across a real FSDP parameter-slot transition.
    """
    import sys
    import types
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.tensor import DTensor
    from distillschedule.runtime import helios_native_safe_history as history
    from distillschedule.runtime import helios_checkpoint_inputs_cpu as runtime
    from distillschedule.runtime.fsdp2_static import fsdp2_metadata_tolerant_checkpoint
    from distillschedule.runtime.fsdp2_variable_forward import fsdp2_variable_forward_phase
    from distillschedule.training import helios_checkpoint_cpu as policy, helios_lifecycle as lifecycle
    from distillschedule.training import helios_safe_stream, host_memory
    from distillschedule.training.native_state import _named

    upstream = types.ModuleType('helios.modules.transformer_helios')
    upstream.HeliosTransformerBlock = TinyBlock
    monkeypatch.setitem(sys.modules, 'helios.modules.transformer_helios', upstream)
    assert not dist.is_initialized() and not torch.cuda.is_initialized()
    dist.init_process_group('gloo', init_method='file://' + str(tmp_path/'cpu-fsdp'), rank=0, world_size=1)
    try:
        root = torch.nn.Module()
        root.blocks = torch.nn.ModuleList(TinyBlock() for _ in range(40))
        root.projection = torch.nn.Linear(4, 4)
        root.float()
        root.gradient_checkpointing = True
        root._gradient_checkpointing_func = fsdp2_metadata_tolerant_checkpoint
        mesh = init_device_mesh('cpu', (1, 1), mesh_dim_names=('replicate', 'shard'))
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for block in root.blocks:
            fully_shard(block, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
        fully_shard(root, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
        engine = NS(module=root, optimizer=torch.optim.AdamW(root.parameters()), global_steps=2)
        owned = tuple(id(p) for p in root.parameters())
        assert all(isinstance(p, DTensor) and p.dtype == torch.float32 for p in root.parameters())
        with pytest.raises(ValueError, match='topology/layout'):
            history.admit_engine(engine)

        cpu_topology = dict(mesh_policy='CPU_PHYSICAL_SUBSTITUTION_ONLY', mesh=[[0]],
            mesh_dim_names=['replicate', 'shard'], device_type='cpu',
            placements=[dict(kind='replicate'), dict(kind='shard', dim=0)])
        monkeypatch.setattr(helios_safe_stream, '_topology', lambda:deepcopy(cpu_topology))
        assert history.admit_engine(engine)['fsdp_units'] == 41
        with fsdp2_variable_forward_phase(engine, enabled=True, with_grad=False,
                expected_units=41, checkpoint_block_type=TinyBlock):
            assert tuple(id(p) for p in root.parameters()) != owned
            assert all(not isinstance(p, DTensor) and p.dtype == torch.bfloat16 for p in root.parameters())
            with pytest.raises(ValueError, match='optimizer owns parameters outside'):
                _named(engine)
        assert tuple(id(p) for p in root.parameters()) == owned
        assert history.admit_engine(engine)['fsdp_units'] == 41

        conf = candidate_config(tmp_path)
        capacity = policy.capacity_template()
        (tmp_path/'capacity.json').write_text(json.dumps(capacity))
        identity = dict(schema_version=policy.IDENTITY, native_history=policy.history_descriptor(),
            execution_policy=policy.descriptor(), checkpoint_input_capacity=capacity,
            source_head='CPU_PHYSICAL_SUBSTITUTION_ONLY', source_fingerprint='b'*64,
            stream_sha256='c'*64, verified_immutable_assets={})
        binding = dict(config=conf, identity=identity, implementation=policy.implementation(),
            effective_configuration=policy.descriptor(), owns_process_group=True, metadata_group=object())
        monkeypatch.setattr(host_memory, 'admit_host_memory',
            lambda *args:dict(scope='CPU_PHYSICAL_SUBSTITUTION_ONLY'))
        with lifecycle.original_training_scope(binding):
            policy.admit_startup(binding, rank=0)
            # Exercise the unchanged production factory itself, not a session
            # constructor standing in for it. This refusal must stay strict.
            with pytest.raises(ValueError, match='CPU fake owner'):
                runtime.generator_scope(engine, task=task())
            # Regression: factory must run while optimizer-owned FP32 DTensor
            # slots are visible, BEFORE materialization swaps compute slots.
            with pytest.raises(ValueError, match='CPU fake owner'):
                with history.generator_phase(engine, task=task(), with_grad=True, action_id='g'):
                    pytest.fail('CPU owner must never install a native registration')
            assert 'checkpoint_input_executions' not in binding
        assert tuple(id(p) for p in root.parameters()) == owned
        assert all(not m._get_fsdp_state()._fsdp_param_group.is_unsharded for m in root.modules()
            if hasattr(m, '_get_fsdp_state') and m._get_fsdp_state()._fsdp_param_group is not None)
        assert root._gradient_checkpointing_func is fsdp2_metadata_tolerant_checkpoint
    finally:
        dist.destroy_process_group()
    assert not dist.is_initialized() and not torch.cuda.is_initialized()


@pytest.mark.parametrize('exits,restores', [((0,1,1),40), ((0,1,0),80), ((0,0,0),120)])
def test_existing_native_phase_reaches_live_120_boundary_copies_not_just_a_flag(
        phase_protocol,tiny_contract,tmp_path,monkeypatch,exits,restores):
    """CPU_SYNTHETIC_TEST_ONLY: mock ONLY physical HSDP/device admission.

    Existing phase ownership, original checkpoint callable, actual checkpoint
    saves/recomputes, real CPU Adam and the new runtime branch still execute.
    This synthetic owner is deliberately rejected by the production factory.
    """
    from distillschedule.training import helios_checkpoint_cpu as policy,helios_lifecycle as lifecycle,host_memory
    from distillschedule.runtime import helios_native_safe_history as history,helios_checkpoint_inputs_cpu as runtime
    from distillschedule.runtime.fsdp2_static import fsdp2_metadata_tolerant_checkpoint
    engine,Unit,BlockUnit=phase_protocol
    spec={**tiny_contract,'checkpoint_blocks':40}
    monkeypatch.setattr(policy,'native_contract',lambda:deepcopy(spec))
    class Block(TinyBlock,BlockUnit):
        def __init__(self,events,index):
            TinyBlock.__init__(self);BlockUnit.__init__(self,events,str(index))
    class Root(torch.nn.Module,Unit):
        def __init__(self):
            torch.nn.Module.__init__(self);Unit.__init__(self,engine.events,'root')
            self.blocks=torch.nn.ModuleList(Block(engine.events,i) for i in range(40))
            self.gradient_checkpointing=True;self._gradient_checkpointing_func=fsdp2_metadata_tolerant_checkpoint
    engine.module=Root()
    from distillschedule.runtime.helios_native_history import admit_engine as exact_callable_admission
    assert exact_callable_admission(engine)['checkpoint_blocks']==40
    monkeypatch.setattr(history,'admit_engine',lambda value:exact_callable_admission(value))
    monkeypatch.setattr(host_memory,'admit_host_memory',lambda *args:dict(scope='CPU_SYNTHETIC_TEST_ONLY'))
    conf=candidate_config(tmp_path);capacity=policy.capacity_template()
    (tmp_path/'capacity.json').write_text(json.dumps(capacity))
    identity=dict(schema_version=policy.IDENTITY,native_history=policy.history_descriptor(),execution_policy=policy.descriptor(),
        checkpoint_input_capacity=capacity,source_head='CPU_SYNTHETIC_TEST_ONLY',source_fingerprint='b'*64,
        stream_sha256='c'*64,verified_immutable_assets={})
    binding=dict(config=conf,identity=identity,implementation=policy.implementation(),effective_configuration=policy.descriptor(),
        owns_process_group=True,metadata_group=object())
    real_factory=runtime.generator_scope
    created=[]
    def cpu_component_factory(engine,*,task):
        assert all(not m.group.is_unsharded for m in engine.module.modules() if isinstance(m,Unit))
        value=runtime.GCheckpointInputSession(owner=engine.module,blocks=tuple(engine.module.blocks),capacity=capacity,
            identity=dict(source=identity['source_head'],source_fingerprint=identity['source_fingerprint'],
                stream_sha256=identity['stream_sha256'],task_sha256=task.digest,generator_version=engine.global_steps+2),device='cpu')
        created.append(value);return value
    with lifecycle.original_training_scope(binding):
        policy.admit_startup(binding,rank=0)
        with pytest.raises(ValueError,match='CPU fake owner'):real_factory(engine,task=task())
        monkeypatch.setattr(runtime,'generator_scope',cpu_component_factory)
        from test_training_end_to_end import equal
        torch.manual_seed(121);expected=train_native_prefixes(deepcopy(engine.module),exits)
        with history.generator_phase(engine,task=task(),with_grad=True,action_id='g') as (view,receipt):
            # The old validator is NOT widened to accept a generic callable or
            # even the known wrapper. It only admits before/after this scope.
            with pytest.raises(ValueError,match='checkpoint function'):exact_callable_admission(engine)
            assert all(m.group.is_unsharded for m in engine.module.modules() if isinstance(m,Unit))
            torch.manual_seed(121);actual_update=train_native_prefixes(engine.module,exits)
            assert equal(actual_update,expected)
            view.backward('component backward already executed');view.step()
        actual=receipt['checkpoint_input_cpu']
        assert binding['checkpoint_input_executions']==[actual]
        assert actual['stats']['checkpoint_calls']==actual['stats']['cpu_copy_count']==actual['stats']['release_count']==120
        assert actual['stats']['restore_count']==actual['stats']['restored_release_count']==restores
        assert actual['stats']['unused_release_count']==120-restores
        assert actual['closed'] and actual['stats']['live_cpu_bytes']==0
        assert receipt['unshard_count']==receipt['reshard_count']==41 and receipt['restored']
        assert exact_callable_admission(engine)['checkpoint_blocks']==40
        with history.generator_phase(engine,task=task(),with_grad=False,action_id='r0') as (view,detached):view(7)
        assert detached['checkpoint_input_cpu']['enabled'] is False and len(created)==1


@pytest.mark.parametrize('mutation',['dtype','shape','version','source','capacity','restore','release','duplicate','callable'])
def test_native_receipt_refuses_wrong_source_version_shape_dtype_or_lifetime(tmp_path,mutation):
    from distillschedule.training import helios_lifecycle as lifecycle
    from distillschedule.runtime.helios_checkpoint_inputs_cpu import validate_phase_receipt
    from distillschedule.workloads.best_known.registry import fingerprint
    row=native_suffix_row((0,0,0),120,7_593_984_000)
    receipt=row['native_history_phase']['checkpoint_input_cpu']
    identity=receipt['identity'];capacity=receipt['capacity'];stage_schedule=row['trace_entries'][0]['stage_schedule']
    binding=dict(config=candidate_config(tmp_path),identity=dict(source_head=identity['source'],
        source_fingerprint=identity['source_fingerprint'],stream_sha256=identity['stream_sha256'],checkpoint_input_capacity=capacity))
    with lifecycle.original_training_scope(binding):
        validate_phase_receipt(receipt,with_grad=True,task_sha256=task().digest,generator_version=2,rank=0,stage_schedule=stage_schedule)
        bad=deepcopy(receipt)
        if mutation=='dtype':bad['boundaries'][0]['dtype']='torch.float32'
        elif mutation=='shape':bad['boundaries'][0]['shape'][1]-=1
        elif mutation=='version':bad['identity']['generator_version']=3
        elif mutation=='source':
            bad['identity']['source']='foreign';new_digest=fingerprint(bad['identity'])
            bad['identity_sha256']=new_digest
            for record in bad['boundaries']:
                record['source']='foreign';record['identity_sha256']=new_digest
        elif mutation=='capacity':bad['capacity']['rank_pageable_limit_bytes']+=1
        elif mutation=='restore':bad['stats']['restore_count']-=1
        elif mutation=='release':bad['stats']['live_cpu_bytes']=1
        elif mutation=='duplicate':bad['boundaries'][0]=bad['boundaries'][1]
        else:bad['original_callable_restored']=False
        with pytest.raises(ValueError):validate_phase_receipt(bad,with_grad=True,task_sha256=task().digest,generator_version=2,rank=0,
            stage_schedule=stage_schedule)


@pytest.mark.parametrize('failure', ['retained_graph', 'wrapper_replaced'])
def test_normal_exit_retention_or_wrapper_replacement_cleans_context_and_all_leases(tiny_contract, failure):
    from distillschedule.runtime.helios_checkpoint_inputs_cpu import GCheckpointInputSession
    owner,value = session(tiny_contract)
    expected = 'retained checkpoint input graph' if failure == 'retained_graph' else 'wrapper replaced'
    with pytest.raises(ValueError, match=expected):
        with value:
            y = owner._gradient_checkpointing_func(owner.blocks[0],
                torch.randn(1,3,4,dtype=torch.float64,requires_grad=True),torch.zeros(1,1,4,dtype=torch.float64))
            if failure == 'wrapper_replaced':
                y.sum().backward()
                owner._gradient_checkpointing_func = lambda *args:None
    receipt = value.receipt()
    assert receipt['closed'] and receipt['original_callable_restored']
    assert receipt['stats']['live_cpu_bytes'] == receipt['stats']['live_leases'] == 0
    if failure == 'retained_graph':
        assert receipt['boundaries'][0]['release_outcome'] == 'cancelled'
        with pytest.raises(RuntimeError, match='cancelled|closed'):y.sum().backward()
    fresh = GCheckpointInputSession(owner=owner, blocks=owner.blocks, capacity=value.capacity,
        identity=value.identity, device='cpu')
    with fresh:
        owner._gradient_checkpointing_func(owner.blocks[0],
            torch.randn(1,3,4,dtype=torch.float64,requires_grad=True),torch.zeros(1,1,4,dtype=torch.float64)).sum().backward()
    assert fresh.receipt()['stats']['live_leases'] == 0


def train_native_prefixes(owner, exits, scope=None, optimizer=None):
    """CPU composition of the original single-section/no-grad-prefix graph.

    Each stage's updating exit runs once, including dead earlier stages. Only
    the last stage supplies a loss; a later no-grad prefix severs earlier exits.
    This is not an invocation of the original sampler or native GPU proof.
    """
    from contextlib import nullcontext
    optimizer = optimizer or torch.optim.AdamW(owner.parameters(), lr=.001)
    optimizer.zero_grad(set_to_none=True)
    noise = [torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        for shape in ((1,3,4), (1,5,4), (1,7,4))]
    shared = torch.randn(1,1,4, dtype=torch.float64, requires_grad=True)
    outputs = []
    with scope if scope is not None else nullcontext():
        previous = None
        for x, exit_index in zip(noise, exits, strict=True):
            if previous is not None:
                x = x + torch.nn.functional.interpolate(previous.transpose(1,2),
                    size=x.shape[1], mode='nearest').transpose(1,2)
            for index in range(exit_index + 1):
                with torch.set_grad_enabled(index == exit_index):
                    for block in owner.blocks:
                        x = owner._gradient_checkpointing_func(block, x, shared)
            previous = x
            # Never retain the dead graph just for an audit output.
            outputs.append(previous.detach().clone())
        previous.square().sum().backward()
        grads = {name:p.grad.clone() for name,p in owner.named_parameters()}
        input_grads = [x.grad.clone() if x.grad is not None else None for x in (*noise,shared)]
        optimizer.step()
    return dict(outputs=outputs, grads=grads, input_grads=input_grads,
        parameters={name:p.detach().clone() for name,p in owner.named_parameters()},
        adam={name:{k:v.clone() if torch.is_tensor(v) else v for k,v in optimizer.state[p].items()}
            for name,p in owner.named_parameters()}, rng=torch.get_rng_state(), calls=[b.calls for b in owner.blocks])


@pytest.mark.parametrize('exits,first_live,restores', [((0,1,1),2,2), ((0,1,0),1,4), ((0,0,0),0,6)])
def test_native_prefix_suffix_preserves_outputs_all_gradients_adam_rng_and_boundary_lifetimes(
        tiny_contract, exits, first_live, restores):
    from test_training_end_to_end import equal
    torch.manual_seed(19);control,_ = session(tiny_contract)
    torch.manual_seed(19);candidate,offload = session(tiny_contract)
    torch.manual_seed(72);expected = train_native_prefixes(control, exits)
    torch.manual_seed(72);actual = train_native_prefixes(candidate, exits, offload)
    assert equal(actual, expected)
    receipt = offload.receipt();stats = receipt['stats']
    assert stats['checkpoint_calls'] == stats['cpu_copy_count'] == stats['release_count'] == 6
    assert stats['restore_count'] == restores
    assert stats['restored_bytes'] == 2*sum(1*t*4*8 for t in (3,5,7)[first_live:])
    assert stats['restored_release_count'] == restores and stats['unused_release_count'] == 6-restores
    assert stats['cancel_count'] == stats['live_cpu_bytes'] == stats['live_leases'] == 0
    assert len(receipt['boundaries']) == 6
    for boundary in receipt['boundaries']:
        live = boundary['stage_index'] >= first_live
        assert boundary['shape'] == tiny_contract['shapes'][boundary['stage_index']]
        assert boundary['restore_count'] == int(live) and boundary['release_count'] == 1
        assert boundary['release_outcome'] == ('restored' if live else 'unused')


def native_suffix_row(exits, restores, restored_bytes):
    """Metadata-only native receipt validator fixture, not CUDA execution."""
    from distillschedule.training import helios_checkpoint_cpu as policy
    from distillschedule.runtime import helios_native_safe_history as history, helios_native_history as original
    from distillschedule.runtime.helios_checkpoint_inputs_cpu import SCHEMA
    from distillschedule.workloads.best_known.registry import fingerprint
    from test_training_helios_native_history import controls, POLICY
    value = task()
    sequence = deepcopy(controls(value)[0].operation('step/10/generator/rollout')['control']['pyramid_stage_call_sequence'])
    for item, exit_index in zip(sequence, exits, strict=True):
        item.update(denoising_exit=exit_index, call_count=exit_index+1, timestep=item['timestep'][:exit_index+1])
    first_live = max([0] + [stage for stage in (1,2) if exits[stage] > 0])
    identity = dict(source='a'*40, source_fingerprint='b'*64, stream_sha256='c'*64,
        task_sha256=value.digest, generator_version=2, rank=0)
    digest = fingerprint(identity)
    boundaries = [dict(**identity, identity_sha256=digest, block_index=block, stage_index=stage,
        shape=shape, dtype='torch.bfloat16', device='cuda:0', input_version=0,
        restore_count=int(stage >= first_live), release_count=1,
        release_outcome='restored' if stage >= first_live else 'unused')
        for stage,shape in enumerate(policy.native_contract()['shapes']) for block in range(40)]
    receipt = dict(schema_version=SCHEMA, policy=PRESET, enabled=True, closed=True, original_callable_restored=True,
        identity=identity, identity_sha256=digest, capacity=policy.capacity_template(),
        native_contract=policy.native_contract(), boundaries=boundaries,
        stats=dict(checkpoint_calls=120, cpu_copy_count=120, restore_count=restores, release_count=120,
            restored_release_count=restores, unused_release_count=120-restores, cancel_count=0,
            live_cpu_bytes=0, live_leases=0, copied_bytes=7_593_984_000, restored_bytes=restored_bytes,
            peak_live_cpu_bytes=7_593_984_000, peak_live_leases=120, failed_pack_count=0))
    entry = dict(native_history_policy=POLICY, native_history_semantics=original.SEMANTICS, source_rank=0,
        all_history_zero=True, stage_schedule=sequence, native_task_sha256=value.digest)
    phase = dict(schema_version=history.SCHEMA, execution_requirement=history.REQUIREMENT, action_id='g',
        task_sha256=value.digest, with_grad=True, unshard_count=41, reshard_count=41,
        backward_count=1, commit_count=1, restored=True, mesh_policy='node-hsdp', mesh_shape=[2,8],
        checkpoint_input_cpu=receipt)
    return dict(action_id='g', source_rank=0, physical_versions_before=dict(generator=2),
        trace_entries=[entry], native_history_phase=phase)


@pytest.mark.parametrize('exits,restores,restored_bytes', [
    ((0,1,1),40,4_521_984_000), ((0,1,0),80,6_389_760_000), ((0,0,0),120,7_593_984_000)])
def test_native_phase_receipt_uses_validated_executed_schedule_not_full_capacity(exits, restores, restored_bytes):
    from distillschedule.runtime.helios_native_safe_history import validate_action
    row = native_suffix_row(exits, restores, restored_bytes)
    before = deepcopy(row)
    validate_action(row)
    assert row == before, 'receipt validation must not rewrite the native schedule or outcomes'


@pytest.mark.parametrize('mutation', ['wrong_schedule', 'missing_required_restore', 'extra_dead_restore',
    'duplicate_unpack', 'balanced_misassigned_restore', 'wrong_stage', 'wrong_release_outcome', 'unvalidated_exit'])
def test_native_suffix_receipt_rejects_wrong_schedule_or_each_missing_required_boundary(mutation):
    from distillschedule.runtime.helios_native_safe_history import validate_action
    row = native_suffix_row((0,1,1), 40, 4_521_984_000)
    validate_action(row)
    bad = deepcopy(row)
    receipt = bad['native_history_phase']['checkpoint_input_cpu']; stats = receipt['stats']
    if mutation == 'wrong_schedule':
        stage = bad['trace_entries'][0]['stage_schedule'][2]
        stage.update(denoising_exit=0, call_count=1, timestep=stage['timestep'][:1])
        # Forge aggregate counts/bytes to match the injected row exit. They
        # must NOT hide the actual stage-1 unused releases in the boundaries.
        stats.update(restore_count=80, restored_release_count=80, unused_release_count=40,
            restored_bytes=6_389_760_000)
    elif mutation == 'unvalidated_exit':
        bad['trace_entries'][0]['stage_schedule'][2]['denoising_exit'] = -1
    else:
        record = receipt['boundaries'][80 if mutation in ('missing_required_restore','duplicate_unpack') else 0]
        size = 1*record['shape'][1]*5120*2
        if mutation == 'missing_required_restore':
            record.update(restore_count=0, release_outcome='unused')
            stats['restore_count'] -= 1; stats['restored_bytes'] -= size
            stats['restored_release_count'] -= 1; stats['unused_release_count'] += 1
        elif mutation == 'extra_dead_restore':
            record.update(restore_count=1, release_outcome='restored')
            stats['restore_count'] += 1; stats['restored_bytes'] += size
            stats['restored_release_count'] += 1; stats['unused_release_count'] -= 1
        elif mutation == 'duplicate_unpack':
            record['restore_count'] = 2
            stats['restore_count'] += 1; stats['restored_bytes'] += size
        elif mutation == 'balanced_misassigned_restore':
            receipt['boundaries'][80].update(restore_count=0, release_outcome='unused')
            receipt['boundaries'][81]['restore_count'] = 2
            # Keep every aggregate count and byte unchanged. Each live block
            # still has its own required replay; a duplicate cannot repay it.
        elif mutation == 'wrong_stage': record['stage_index'] = 2
        else: record['release_outcome'] = 'unused' if record['release_outcome'] == 'restored' else 'restored'
    message = 'shape/dtype/version/source/capacity census' if mutation in (
        'wrong_schedule','balanced_misassigned_restore') else None
    with pytest.raises(ValueError, match=message): validate_action(bad)
