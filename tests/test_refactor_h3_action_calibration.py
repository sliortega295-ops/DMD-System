"""H3 calibration metadata uses the shared collector, not a model fixture cost."""
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest

from test_refactor_wan_stage_calibration import FakeBackend, SOURCE, DEVICE
from test_refactor_wan_action_calibration import Tensor
from distillschedule.search.h3_action_calibration import H3ActionCalibrator, shallow_prepared
from distillschedule.search.h3_action_costs import action_graph, component_contract, PRECISION
from distillschedule.search.h3_implementation import H3Implementation
from distillschedule.search.wan_action_calibration import validate_lifetimes


def fixture():
    backend = FakeBackend()
    backend.is_tensor = lambda x: isinstance(x, Tensor)
    impl = H3Implementation(2, 128, 3)
    c = H3ActionCalibrator(torch=backend, distributed=backend, device=DEVICE, rank=0,
        implementation=impl, source_head=SOURCE, clock=backend.clock, evidence_scope="CPU_FIXTURE_ONLY")
    return backend, c, impl


def work(kind):
    roles = ([('student', False)]*3 + [('student', True), ('fake', False), ('teacher', False)]
             if kind == 'g' else [('student', False)]*4 if kind == 'r' else [('fake', True)])
    return [dict(role=r, grad_enabled=g, latents=[{'shape':[1, 2], 'dtype':'torch.bfloat16'}],
                 condition={'prompt_embeds': {'shape':[1, 12, 16], 'dtype':'torch.bfloat16'}},
                 latent_shape={'video': [1, 2]}, sigmas=[{'shape':[], 'dtype':'torch.float32'}]) for r,g in roles]


def test_h3_22_actions_and_real_retained_storage_metadata_no_tensor_copies():
    b,c,impl=fixture()
    for n,op in enumerate(action_graph(3).operations):
        kind=op.metadata['action'][0]
        with c.stage(op.metadata['cycle'],op.metadata['action']) as sample:
            for row in work(kind): c.note_forward(row)
            b.body()
        if kind=='r': sample.record_output(Tensor(100+n,20))
    value=c.finish()
    assert value['status']=='PASS' and len(value['stages'])==22
    assert value['workload']=='minimax_h3_grid1000_best'
    assert value['precision_fingerprint']==PRECISION
    assert len(value['output_lifetimes'])==10 and value['peak_retained_input_bytes']==60
    assert value['tensor_value_reads']==0
    validate_lifetimes(value)
    for op,row in zip(action_graph(3).operations,value['stages']):
        assert row['component_contract']==component_contract(op,impl,value['work_signatures'][op.operation_id]).to_dict()
    # These are measured allocation envelopes, NOT proof that framework/cache
    # state growth is pure ephemeral workspace.
    assert 'PERSISTENT_GROWTH' in value['allocation_limitations']


def test_missing_call_inventory_is_a_failed_calibration_not_an_inferred_cost():
    _,c,_=fixture()
    with pytest.raises(ValueError,match='inventory'):
        with c.stage(1,'g'): pass
    assert c.snapshot()['status']=='FAIL'


def test_prepared_projection_is_shallow_not_dataclasses_asdict_tensor_copy():
    @dataclass
    class Item:
        generated: object
        condition: object
        cache_key: tuple
    t=Tensor(7,100)
    item=Item(t,{'x':t},('score',1))
    projected=shallow_prepared([item])
    assert projected[0]['generated'] is t and projected[0]['condition']['x'] is t


def test_metadata_draw_observer_never_reads_tensor_values_or_masquerades_as_full_proof():
    from distillschedule.backends.training.minimax_h3.cycle_actions import tensor_evidence
    t=Tensor(1,20)
    t.numel=lambda:10
    t.element_size=lambda:2
    backend=NS(is_tensor=lambda x:x is t)
    value=tensor_evidence(backend,(t,t),value_mode='metadata')
    assert len(value)==2 and value[0]['bytes']==20
    assert 'sha256' not in value[0] and 'finite' not in value[0]
    # Default correctness path must still access values, not silently default off.
    with pytest.raises(AttributeError,match='detach'):
        tensor_evidence(backend,t)


def test_metadata_receipt_validates_actual_inventory_but_is_not_value_parity():
    from test_refactor_h3_cycle_actions import action_receipt
    from distillschedule.backends.training.minimax_h3.cycle_actions import validate_cycle_actions, compare_action_inputs
    full=action_receipt(3)
    from copy import deepcopy
    metadata=deepcopy(full)
    metadata['tensor_value_reads']='DISABLED_METADATA_ONLY_PRIOR_CORRECTNESS_REQUIRED'
    metadata['status']='PASS_NATIVE_METADATA_ACTION_BINDING_VALUES_NOT_RECHECKED'
    for row in metadata['rows']:
        for draw in row['draws']:
            for tensor in draw['tensors']:
                del tensor['sha256'];del tensor['finite']
    validate_cycle_actions(metadata)
    assert compare_action_inputs(metadata,full)['status']=='FAIL'
    bad=deepcopy(metadata)
    bad['rows'][0]['draws'].pop()
    with pytest.raises(ValueError,match='inventory'):validate_cycle_actions(bad)
