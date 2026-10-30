"""Real stage2 helper parity and causal checkpoint-saved storage liveness."""
import itertools
import json
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")

from helios_rollout_liveness_fixtures import (ROOT, POST, REARRANGE_PROVIDER, alias_audit, byte_equal,
    execute, load, paired, rearrange, saved_at_grad_entry)


@pytest.fixture(scope="module", autouse=True)
def _source_fixture(tmp_path_factory):
    """Use bundled live bytes; preserve the original source by exact inverse patch."""
    import hashlib
    import helios_rollout_liveness_fixtures as fixture
    global ROOT
    repository = Path(__file__).resolve().parents[1]
    root = tmp_path_factory.mktemp("helios-rollout-liveness")
    (root / "reports").mkdir()
    names = (POST, fixture.BASE, fixture.HISTORY, fixture.SCHEDULER, "runtime/official_baseline.py")
    for name in names:
        current = (repository / name).read_bytes()
        destination = root / "original" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name == POST:
            assert hashlib.sha256(current).hexdigest() == "3bb57f4aaa65c59cf2f928765be67e601d8bec69bd3aeb384cbe4e8982724488"
            text = current.decode()
            for after, before in (
                ("start_point_list = [noisy_model_input.detach()]", "start_point_list = [noisy_model_input]"),
                ("start_point_list.append(noisy_model_input.detach())", "start_point_list.append(noisy_model_input)"),
            ):
                assert text.count(after) == 1
                text = text.replace(after, before)
            original = text.encode()
            assert hashlib.sha256(original).hexdigest() == fixture.EXPECTED_POST
            destination.write_bytes(original)
            candidate = root / "candidate" / name
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(current)
        else:
            destination.write_bytes(current)
    previous = fixture.ROOT
    initialized = torch.cuda.is_initialized()
    fixture.ROOT = ROOT = root
    try:
        yield
    finally:
        fixture.ROOT = previous
        assert torch.cuda.is_initialized() == initialized


@pytest.fixture(autouse=True)
def _execution_state_isolation(monkeypatch):
    # The native contract uses the existing PSD-safe block-noise sampler.
    # Do not rely on a caller shell, or change the upstream covariance/math.
    monkeypatch.setenv("DISTILLSCHEDULE_STABLE_BLOCK_NOISE", "1")
    from helios_rollout_liveness_fixtures import isolated_cpu_test_state
    with isolated_cpu_test_state():
        yield


def test_liveness_fixture_restores_rng_threads_and_import_namespaces():
    import random
    import sys
    import numpy as np
    from helios_rollout_liveness_fixtures import isolated_cpu_test_state
    snapshot = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state(), torch.get_num_threads())
    before = {name: module for name, module in sys.modules.copy().items()
              if name == "runtime" or name.startswith("runtime.") or name.startswith("helios_liveness_scheduler_")}
    with isolated_cpu_test_state():
        paired((0, 1, 1), hooks=False)
    after = {name: module for name, module in sys.modules.copy().items()
             if name == "runtime" or name.startswith("runtime.") or name.startswith("helios_liveness_scheduler_")}
    assert before == after
    assert torch.equal(snapshot[0], torch.get_rng_state())
    assert snapshot[1] == random.getstate()
    numpy = np.random.get_state()
    assert snapshot[2][0] == numpy[0] and np.array_equal(snapshot[2][1], numpy[1]) and snapshot[2][2:] == numpy[2:]
    assert snapshot[3] == torch.get_num_threads()


def test_original_alias_audit():
    alias_audit(ROOT/'original'/POST,candidate=False)


def test_candidate_changes_only_two_stored_aliases():
    original=(ROOT/'original'/POST).read_text()
    expected=original.replace('start_point_list = [noisy_model_input]',
        'start_point_list = [noisy_model_input.detach()]').replace(
        'start_point_list.append(noisy_model_input)',
        'start_point_list.append(noisy_model_input.detach())')
    assert (ROOT/'candidate'/POST).read_text()==expected
    alias_audit(ROOT/'candidate'/POST,candidate=True)


@pytest.mark.parametrize('batch',[1,2])
def test_rearrange_dependency_exact_values_grads_aliases(batch):
    x=torch.arange(batch*2*2*8*8,dtype=torch.float32).reshape(batch,2,2,8,8).requires_grad_()
    left=rearrange(x,'b c t h w -> (b t) c h w')
    expected=x.permute(0,2,1,3,4).reshape(batch*2,2,8,8)
    assert byte_equal(left,expected)
    right=rearrange(left,'(b t) c h w -> b c t h w',t=2)
    assert byte_equal(right,x)
    first=torch.autograd.grad(left.square().sum(),x,retain_graph=True)[0]
    second=torch.autograd.grad(expected.square().sum(),x,retain_graph=True)[0]
    assert byte_equal(first,second)
    assert left.untyped_storage()._cdata==expected.untyped_storage()._cdata if batch==1 else True


NORMAL=list(itertools.product(range(2),repeat=3))
AMPLIFIED=list(itertools.product(range(4),repeat=3))


@pytest.mark.parametrize('exits',NORMAL)
@pytest.mark.parametrize('consistency',[False,True])
@pytest.mark.parametrize('dynamic',[False,True])
def test_all_legal_two_step_exits_bitwise_outputs_all_grads_rng_and_order(exits,consistency,dynamic):
    paired(exits,consistency=consistency,dynamic=dynamic)


@pytest.mark.parametrize('exits',AMPLIFIED)
@pytest.mark.parametrize('consistency',[False,True])
def test_all_legal_amplified_four_step_exits_bitwise(exits,consistency):
    paired(exits,amplify=True,consistency=consistency,zero_history=True,dynamic=True)


@pytest.mark.parametrize('pyramid',[0,1,2])
@pytest.mark.parametrize('exits',NORMAL)
@pytest.mark.parametrize('consistency',[False,True])
def test_multi_pyramid_intended_gradient_consumers_preserved(exits,pyramid,consistency):
    paired(exits,multi=True,pyramid=pyramid,consistency=consistency)


@pytest.mark.parametrize('seed',range(8))
@pytest.mark.parametrize('branch',['gt-nonzero','gt-zero-amplified','generated-history','generated-amplified'])
def test_actual_rng_exit_flag_block_noise_and_multisection_history(seed,branch):
    paired((0,0,0),trace=False,seed=seed+51,gt=branch.startswith('gt'),
           zero_history=branch=='gt-zero-amplified',amplify='amplified' in branch,
           sections=2,consistency=True,dynamic=True)


@pytest.mark.parametrize('options',[
    dict(trace=False,gt=False,sections=3,last_section=True,skip_first=True),
    dict(trace=False,gt=False,sections=3,last_section=False,vae=True,kv=False),
    dict(trace=False,gt=False,sections=2,corrupt=True,saturation=True),
    dict(trace=False,gt=False,sections=2,amplify=True,last_step=True),
    dict(trace=False,gt=True,zero_history=False,amplify=True,last_step=True),
    dict(trace=False,gt=True,zero_history=True,amplify=True,last_step=True),
    dict(dynamic=True,shift_type='exponential'),
    dict(keep=False,gt=True),
])
def test_additional_supported_flags_match(options):
    paired((0,1,1),**options)


@pytest.mark.parametrize('stages',[1,2])
@pytest.mark.parametrize('consistency',[False,True])
def test_one_and_two_stage_legal_exit_patterns(stages,consistency):
    for exits in itertools.product(range(2),repeat=stages):
        paired(exits,stages=stages,consistency=consistency)


def test_exit_zero_keeps_every_true_upstream_parameter_and_input_gradient():
    payload,old,new=paired((0,0,0))
    for stage in range(3):
        for kind in ('scale','a','b','bias'):
            grad=payload['grads']['parameters'][f'{kind}.{stage}']
            assert grad is not None and torch.count_nonzero(grad), (kind,stage)
    for name in ('prompt','history_short','history_mid','history_long','rollout_noise','block_noise/1','block_noise/2'):
        grad=payload['grads']['inputs'][name]
        assert grad is not None and torch.count_nonzero(grad),name
    for proof in (old,new):
        entry=saved_at_grad_entry(proof,2)
        assert all(r['storage_alive'] and r['checkpoint_box_alive'] for r in entry)


@pytest.mark.parametrize('exits',[(0,1,1),(0,0,1),(1,1,1),(0,1,0)])
def test_dead_checkpoint_saved_state_released_before_later_grad_forward(exits):
    _,old,new=paired(exits)
    target_stage=1 if exits[1]>0 else 2
    old_entry=saved_at_grad_entry(old,target_stage)
    new_entry=saved_at_grad_entry(new,target_stage)
    old_old=[r for r in old_entry if int(r['label'][1])<target_stage]
    new_old=[r for r in new_entry if int(r['label'][1])<target_stage]
    assert old_old and len(old_old)==len(new_old)
    assert all(r['storage_alive'] and r['tensor_alive'] and r['checkpoint_box_alive'] for r in old_old)
    assert all(not r['storage_alive'] and not r['tensor_alive'] and not r['checkpoint_box_alive'] for r in new_old), (
        'dead checkpoint graph still retained by stored aliases',exits,new_old)


@pytest.mark.parametrize('exits',[(0,1,1),(0,0,1),(1,1,1),(0,1,0)])
def test_default_checkpoint_cpp_storage_lifetime_without_saved_tensor_hooks(exits):
    # This control uses default PyTorch SavedVariable ownership. No outer pack
    # hook, SavedBox or strong Python reference to boundary is involved.
    _,old,new=paired(exits,hooks=False)
    target=1 if exits[1]>0 else 2
    old_entry=saved_at_grad_entry(old,target)
    new_entry=saved_at_grad_entry(new,target)
    assert old_entry and all(r['storage_alive'] for r in old_entry)
    assert all(not r['storage_alive'] for r in new_entry)
    assert all(not r['checkpoint_box_alive'] for r in old_entry+new_entry)


def test_release_occurs_after_prefix_not_prematurely_at_stage_transition():
    _,old,new=paired((0,1,1),hooks=False)
    for proof in (old,new):
        first_prefix=next(c for c in proof['calls'] if c['stage']==1 and not c['grad'])
        assert first_prefix['saved_state_at_entry']
        assert all(r['storage_alive'] for r in first_prefix['saved_state_at_entry'])
    assert all(not r['storage_alive'] for r in saved_at_grad_entry(new,1))
    assert all(r['storage_alive'] for r in saved_at_grad_entry(old,1))


def test_default_checkpoint_preserves_true_zero_exit_and_consistency_saved_storage():
    for exits,options in [((0,0,0),{}),((1,1,1),dict(consistency=True)),
                         ((0,1,1),dict(multi=True,pyramid=0))]:
        _,old,new=paired(exits,hooks=False,**options)
        for proof in (old,new):
            rows=saved_at_grad_entry(proof,2)
            stage0=[r for r in rows if r['label'].startswith('s0-')]
            assert stage0 and all(r['storage_alive'] for r in stage0)


def test_consistency_retains_its_intended_old_stage_saved_state_and_gradients():
    payload,old,new=paired((1,1,1),consistency=True)
    assert payload['outputs'][3].item()>0
    for proof in (old,new):
        entry=saved_at_grad_entry(proof,2)
        assert all(r['storage_alive'] and r['checkpoint_box_alive'] for r in entry)
    for stage in (0,1):
        assert payload['grads']['parameters'][f'a.{stage}'] is not None


def test_multi_pyramid_output_retains_its_intended_old_stage_saved_state():
    payload,old,new=paired((0,1,1),multi=True,pyramid=0)
    for proof in (old,new):
        stage0=[r for r in saved_at_grad_entry(proof,2) if r['label'].startswith('s0-')]
        assert stage0 and all(r['storage_alive'] and r['checkpoint_box_alive'] for r in stage0)
    assert payload['grads']['parameters']['a.0'] is not None


def test_unsupported_non_gt_keep_false_fails_identically():
    for mode in ('original','candidate'):
        with pytest.raises(NotImplementedError):
            execute(load(mode),(0,1,1),gt=False,keep=False)


def test_liveness_evidence_export():
    rows=[]
    for exits,options in [((0,1,1),{}),((0,0,1),{}),((0,0,0),{}),
                         ((1,1,1),dict(consistency=True)),((0,1,1),dict(multi=True,pyramid=0)),
                         ((0,1,1),dict(hooks=False)),((0,0,1),dict(hooks=False)),
                         ((0,0,0),dict(hooks=False)),((1,1,1),dict(hooks=False,consistency=True))]:
        payload,old,new=paired(exits,**options)
        def scalar_proof(p):
            return dict(calls=[{k:v for k,v in c.items() if k in ('label','stage','grad','saved_state_at_entry')}
                               for c in p['calls']],helper_return=p['helper_return'],
                        before_backward=p['before_backward'],after_backward=p['after_backward'],unpacks=p['unpacks'])
        rows.append(dict(exits=exits,options=options,original=scalar_proof(old),candidate=scalar_proof(new),
                         parameter_grad_present={k:v is not None for k,v in payload['grads']['parameters'].items()},
                         core_call_order=payload['core_calls'],helper_call_order=payload['helper_call_order']))
    (ROOT/'reports/liveness-witnesses.json').write_text(json.dumps(dict(
        evidence='actual CPU non-reentrant torch checkpoint saved-state ownership + C++ StorageWeakRef expiration',
        rearrange_provider=REARRANGE_PROVIDER,torch=torch.__version__,cases=rows),indent=2)+'\n')



def nonzero(tensor):
    return tensor is not None and bool(torch.count_nonzero(tensor))


@pytest.mark.parametrize('sections,critic_frames',[(2,4),(3,4),(3,6)])
@pytest.mark.parametrize('hooks',[False,True])
def test_cross_section_generated_history_keeps_required_grads_and_only_frees_dead_state(sections,critic_frames,hooks):
    payload,old,new=paired((0,0,0),trace=False,gt=False,sections=sections,critic_frames=critic_frames,
        consistency=False,hooks=hooks,seed=1,observe_sections=True,objective_last_frames=2)
    # Actual native flags, not synthesized denoising control. Stage1's prefix
    # makes current-section stage0 dead; stage2 exit0 preserves stage1's path.
    assert [row['value'] for row in payload['flag_calls']]==[1,1,0]
    assert payload['outputs'][0].shape[2]==critic_frames
    assert payload['outputs'][3].item()==0., 'consistency must not explain old graph retention'
    observer=payload['section_observer']
    section_by_label=observer['section_by_label']
    expected_grad_sections=list(range(max(0,sections-(critic_frames+1)//2),sections))
    actual_grad_sections=sorted({section_by_label[c['label']] for c in old['calls'] if c['grad']})
    assert actual_grad_sections==expected_grad_sections and len(actual_grad_sections)>=2
    prior_sections=actual_grad_sections[:-1]
    witnesses=[]
    for section in prior_sections:
        prior_stage2=next(c for c in old['calls'] if c['grad'] and c['stage']==2
                          and section_by_label[c['label']]==section)
        label=prior_stage2['label']
        # Distinct per-forward parameter-use identity view: not a shared-stage
        # Parameter's aggregate gradient. The actual model input is observed
        # with a tensor hook that captures no tensor/root reference.
        assert nonzero(observer['parameter_use_grads'].get(label)), label
        assert nonzero(observer['input_grads'].get(label)), label
        later=next(c for c in new['calls'] if c['grad'] and c['stage']==1
                   and section_by_label[c['label']]==section+1)
        for proof in (old,new):
            call=next(c for c in proof['calls'] if c['label']==later['label'])
            required=next(r for r in call['saved_state_at_entry'] if r['label']==label)
            assert required['storage_alive'], (section,proof is new,required)
            if hooks:
                assert required['checkpoint_box_alive']
        witnesses.append(dict(prior_section=section,label=label,
            parameter_use_gradient=float(observer['parameter_use_grads'][label]),
            model_input_gradient_abs_sum=float(observer['input_grads'][label].abs().sum()),
            required_storage_live_at_next_section_stage1_grad=True))

    final_section=sections-1
    history_call=next(c for c in new['calls'] if c['grad'] and c['stage']==2
                      and section_by_label[c['label']]==final_section)
    history_grad=observer['history_short_grads'][history_call['label']]
    assert nonzero(history_grad[:,:,:1]), 'image_latents prefix input gradient missing'
    assert nonzero(history_grad[:,:,1:]), 'generated history1x input gradient missing'

    for section in actual_grad_sections:
        old_stage0=next(c for c in old['calls'] if c['grad'] and c['stage']==0
                       and section_by_label[c['label']]==section)['label']
        old_stage1=next(c for c in old['calls'] if c['grad'] and c['stage']==1
                       and section_by_label[c['label']]==section)
        new_stage1=next(c for c in new['calls'] if c['label']==old_stage1['label'])
        before=next(r for r in old_stage1['saved_state_at_entry'] if r['label']==old_stage0)
        after=next(r for r in new_stage1['saved_state_at_entry'] if r['label']==old_stage0)
        assert before['storage_alive'] and not after['storage_alive']
        assert old_stage0 not in observer['parameter_use_grads'], 'supposedly dead forward received gradient'

    def timeline(proof):
        return [dict(label=c['label'],section=section_by_label[c['label']],stage=c['stage'],grad=c['grad'],
                     saved_state_at_entry=c['saved_state_at_entry']) for c in proof['calls']]
    report=dict(scope='tiny CPU actual stage2 helper; generated history, >=2 grad sections; not full-model/native proof',
        sections=sections,critic_frames=critic_frames,latent_window=2,consistency=False,
        objective='same scalar formula on last2 returned frames ONLY; no direct earlier-frame objective',
        actual_exits=[1,1,0],gradient_sections=actual_grad_sections,
        default_checkpoint_without_saved_tensor_hooks=not hooks,prior_section_grad_witnesses=witnesses,
        final_history_image_prefix_gradient_abs_sum=float(history_grad[:,:,:1].abs().sum()),
        final_history1x_gradient_abs_sum=float(history_grad[:,:,1:].abs().sum()),
        original_timeline=timeline(old),candidate_timeline=timeline(new),
        equality='bitwise all outputs/loss/all parameter+input+per-use gradients/RNG/helper+model+recompute+scheduler order',
        parameter_observer='distinct per-forward scalar view, same value/storage; no extra parameter or strong graph root',
        image_prefix_origin_is_gradient_bearing=0 in actual_grad_sections)
    path=ROOT/'reports'/f'history-grad-sections{sections}-critic{critic_frames}-hooks{int(hooks)}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
