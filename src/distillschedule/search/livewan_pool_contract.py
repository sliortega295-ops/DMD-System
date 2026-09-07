"""Source-bound LiveWan C34 pool DAG, NOT a calibrated/executable search arm.

This is the missing structural bridge from native action observations. It uses
ExecutionGraph, but deliberately cannot construct a PlanningProblem: output
aliases, workspace and native release times are not measured yet. Never replace
those unknowns with zero or reuse coupled commit costs as independent services.
"""
from __future__ import annotations

from copy import deepcopy
import re

from distillschedule.core.enums import GradientMode as Grad, OperationKind as Kind
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation
from distillschedule.workloads.best_known.registry import fingerprint
from distillschedule.workloads.livewan_crossscale import TRACE_BALANCED_8G8S_SLOT_PAIRS
from .contracts import require
from .livewan_action_observation import validate_action_observations
from .livewan_implementation import LiveWanImplementation
from .plan_ir import graph_from_dict, validate_versions

SCHEMA = 'distillschedule.livewan_observed_pool_contract.v1'
G, S = tuple(range(8)), tuple(range(8, 16))


def build_pool_contract(raws, *, expected_head):
    """Reuse completed native observations; this function launches no workers."""
    require(isinstance(expected_head,str) and re.fullmatch('[0-9a-f]{40}',expected_head), 'invalid observed source HEAD')
    gate=validate_action_observations(raws,expected_head=expected_head)
    raws=sorted(raws,key=lambda r:r['rank'])
    require(raws[0]['implementation']==LiveWanImplementation('pipeline',True,True,True).id, 'only observed C34 is admitted')
    samples={}
    for raw in raws:
        if raw['rank'] >= 8:
            continue
        lane=raw['rank']
        for row in raw['rows']:
            m=row.get('metadata',{})
            if m.get('action_kind') != 'generator_rollout':
                continue
            sample=m['rows'][0]
            wave=TRACE_BALANCED_8G8S_SLOT_PAIRS[lane].index(sample['logical_slot'])
            samples.setdefault(f"{sample['logical_step']}.{wave}",{})[str(lane)]=deepcopy(sample)
    evidence=dict(source_head=expected_head,implementation=raws[0]['implementation'],
        native_observations_sha256=fingerprint(sorted(raws,key=lambda r:r['rank'])),
        samples=samples,latent_contract=deepcopy(raws[0]['native_transport_contract']['runahead']),
        shadow_contract=deepcopy(raws[0]['native_transport_contract']['shadow']),
        gate_status=gate['status'])
    result=_build(evidence)
    validate_pool_contract(result)
    return result


def _build(evidence):
    """Only the frozen C34 order; labels distinguish policy from true data edges."""
    ops, edges, messages, storage = [], [], [], []
    edge_keys=set()
    latent=evidence['latent_contract']['payload_bytes']
    shadow=evidence['shadow_contract']['payload_bytes']
    host_keys={(s,w) for s in range(10) if s%5 for w in (0,1)}|{(5,0)}

    def op(key, step, role, kind, family, source, *, read=None, write=None, batch=1, **meta):
        ranks=G if role.startswith('generator') else S if role.startswith('score') else G+S
        ops.append(Operation(key,step,role,kind,
            Grad.GRAD if family in ('generator_backward','critic_backward') else Grad.NO_GRAD,
            required_versions=read or {},produced_versions=write or {},
            metadata=dict(cycle=step//5+1,family=family,native_source_boundary=source,ranks=ranks,
                          physical_batch_per_rank=batch,**meta)))
        return key

    def edge(a,b,classification='semantic',reason='native_data_or_version'):
        if (a,b,classification,reason) not in edge_keys:
            edges.append(Dependency(a,b,reason,dict(classification=classification)))
            edge_keys.add((a,b,classification,reason))

    def chain(keys,classification='frozen_native_order'):
        for a,b in zip(keys,keys[1:]):edge(a,b,classification,'source_invocation_order')

    def store(id, *, owner, producer, consumers, release, space='cuda', observed=None, **meta):
        storage.append(dict(id=id,owner_pool=owner,memory_space=space,producer=producer,
            consumers=list(consumers),last_semantic_consumer=consumers[-1] if consumers else None,
            native_release_boundary=release,observed_bytes=observed,
            owning_ranks=G if owner in ('G','generator_pool') else S if owner in ('Score','score_pool') else G+S,
            byte_scope='PER_OWNING_RANK_NOT_POOL_SUM',
            native_release_scope='NAMED_REFERENCE_ONLY_ALIASED_STORAGE_RELEASE_UNRESOLVED',
            actual_storage_release='UNRESOLVED',
            observation_status='MISSING' if observed is None else 'NATIVE_PAYLOAD_CENSUS_ONLY',
            allocation_boundary='producer',async_completion_handles=[],**meta))

    # One pool invocation represents eight native rank calls, not a physical B8.
    for step in range(10):
        gv=step//5+int(step%5!=0)
        for wave in (0,1):
            p=f's{step}';suffix=f'w{wave}';r=f'{p}.g.rollout.{suffix}'
            rs=evidence['samples'][f'{step}.{wave}']
            op(r,step,'generator_pool',Kind.GENERATOR_ROLLOUT,'rollout','prepare_rollout',
               read={'generator':gv},rank_samples=rs,produced_with_generator_version=gv)
            host=(step,wave) in host_keys
            issue=f'{p}.latent.issue.{suffix}';ready=f'{p}.latent.ready.{suffix}';done=f'{p}.latent.complete.{suffix}'
            delivery=f'{p}.latent.delivery.{suffix}'
            op(delivery,step,'link',Kind.VERSION_SYNC,'protocol_delivery',
               'LOGICAL_MESSAGE_DELIVERY_NOT_OBSERVED_HOST_RETURN_ORDER',wave=wave)
            op(issue,step,'generator_pool',Kind.VERSION_SYNC,'latent_submit',
               'PinnedHostRunaheadTransport.enqueue_send' if host else '_send_pair',wave=wave)
            op(ready,step,'score_pool',Kind.VERSION_SYNC,'latent_ready',
               'PinnedHostRunaheadTransport.receive_to_device' if host else '_recv_pair',wave=wave)
            op(done,step,'generator_transport',Kind.VERSION_SYNC,'latent_complete',
               'PinnedHostRunaheadTransport._send_after_copy' if host else 'native NCCL send completion',wave=wave)
            edge(r,issue);edge(issue,done,'native_transport_safety','send_before_completion')
            if host:
                copy=f'{p}.latent.d2h_ready.{suffix}'
                op(copy,step,'generator_transport',Kind.VERSION_SYNC,'latent_d2h_ready','native copy event',wave=wave)
                edge(issue,copy,'native_transport_safety','source_stream_order')
                edge(copy,delivery,'native_transport_safety','copy_before_gloo_delivery')
                edge(delivery,done,'native_transport_safety','delivery_before_sender_completion')
                edge(delivery,ready,'native_transport_safety','delivery_before_receiver_h2d')
                store(f'host_source.{p}.{suffix}',owner='generator_pool',producer=issue,
                      consumers=[copy,done],release=done,known_payload_bytes=latent,
                      alias_status='UNRESOLVED_CONTIGUOUS_MAY_ALIAS_PREPARED_X0')
                storage[-1]['async_completion_handles']=[done]
            else:
                edge(issue,delivery,'native_transport_safety','matched_nccl_delivery')
                edge(delivery,ready,'native_transport_safety','delivery_before_receiver_return')
                edge(delivery,done,'native_transport_safety','delivery_before_sender_return')
            messages.append(dict(key=[step,wave],path='pinned_gloo' if host else 'nccl_latent',
                payload='detached_rollout_latent',bytes_per_lane=latent,lanes=8,
                source=r,submit=issue,delivery=delivery,ready=ready,completion=done,
                peer_host_return_order='NOT_ESTABLISHED',
                sample_bindings=deepcopy(rs),produced_with_generator_version=gv))
            consumers=([f'{p}.g.backward.{suffix}'] if step%5==0 else [done])
            store(f'prepared.{p}.{suffix}',owner='generator_pool',producer=r,consumers=consumers,
                  release=('cycle2.g.clear_precomputed_first' if (step,wave)==(5,0) else
                           f'cycle{step//5+1}.g.role_return' if step%5==0 else 'UNRESOLVED_NATIVE_LOCAL_AND_ASYNC_REFERENCES'),
                  known_x0_payload_bytes=latent,includes_unmeasured=['raw_context','replay_records','streamer_KV_aliases'])
            store(f'score_latent.{p}.{suffix}',owner='score_pool',producer=ready,
                  consumers=([f'{p}.s.teacher.{suffix}'] if step%5==0 else [])+[f'{p}.c.backward'],
                  release='UNRESOLVED_NATIVE_LOCAL_REFERENCES',known_payload_bytes=latent)
        cb=f's{step}.c.backward';cc=f's{step}.c.commit'
        op(cb,step,'score_pool',Kind.CRITIC_BACKWARD,'critic_backward','critic_backward_for_batch',
           read={'critic':step},batch=2,produced_with_generator_version=gv,
           rank_samples={str(i+8):[evidence['samples'][f'{step}.{w}'][str(i)] for w in (0,1)] for i in G},
           coupled_native_reducer='ASYNC_BUCKETS_INSIDE_BACKWARD_FINISH_AT_COMMIT')
        op(cc,step,'score_pool',Kind.CRITIC_UPDATE,'critic_commit','reducer.finish -> clip -> optimizer.step',
           read={'critic':step},write={'critic':step+1},batch=2)
        for w in (0,1):edge(f's{step}.latent.ready.w{w}',cb)
        edge(cb,cc)
        if step:
            edge(f's{step-1}.c.commit',f's{step}.latent.ready.w0','frozen_native_order')
            edge(f's{step-1}.c.commit',cb,'semantic','critic_optimizer_order')
        if step%5:
            chain([f's{step}.latent.ready.w0',f's{step}.latent.ready.w1',cb])
        store(f'b2_inputs.s{step}',owner='score_pool',producer='UNRESOLVED_NATIVE_B2_ASSEMBLY_BEFORE_BACKWARD',
              availability_anchors=[f's{step}.latent.ready.w0',f's{step}.latent.ready.w1'],
              consumers=[cb],release='UNRESOLVED_NATIVE_B2_ASSEMBLY_AND_CONTEXT_REFERENCES',
              includes_unmeasured=['x0_cat','context_cat','flow','gradient_workspace'])
        storage[-1]['allocation_boundary']='UNRESOLVED_NATIVE_ASSEMBLY_AFTER_TEACHER1_SUBMIT_FOR_STEP0'

    for cycle in (0,1):
        s=cycle*5;p=f's{s}'
        for w in (0,1):
            suffix=f'w{w}';f=f'{p}.g.fake.{suffix}';t=f'{p}.s.teacher.{suffix}'
            combine=f'{p}.g.combine.{suffix}';back=f'{p}.g.backward.{suffix}'
            ts=f'{p}.teacher.issue.{suffix}';tr=f'{p}.teacher.ready.{suffix}';td=f'{p}.teacher.complete.{suffix}'
            delivery=f'{p}.teacher.delivery.{suffix}'
            op(delivery,s,'link',Kind.VERSION_SYNC,'protocol_delivery',
               'LOGICAL_MESSAGE_DELIVERY_NOT_OBSERVED_HOST_RETURN_ORDER')
            op(f,s,'generator_pool',Kind.SCORE_FAKE,'fake_velocity','context -> flow -> _dmd_fake_velocity',
               read={'shadow_epoch':cycle},native_critic_version=s)
            op(t,s,'score_pool',Kind.SCORE_REAL_CONDITIONAL,'teacher_velocity','local context -> flow -> _dmd_teacher_velocity',read={'teacher':0})
            op(ts,s,'score_pool',Kind.VERSION_SYNC,'teacher_submit','dist.send' if w==0 else 'dist.isend')
            op(tr,s,'generator_pool',Kind.VERSION_SYNC,'teacher_ready','dist.recv')
            op(td,s,'score_pool',Kind.VERSION_SYNC,'teacher_complete','dist.send return' if w==0 else 'grad1_work.wait')
            op(combine,s,'generator_pool',Kind.DMD_GRADIENT,'combine','_combine_dmd_gradient')
            op(back,s,'generator_pool',Kind.GENERATOR_BACKWARD,'generator_backward','generator_backward_for_sample',read={'generator':cycle})
            edge(f'{p}.g.rollout.{suffix}',f);edge(f'{p}.g.rollout.{suffix}',back)
            edge(f'{p}.latent.ready.{suffix}',t);edge(t,ts);edge(ts,delivery,'native_transport_safety')
            edge(delivery,tr,'native_transport_safety');edge(delivery,td,'native_transport_safety');edge(tr,combine);edge(f,combine);edge(combine,back)
            messages.append(dict(key=[s,w],path='nccl_teacher_velocity',payload='teacher_velocity',
                bytes_per_lane=latent,lanes=8,source=t,submit=ts,delivery=delivery,ready=tr,completion=td,
                peer_host_return_order='NOT_ESTABLISHED'))
            if w==1:
                edge(f'{p}.c.backward',td,'frozen_native_order','post_backward_send_wait')
                edge(td,f'{p}.c.commit','native_transport_safety','retain_send_until_complete')
            store(f'teacher_send.{p}.{suffix}',owner='score_pool',producer=ts,consumers=[tr,td],
                  release=td,known_payload_bytes=latent,alias_status='UNRESOLVED_CONTIGUOUS_ALIAS')
            storage[-1]['async_completion_handles']=[td]
        gc=f'{p}.g.commit'
        op(gc,s,'generator_pool',Kind.GENERATOR_UPDATE,'generator_commit','gradient mean -> clip -> optimizer.step',
           read={'generator':cycle},write={'generator':cycle+1},batch=1,accumulation_waves=2)
        edge(f'{p}.g.backward.w0',gc);edge(f'{p}.g.backward.w1',gc)
        chain([f'{p}.g.rollout.w1',f'{p}.g.fake.w0',f'{p}.teacher.ready.w0',
               f'{p}.g.combine.w0',f'{p}.latent.issue.w1',f'{p}.g.fake.w1',f'{p}.g.backward.w0',
               f'{p}.latent.complete.w1',f'{p}.teacher.ready.w1',f'{p}.g.combine.w1',f'{p}.g.backward.w1',gc])
        edge(f'{p}.g.fake.w0',f'{p}.teacher.delivery.w0','frozen_native_order','G_posts_teacher_recv_after_fake')
        edge(f'{p}.latent.complete.w1',f'{p}.teacher.delivery.w1','frozen_native_order','G_posts_teacher_recv_after_backward0_and_latent_wait')
        chain([f'{p}.teacher.complete.w0',f'{p}.latent.ready.w1',f'{p}.s.teacher.w1',
               f'{p}.teacher.issue.w1',f'{p}.c.backward'])
        last=gc
        for step in range(s+1,s+5):
            for wave in (0,1):
                r=f's{step}.g.rollout.w{wave}'
                edge(gc,r,'semantic','post_update_rollout')
                edge(last,r,'frozen_native_order','rollout_preparation_order')
                if step>s+1:
                    edge(f's{step-1}.latent.d2h_ready.w{wave}',r,'native_transport_safety','streamer_reuse_after_copy')
                last=f's{step}.latent.issue.w{wave}'
        ret=f'cycle{cycle+1}.g.role_return'
        op(ret,s+4,'generator_pool',Kind.BARRIER,'role_return','_run_pipeline_8x8 return')
        edge(last,ret,'frozen_native_order','native_local_reference_release')
        clear=f'cycle{cycle+1}.g.clear_precomputed_first'
        op(clear,s+4,'generator_pool',Kind.BARRIER,'clear_precomputed_handle',
           '_run_pipeline_8x8_continuous: endpoint/trajectory anchor -> precomputed_first=None',
           not_an_added_cuda_barrier=True)
        edge(ret,clear,'frozen_native_order','outer_handle_survives_role_return')
    # The first blocking latent send; next-cycle wave0 is a preproduced lookahead.
    edge('s0.latent.complete.w0','s0.g.rollout.w1','frozen_native_order')
    edge('cycle1.g.clear_precomputed_first','s5.g.rollout.w0','frozen_native_order')
    edge('s4.latent.d2h_ready.w0','s5.g.rollout.w0','native_transport_safety','streamer_reuse_after_copy')
    edge('s0.g.commit','s5.g.rollout.w0','semantic','lookahead_generator_version')
    op('shadow.pack.C5',5,'score_pool',Kind.VERSION_SYNC,'shadow_pack','PairedCriticShadowSync.begin_source',read={'critic':5})
    op('shadow.load.C5',5,'generator_pool',Kind.VERSION_SYNC,'shadow_load','PairedCriticShadowSync.receive_and_load',
       read={'shadow_epoch':0},write={'shadow_epoch':1},native_critic_version=5)
    op('shadow.source_wait.C5',5,'score_pool',Kind.VERSION_SYNC,'shadow_source_wait','PairedCriticShadowSync.wait_source')
    edge('s4.c.commit','shadow.pack.C5');edge('shadow.pack.C5','shadow.load.C5','native_transport_safety')
    edge('shadow.pack.C5','s5.latent.ready.w0','frozen_native_order','shadow_submit_before_next_role_entry')
    edge('s5.latent.issue.w0','shadow.load.C5','depth1_policy','only_wave0_may_look_ahead')
    edge('shadow.load.C5','s5.g.rollout.w1','depth1_policy','no_second_lookahead_wave')
    edge('shadow.load.C5','s5.g.fake.w0');edge('shadow.load.C5','s5.g.fake.w1')
    edge('s0.g.fake.w0','shadow.load.C5');edge('s0.g.fake.w1','shadow.load.C5')
    edge('shadow.pack.C5','shadow.source_wait.C5','native_transport_safety')
    edge('shadow.load.C5','shadow.source_wait.C5','native_transport_safety')
    edge('s5.teacher.complete.w0','shadow.source_wait.C5','frozen_native_order')
    edge('shadow.source_wait.C5','s5.latent.ready.w1','frozen_native_order')
    edge('shadow.source_wait.C5','s5.c.commit','native_transport_safety')
    messages.append(dict(key=[5,0],path='paired_nccl_shadow',payload='critic_lora_C5',
        bytes_per_lane=shadow,lanes=8,source='shadow.pack.C5',ready='shadow.load.C5',completion='shadow.source_wait.C5'))
    # These are native queue/stream ordering, not extra GPU synchronizations.
    keys=sorted(host_keys)
    for (a,aw),(b,bw) in zip(keys,keys[1:]):
        edge(f's{a}.latent.d2h_ready.w{aw}',f's{b}.latent.d2h_ready.w{bw}',
             'native_transport_safety','single_native_copy_stream')
        edge(f's{a}.latent.complete.w{aw}',f's{b}.latent.delivery.w{bw}',
             'native_transport_safety','single_native_sender_fifo')
    for pool,role in (('G','generator_pool'),('Score','score_pool')):
        drain=f'transport.finish.{pool}'
        op(drain,9,role,Kind.BARRIER,'transport_drain','PinnedHostRunaheadTransport.finish',
           not_an_added_cuda_barrier=True)
        edge('cycle2.g.clear_precomputed_first' if pool=='G' else 's9.c.commit',drain,'frozen_native_order')
        for step,wave in keys:
            edge(f's{step}.latent.'+('complete' if pool=='G' else 'ready')+f'.w{wave}',drain,
                 'native_transport_safety','all_native_message_keys_drained')
    for pool in ('G','Score'):
        store(f'pinned_buffers.{pool}',owner=pool,producer='setup',consumers=[],release='transport_teardown',
              space='cpu_pinned',observed=17*latent)
        storage[-1]['allocation_boundary']='setup'
    for name in ('generator_streamers_KV','score_reducer_flat_and_gradient_views','shadow_flat_and_views',
                 'model_optimizer_state','context_flow_fake_real_gradient_temporaries'):
        store(name,owner='native',producer='UNRESOLVED',consumers=[],release='UNRESOLVED',
              alias_status='UNRESOLVED',includes_unmeasured=['storage_aliases','native_high_water'])
    graph=ExecutionGraph(tuple(ops),tuple(edges),dict(
        initial_versions=dict(generator=0,critic=0,teacher=0,shadow_epoch=0,ema=0),
        final_versions=dict(generator=2,critic=10,teacher=0,shadow_epoch=1,ema=0),
        source_head=evidence['source_head'],native_lowering='NOT_RUN',
        cycle_semantics='TWO_CYCLES_B16_B1_G_AND_B2_CRITIC_DEPTH1_C34',
        shadow_epoch_to_critic_version={'0':0,'1':5},
        absence_of_edge='NOT_PERMISSION_TO_REORDER_UNLOWERED_NATIVE_ACTIONS'))
    return dict(schema_version=SCHEMA,status='CPU_SOURCE_CONTRACT_ONLY',cost_admission='UNADMITTED',
        native_lowering='NOT_RUN',memory_feasibility='UNRESOLVED',evidence=deepcopy(evidence),
        graph=graph.to_dict(),messages=messages,storage=storage,
        missing=['MEASURED_STORAGE_ALIAS_AND_NATIVE_RELEASE_LIFETIMES','CONTEXT_BOUND_SERVICE_COSTS',
                 'EXPLICIT_ASYNC_POOL_ACTION_LOWERING','HELDOUT_NATIVE_FEEDBACK'])


def validate_pool_contract(value, *, observations=None):
    """Structural check; source proof requires the original hashed observations.

    A serialized PASS field and well-formed SHA alone never revalidate raw
    evidence. Even with that evidence this is not measured-cost admission.
    """
    require(value.get('schema_version')==SCHEMA,'pool contract schema mismatch')
    evidence=value['evidence']
    require(evidence.get('gate_status')=='PASS_NATIVE_ACTION_OBSERVATIONS', 'native observation gate is not PASS')
    require(evidence.get('implementation')==LiveWanImplementation('pipeline',True,True,True).id, 'unadmitted native implementation')
    require(isinstance(evidence.get('source_head'),str) and re.fullmatch('[0-9a-f]{40}',evidence['source_head']),
            'pool contract lacks exact source identity')
    require(isinstance(evidence.get('native_observations_sha256'),str) and
            re.fullmatch('[0-9a-f]{64}',evidence['native_observations_sha256']), 'observation content binding absent')
    require(set(evidence['samples'])=={f'{s}.{w}' for s in range(10) for w in (0,1)}, 'missing logical samples')
    for key,rows in evidence['samples'].items():
        step,wave=map(int,key.split('.'))
        require(set(rows)=={str(i) for i in G}, 'missing sample owner rank')
        for lane,sample in rows.items():
            require(sample.get('logical_step')==step and sample.get('logical_slot')==TRACE_BALANCED_8G8S_SLOT_PAIRS[int(lane)][wave]
                    and sample.get('operation_key') and sample.get('seeds'), 'sample binding changed')
    require(type(evidence['latent_contract'].get('payload_bytes')) is int and evidence['latent_contract']['payload_bytes']>0,
            'latent payload absent')
    require(type(evidence['shadow_contract'].get('payload_bytes')) is int and evidence['shadow_contract']['payload_bytes']>0,
            'shadow payload absent')
    expected=_build(evidence)
    require(fingerprint(value)==fingerprint(expected),'pool contract changed; missing memory/cost/native evidence cannot be waived')
    validate_versions(graph_from_dict(value['graph']))
    if observations is not None:
        gate=validate_action_observations(observations,expected_head=evidence['source_head'])
        require(gate['status']==evidence['gate_status'] and
                fingerprint(sorted(observations,key=lambda r:r['rank']))==evidence['native_observations_sha256'],
                'original observation evidence content changed')
        rebuilt=build_pool_contract(observations,expected_head=evidence['source_head'])
        require(fingerprint(value)==fingerprint(rebuilt), 'extracted evidence differs from original native observations')
    return {'status':'PASS_CPU_STRUCTURE_ONLY','native_lowering':'NOT_RUN','cost_admission':'UNADMITTED',
            'native_evidence_check':'PASS_REUSED_NATIVE_RECEIPTS' if observations is not None else 'NOT_RECHECKED'}
