#!/usr/bin/env python3
"""Native TP4/DP2/PP2 reference, original G1/C5/EMA5 and eight B1 microbatches.

This is not a planner simulation. All native DiT stage calls are B1; TP/PP
ranks collaborate rather than becoming additional independent samples. Only
parallel placement changes; no DMD kernel/cache/merge/reorder flags admitted.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from distillschedule.runtime.static_3d_layout import Static3DLayout
from distillschedule.runtime.static_3d_state import StaticDataParallelState, placed_trainables
from distillschedule.runtime.static_native_forward import create_admitted_forward_split
from distillschedule.runtime.static_native_pp_stage import NativeSourcePPStage
from distillschedule.runtime.static_native_proxy import NativeStaticPPProxy
from distillschedule.runtime.static_native_cooperative import run_native_B1_cohort
from distillschedule.runtime.static_tp_model import bind_wan_helios_tp4
from distillschedule.runtime.static_trace_cohort import cohort_entry
from distillschedule.runtime.fsdp2_static import configure_fsdp2_checkpointing
from distillschedule.runtime.wan13b_upper_bound import (
    FreshTrainableEMA, Wan13BBootstrap, WAN13B_PYRAMID_PATCH_SEED, materialize_adam_state,
)
from runtime.critic_pool_helios import (
    build_helios_pool_model, prepare_student_batch, student_forward_loss,
    run_generator_trace_rollout,
)
from runtime.dmd_boundary import detached_dmd_surrogate_loss
from runtime.gst_pool_helios import (
    construct_dmd_gradient, prepare_dmd_score_batch, run_generator_update_trace_rollout, score_forward,
)
from runtime.gst_pool_contract import NegativePromptAsset
from runtime.official_baseline import suspend_transformers_zero3_init
from runtime.static_world16_trace import StaticWorld16Trace

TRANSFORMER_SHA = '5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.publish.' + str(os.getpid()))
    try:
        with tmp.open('x') as handle:
            json.dump(value, handle, indent=2); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.link(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def memory(device):
    free, total = torch.cuda.mem_get_info(device)
    events = dict(line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())
    return dict(allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device),
        peak_allocated=torch.cuda.max_memory_allocated(device), peak_reserved=torch.cuda.max_memory_reserved(device),
        driver_free=free, driver_total=total,
        RSS_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
        cgroup_current=int(Path('/sys/fs/cgroup/memory.current').read_text()),
        cgroup_peak=int(Path('/sys/fs/cgroup/memory.peak').read_text()),
        cgroup_events={k: int(v) for k,v in events.items()})


class Native3DEngine:
    def __init__(self, module, state, *, lr, physical_step):
        self.module, self.state = module, state
        self.optimizer = torch.optim.AdamW([x.parameter for x in state.placements],
            lr=lr, betas=(0.,.999), eps=1e-8, weight_decay=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)
        self.optimizer_restore = materialize_adam_state(optimizer=self.optimizer,
            scheduler=self.lr_scheduler, physical_step=physical_step)
        self.global_steps, self.step_calls = physical_step, 0
        self.grad_norms = []

    def __call__(self, *args, **kwargs): return self.module(*args, **kwargs)

    def backward(self, loss):
        loss.backward()
        self.module.note_memory_peak()
        # The native batch loss already averages eight B1 samples. The GPipe
        # output-gradient bridge deliberately performs no second division.
        self.state.note_gpipe_mean_backward()

    def step(self):
        norm = self.state.synchronize_and_clip()
        self.grad_norms.append(norm.detach())
        self.optimizer.step(); self.lr_scheduler.step(); self.optimizer.zero_grad(set_to_none=True)
        self.module.note_memory_peak()
        self.global_steps += 1; self.step_calls += 1


def build_role(args, boot, role, device, pp, tp_group, pp_group, dp_group, model_group):
    start = time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(boot.component(role)['seed']))
        with suspend_transformers_zero3_init():
            native, metadata = build_helios_pool_model(role=role, model_root=args.model_root,
                subfolder='transformer', lora_rank=256, enable_gan=False,
                chunked_residual_storage='cuda_out_of_place',
                missing_pyramid_patch_seed=WAN13B_PYRAMID_PATCH_SEED)
    if len(native.blocks) != 30: raise ValueError('actual Wan30 model required')
    restore = boot.restore_trainables(native, component=role)
    # This DMD-specific representation is not part of the static reference.
    native._distillschedule_split_modulation = False
    for block in native.blocks: block._distillschedule_split_modulation = False
    configure_fsdp2_checkpointing(native)
    root_names = tuple(n for n,p in native.named_parameters() if p.requires_grad and not n.startswith('blocks.'))
    tp_receipt = bind_wan_helios_tp4(native, group=tp_group,
                                  expected_transformer_source_sha256=TRANSFORMER_SHA)
    split = create_admitted_forward_split(native, expected_source_sha256=TRANSFORMER_SHA, expected_blocks=30)
    with torch.no_grad():
        template = split.prefix(native, torch.zeros(1,16,1,4,4,dtype=torch.bfloat16),
            torch.tensor([500.],dtype=torch.float64), torch.zeros(1,3,4096,dtype=torch.bfloat16), return_dict=False)
    stage = NativeSourcePPStage(native, split=split, stage_index=pp, context_template=template,
        remove_root_modules=('norm_out','proj_out') if pp == 0 else
            ('condition_embedder','patch_embedding','patch_short','patch_mid','patch_long'))
    del template
    native.to(device)
    proxy = NativeStaticPPProxy(stage, pp_group=pp_group, device=device, role=role)
    actual_names = dict(native.named_parameters())
    placements = placed_trainables(native, replicated_root_names=tuple(n for n in root_names if n in actual_names))
    state = StaticDataParallelState(placements, dp_group=dp_group, model_group=model_group, max_norm=1.)
    engine = Native3DEngine(proxy, state, lr=2e-6 if role=='generator' else 4e-7,
                           physical_step=int(boot.component(role)['optimizer_step']))
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    return engine, dict(role=role, cold_construction_seconds=time.perf_counter()-start,
        source_metadata=metadata, bootstrap_restore=restore, TP=tp_receipt, PP=stage.receipt(),
        optimizer=engine.optimizer_restore, after_setup=memory(device),
        static_baseline_split_modulation=False,
        placements=[dict(name=x.name,tp=x.tp_placement,partition_dim=x.partition_dim) for x in placements])


def main():
    ap=argparse.ArgumentParser()
    for flag in ('run-id','expected-head','model-root','bootstrap','expected-bootstrap-hash',
                 'trace-bundle','expected-trace-hash','negative-prompt','output-dir'):
        ap.add_argument('--'+flag,required=True)
    ap.add_argument('--seed',type=int,default=20260903)
    ap.add_argument('--validate-only',action='store_true')
    args=ap.parse_args()
    rank=int(os.environ.get('RANK','0')); local=int(os.environ.get('LOCAL_RANK','0'))
    root=Path(args.output_dir); where='preflight'; cycle_index=0
    row=dict(schema='distillschedule.native_static_3d_wan.v1',run_id=args.run_id,rank=rank,
        hostname=socket.gethostname(),source_head=args.expected_head,profiler='NOT_RUN',cycles=[],
        numerical='DEFERRED_BY_USER_NOT_PROMOTED_TO_FULL_STATE_EQUIVALENCE')
    try:
        actual=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        if actual!=args.expected_head or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
            raise ValueError('native run requires the exact clean source HEAD')
        if torch.__version__!='2.10.0+cu128': raise ValueError('unadmitted pipeline ABI')
        for k,v in os.environ.items():
            if k.startswith(('DISTILLSCHEDULE_WAN13B_','H3_')) and v not in ('','0','off','false'):
                raise ValueError('DMD optimization env leaked into static baseline: '+k)
        boot=Wan13BBootstrap.load(args.bootstrap, expected_hash=args.expected_bootstrap_hash)
        trace=StaticWorld16Trace(args.trace_bundle,variant='no-gan')
        if trace.bundle_hash!=args.expected_trace_hash: raise ValueError('trace identity mismatch')
        negative_asset=NegativePromptAsset.load(args.negative_prompt)
        from helios.modules.transformer_helios import HeliosTransformer3DModel
        transformer_source=Path(inspect.getsourcefile(HeliosTransformer3DModel))
        if hashlib.sha256(transformer_source.read_bytes()).hexdigest()!=TRANSFORMER_SHA:
            raise ValueError('actual external native transformer source changed')
        model_config=Path(args.model_root)/'transformer/config.json'
        model_identity=json.loads(model_config.read_text())
        if model_identity['num_layers']!=30 or model_identity['num_attention_heads']!=12:
            raise ValueError('actual Wan checkpoint architecture changed')
        layout=Static3DLayout()
        pp,dp,tp=layout.coordinate(rank)
        source_ranks=tuple(layout.logical_sample_rank(dp,i) for i in range(8))
        row.update(layout=layout.receipt(),coordinate=dict(pp=pp,dp=dp,tp=tp),source_ranks=source_ranks,
            bootstrap_hash=boot.manifest_hash,trace_hash=trace.bundle_hash,
            static_layout='TP4_DP2_PP2',DMD_optimizations=False,merged_lora_count=0,
            transformer_source_sha256=TRANSFORMER_SHA,model_config_sha256=hashlib.sha256(model_config.read_bytes()).hexdigest(),
            software=dict(torch=torch.__version__,torch_cuda=torch.version.cuda,NCCL=torch.cuda.nccl.version()),
            update_order='G_UPDATE_THEN_G_ROLLOUT_C_UPDATE_EMA_REPEAT5')
        if args.validate_only:
            if torch.cuda.is_initialized(): raise ValueError('validate-only initialized CUDA')
            row['status']='PASS_CPU_NATIVE_STATIC_3D_SOURCE_ENV_NO_GPU'
            print(json.dumps(row)); return 0
        if int(os.environ['WORLD_SIZE'])!=16 or torch.cuda.device_count()!=8:
            raise ValueError('actual world16 with eight assigned devices per node required')
        device=torch.device('cuda',local); torch.cuda.set_device(device)
        dist.init_process_group('nccl',timeout=timedelta(seconds=240),device_id=device)
        mesh=init_device_mesh('cuda',(2,2,4),mesh_dim_names=('pp','dp','tp'))
        tp_group,pp_group,dp_group=(mesh[name].get_group() for name in ('tp','pp','dp'))
        model_groups=[dist.new_group([layout.physical_rank(p,d,t) for p in range(2) for t in range(4)]) for d in range(2)]
        model_group=model_groups[dp]
        random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        row['before']=memory(device)
        where='generator construction'
        generator,g_build=build_role(args,boot,'generator',device,pp,tp_group,pp_group,dp_group,model_group)
        dist.barrier()
        where='critic construction'
        critic,c_build=build_role(args,boot,'critic',device,pp,tp_group,pp_group,dp_group,model_group)
        ema=FreshTrainableEMA(generator.module,optimization_step=int(boot.manifest['ema']['optimization_step']),
            decay=float(boot.manifest['ema']['decay']),update_after_step=int(boot.manifest['ema']['update_after_step']))
        negative=negative_asset.embedding.to(device=device,dtype=torch.bfloat16)
        row['build']=dict(generator=g_build,critic=c_build)
        row['after_setup']=memory(device)
        for cycle_index in (1,2):
            g_before=generator.module.receipt();c_before=critic.module.receipt()
            losses=[]; census=[]; identities=[]
            dist.barrier();torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            begin=time.perf_counter()
            where='generator native gradient rollout'
            members=[trace.load_generator_entry(rank=r,device=device) for r in source_ranks]
            entry=cohort_entry(members,role='generator')
            generator.module.enable_adapters();generator.module.train()
            results, dispatch=run_native_B1_cohort(generator.module,
                lambda view,i:run_generator_update_trace_rollout(engine=view,entry=members[i],device=device,
                    detailed_profile=False,collect_diagnostics=False),device=device)
            x0=torch.cat([result[0] for result in results])
            if any(result[1] is not None for result in results):
                raise ValueError('native Generator trace unexpectedly selected a gradient mask')
            mask=None
            census.append(dict(role='generator',step=10,inventory=results[0][2],
                rank_qualified_inventories=[result[2] for result in results],dispatcher=dispatch))
            identities.extend([f'cycle{cycle_index}/generator/step10/rank{r}' for r in source_ranks])
            where='generator DMD score roles'
            batch=prepare_dmd_score_batch(generated_x0=x0,entry=entry,device=device)
            fake=score_forward(critic.module,batch,condition=batch['encoder_hidden_states'],adapter_enabled=True,validate_finite=False)
            real_cond=score_forward(critic.module,batch,condition=batch['encoder_hidden_states'],adapter_enabled=False,validate_finite=False)
            real_uncond=score_forward(critic.module,batch,condition=negative,adapter_enabled=False,validate_finite=False)
            gradient,_aux=construct_dmd_gradient(batch=batch,fake_flow_cond=fake,real_flow_cond=real_cond,real_flow_uncond=real_uncond,validate_finite=False)
            loss=detached_dmd_surrogate_loss(x0,gradient,mask);losses.append(loss.detach())
            where='generator native checkpoint backward'; generator.backward(loss)
            where='generator optimizer';generator.step()
            del members,entry,x0,mask,batch,fake,real_cond,real_uncond,gradient,_aux,loss,results
            for step in range(10,15):
                where=f'critic{step} native Generator rollout'
                members=[trace.load_critic_entry(rank=r,step=step,device=device) for r in source_ranks]
                generator.module.enable_adapters();generator.module.train()
                results,dispatch=run_native_B1_cohort(generator.module,
                    lambda view,i:run_generator_trace_rollout(model=view,loaded=members[i],device=device,
                        detailed_profile=False,record_output_sha256=False,collect_diagnostics=False),device=device)
                outputs=tuple(result[0] for result in results)
                census.append(dict(role='critic',step=step,inventory=results[0][1],
                    rank_qualified_inventories=[result[1] for result in results],dispatcher=dispatch))
                identities.extend([f'cycle{cycle_index}/critic/step{step}/rank{r}' for r in source_ranks])
                entry=cohort_entry(members,role='critic');x0=torch.cat(outputs)
                batch=prepare_student_batch(generated_x0=x0,loaded=entry,device=device)
                where=f'critic{step} native gradient forward'
                _,loss=student_forward_loss(critic,batch,validate_finite=False)
                losses.append(loss.detach())
                where=f'critic{step} native checkpoint backward';critic.backward(loss)
                where=f'critic{step} optimizer';critic.step()
                where=f'critic{step} EMA';ema.step()
                del members,outputs,entry,x0,batch,loss,results
            torch.cuda.synchronize(device);dist.barrier(); wall=time.perf_counter()-begin
            # All value extraction, census serialization, and world validation
            # are OUTSIDE the native timer. No profiler or anchor probe.
            where='post-cycle semantic gate'
            finite=bool(torch.stack([torch.isfinite(l) for l in losses]+[
                torch.isfinite(n) for n in generator.grad_norms+critic.grad_norms]).all().cpu())
            if not finite:raise ValueError('nonfinite native losses/grad norms')
            g_after=generator.module.receipt(); c_after=critic.module.receipt()
            g_calls=g_after['application_forward_per_logical_sample']-g_before['application_forward_per_logical_sample']
            c_calls=c_after['application_forward_per_logical_sample']-c_before['application_forward_per_logical_sample']
            expected_g=sum(int(x['inventory']['call_count']) for x in census)
            if g_calls!=expected_g or c_calls!=8 or g_after['pending_backward_schedules'] or c_after['pending_backward_schedules']:
                raise ValueError('actual native PP DiT application census mismatch')
            if (g_after['grad_forward_per_logical_sample']-g_before['grad_forward_per_logical_sample']!=3
                    or c_after['grad_forward_per_logical_sample']-c_before['grad_forward_per_logical_sample']!=5):
                raise ValueError('selected gradient-forward census changed')
            if len(identities)!=48 or len(set(identities))!=48: raise ValueError('DP replica sample identity corrupted')
            item=dict(cycle=cycle_index,wall_seconds=wall,finite_losses=finite,losses=[float(l.cpu()) for l in losses],
                generator_calls=g_calls,critic_calls=c_calls,actual_B1_stage_G_calls=g_calls*8,
                actual_B1_stage_C_calls=c_calls*8,source_sample_identities=identities,rollout_census=census,
                memory=memory(device),generator_optimizer_updates=1,critic_optimizer_updates=5,ema_updates=5,
                versions=dict(generator=generator.global_steps,critic=critic.global_steps,ema=ema.optimization_step))
            row['cycles'].append(item)
            write(root/'cycles'/f'rank_{rank:03d}_cycle_{cycle_index}.json',item)
            print(json.dumps(dict(rank=rank,phase='NATIVE_CYCLE_COMPLETE',cycle=cycle_index,wall_seconds=wall,
                memory_allocated=item['memory']['allocated'],versions=item['versions'])),flush=True)
        row['final_versions']=dict(generator=generator.global_steps,critic=critic.global_steps,ema=ema.optimization_step)
        row['final_runtime_receipts']=dict(generator=generator.module.receipt(),critic=critic.module.receipt(),
            generator_state=generator.state.receipt(),critic_state=critic.state.receipt())
        row['after']=memory(device)
        if any(row['after']['cgroup_events'][k]-row['before']['cgroup_events'][k] for k in ('oom','oom_kill')):
            raise ValueError('new cgroup OOM')
        row['status']='PASS_NATIVE_RUNTIME_CONTRACTS'
        write(root/'ranks'/f'rank_{rank:03d}.json',row)
        return 0
    except BaseException as e:
        row.update(status='FAIL_NATIVE_STATIC_3D',failure_point=where,cycle=cycle_index,error=repr(e),traceback=traceback.format_exc())
        if torch.cuda.is_initialized():
            try:row['failure_memory']=memory(torch.device('cuda',local))
            except BaseException as memerror:row['memory_error']=repr(memerror)
        if not args.validate_only: write(root/'failures'/f'rank_{rank:03d}.json',row)
        print(json.dumps(row),flush=True);raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':raise SystemExit(main())
