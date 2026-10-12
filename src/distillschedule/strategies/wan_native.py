"""Original world16 Wan engines and native actions, selected before construction.

This is a binding to the existing worker, not a new DMD implementation. Explicit
execution under torchrun and original local assets are required. Calibration
ranges are synchronized diagnostics, not formal performance measurements.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time

from distillschedule.search.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint, file_sha256, repository_root, write_new_json
from .wan import OriginalWanTask, OriginalWanStrategy, PRECISION
from .wan_planning import action_graph, component_contract


@dataclass(frozen=True)
class WanAssets:
    model_root: str
    bootstrap: str
    bootstrap_hash: str
    trace_bundle: str
    trace_hash: str
    negative_prompt: str
    runtime_seed: int = 20260903
    allow_legacy_trace: bool = False
    gpu_reserve_bytes: int = 2147483648

    def __post_init__(self):
        for key in ('model_root','bootstrap','trace_bundle','negative_prompt'):
            require(isinstance(getattr(self,key),str) and getattr(self,key),'missing local asset path')
        for key in ('bootstrap_hash','trace_hash'):
            value=getattr(self,key)
            require(len(value)==64 and all(c in '0123456789abcdef' for c in value),'invalid asset SHA256')
        require(type(self.runtime_seed) is int and self.runtime_seed>=0,'invalid runtime seed')
        require(type(self.allow_legacy_trace) is bool,'legacy admission must be explicit')
        require(type(self.gpu_reserve_bytes) is int and self.gpu_reserve_bytes>=0,'invalid memory reserve')


def source_identity():
    root=repository_root()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    subprocess.run(['git','diff','--exit-code','HEAD','--','src','runtime','scripts','third_party'],
                   cwd=root,check=True,stdout=subprocess.DEVNULL)
    return head


def apply_environment(task, strategy):
    """Apply declared controls before NCCL/model imports, rejecting unrelated knobs."""
    strategy.validate(task)
    values=strategy.environment(task)
    for key,value in os.environ.items():
        if key.startswith(('DISTILLSCHEDULE_WAN13B_','DISTILLSCHEDULE_WAN_ROLLOUT_')) and key not in values:
            require(value in ('','0'),'undeclared Wan runtime override: '+key)
    for key in ('DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE',):
        require(os.environ.get(key,'') in ('','0'),'profiling cannot enter native calibration')
    for key,value in values.items():
        if value: os.environ[key]=value
        else: os.environ.pop(key,None)
    return values


def load_original_worker():
    """The original fair-worker patches its original base builder on import."""
    root=repository_root(); name='distillschedule_original_wan_fair_binding'
    compat=root/'third_party/helios_compat/source'
    require((compat/'helios').is_dir(),'pinned Helios compatibility source missing')
    for path in (str(root/'src'),str(root),str(compat)):
        if path not in sys.path:sys.path.insert(0,path)
    if name not in sys.modules:
        path=root/'scripts/distillschedule/probe_wan13b_fair_world16.py'
        spec=importlib.util.spec_from_file_location(name,path)
        require(spec is not None and spec.loader is not None,'original worker import unavailable')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name,None)
            raise
    return sys.modules[name]


def build_engines(task, strategy, assets, bootstrap, device, *, worker=None):
    """Testable seam: the selected existing placement reaches its real factory."""
    strategy.validate(task)
    require(os.environ.get('DISTILLSCHEDULE_SEARCH_CANDIDATE_ID')==strategy.implementation.id,
            'selected implementation not installed before model construction')
    worker=load_original_worker() if worker is None else worker
    args=SimpleNamespace(model_root=Path(assets.model_root),placement=strategy.implementation.native_placement,
                         timing_mode=True,profile_mode=False)
    return worker.base._build_engines(args=args,bootstrap=bootstrap,device=device)


def validate_build(build, strategy, engines):
    expected='FSDP2EngineAdapter' if strategy.placement=='fsdp' else 'GradReadyAsyncResidentEngine'
    require(all(type(e).__name__==expected for e in engines),'selected strategy did not create its native engine')
    for role in ('generator','critic'):
        row=build[role];metadata=row['metadata'];chunk=metadata['modulation_chunk_runtime']
        require(chunk['effective_tokens']==[strategy.chunk_tokens] and chunk['all_blocks_match'] and
                chunk['block_count']==30,'selected chunk did not reach all native blocks')
        require(metadata['planner_execution_precision']['precision_fingerprint']==PRECISION.digest,
                'native precision differs from selected implementation')
        fused=metadata['generator_nograd_lora_fused_epilogue']
        require(fused['enabled']==(strategy.lora_epilogue and role=='generator'),'epilogue enablement differs')
        require(row['placement']['gradient_reduction']=='SUM_THEN_DIVIDE_WORLD_SIZE','gradient scaling changed')
    return {'status':'PASS_BUILD_BINDING','engine_class':expected,'strategy':strategy.to_dict()}


def reserve_trial(session, identity, output):
    """Persistent explicit-session budget. Failed starts consume a slot too."""
    import fcntl
    session=Path(session).resolve();session.mkdir(parents=True,exist_ok=True)
    with (session/'budget.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=session/'budget.json'
        value=json.loads(path.read_text()) if path.exists() else dict(identity=identity,limit=8,trials=[])
        require(value['identity']==identity and value['limit']==8,'budget session identity changed')
        require(len(value['trials'])<value['limit'],'original-Wan native session budget exhausted')
        require(str(output) not in value['trials'],'native output already used in this budget')
        value['trials'].append(str(output))
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
        return len(value['trials'])


def _full_state(torch, generator, critic, ema, rank, output):
    from safetensors.torch import save_file
    state={}
    def save(key,value):
        full=value.detach().full_tensor() if hasattr(value,'full_tensor') else value.detach()
        if rank==0:state[key]=full.float().cpu().contiguous()
    for role,engine in (('generator',generator),('critic',critic)):
        for name,p in engine.module.named_parameters():
            if not p.requires_grad:continue
            save(role+'/parameter/'+name,p)
            values=engine.optimizer.state[p]
            for key in ('exp_avg','exp_avg_sq'):save(role+'/'+key+'/'+name,values[key])
            step=values.get('step')
            require(step is not None,'Adam step evidence missing')
            if rank==0:state[role+'/step/'+name]=torch.as_tensor(step).float().cpu().clone()
    shadows=getattr(ema,'_shadows',None);parameters=getattr(ema,'_parameters',None)
    require(shadows is not None and parameters is not None and len(shadows)==len(parameters),'unknown native EMA state')
    for i,shadow in enumerate(shadows):save('ema/'+str(i),shadow)
    if rank!=0:return {'writer_rank':0}
    path=output/'final-state.safetensors';save_file(state,str(path))
    return {'writer_rank':0,'path':str(path),'sha256':file_sha256(path)}


def execute(task, strategy, assets, output, *, session, calibration, expected_binding=None):
    """Run explicitly under an existing torchrun world16. No process spawning."""
    apply_environment(task,strategy)
    require(os.environ.get('WORLD_SIZE')=='16','original execution requires torchrun world16')
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    require(0<=rank<16 and 0<=local_rank<8,'unsupported original two-node rank topology')
    head=source_identity()
    import torch
    import torch.distributed as dist
    require(torch.cuda.is_available() and not dist.is_initialized(),'fresh CUDA/NCCL process required')
    torch.cuda.set_device(local_rank);device=torch.device('cuda',local_rank)
    worker=load_original_worker();base=worker.base
    from .wan_trace import open_trace
    bootstrap=base.Wan13BBootstrap.load(assets.bootstrap,expected_hash=assets.bootstrap_hash,verify_payloads=True)
    trace=open_trace(assets.trace_bundle,task,expected_hash=assets.trace_hash,allow_legacy=assets.allow_legacy_trace)
    # Load this rank's actual tensors before constructing models; no trace resizing.
    trace._load(rank)
    negative_asset=base.NegativePromptAsset.load(assets.negative_prompt)
    require(Path(bootstrap.manifest['model_root']).resolve()==Path(assets.model_root).resolve(),'bootstrap model root differs')
    model_files=sorted(p for p in (Path(assets.model_root)/'transformer').rglob('*')
                       if p.is_file() and (p.suffix in ('.json','.safetensors','.bin')))
    require(model_files,'no original model weight/config files')
    model_hash=fingerprint({str(p.relative_to(assets.model_root)):file_sha256(p) for p in model_files})
    asset_identity=fingerprint(dict(bootstrap=bootstrap.manifest_hash,model=model_hash,trace=trace.binding_hash,
                negative=negative_asset.manifest['manifest_hash'],seed=assets.runtime_seed))
    from distillschedule.search.wan_implementation import verify_execution_precision
    precision_receipt=verify_execution_precision(torch)
    attention_receipt=base.p0._attention_backend_receipt('local-fa2')
    runtime={'attention':attention_receipt,'precision_receipt':precision_receipt,'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),
             'nccl':torch.cuda.nccl.version(),'precision':PRECISION.digest}
    prop=torch.cuda.get_device_properties(device)
    hardware={'rank':rank,'name':prop.name,'uuid':str(getattr(prop,'uuid','UNAVAILABLE')),
              'total_memory':prop.total_memory,'host':platform.node(),'local_rank':local_rank}
    dist.init_process_group('nccl')
    output=Path(output).resolve()
    require(not output.exists(),'native output already exists; never overwrite or reuse another trial')
    receipt={'status':'RUNNING','gpu_execution':'NATIVE_STARTED','claimable_speedup':False}
    observer=None
    try:
        identities=[None]*16;dist.all_gather_object(identities,(hardware,runtime,asset_identity,task.digest,head))
        require(all(x[1:]==identities[0][1:] for x in identities),'ranks disagree on source/task/assets/runtime')
        binding=dict(source_head=head,hardware_fingerprint=fingerprint([x[0] for x in identities]),
                     environment_fingerprint=fingerprint(runtime),asset_fingerprint=asset_identity,
                     semantic_fingerprint=task.digest,precision_fingerprint=PRECISION.digest)
        require(expected_binding is None or binding==expected_binding,'selected plan calibration binding changed')
        if rank==0:
            slot=reserve_trial(session,fingerprint(binding),output)
            output.mkdir(parents=True,exist_ok=False)
            (output/'ranks').mkdir()
            write_new_json(output/'trial.json',dict(binding=binding,task=task.to_dict(),assets=asdict(assets),
                 strategy=strategy.to_dict(),slot=slot,calibration=calibration,status='STARTED',historical_budget_reset=False))
        dist.barrier()
        torch.manual_seed(assets.runtime_seed+rank);torch.cuda.manual_seed_all(assets.runtime_seed+rank)
        import random
        random.seed(assets.runtime_seed+rank)
        # Existing precision helper owns the actual policy; require it rather than guessing APIs.
        from distillschedule.search.wan_implementation import verify_execution_precision
        verify_execution_precision(torch)
        generator,critic,ema,build=build_engines(task,strategy,assets,bootstrap,device,worker=worker)
        build_receipt=validate_build(build,strategy,(generator,critic))
        negative=negative_asset.embedding.to(device=device,dtype=torch.bfloat16)
        from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
        from distillschedule.profiling.static_profile_diagnostics import StaticProfileDiagnostics
        from distillschedule.runtime.wan_cycle_plan import prepare_parameterized_wan_cycle
        policy=StaticFSDP2P0Policy(mesh_policy='world16-fsdp',rollout_batch_size=1,
                                   pack_real_score_pair=False,rollout_backend='native-fsdp2-batch')
        diagnostics=StaticProfileDiagnostics(enabled=False,rank=rank,output_path=None,collect_cpu_diagnostics=False)
        versions=lambda:dict(generator=int(generator.global_steps),critic=int(critic.global_steps),ema=int(ema.optimization_step))
        initial=versions();cycles=[];finite=[];cache={}
        if calibration:
            from .wan_observer import OriginalWanCalibrator
            observer=OriginalWanCalibrator(task=task,strategy=strategy,initial_versions=initial,
                torch=torch,distributed=dist,device=device,rank=rank,source_head=head)
        for cycle in (1,2):
            scope=(lambda op:observer.stage(cycle,op.operation_id)) if observer else None
            prepared=prepare_parameterized_wan_cycle(p0=base.p0,source_steps=task.source_steps,
                max_live_rollouts=strategy.rollout_window,trace_contract=task.bridge_contract(),ema_after=task.ema_after,
                rank=rank,device=device,generator_engine=generator,critic_engine=critic,ema=ema,trace=trace,negative=negative,
                diagnostics=diagnostics,policy=policy,clean_runtime=True,runtime_cache=cache,
                deferred_finite_checks=finite,operation_scope=scope)
            dist.barrier();torch.cuda.synchronize(device);start=time.perf_counter_ns()
            raw=prepared.run()
            torch.cuda.synchronize(device);dist.barrier();duration=time.perf_counter_ns()-start
            require(len(raw['critic'])==len(raw['rollouts'])==task.fake_updates,'native call inventory changed')
            cycles.append(dict(cycle=cycle,duration_ns=duration,dispatch=prepared.receipt(),versions=versions()))
        require(finite and bool(torch.stack(finite).all().cpu()),'native finite checks failed')
        observed=observer.finish() if observer else None
        state=_full_state(torch,generator,critic,ema,rank,output)
        stats=base._memory(device)
        require(stats['driver_free_bytes']>=assets.gpu_reserve_bytes,'native reserve not satisfied at exit')
        receipt.update(status='PASS_NATIVE_EXECUTION',binding=binding,task=task.to_dict(),strategy=strategy.to_dict(),
            calibration=calibration,calibration_receipt=observed,cycles=cycles,initial_versions=initial,
            final_versions=versions(),state=state,build=build_receipt,hardware=hardware,memory=stats,
            memory_limit_bytes=prop.total_memory-assets.gpu_reserve_bytes,numerical_status='REQUIRES_MATCHED_FSDP_REFERENCE',
            training_semantics='ORIGINAL_WAN_THREE_STAGE_NOGAN',trace_policy='MATCHED_COHORT_REPLAY',quality_validation='NOT_RUN')
        write_new_json(output/'ranks'/f'{rank:02d}.json',receipt)
        dist.barrier()
        return receipt
    except BaseException as exc:
        if output.is_dir():
            failure=output/f'failure-rank{rank:02d}.json'
            if not failure.exists():write_new_json(failure,{**receipt,'status':'FAIL','error':str(exc),
                      'partial_calibration':observer.snapshot() if observer else None})
        raise
    finally:
        dist.destroy_process_group()
