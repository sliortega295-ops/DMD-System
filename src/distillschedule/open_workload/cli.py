"""Public factory entry, with no model or workload name in the dispatcher.

The explicitly supplied Python factory describes training math/state. It never
needs to import the planner or construct optimization candidates. Factories are
trusted user code, executed only with --execute (including model construction).
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import file_sha256,write_new_json


def load_factory(name,parameters):
    from .torch_backend import Workload
    path,sep,symbol=name.rpartition(':')
    require(sep and symbol.isidentifier(),'factory must be a local.py:callable',path='open.cli')
    source=Path(path).expanduser().resolve()
    require(source.is_file() and source.suffix=='.py','local Python factory required',path='open.cli')
    sha=file_sha256(source)
    module_name='distillschedule_user_'+sha
    spec=importlib.util.spec_from_file_location(module_name,source)
    require(spec is not None and spec.loader is not None,'factory cannot be loaded',path='open.cli')
    module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module
    spec.loader.exec_module(module)
    factory=getattr(module,symbol,None)
    require(callable(factory),'factory entry not callable',path='open.cli')
    workload=factory(parameters)
    require(isinstance(workload,Workload),'factory must return Workload, not a schedule/recipe',path='open.cli')
    require(file_sha256(source)==sha,'factory source changed during construction',path='open.cli')
    workload.identity_extra={**workload.identity_extra,'factory_sha256':sha,'factory_entry':symbol,'factory_parameters':parameters}
    return workload.validate()


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {'role-describe','role-auto','role-plan','role-run'}:
        from .role_cli import main as role_main
        return role_main(argv)
    p=argparse.ArgumentParser(prog='distillctl workloads native program')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('describe','calibrate','plan','run','auto','hardware'):
        q=sub.add_parser(name)
        q.add_argument('--output',type=Path,required=name!='describe')
        if name not in ('plan','hardware'):
            q.add_argument('--factory',required=True)
            q.add_argument('--parameters',type=Path,required=True)
            q.add_argument('--execute',action='store_true')
            q.add_argument('--distributed',choices=('gloo','nccl'),help='join an explicitly allocated torchrun group')
            q.add_argument('--group-timeout',type=int,default=180)
        if name=='hardware':
            q.add_argument('--execute',action='store_true',
                           help='run bounded hardware probes without constructing a workload')
            q.add_argument('--distributed',choices=('gloo','nccl'))
            q.add_argument('--group-timeout',type=int,default=180)
            q.add_argument('--probe-samples',type=int,default=3)
        if name in ('calibrate','auto'):
            q.add_argument('--no-hardware-probe',action='store_true',
                           help='disable bounded, automatic A100/H100 H2D/NCCL microprofiling')
            q.add_argument('--max-candidates',type=int,default=8)
            q.add_argument('--samples',type=int,default=2)
            q.add_argument('--max-program-runs',type=int,default=32)
        if name in ('plan','auto'):
            q.add_argument('--memory-limit-bytes',type=int,required=name=='plan',
                           help='optional for CUDA auto: infer conservative free+allocated budget; required for CPU')
            if name=='auto':
                q.add_argument('--gpu-reserve-gib',type=int,default=2,
                               help='real GPU reserve (0..16 GiB), applied before automatic layout search')
            q.add_argument('--max-states',type=int,default=50000)
            q.add_argument('--beam-width',type=int,default=0)
        if name=='plan': q.add_argument('--calibration',type=Path,required=True)
        if name in ('auto','run'):
            q.add_argument('--deep-profile',action='store_true',
                           help='trace only selected execution; not part of measured search costs')
        if name=='run': q.add_argument('--plan',type=Path,required=True)
    args=p.parse_args(argv)
    owned_group=False
    try:
        if args.command!='plan' and not args.execute:
            print(json.dumps({'status':'NOT_EXECUTED','reason':'factory is user code; --execute required','model_constructed':False}))
            return 0
        if getattr(args,'distributed',None):
            import os
            from datetime import timedelta
            import torch
            import torch.distributed as dist
            require(1<=args.group_timeout<=3600,'invalid collective timeout',path='open.cli')
            require(all(k in os.environ for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
                    and int(os.environ['WORLD_SIZE'])>=2,'torchrun rank environment required',path='open.cli')
            if args.distributed=='nccl':
                require(torch.cuda.is_available(),'CUDA/NCCL requested but unavailable',path='open.cli')
                torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            if not dist.is_initialized():
                dist.init_process_group(args.distributed,timeout=timedelta(seconds=args.group_timeout));owned_group=True
            require(str(dist.get_backend())==args.distributed,'process group backend mismatch',path='open.cli')
        from .planning import calibrate,plan,run
        if args.command=='hardware':
            import torch
            import torch.distributed as dist
            from .hardware_probe import probe, ProbeConfig
            device=torch.device('cuda',torch.cuda.current_device()) if args.distributed=='nccl' else torch.device('cpu')
            local=probe(device,distributed=dist.is_initialized() and dist.get_world_size()>1,
                        config=ProbeConfig(samples=args.probe_samples))
            if dist.is_initialized() and dist.get_world_size()>1:
                from .replicated import gather,checked
                ranks=gather(local)
                result={'schema_version':'distillschedule.hardware_profile_collection.v1',
                        'status':'MEASURED_GPU_ALLOCATION' if device.type=='cuda' else 'NOT_GPU_PROFILE',
                        'rank_profiles':ranks, 'claimable_speedup':False,
                        'gpu_execution':'CUDA_PROBE_ONLY' if device.type=='cuda' else 'NOT_RUN'}
                checked(lambda:write_new_json(args.output,result) if dist.get_rank()==0 else None,
                        'hardware profile publication')
            else:
                result={'schema_version':'distillschedule.hardware_profile_collection.v1',
                        'status':local['status'], 'rank_profiles':[local],
                        'claimable_speedup':False,'gpu_execution':local['gpu_execution']}
                write_new_json(args.output,result)
        elif args.command=='plan':
            result=plan(args.calibration,memory_limit_bytes=args.memory_limit_bytes,max_states=args.max_states,beam_width=args.beam_width)
            write_new_json(args.output,result)
        else:
            parameters=json.loads(args.parameters.read_text())
            require(isinstance(parameters,dict),'parameter object required',path='open.cli')
            # Factories remain model/semantic bindings. All ranks report a bad
            # factory before another rank starts DDP initialization.
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_world_size()>1:
                from .replicated import checked
                workload=checked(lambda:load_factory(args.factory,parameters),'factory construction')
            else: workload=load_factory(args.factory,parameters)
            from .execution import ExecutionBackend
            backend=ExecutionBackend(workload)
            if args.command=='describe':
                result={'program':workload.program.to_dict(),'graph':workload.program.compile().to_dict(),
                        'capabilities':backend.inventory(args.max_candidates if hasattr(args,'max_candidates') else 16),'gpu_execution':'NOT_RUN'}
            elif args.command=='calibrate':
                result=calibrate(workload,args.output,max_candidates=args.max_candidates,
                                 samples=args.samples,max_program_runs=args.max_program_runs,
                                 profile_hardware=False if args.no_hardware_probe else None)
            elif args.command=='run':
                result,_,_=run(workload,args.plan,args.output,deep_profile=args.deep_profile)
            else:
                # Two additional executions are the independent reference and
                # selected-plan confirmation, not free out-of-budget trials.
                required=3+args.samples*len(backend.inventory(args.max_candidates)['candidates'])
                require(required<=args.max_program_runs,'auto budget includes calibration and two confirmation runs',path='open.cli')
                from .hardware_probe import plan_memory_limit
                budget = plan_memory_limit(workload.device,distributed=backend.distributed,
                    consensus=backend.consensus,requested=args.memory_limit_bytes,
                    reserve_bytes=args.gpu_reserve_gib * 2**30)
                # Same device/rank budget is consumed for both search and native
                # admission; re-evaluated against current headroom on replay.
                limit_bytes=budget['limit_bytes']
                root=args.output.resolve();backend.mkdir(root)
                calibration=calibrate(workload,root/'calibration',max_candidates=args.max_candidates,
                            samples=args.samples,max_program_runs=args.max_program_runs-2,
                            profile_hardware=False if args.no_hardware_probe else None)
                selected=backend.select_plan(lambda:plan(root/'calibration/calibration.json',memory_limit_bytes=limit_bytes,
                              max_states=args.max_states,beam_width=args.beam_width))
                from distillschedule.workloads.best_known.registry import fingerprint
                selected['hardware_memory_budget'] = budget
                selected.pop('plan_sha256')
                selected['plan_sha256']=fingerprint(selected)
                backend.publish(root/'plan.json',selected)
                result,_,_=run(workload,root/'plan.json',root/'execution',deep_profile=args.deep_profile)
                result['program_runs']=calibration['budget']['actual']+2
                result['memory_budget']=budget
                backend.publish(root/'result.json',result)
        print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
        return 0 if result.get('status')!='REVIEW_SELECTED_PROGRAM' else 3
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 2
    finally:
        if owned_group:
            import torch.distributed as dist
            dist.destroy_process_group()


if __name__=='__main__': raise SystemExit(main())
