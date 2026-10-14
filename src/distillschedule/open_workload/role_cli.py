"""Single opt-in public route from new role-math bindings to measured pool search.

The user supplies training SEMANTICS in an inspectable factory. Only the system
chooses physical G/Score ranks, in-flight window and CPU activation policy.
No GPU allocation, pretrained model downloads or background orchestration.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from datetime import timedelta

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint,write_new_json
from .role_program import RoleTask, AllocatedWorld
from .role_transport import RoleFunctions


def load_factory(entry,parameters):
    path,sep,name=entry.rpartition(':')
    require(sep and name.isidentifier(),'role factory must be local.py:callable',path='role.cli')
    source=Path(path).expanduser().resolve()
    require(source.is_file() and source.suffix=='.py','local role factory module required',path='role.cli')
    source_sha=file_sha256(source)
    spec=importlib.util.spec_from_file_location('distillschedule_role_user_'+source_sha,source)
    require(spec is not None and spec.loader is not None,'role factory cannot load',path='role.cli')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    factory=getattr(module,name,None)
    require(callable(factory),'unknown role factory entry',path='role.cli')
    value=factory(parameters)
    require(isinstance(value,(list,tuple)) and len(value)==2 and
            isinstance(value[0],RoleTask) and isinstance(value[1],RoleFunctions),
            'factory must produce (RoleTask, RoleFunctions), not a pool recipe',path='role.cli')
    task,bindings=value;bindings.validate(task)
    require(file_sha256(source)==source_sha,'role source changed while loading',path='role.cli')
    require(bool(bindings.data_identity) and len(bindings.data_identity)==64 and
            all(c in '0123456789abcdef' for c in bindings.data_identity),
            'explicit content-hashed data cohort required',path='role.cli')
    return task,bindings,{'factory_sha256':source_sha,'factory_name':name,
                          'parameters_sha256':fingerprint(parameters)}


def allocated_hardware(args,device):
    import torch
    import torch.distributed as dist
    require(dist.is_initialized() and dist.get_world_size()>=2,
            'role pools require explicit torchrun rank allocation',path='role.cli')
    if device.type=='cuda':
        free,total=torch.cuda.mem_get_info(device)
        reserved=torch.cuda.memory_reserved(device)
        rank_capacity=int(free+reserved-args.gpu_reserve_gib*2**30)
        require(rank_capacity>0,'GPU headroom below requested reserve',path='role.cli')
    else:
        require(args.memory_limit_bytes is not None and args.memory_limit_bytes>0,
                'CPU test storage budget must be explicit',path='role.cli')
        rank_capacity=args.memory_limit_bytes
    if args.host_memory_limit_bytes is None:
        # Host memory is a node resource, never one unlimited host pool per rank.
        pages=os.sysconf('SC_AVPHYS_PAGES')
        bytes_per_page=os.sysconf('SC_PAGE_SIZE')
        local_world=int(os.getenv('LOCAL_WORLD_SIZE','1'))
        rank_host=max(0, int(pages*bytes_per_page/local_world)-2**30)
    else:rank_host=args.host_memory_limit_bytes
    record={'rank':dist.get_rank(),'gpu':rank_capacity,'host':rank_host,'hostname':socket.gethostname(),
            'node':int(os.getenv('GROUP_RANK','0'))}
    rows=[None]*dist.get_world_size();dist.all_gather_object(rows,record)
    require({r['rank'] for r in rows}==set(range(dist.get_world_size())),'missing resource rank',path='role.cli')
    # Physical rank order determines GPU ownership and native P2P peers.
    rows.sort(key=lambda x:x['rank'])
    return AllocatedWorld(tuple(r['gpu'] for r in rows),tuple(r['node'] for r in rows),
                          tuple(r['host'] for r in rows))


def main(argv=None):
    parser=argparse.ArgumentParser(prog='distillctl workloads native program role-*')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('role-describe','role-auto','role-plan','role-run'):
        p=sub.add_parser(name)
        p.add_argument('--output',type=Path,required=name!='role-describe')
        if name!='role-plan':
            p.add_argument('--factory',required=True)
            p.add_argument('--parameters',type=Path,required=True)
            p.add_argument('--execute',action='store_true')
        if name in ('role-auto','role-run'):
            p.add_argument('--distributed',choices=('gloo','nccl'),required=True)
            p.add_argument('--group-timeout',type=int,default=180)
            p.add_argument('--memory-limit-bytes',type=int)
            p.add_argument('--host-memory-limit-bytes',type=int)
            p.add_argument('--gpu-reserve-gib',type=int,default=2)
        if name=='role-run':p.add_argument('--cycles',type=int,default=1)
        if name=='role-auto':
            p.add_argument('--max-candidates',type=int,default=4)
            p.add_argument('--samples',type=int,default=1)
            p.add_argument('--max-native-runs',type=int,default=8)
            p.add_argument('--disable-offload',action='store_true')
            p.add_argument('--allow-generator-fsdp',action='store_true',
                           help='admit native FSDP2 Generator-pool alternatives')
        if name in ('role-run','role-plan'):p.add_argument('--plan' if name=='role-run' else '--calibration',type=Path,required=True)
    args=parser.parse_args(argv)
    owned=False
    try:
        if args.command=='role-plan':
            from .role_planning import search
            result=search(args.calibration)
            write_new_json(args.output,result)
            print(json.dumps(result,sort_keys=True));return 0
        if not args.execute:
            print(json.dumps({'status':'NOT_EXECUTED','reason':'trusted factory requires explicit --execute',
                              'model_constructed':False}));return 0
        values=json.loads(args.parameters.read_text())
        require(isinstance(values,dict),'JSON parameter object required',path='role.cli')
        task,bindings,code=load_factory(args.factory,values)
        if args.command=='role-describe':
            result={'task':task.to_dict(),'program':task.program().to_dict(),
                    'factory':code,'role_estimates':bindings.role_storage_estimate_bytes,
                    'data_identity':bindings.data_identity,'gpu_execution':'NOT_RUN'}
            print(json.dumps(result,sort_keys=True));return 0
        import torch
        import torch.distributed as dist
        require(1<=args.group_timeout<=3600 and 0<=args.gpu_reserve_gib<=16,
                'invalid group timeout/reserve',path='role.cli')
        require(all(k in os.environ for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
                and int(os.environ['WORLD_SIZE'])>=2,
                'must join a preallocated torchrun group',path='role.cli')
        if args.distributed=='nccl':
            require(torch.cuda.is_available(),'CUDA/NCCL not available',path='role.cli')
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        if not dist.is_initialized():
            dist.init_process_group(args.distributed,timeout=timedelta(seconds=args.group_timeout))
            owned=True
        require(dist.get_backend()==args.distributed,'group backend mismatch',path='role.cli')
        device=torch.device('cuda',torch.cuda.current_device()) if args.distributed=='nccl' else torch.device('cpu')
        hardware=allocated_hardware(args,device)
        from .role_planning import calibrate,search,run_selected
        root=args.output.resolve()
        if args.command=='role-auto':
            require(bindings.role_storage_estimate_bytes is not None,
                    'model residency estimates required before cross-pool search',path='role.cli')
            reserve=bindings.role_storage_estimate_bytes
            calibration=calibrate(task,hardware,bindings,root/'calibration',device=device,
                generator_bytes=reserve['generator'],score_bytes=reserve['score'],
                max_candidates=args.max_candidates,samples=args.samples,max_native_runs=args.max_native_runs,
                allow_offload=not args.disable_offload,
                allow_generator_fsdp=args.allow_generator_fsdp)
            # Only rank zero writes the common plan. The runtime rechecks all
            # source/hardware/semantics identities prior to ANY selected update.
            if dist.get_rank()==0:
                selected=search(root/'calibration/calibration.json')
                write_new_json(root/'plan.json',selected)
            dist.barrier()
            execution,_,_=run_selected(task,hardware,bindings,root/'plan.json',root/'selected',device=device)
            result={'status':'EXECUTED_ROLE_POOL_NEEDS_INDEPENDENT_REFERENCE',
                    'selected_implementation':execution['layout'],
                    'plan_path':str(root/'plan.json'),
                    'selected_wall_ns':execution['wall_ns'],
                    'hardware':calibration['identity']['hardware'],
                    'calibration_budget':calibration['budget'],
                    'gpu_execution':execution['gpu_execution'],'claimable_speedup':False}
            if dist.get_rank()==0:write_new_json(root/'result.json',result)
        else:
            execution,_,_=run_selected(task,hardware,bindings,args.plan,root,device=device,cycles=args.cycles)
            result={'status':execution['status'],'plan_path':str(args.plan),
                    'selected_implementation':execution['layout'],
                    'gpu_execution':execution['gpu_execution'],'claimable_speedup':False}
        if dist.get_rank()==0:print(json.dumps(result,indent=2,sort_keys=True))
        return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print('ROLE_ERROR:',exc,file=sys.stderr);return 2
    finally:
        if owned:
            import torch.distributed as dist
            dist.destroy_process_group()
