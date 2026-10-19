#!/usr/bin/env python3
"""W6 node glue only. Delegate unchanged r2 flock/birth/FD/cancel owner.

No SSH, trainer, timing code, GPU-PID signal authority or workload fallback.
Preparation delegates the original native GPU metadata CPU qualifier. The
parent owner must remain Torch-free. Default invocation writes nothing.
"""
from __future__ import annotations
import argparse
import importlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest

sys.dont_write_bytecode=True
from h3_ar_match_controller import (GENERATOR_SHA, PHASES, bound, environment, gate_admission,
    gpu_command, implementation, load_config, native_manifest, paths, read, ref, require, sha,
    source_state, torch_library_dir, validate_ranks, write, native_dependency_identity,
    native_dependency_import_identity)


def emit(value):
    try:print('H3_AR_OWNER='+json.dumps(value,sort_keys=True,allow_nan=False),flush=True)
    except BrokenPipeError:pass  # The small durable proof was written first.


def owner_api(cfg):
    packet=Path(cfg['owner_packet']).resolve(strict=True)
    for name,digest in cfg['owner_files'].items():require(sha(packet/name)==digest,'immutable r2 owner changed: '+name)
    sys.path.insert(0,str(packet/'scripts'))
    result={name:importlib.import_module(name) for name in ('node_stage','fd_ownership','packet_common')}
    require(all(Path(module.__file__).resolve()==packet/'scripts'/(name+'.py') for name,module in result.items()),'mixed owner import')
    require('torch' not in sys.modules,'node owner must remain Torch-free')
    return result['node_stage'],result['fd_ownership']


def package_identity(manifest):
    """Original owner's small dependency-source tree admission, never weights."""
    root=Path(manifest['dependency_package_root']);digest=__import__('hashlib').sha256();stats={}
    for path in sorted(root.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
            before=path.stat()
            name=str(path.relative_to(root));digest.update(name.encode()+b'\0');digest.update(path.read_bytes())
            st=path.stat();stats[name]=[st.st_size,st.st_mtime_ns,st.st_ctime_ns]
            require((before.st_size,before.st_mtime_ns,before.st_ctime_ns)==tuple(stats[name]),
                'native dependency source changed during hash admission')
    require(len(stats)==manifest['dependency_package_file_count']
        and digest.hexdigest()==manifest['dependency_package_tree_sha256'],'native dependency source identity changed')
    return dict(root=str(root),sha256=digest.hexdigest(),files=stats,scope='ORIGINAL_DEPENDENCY_SOURCE_TREE_NOT_MODEL_PAYLOAD')


def check_package_stat(binding):
    root=Path(binding['root']);now={}
    for path in sorted(root.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
            st=path.stat();now[str(path.relative_to(root))]=[st.st_size,st.st_mtime_ns,st.st_ctime_ns]
    require(now==binding['files'],'admitted dependency-source stat inventory changed')


def bind_startup_dependency(cfg,original,derived,identity):
    """Preserve original generator receipt; hash-bound derivative is an asset."""
    require(Path(original)!=Path(derived),'original startup metadata must not be overwritten')
    row=read(original)
    require('w6_native_dependency' not in row and 'original_native_metadata' not in row,
        'startup dependency derivative must originate from native generator')
    write(derived,dict(row,original_native_metadata=ref(original),
        w6_native_dependency=dict(contract=cfg['native_dependency'],import_identity=identity)))
    return ref(derived)


def cpu_child(cfg,node,digest):
    """Fresh hidden child: real native imports/library ABI, no construction."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU child must hide CUDA')
    source=source_state(cfg);manifest=native_manifest(cfg)
    bound(cfg['startup_generator']);require(cfg['startup_generator']['sha256']==GENERATOR_SHA,'original metadata CPU generator changed')
    require(manifest['file_sha256'].get(cfg['startup_generator']['path'])==GENERATOR_SHA,'metadata generator missing native provenance')
    # Pin the actual module file, not a shim advertising the desired version.
    import torch
    from distillschedule.strategies.h3_ar_worker import runtime_identity
    runtime=runtime_identity(torch)
    require(Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        and not torch.cuda.is_initialized(),'real Torch 2.10 prefix/no-CUDA proof required')
    torch_library=torch_library_dir(cfg['nodes'][node]['python']['path'],torch_file=torch.__file__)
    native_import=(native_dependency_import_identity(cfg,torch,node=node) if 'native_dependency' in cfg else None)
    import runpy
    startup=paths(cfg)['root']/f'reports/startup-{"native-" if native_import else ""}node{node}.json'
    startup.parent.mkdir(parents=True,exist_ok=True)
    original=sys.argv;sys.argv=[cfg['startup_generator']['path'],str(startup)]
    try:runpy.run_path(cfg['startup_generator']['path'],run_name='__main__')
    finally:sys.argv=original
    from distillschedule.strategies.h3_ar_assets import HELPER_PINS,LIBRARY_PINS,checked,validate_source_tree
    source_root=Path(manifest['source_root']).resolve(strict=True);receipt=ref(source_root/'source_receipt.json')
    inventory={item['relative_path']:item['sha256'] for item in read(source_root/'source_receipt.json')['objects']
        if item['relative_path'].endswith('.py')}
    validate_source_tree(source_root,receipt,inventory)
    roots=set()
    for name,pin in HELPER_PINS.items():
        matches=[p for p,h in manifest['file_sha256'].items() if Path(p).stem==name and h==pin]
        require(matches,'native helper source missing: '+name);roots.add(str(checked(dict(path=matches[0],sha256=pin)).parent))
    sys.path[:0]=[str(Path(manifest['source_root'])/'examples/minimax_h3/model_training'),manifest['source_root'],*sorted(roots)]
    modules=[importlib.import_module(name) for name in ('h3_dmd2','train_dmd2','dmd2_zero3','diffsynth.models.minimax_h3_dit')]
    require(all(Path(m.__file__).resolve().is_relative_to(source_root) for m in modules),'native source import shadowed')
    flags=manifest['worker_flags'];libraries={}
    for name,flag in (('pinned','--pinned-audit-library'),('rms','--rms-library'),('cutlass','--cutlass-lora-library')):
        require(flags.count(flag)==1,'native library flag missing/duplicate')
        path=checked(dict(path=flags[flags.index(flag)+1],sha256=LIBRARY_PINS[name]))
        torch.ops.load_library(str(path));libraries[name]=ref(path)
    import deepspeed
    require(Path(deepspeed.__file__).resolve().is_relative_to(Path(manifest['dependency_package_root']).resolve()),
        'actual DeepSpeed module differs from accepted source tree')
    tests=importlib.import_module('test_train_dmd2')
    require(Path(tests.__file__).resolve().is_relative_to(source_root),'native CPU test module shadowed')
    test_log=io.StringIO();native_tests=unittest.TextTestRunner(stream=test_log,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests))
    require(native_tests.testsRun==13 and native_tests.wasSuccessful() and not native_tests.skipped,
        'original13 native CPU tests failed: '+test_log.getvalue())
    replay=importlib.import_module('zero3_checkpoint_tensor_identity_r2').cpu_tests()
    require(replay['status']=='PASS_ORIGINAL_BUG_REPRO_AND_BITWISE_CPU_OUTPUT_INPUT_LORA_GRAD'
        and replay['metadata_check_disabled'] is False,'original checkpoint identity CPU parity failed')
    from distillschedule.strategies.h3_ar_task import H3ARTask
    from distillschedule.strategies.h3_ar_inputs import preflight_native_inputs
    task=H3ARTask.from_dict(read(bound(cfg['task'])))
    # Every original raw sample must pass the SAME validator used by Session,
    # after the unchanged original prepare_batch, before this owner admits GPUs.
    input_proof=preflight_native_inputs(task,ref(manifest['cache_manifest']),modules[0],torch)
    input_path=paths(cfg)['root']/f'reports/native-inputs-node{node}.json'
    write(input_path,input_proof)
    require(not torch.cuda.is_initialized(),'CPU source/ABI qualification created CUDA context')
    native_dependency=None
    if native_import is not None:
        native_dependency=native_dependency_identity(cfg)
        derived=paths(cfg)['root']/f'reports/startup-node{node}.json'
        bind_startup_dependency(cfg,startup,derived,native_import);startup=derived
    row=dict(status='PASS_REAL_TORCH210_NATIVE_IMPORT_ABI_NO_CUDA',source=source,runtime=runtime,
        torch_file=str(Path(torch.__file__).resolve()),torch_library_dir=str(torch_library),
        python_prefix=str(Path(sys.prefix).resolve()),deepspeed_file=str(Path(deepspeed.__file__).resolve()),
        libraries=libraries,config_sha256=digest,node=node,startup_gpu_info=ref(startup),
        native_input_preflight=ref(input_path),input_contract_sha256=task.input_contract_sha256,
        native_cpu_tests=dict(run=native_tests.testsRun,failures=len(native_tests.failures),errors=len(native_tests.errors),log=test_log.getvalue()),
        checkpoint_identity_cpu=replay,full_model_constructed=False,CUDA_initialized=False)
    if native_import is not None:
        row.update(native_dependency=native_dependency,native_dependency_import=native_import)
    write(paths(cfg)['root']/f'reports/prepare-child-node{node}.json',row)
    return row


def local_proof(cfg,node,phase,digest):
    root=paths(cfg)['root'];stage=root/'lifecycle'/phase/f'node{node}';lifecycle=read(stage/'lifecycle.json')
    require(lifecycle['cleanup_pass'] is True and lifecycle['status']=='PROCESS_EXIT_ONLY_REQUIRES_NATIVE_RECEIPTS'
        and lifecycle['binding']['config_sha256']==digest and lifecycle['binding']['phase']==phase
        and lifecycle['binding']['node']==node,'owned lifecycle missing/failed')
    after=read(stage/'post-teardown-audit.json')
    require(after['oom_delta']==dict(oom=0,oom_kill=0),'owner observed OOM increment')
    if phase=='prepare':
        child=read(root/f'reports/prepare-child-node{node}.json');require(child['config_sha256']==digest
            and child['node']==node and child['source']==source_state(cfg)
            and child['status']=='PASS_REAL_TORCH210_NATIVE_IMPORT_ABI_NO_CUDA','CPU child proof differs')
        from distillschedule.strategies.h3_ar_task import H3ARTask
        from distillschedule.strategies.h3_ar_inputs import validate_preflight
        task=H3ARTask.from_dict(read(bound(cfg['task'])))
        validate_preflight(task,read(bound(child['native_input_preflight'])))
        require(child['input_contract_sha256']==task.input_contract_sha256,'CPU child input contract differs')
        startup=read(bound(child['startup_gpu_info']))
        require(startup['status']=='PASS_ORIGINAL_GPU_INFO_NO_CUDA_CONTEXT'
            and startup['hostname']==cfg['nodes'][node]['hostname']
            and startup['ordered_gpu_uuids']==cfg['nodes'][node]['assigned_gpu_uuids'],'native startup UUID/host differs')
        proof=dict(status='PASS_CPU_PREPARE_NO_CUDA',node=node,phase=phase,config_sha256=digest,
            source=source_state(cfg),cleanup_pass=True,lifecycle=ref(stage/'lifecycle.json'),
            child=ref(root/f'reports/prepare-child-node{node}.json'),startup_gpu_info=child['startup_gpu_info'],
            native_input_preflight=child['native_input_preflight'],input_contract_sha256=task.input_contract_sha256,
            capacity=after['host_capacity'],dependency_source=after['dependency_source'],GPU_started=0)
        if 'native_dependency' in cfg:
            require(child.get('native_dependency')==after.get('native_dependency')
                and startup.get('w6_native_dependency')==dict(contract=cfg['native_dependency'],
                    import_identity=child['native_dependency_import']),
                'CPU native dependency source/import/startup asset differs')
            bound(startup['original_native_metadata'])
            proof['native_dependency']=after['native_dependency']
        return proof
    rows=validate_ranks(cfg,phase,node)
    return dict(status='PASS_NATIVE_OWNER',node=node,phase=phase,config_sha256=digest,cleanup_pass=True,
        lifecycle=ref(stage/'lifecycle.json'),post_teardown=ref(stage/'post-teardown-audit.json'),
        native_rank_sources=[ref(path) for path,_ in rows],numerical_status='REVIEW_REQUIRES_MATCHED_REFERENCE')


def run(cfg,node,phase,digest,config_path,admission_sha=None):
    source_state(cfg);manifest=native_manifest(cfg);spec=cfg['nodes'][node];bound(spec['python'])
    require(socket.gethostname()==spec['hostname'],'assigned hostname changed')
    require(sha(__file__)==cfg['scripts']['node']['sha256'],'node wrapper changed')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','parent owner must not see CUDA')
    owner,fd_module=owner_api(cfg)
    from distillschedule.strategies.h3_ar_worker import host_admission,host_capacity_signature,observe_host_memory
    from distillschedule.strategies.h3_ar_planning import validate_policy
    policy=read(bound(cfg['capacity_policy']));validate_policy(policy)
    if phase!='prepare':
        require(admission_sha is not None and sha(paths(cfg)['admission'])==admission_sha,
            'explicit CPU asset admission hash missing/changed')
        gate_admission(cfg,digest)
    root=paths(cfg)['root'];stage=root/'lifecycle'/phase/f'node{node}';coordination=root/'coordination'/phase
    coordination.mkdir(parents=True,exist_ok=True)
    env=environment(cfg,node,phase)
    for key in ('TMPDIR','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR'):Path(env[key]).mkdir(parents=True,exist_ok=True)
    cmd=([spec['python']['path'],'-s','-B',str(Path(__file__).resolve()),'--config',str(config_path),
        '--sha256',digest,'--node',str(node),'--phase','prepare','--cpu-child','--execute']
        if phase=='prepare' else gpu_command(cfg,node,phase))
    before={};entry={};fd=[];audits=[];context={};next_audit=[0.];next_status=[0.];dependency={}
    native_dependency={}
    start=time.monotonic()
    def remaining():
        require(time.monotonic()-start<(180 if phase=='prepare' else cfg['work_seconds']),
            'bounded original work deadline exceeded')
    def snapshot(idle):
        gpu=owner.gpu_audit(spec['assigned_gpu_uuids'],require_idle=idle)
        host=observe_host_memory(policy,node_rank=node)
        require(host_capacity_signature(host)==entry['host_capacity'],'host capacity/capability changed under owner')
        require(all(host['cgroup_events'][k]==before[k] for k in ('oom','oom_kill')),'host OOM increment')
        return gpu,host
    def preflight():
        remaining()
        require(sha(config_path)==digest,'config changed under node lock');source_state(cfg);native_manifest(cfg)
        bound(spec['python'])
        if phase!='prepare':require(sha(paths(cfg)['admission'])==admission_sha,'CPU asset admission changed under lock')
        entry.update(host_admission(policy,node_rank=node));before.update(entry['initial_events'])
        gpu=owner.gpu_audit(spec['assigned_gpu_uuids'],require_idle=True)
        fd.append(fd_module.FDOwnership(cfg['owner_packet'],'PINNED_VERIFIED_SSH_ANCESTOR'));observed=fd[0].audit()
        if phase=='prepare':dependency.update(package_identity(manifest))
        else:
            prepared=read(root/f'reports/prepare-node{node}.json');require(prepared['config_sha256']==digest,'CPU preparation config differs')
            dependency.update(package_identity(manifest))
            require(dependency==prepared['dependency_source'],'native dependency source differs from CPU admission')
            require(prepared['capacity']==entry['host_capacity'],'actual host capability differs from prepared admission')
            from distillschedule.strategies.h3_ar_assets import H3ARAssets
            H3ARAssets.from_dict(read(paths(cfg)['assets'])).validate()
            from h3_ar_match_controller import verified_owner_proof
            for previous in PHASES[:PHASES.index(phase)]:
                for peer_node in (0,1):verified_owner_proof(cfg,previous,peer_node,digest)
            if phase in PHASES[2:]:
                gate=read(paths(cfg)['plan_gate']);require(gate['config_sha256']==digest,'CPU plan gate changed');bound(gate['saved_plan'])
                from distillschedule.strategies.h3_ar_planning import validate_plan
                from distillschedule.strategies.h3_ar_task import H3ARTask
                saved,_=validate_plan(H3ARTask.from_dict(read(bound(cfg['task']))),paths(cfg)['plan'])
                require(saved['identity']==gate_admission(cfg,digest)['identity'],
                    'manual/selected assets/source differ from calibrated saved plan')
        if 'native_dependency' in cfg:
            native_dependency.update(native_dependency_identity(cfg))
            if phase!='prepare':
                require(prepared.get('native_dependency')==native_dependency,
                    'native Accelerate source assembly differs from CPU admission')
        if phase!='prepare':
            if node==0:
                with socket.socket() as sock:sock.bind(('0.0.0.0',cfg['master_ports'][phase]))
            write(coordination/f'ready{node}.json',dict(config_sha256=digest,node=node,phase=phase,
                host=socket.gethostname(),lock_held=True,owner_birth=owner.birth(os.getpid())))
            peer=coordination/f'ready{1-node}.json';deadline=time.monotonic()+90
            while not peer.exists():
                remaining()
                require(not (coordination/f'abort{1-node}.json').exists(),'peer aborted before launch')
                require(time.monotonic()<deadline,'paired held-lock readiness timeout');time.sleep(.25)
            row=read(peer);require(row['config_sha256']==digest and row['node']==1-node and row['phase']==phase
                and row['host']==cfg['nodes'][1-node]['hostname'] and row['lock_held'] is True,'peer lock/config/host changed')
            require(not (coordination/f'abort{1-node}.json').exists(),'peer aborted before Popen')
            snapshot(True);fd[0].audit()  # Last-moment admission after peer wait.
        remaining();return dict(gpu=gpu,fd=observed,host_admission=entry,dependency_source=dependency)
    def heartbeat(known):
        remaining()
        if phase!='prepare':require(not (coordination/f'abort{1-node}.json').exists(),'peer abort; cancel owned births only')
        if time.monotonic()>=next_audit[0]:
            gpu,host=snapshot(False)
            observed=fd[0].audit(known,refresh=lambda:owner.discover(owner.birth(os.getpid()),known,owner_pid=os.getpid()),
                owner_context=context,fresh_gpu=lambda:owner.gpu_audit(spec['assigned_gpu_uuids'],require_idle=False),gpu_observation=gpu)
            free=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=30)
            values=[line.split(',') for line in free.strip().splitlines()]
            require(len(values)==8 and [r[1].strip() for r in values]==spec['assigned_gpu_uuids'],'sampled free UUID inventory differs')
            minimum=[int(r[2].strip())<<20 for r in values]
            require(all(v>=policy['gpu_reserve_bytes'] for v in minimum),'sampled driver reserve below frozen policy')
            audits.append(dict(monotonic=time.monotonic(),gpu=observed.get('fresh_gpu_observation',gpu),fd=observed,
                host=host,driver_free_bytes=minimum))
            require(len(audits)<=360,'bounded lifecycle inventory exceeded');next_audit[0]=time.monotonic()+5
        if time.monotonic()>=next_status[0]:
            try:print('H3_AR_HEARTBEAT='+json.dumps(dict(phase=phase,node=node,elapsed=time.monotonic()-start,
                local_rank_receipts=len(list((root/phase).glob('rank_*.json'))),GPU_auxiliary_SSH=0)),flush=True)
            except BrokenPipeError:pass
            next_status[0]=time.monotonic()+30
    def finalizer():
        require(before,'preflight was not admitted')
        gpu,host=snapshot(True);final_fd=fd[0].audit();check_package_stat(dependency)
        result=dict(gpu=gpu,fd=final_fd,oom_delta={k:host['cgroup_events'][k]-before[k] for k in before},
            host_capacity=host_capacity_signature(host),dependency_source=dependency,audits=audits,
            driver_sampling_scope='actual memory.free approximately5s; unsampled minima UNKNOWN',
            host_peak_status=host['host_peak_status'],host_peak_bytes=host['host_peak_bytes'])
        if 'native_dependency' in cfg:
            require(native_dependency_identity(cfg)==native_dependency,
                'native Accelerate source changed under owner')
            result['native_dependency']=native_dependency
        result['held_seconds']=time.monotonic()-start
        write(stage/'post-teardown-audit.json',result)
        require(result['held_seconds']<cfg['lease_seconds'],'original owner lease exceeded after safe cleanup')
        return dict(post_teardown=ref(stage/'post-teardown-audit.json'))
    try:
        result=owner.run_owned(Path(cfg['source']),cmd,env,stage,timeout=180 if phase=='prepare' else cfg['work_seconds'],
            preflight=preflight,heartbeat=heartbeat,finalizer=finalizer,fd_context=context,
            binding=dict(config_sha256=digest,source_head=cfg['expected_head'],node=node,phase=phase))
        require(result['cleanup_pass'] is True,'owned cleanup failed')
        proof=local_proof(cfg,node,phase,digest)
        target=root/(f'reports/prepare-node{node}.json' if phase=='prepare' else f'reports/{phase}-node{node}.json')
        write(target,proof);emit(proof);return proof
    except BaseException as error:
        path=coordination/f'abort{node}.json'
        if not path.exists():write(path,dict(config_sha256=digest,node=node,phase=phase,error=repr(error)))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--node',type=int,choices=(0,1),required=True)
    parser.add_argument('--phase',choices=('prepare',*PHASES),required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--cpu-child',action='store_true')
    parser.add_argument('--admission-sha256')
    parser.add_argument('--recover-proof',action='store_true')
    args=parser.parse_args(argv)
    try:
        require(sha(args.config)==args.sha256,'config SHA changed');cfg=load_config(args.config)
        cfg['_config_path']=str(args.config.resolve())
        if not args.execute:
            print(json.dumps(dict(status='PLAN_ONLY_NO_SSH_GPU_LOCK_WRITE',gpu_execution='NOT_RUN',phase=args.phase,node=args.node)));return 0
        if args.cpu_child:
            require(args.phase=='prepare' and not args.recover_proof,'CPU-child mode must be preparation only');cpu_child(cfg,args.node,args.sha256)
        elif args.recover_proof:
            source_state(cfg);proof=local_proof(cfg,args.node,args.phase,args.sha256)
            target=paths(cfg)['root']/(f'reports/prepare-node{args.node}.json' if args.phase=='prepare' else f'reports/{args.phase}-node{args.node}.json')
            if target.exists():require(read(target)==proof,'existing recovered proof changed')
            else:write(target,proof)
            emit(proof)
        else:run(cfg,args.node,args.phase,args.sha256,args.config,args.admission_sha256)
        return 0
    except (ValueError,KeyError,TypeError,OSError,RuntimeError,subprocess.SubprocessError) as error:
        print('ERROR: '+str(error),file=sys.stderr,flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
