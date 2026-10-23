#!/usr/bin/env python3
"""Versioned two-arm W6 storage experiment; original atomic owner unchanged. No trainer, timer, retry, or auxiliary SSH.

GPU phases use only the two r2-style long owner streams. Receipt-host CPU
subcommands are invoked through the root's existing remote_io transport.
Default invocation is read-only intent; execution always requires --execute.
"""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import replace
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
SCHEMA='distillschedule.h3_ar_metadata_barrier_controller.v1'
DEPENDENCY_SCHEMA='distillschedule.h3_ar_metadata_barrier_controller.v2'
NATIVE_ACCELERATE={
    'version':'1.14.0',
    'package':dict(tree_sha256='3ee3c0d50df3d29faa8a07034a4ec26da94e964a7f9e4336e790457e52be1b6f',files=84,bytes=1491684),
    'dist_info':dict(tree_sha256='104108264e27a84ff340ddcb95e16839a5f199bc3464d9434a52bb0dbde601c7',files=8,bytes=45656),
    'environment_sha256':'5a161dd2ba6b27a26d0f1a36bf94d48a52bef26eed4e0c355445776c821ecc84',
    'metadata_sha256':'ed1c0dd9cca2d1449bbae7d641208431f5fccc7d5e06e20bec802c1d81340f6c',
    'provenance_probe_sha256':['9a6340983b9c52a0eb9946ed640809fe07494522076d9f1d7e4c6cbd1eb9cfa8',
        '78065e73757775a2b277e132b7e74d374f51a3079b6cc43895aa8bc38f6ae354']}
PHASES=('manual-control','manual-child')
MANUAL_WORKER_SHA='8812bed6a7c7b9eda2beb4b58b32cd3264ed2e5e0affa3ec8e092713e47c4590'
CANARY_SHA='3c93aea640b72b82589cec3460162b02cb2014dc2dec3d226b751a861b0761a5'
OWNER_FILES={
 'scripts/node_stage.py':'b4f013a96c371f8c54e47b56b0e04b9375e23c4bcde8a8b30e6b3fe09c27c71a',
 'scripts/fd_ownership.py':'c6e411de4cd65f3213b39e0c4a8ba7bb3d3b9ca4d31ed74ce658529880cb218e',
 'scripts/packet_common.py':'f950571dd591bf5a4d45b63d9071541644f56acf71edd279430e4f4577d09917',
 'scripts/stage_command.py':'60074c66282ee53de07839f46855291c95b452d8cec6cdeaaec01ddaa867a311',
 'scripts/check_outputs.py':'59805def5ae5c472947bb14b913ddade9b30251fb4b42faf26e807b540cd817b'}
GENERATOR_SHA='49f7496bf64ce75137cd72e85c901d905a1f7bbde683529d97ad87b511b15d52'
NATIVE_WORKER_SHA='20853d3c8a5a0457fa56c1e395032ad97c72fd3ca94e058a24bb90e80e9fe2e1'
SOURCE_RECEIPT_SHA='9ca4ad8ca13b0a9e7397f6313e4eca225f10f52802e90423b06f7e786a58b989'


def require(ok, why):
    if not ok: raise ValueError(why)


def read(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON key: '+key); result[key]=value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def ref(path): return dict(path=str(Path(path).resolve(strict=True)),sha256=sha(path))


def executable_ref(path):
    # A venv's python is commonly a symlink. Resolving it in argv silently
    # drops the venv even though the executable payload SHA is unchanged.
    return dict(path=str(Path(path).absolute()),sha256=sha(path))


def bound(value):
    require(isinstance(value,dict) and set(value)=={'path','sha256'}
        and Path(value['path']).is_absolute() and re.fullmatch('[0-9a-f]{64}',value['sha256']), 'invalid file binding')
    require(sha(value['path'])==value['sha256'],'file changed: '+value['path'])
    return Path(value['path'])


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    # Atomic exclusive publication, matching the existing packet. A failed
    # partial temporary remains evidence; CPU recovery can publish afresh.
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.publish-',delete=False) as stream:
        temporary=Path(stream.name)
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    try:os.link(temporary,path)
    finally:temporary.unlink()


def paths(cfg):
    root=Path(cfg['run_root'])
    return dict(root=root,assets=root/'configs/assets.json',admission=root/'configs/assets-admission.json',
        checkpoint_audit=root/'artifacts/checkpoint-audit.json',plan=root/'artifacts/selected-plan.json',
        plan_gate=root/'reports/plan-gate.json')


def load_config(path):
    cfg=read(path)
    fields={'schema_version','source','expected_head','native_manifest','task','capacity_policy','nodes',
        'owner_packet','owner_files','startup_generator','scripts','local_root','run_root','cache_root',
        'master_addr','master_ports','cycles','max_native_starts','profiler','work_seconds','lease_seconds','manual_packet'}
    require(isinstance(cfg,dict),'controller object required')
    if cfg.get('schema_version')==DEPENDENCY_SCHEMA:fields.add('native_dependency')
    require(set(cfg)==fields and cfg['schema_version'] in (SCHEMA,DEPENDENCY_SCHEMA),'controller fields/schema changed')
    require(type(cfg['cycles'])is int and cfg['cycles']==2 and type(cfg['max_native_starts'])is int
        and cfg['max_native_starts']==2 and cfg['profiler'] is False,'two-start manual-only unprofiled C1/C2 bound changed')
    require(cfg['owner_files']==OWNER_FILES,'existing r2 owner policy changed')
    require(re.fullmatch('[0-9a-f]{40}',cfg['expected_head']) is not None,'explicit clean integration HEAD required')
    for key in ('source','local_root','run_root','cache_root','owner_packet'):
        require(Path(cfg[key]).is_absolute() and Path(cfg[key]) not in (Path('/'),Path('/home/lyy')),'absolute project-scoped '+key+' required')
    require(not Path(cfg['run_root']).is_relative_to(cfg['source']),'outputs must be outside source')
    require(set(cfg['master_ports'])==set(PHASES) and len(set(cfg['master_ports'].values()))==2
        and all(type(v)is int and 1024<=v<=65535 for v in cfg['master_ports'].values()),'two explicit distinct rendezvous ports required')
    require(0<cfg['work_seconds']<=1200 and cfg['work_seconds']<cfg['lease_seconds']<=1260,'original work/lease bound exceeded')
    require(len(cfg['nodes'])==2 and len({n['hostname'] for n in cfg['nodes']})==2,'exact two assigned hosts required')
    for node in cfg['nodes']:
        require(set(node)=={'hostname','python','assigned_gpu_uuids','ssh_target','ssh_port'},'node fields changed')
        require(len(node['assigned_gpu_uuids'])==len(set(node['assigned_gpu_uuids']))==8,'all-eight UUID order required')
        require(isinstance(node['ssh_target'],str) and not node['ssh_target'].startswith('-')
            and type(node['ssh_port'])is int and 1<=node['ssh_port']<=65535,'invalid existing transport endpoint')
    require(set(cfg['scripts'])=={'controller','node'},'script freeze missing')
    require(sha(__file__)==cfg['scripts']['controller']['sha256'],'controller source changed')
    return cfg


def source_state(cfg):
    source=Path(cfg['source'])
    require(subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==cfg['expected_head']
        and not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip(), 'source changed or dirty')
    sys.path[:0]=[str(source/'src'),str(source)]
    from distillschedule.workloads.best_known.registry import repository_root
    require(repository_root().resolve()==source.resolve(),'mixed source import')
    return importlib.import_module('distillschedule.strategies.h3_ar_planning').source_identity()


def native_manifest(cfg):
    manifest=read(bound(cfg['native_manifest']))
    require(manifest.get('cycles')==2 and manifest.get('worker_sha256')==NATIVE_WORKER_SHA
        and manifest.get('source_receipt_sha256')==SOURCE_RECEIPT_SHA,'not the accepted original two-cycle manifest')
    require(manifest['hosts']==[n['hostname'] for n in cfg['nodes']]
        and manifest['ordered_UUIDs_by_node']==[n['assigned_gpu_uuids'] for n in cfg['nodes']], 'original host/UUID assignment changed')
    require(manifest.get('profile',False) is False,'profiled manifest not admitted')
    require(cfg['work_seconds']==manifest['work_seconds'] and cfg['lease_seconds']==manifest['lease_seconds'],
        'original owner work/lease policy changed')
    for name,pin in manifest['file_sha256'].items():
        require(Path(name).suffix not in ('.safetensors','.pt','.bin','.nsys-rep','.sqlite'),
            'native owner metadata inventory unexpectedly contains a large payload')
        require(sha(name)==pin,'original owner metadata/source pin changed: '+name)
    return manifest


def allocator(implementation):
    require(implementation in ('zero3-native','zero3-ap','zero3-ap-gpu-tail1'),'only native/AP/versioned GPU-tail1 implementation admitted')
    return 'expandable_segments:True'+(',pinned_use_cuda_host_register:True,pinned_num_register_threads:4' if implementation!='zero3-native' else '')


def implementation(cfg,phase):
    require(phase in PHASES or phase=='prepare', 'explicit manual phase required')
    return 'zero3-ap'


def gpu_command(cfg,node,phase):
    require(phase in PHASES and node in (0,1),'invalid manual phase/node')
    p=paths(cfg)
    worker=Path(__file__).with_name('metadata_barrier_worker.py')
    require(sha(worker)==MANUAL_WORKER_SHA,'manual worker changed')
    command = [cfg['nodes'][node]['python']['path'],'-m','torch.distributed.run','--nnodes=2','--nproc-per-node=8',
        '--max-restarts=0','--node-rank='+str(node),'--master-addr='+cfg['master_addr'],
        '--master-port='+str(cfg['master_ports'][phase]),str(worker),
        '--task',cfg['task']['path'],'--assets',str(p['assets']),
        '--capacity-policy',cfg['capacity_policy']['path'],'--output',str(p['root']/phase),
        '--manual-policy','uncached-rollout-fa3-varlen-v1','--execute']
    command+=['--fa3-kernel-root','/share/project/shuchenweng/yongyan_liu/DMD-System_assets/lightx2v_minimax_dmd_20260817/hf_home/hub/models--kernels-community--flash-attn3/snapshots/acb0f42e6901d3e9effa8a83bd32bc2bf0cfdfc2/build/torch210-cxx11-cu128-x86_64-linux']
    packet=read(bound(cfg['manual_packet']))
    require(packet['schema']=='dmd.h3_ar_manual_metadata_barrier.v1'
        and packet['source_head']==cfg['expected_head'],'explicit external manual source packet required')
    require(sha(bound(packet['scripts']['worker']))==MANUAL_WORKER_SHA,'manual worker identity differs')
    bound(packet['scripts']['counter'])
    command+=['--packet',cfg['manual_packet']['path'],'--packet-sha256',cfg['manual_packet']['sha256'],
        '--restore-native-barrier','1'if phase=='manual-child'else'0']
    return command


def torch_library_dir(python, *, torch_file=None):
    # Keep the qualified venv entry, not its resolved system-python binary.
    prefix=Path(python).parents[1].resolve(strict=True);targets=set()
    for alias in Path(python).parents[1].glob('lib/python*/site-packages/torch/lib'):
        target=alias.resolve(strict=True)
        require(target.is_dir() and target.is_relative_to(prefix),
            'Torch library alias escaped qualified prefix or is not a directory')
        targets.add(target)
    require(len(targets)==1,'validated real Torch library directory missing/ambiguous')
    target=next(iter(targets))
    if torch_file is not None:
        module=Path(torch_file).resolve(strict=True)
        require(module.is_file() and module.is_relative_to(prefix)
            and (module.parent/'lib').resolve(strict=True)==target,
            'actual Torch module/library differs from qualified canonical path')
    return target


def native_accelerate_contract(overlay,manifest):
    """Exactly the complete original package + dist-info, never old Torch."""
    prefix=Path(manifest['python']).parents[1].resolve(strict=True)
    site=(prefix/'lib/python3.10/site-packages').resolve(strict=True)
    require(site.is_dir() and site.is_relative_to(prefix),'native package root escaped original interpreter prefix')
    overlay=Path(overlay).absolute()
    require('..' not in overlay.parts,'native dependency overlay traversal')
    return dict(schema_version='distillschedule.h3_ar_native_accelerate.v1',
        source_site_packages=str(site),overlay_root=str(overlay),**json.loads(json.dumps(NATIVE_ACCELERATE)))


def small_dependency_tree(root):
    """Strong-hash <=16MB source only; no weight or environment payload scan."""
    root=Path(root).resolve(strict=True);digest=hashlib.sha256();files={};size=0
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts or path.suffix=='.pyc':continue
        require(not path.is_symlink(),'native package nested symlink is not in admitted source')
        if not path.is_file():continue
        require(path.suffix not in ('.so','.bin','.pt','.safetensors','.nsys-rep','.sqlite'),
            'native dependency admission is small source only')
        before=path.stat();size+=before.st_size
        require(len(files)<2048 and before.st_size<=2_000_000 and size<=16_000_000,'bounded native package source exceeded')
        data=path.read_bytes();after=path.stat()
        stat=lambda st:[st.st_size,st.st_mtime_ns,st.st_ctime_ns]
        require(stat(before)==stat(after),'native package source changed during admission')
        name=str(path.relative_to(root));digest.update(name.encode()+b'\0');digest.update(data);files[name]=stat(after)
    return dict(tree_sha256=digest.hexdigest(),files=len(files),bytes=size,stat_inventory=files)


def native_dependency_identity(cfg, *, create=False):
    require('native_dependency' in cfg,'explicit native dependency contract required')
    contract=cfg['native_dependency'];overlay=Path(contract['overlay_root'])
    require(contract==native_accelerate_contract(overlay,native_manifest(cfg)),'native dependency source/version/provenance contract changed')
    parent=Path(cfg['run_root']).parent/'assets'
    require(overlay.parent.resolve()==parent.resolve() and not overlay.is_symlink(),
        'native overlay must be a real new-run assets child, never a shared environment')
    site=Path(contract['source_site_packages'])
    targets={'accelerate':site/'accelerate','accelerate-'+contract['version']+'.dist-info':site/('accelerate-'+contract['version']+'.dist-info')}
    if create:
        overlay.mkdir(parents=True,exist_ok=True)
        for name,target in targets.items():
            link=overlay/name
            if not link.exists() and not link.is_symlink():link.symlink_to(target,target_is_directory=True)
    require(overlay.is_dir() and {p.name for p in overlay.iterdir()}==set(targets),
        'native overlay must contain exactly Accelerate package and matching dist-info, no Torch/extra files')
    for name,target in targets.items():
        link=overlay/name
        require(target.is_dir() and target.resolve()==target and link.is_symlink()
            and link.resolve(strict=True)==target,'native dependency link/path changed')
    package=small_dependency_tree(targets['accelerate'])
    info=small_dependency_tree(targets['accelerate-'+contract['version']+'.dist-info'])
    for expected,actual in ((contract['package'],package),(contract['dist_info'],info)):
        require({k:actual[k] for k in expected}==expected,'native dependency full small-source SHA/count/bytes changed')
    require(sha(targets['accelerate']/'utils/environment.py')==contract['environment_sha256']
        and sha(site/('accelerate-'+contract['version']+'.dist-info')/'METADATA')==contract['metadata_sha256'],
        'native Accelerate original environment or distribution metadata SHA changed')
    from email.parser import BytesParser
    metadata=BytesParser().parsebytes((site/('accelerate-'+contract['version']+'.dist-info')/'METADATA').read_bytes())
    require(metadata.get('Name')=='accelerate' and metadata.get('Version')==contract['version'],
        'native Accelerate package/distribution version differs')
    if create:overlay.chmod(0o555)  # Only new links directory, never shared package files.
    return dict(contract=contract,package=package,dist_info=info,
        scope='SMALL_ORIGINAL_ACCELERATE_SOURCE_ONLY_NO_TORCH_ENV_INSTALL_OR_COPY')


def native_dependency_import_identity(cfg,torch, *, node=0):
    """Fresh CPU child proves real imports; original GPU-info API not called."""
    native_dependency_identity(cfg)
    python=cfg['nodes'][node]['python']['path']
    prefix=Path(python).parents[1].resolve(strict=True)
    torch_library_dir(python,torch_file=torch.__file__)
    require(Path(sys.prefix).resolve()==prefix and not torch.cuda.is_initialized(),
        'native dependency child must retain qualified Torch prefix and no CUDA')
    import inspect
    import importlib.metadata as metadata
    import accelerate
    import accelerate.utils.environment as environment_module
    contract=cfg['native_dependency'];site=Path(contract['source_site_packages'])
    distribution=metadata.distribution('accelerate')
    package=Path(accelerate.__file__).resolve(strict=True)
    environment_file=Path(inspect.getfile(environment_module.get_gpu_info)).resolve(strict=True)
    info=Path(distribution._path).resolve(strict=True)
    require(package==site/'accelerate/__init__.py' and environment_file==site/'accelerate/utils/environment.py'
        and info==site/('accelerate-'+contract['version']+'.dist-info')
        and accelerate.__version__==distribution.version==contract['version']
        and sha(environment_file)==contract['environment_sha256'],
        'actual native Accelerate package/function/distribution differs from original source assembly')
    require(not torch.cuda.is_initialized(),'native dependency import initialized CUDA')
    return dict(accelerate_version=accelerate.__version__,accelerate_file=str(package),
        environment_file=str(environment_file),environment_sha256=sha(environment_file),distribution_path=str(info),
        torch_file=str(Path(torch.__file__).resolve()),python_prefix=str(prefix),CUDA_initialized=False)


def environment(cfg,node,phase):
    manifest=native_manifest(cfg); python=Path(cfg['nodes'][node]['python']['path']); source=Path(cfg['source'])
    env={k:os.environ[k] for k in ('HOME','USER','LOGNAME','LANG','NVIDIA_VISIBLE_DEVICES') if k in os.environ}
    env.update(manifest['environment'])
    extra=[]
    if 'native_dependency' in cfg:
        native_dependency_identity(cfg)
        extra=[cfg['native_dependency']['overlay_root']]
    env['PYTHONPATH']=':'.join([str(source/'src'),str(source),*extra,manifest['environment']['PYTHONPATH']])
    env['PATH']=str(python.parent)+':/usr/local/bin:/usr/bin:/bin'
    # The old runtime prefix is never placed ahead of the validated real 2.10 libs.
    old=Path(manifest['python']).parents[1]
    target=torch_library_dir(python)
    if extra:
        for root in env['PYTHONPATH'].split(':'):
            path=Path(root)
            require(path.is_dir(),'native dependency PYTHONPATH must contain only existing source directories')
            candidates=[path/'torch',path/'torch.py',path/'torch.pyc',path/'torch.so',*path.glob('torch.*.so')]
            for candidate in candidates:
                if candidate.exists() or candidate.is_symlink():
                    require(candidate.resolve(strict=True)==target.parent,
                        'nonqualified Torch shadow in native dependency PYTHONPATH')
    env['LD_LIBRARY_PATH']=':'.join([str(target),str(python.parents[1]/'lib')]
        +[s for s in env.get('LD_LIBRARY_PATH','').split(':') if s and not Path(s).is_relative_to(old)])
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES='' if phase=='prepare' else ','.join(cfg['nodes'][node]['assigned_gpu_uuids']),
        DISTILLSCHEDULE_RUN_ID=Path(cfg['run_root']).name+'-'+phase,
        DISTILLSCHEDULE_EXPECTED_SOURCE_RECEIPT=manifest['source_receipt_sha256'],DISTILLSCHEDULE_RUN_ROOT=cfg['run_root'])
    selected='zero3-native' if phase=='prepare' else implementation(cfg,phase)
    env['PYTORCH_CUDA_ALLOC_CONF']=env['PYTORCH_ALLOC_CONF']=allocator(selected)
    cache=Path(cfg['cache_root'])/('node'+str(node))/phase
    for key,subdir in (('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton'),('TORCH_EXTENSIONS_DIR','extensions'),('TORCHINDUCTOR_CACHE_DIR','inductor')):
        env[key]=str(cache/subdir)
    return env


def node_command(cfg,node,phase,digest,remote_config,admission_sha=None):
    spec=cfg['nodes'][node]
    cmd=['env','-u','PYTHONPATH','-u','PYTHONHOME','CUDA_VISIBLE_DEVICES=',spec['python']['path'],'-s','-B',
        cfg['scripts']['node']['path'],'--config',remote_config,'--sha256',digest,'--node',str(node),'--phase',phase,'--execute']
    if admission_sha is not None:cmd+=['--admission-sha256',admission_sha]
    return ['ssh','-n','-tt','-oBatchMode=yes','-oControlMaster=no','-oControlPath=none','-oConnectTimeout=15',
        '-oServerAliveInterval=30','-oServerAliveCountMax=20','-p',str(spec['ssh_port']),spec['ssh_target'],shlex.join(cmd)+'; exit']


def intent(cfg,phase,digest,remote_config='/REMOTE/configs/controller.json'):
    return dict(status='PLAN_ONLY_NO_SSH_GPU_LOCK_OR_OUTPUT_WRITE',gpu_execution='NOT_RUN',phase=phase,
        config_sha256=digest,head=cfg['expected_head'],max_native_starts=4,
        node_commands=[node_command(cfg,n,phase,digest,remote_config) for n in (0,1)])


def check_sequence(cfg,phase,used,eofs):
    require(phase not in [r['phase'] for r in used],'repeat native start forbidden, failures count')
    require(len(used)<2,'manual two-start budget consumed')
    require(all(r['phase'] in eofs for r in used),'unknown owner terminal; no new SSH or retry')
    require([r['phase'] for r in used]==list(PHASES[:PHASES.index(phase)]),'native phase order changed')
    require(all(eofs[r['phase']]['status']=='PASS_CLEAN_NATIVE_PHASE' for r in used),'prior phase failed; stop and preserve')


def gate_admission(cfg,digest):
    p=paths(cfg); admission=read(p['admission'])
    require(admission['config_sha256']==digest and admission['source']==source_state(cfg),'asset admission source/config changed')
    require(bound(admission['assets'])==p['assets'],'asset admission path changed')
    bound(cfg['task']);bound(cfg['capacity_policy'])
    from distillschedule.strategies.h3_ar_task import H3ARTask
    from distillschedule.strategies.h3_ar_inputs import validate_preflight
    task=H3ARTask.from_dict(read(bound(cfg['task'])))
    require(len(admission['prepare_proofs'])==2,'both node CPU preparations required')
    for node,proof in enumerate(admission['prepare_proofs']):
        row=read(bound(proof))
        require(row['node']==node and row['phase']=='prepare' and row['config_sha256']==digest
            and row['source']==admission['source'] and row['status']=='PASS_CPU_PREPARE_NO_CUDA'
            and row['cleanup_pass'] is True,'admitted node CPU proof binding differs')
        require(bound(row['native_input_preflight'])==p['root']/f'reports/native-inputs-node{node}.json',
            'node-qualified original16 CPU input proof path differs')
        validate_preflight(task,read(bound(row['native_input_preflight'])))
        require(row['input_contract_sha256']==task.input_contract_sha256,'admitted original rank-input contract differs')
    if 'native_dependency' in cfg:
        require(admission.get('native_dependency')==cfg['native_dependency'],'asset admission native dependency changed')
    return admission


def verified_owner_proof(cfg,phase,node,digest):
    p=paths(cfg);path=p['root']/f'reports/{phase}-node{node}.json';row=read(path)
    stage=p['root']/'lifecycle'/phase/f'node{node}'
    require(row['status']=='PASS_NATIVE_OWNER' and row['cleanup_pass'] is True and row['config_sha256']==digest
        and row['node']==node and row['phase']==phase,'native owner proof binding failed')
    require(bound(row['lifecycle'])==stage/'lifecycle.json' and bound(row['post_teardown'])==stage/'post-teardown-audit.json',
        'native owner proof path changed')
    lifecycle=read(stage/'lifecycle.json');after=read(stage/'post-teardown-audit.json')
    require(lifecycle['cleanup_pass'] is True and lifecycle['status']=='PROCESS_EXIT_ONLY_REQUIRES_NATIVE_RECEIPTS'
        and lifecycle['binding']==dict(config_sha256=digest,source_head=cfg['expected_head'],node=node,phase=phase)
        and after['oom_delta']==dict(oom=0,oom_kill=0),'native owner cleanup/lifecycle failed')
    require([bound(value) for value in row['native_rank_sources']]
        ==[p['root']/phase/f'rank_{rank:03d}.json' for rank in range(node*8,node*8+8)],'owner rank source binding changed')
    return ref(path)


def validate_ranks(cfg,phase,node=None):
    from distillschedule.strategies.h3_ar_task import H3ARTask,H3ARImplementation
    from distillschedule.strategies.h3_ar_worker import capacity_ok
    from distillschedule.strategies.h3_ar_inputs import validate_input_receipt
    task=H3ARTask.from_dict(read(bound(cfg['task'])));policy=read(bound(cfg['capacity_policy']))
    admission=gate_admission(cfg,sha(cfg['_config_path']))
    expected_impl=H3ARImplementation(implementation(cfg,phase)).to_dict()
    saved=read(paths(cfg)['plan']) if phase=='selected' else None
    wanted=range(16) if node is None else range(node*8,node*8+8);rows=[]
    for rank in wanted:
        path=paths(cfg)['root']/phase/f'rank_{rank:03d}.json';row=read(path)
        require(row['schema_version']=='distillschedule.manual_h3_ar_metadata_barrier_execution.v1'
            and row['status']=='COMPLETED_ORIGINAL_H3_AR_WORK' and row['gpu_execution']=='NATIVE_CUDA_RUN'
            and row['mode']=='manual'
            and row['rank']==rank and row['task']==task.to_dict() and row['implementation']==expected_impl
            and row['identity']==admission['identity'] and row['capacity_policy']==policy
            and row['runtime']['torch']=='2.10.0+cu128' and row['runtime']['cuda']=='12.8'
            and row['hardware']['hostname']==cfg['nodes'][rank//8]['hostname']
            and row['profiled'] is False and row['finite'] is True,'native rank identity/runtime differs')
        require(row['manual_policy_id']=='uncached-rollout-fa3-varlen-v1'
            and row['automatic_planning_eligible'] is False and row['weights_merged'] is False,
            'manual mechanism binding differs')
        require(row['manual_external_packet']==cfg['manual_packet']
            and row['restore_native_barrier']==(phase=='manual-child')and row['timer_unchanged']is True,
            'external source/arm/timer binding differs')
        for index, cycle in enumerate(row['cycles'],1):
            barrier=cycle['backend_effects']['manual_native_metadata_barrier']
            require(barrier==dict(policy='restore-native-cu-metadata-barrier.v1',enabled=phase=='manual-child',
                cycle=index,actual_native_metadata_reads=350 if phase=='manual-child'else 0,
                contract_failures=0,scope='ONLY_ALREADY_ELIGIBLE_R43_NOGRAD_STUDENT_FA3_CALLS',
                tensor_math_changed=False,original_FA3_backend_unchanged=True,
                original_parent_dispatch_value_reads=0,effective_backend_host_value_reads=350 if phase=='manual-child'else 0),
                'actual native metadata read/unchanged math census not proved')
            effect=cycle['backend_effects'].get('manual_uncached_rollout_attention')
            require(bool(effect),'actual manual attention dispatch missing; both arms require same FA3')
            require('manual_frozen_adaln_bank' not in cycle['backend_effects'], 'rejected bank must not be combined')
            require('leaf_runtime' in cycle and len(cycle['leaf_runtime']['leaf_blocks'])==50
                and all(b['logical_parameter_numel']==657499392 for b in cycle['leaf_runtime']['leaf_blocks']),
                'original native block parameters must not move')
            if effect:
                require(effect['policy_id']=='uncached-rollout-fa3-varlen-v1'
                    and effect['per_cycle_counts']==dict(main_forward_calls=700,application_predict_calls=12,
                        actual_FA3_varlen_calls=350,main_native_fallback=350,refiner_native_fallback=24)
                    and effect['heads']==56 and effect['head_dim']==128
                    and effect['grad_and_selected_cache_paths_unchanged'] is True
                    and effect['params_relocated'] is False and effect['weights_merged'] is False
                    and effect['hot_path_host_value_reads']==0 and len(effect['bounds'])==3
                    and effect['bounds'][0]==0<effect['bounds'][1]<effect['bounds'][2],
                    'actual manual attention/cached-source/shape/version census failed')
        if 'native_dependency' in cfg:
            require(row['runtime']['packages']['accelerate']==cfg['native_dependency']['version'],
                'native rank Accelerate differs from admitted source assembly')
        require(row['initial_versions']==dict(student=0,fake=0,ema=0)
            and row['final_versions']==dict(student=2,fake=2,ema=0) and len(row['cycles'])==2
            and row['build']['native_implementation_installed'] is True
            and row['build']['implementation']==expected_impl,
            'native build/update bounds changed')
        validate_input_receipt(task,rank,row['input_identity'],row['build'])
        for cycle,index in zip(row['cycles'],(1,2)):
            require(cycle['cycle']==index and type(cycle['duration_ns'])is int and cycle['duration_ns']>0
                and cycle['census']==task.census and cycle['updates']==cycle['adam_updates']==dict(student=1,critic=1)
                and cycle['final_versions']==dict(student=index,fake=index,ema=0)
                and cycle['prepared']['status']=='PASS' and cycle['prepared']['executed_operations']==['native_update']
                and cycle['RNG_reseeded_between_cycles'] is False and cycle['state']
                and cycle['capacity_pass'] is True and capacity_ok(cycle['memory'],policy,node_rank=rank//8)
                and cycle['backend_effects']['actual_compute_observed'] is True
                and cycle['backend_effects']['implementation']==expected_impl,'native cycle/capacity/dispatch differs')
            cache=cycle['selected_cache'];require(bool(cache)==H3ARImplementation(implementation(cfg,phase)).optimized,'wrong actual attention-cache branch')
            if cache:
                require(cache['cache_epoch']==index-1 and cache['source_student_version']==index-1
                    and cache['intervening_fake_version']==index and cache['same_process_pinned_pool_reuse'] is True
                    and cache['stale_result_reuse'] is False,'native selected cache lifetime differs')
        require(row['plan_sha256']==(saved['plan_sha256'] if saved else None)
            and row['selected_order']==['native_update'],'saved plan not actually consumed')
        rows.append((path,row))
    return rows


def freeze_assets(cfg,digest):
    source=source_state(cfg);native_manifest(cfg);p=paths(cfg)
    from distillschedule.strategies.h3_ar_assets import H3ARAssets,prepare_from_native_manifest
    from distillschedule.strategies.h3_ar_task import H3ARTask
    from distillschedule.strategies.h3_ar_inputs import validate_preflight
    task=H3ARTask.from_dict(read(bound(cfg['task'])))
    proofs=[];startup={}
    for node in (0,1):
        proof=p['root']/f'reports/prepare-node{node}.json';row=read(proof)
        require(row['status']=='PASS_CPU_PREPARE_NO_CUDA' and row['config_sha256']==digest
            and row['source']==source and row['node']==node and row['cleanup_pass'] is True,'node CPU prepare missing/changed')
        require(bound(row['native_input_preflight'])==p['root']/f'reports/native-inputs-node{node}.json',
            'node-qualified original16 CPU input proof path differs')
        validate_preflight(task,read(bound(row['native_input_preflight'])))
        require(row['input_contract_sha256']==task.input_contract_sha256,'original16 CPU input preflight missing/changed')
        if 'native_dependency' in cfg:
            require(row.get('native_dependency',{}).get('contract')==cfg['native_dependency'],
                'CPU preparation native dependency contract differs')
            metadata=read(bound(row['startup_gpu_info']))
            require(metadata.get('w6_native_dependency',{}).get('contract')==cfg['native_dependency'],
                'startup asset does not bind native dependency source assembly')
            bound(metadata['original_native_metadata'])
        startup[str(node)]=ref(bound(row['startup_gpu_info']));proofs.append(ref(proof))
    if p['admission'].exists(): return gate_admission(cfg,digest)
    # Only this owner CPU admission scans checkpoint payload, never per-rank.
    old=prepare_from_native_manifest(bound(cfg['native_manifest']),checkpoint_audit_path=p['checkpoint_audit'])
    assets=replace(old,startup_gpu_info=startup);validation=assets.validate()
    if p['assets'].exists(): require(read(p['assets'])==assets.to_dict(),'existing admitted assets changed')
    else:write(p['assets'],assets.to_dict())
    result=dict(status='PASS_ASSET_ADMISSION_CPU_ONLY',config_sha256=digest,source=source,
        assets=ref(p['assets']),prepare_proofs=proofs,validation=validation,
        identity=dict(source=source,assets_sha256=assets.digest,task_sha256=task.digest),
        native_manifest=cfg['native_manifest'],historical_cost_used=False,gpu_execution='NOT_RUN')
    if 'native_dependency' in cfg:result['native_dependency']=cfg['native_dependency']
    write(p['admission'],result);return result


def assemble_plan(cfg,digest):
    gate_admission(cfg,digest);p=paths(cfg)
    from distillschedule.native.cli import main as common_cli
    from distillschedule.strategies.h3_ar_planning import validate_plan
    from distillschedule.strategies.h3_ar_task import H3ARTask
    def cli(argv):
        logfile=p['root']/'logs/common-assemble-plan.log';logfile.parent.mkdir(parents=True,exist_ok=True)
        with logfile.open('a') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            print(json.dumps(dict(command=['distillctl','workloads','native',*argv])),flush=True)
            code=common_cli(argv)
        require(code==0,'common native CPU operation failed; preserve '+str(logfile))
    collections=[]
    for phase in PHASES[:2]:
        rows=validate_ranks(cfg,phase)
        for node in (0,1):verified_owner_proof(cfg,phase,node,digest)
        # A report failure cannot authorize another GPU run. Preserve partial
        # CPU output and make at most three fresh CPU-only recovery folders.
        for attempt in range(4):
            suffix='' if attempt==0 else '-cpu-recovery-'+str(attempt)
            folder=p['root']/('assembled-'+phase+suffix)
            if (folder/'collection.json').exists() or not folder.exists():break
        else:raise ValueError('CPU assembly recovery exhausted; preserve evidence, no native retry')
        if not (folder/'collection.json').exists():
            cli(['assemble','--task',cfg['task']['path'],'--rank-receipts',*[str(path) for path,_ in rows],
                '--output',str(folder)])
        collections.append(str(folder/'collection.json'))
    if not p['plan'].exists():
        cli(['plan','--task',cfg['task']['path'],'--calibration',*collections,'--output',str(p['plan'])])
    saved,_=validate_plan(H3ARTask.from_dict(read(bound(cfg['task']))),p['plan'])
    require(saved['candidate_count']==2 and set(r['candidate_id'] for r in saved['ranked'])=={'zero3-native','zero3-ap'},
        'both measured legal arms must be present and feasible')
    result=dict(status='PASS_REAL_COMMON_NATIVE_PLAN',config_sha256=digest,source=source_state(cfg),
        saved_plan=ref(p['plan']),plan_sha256=saved['plan_sha256'],selected=saved['selected']['implementation'],
        collections=[ref(path) for path in collections],historical_cost_used=False,numerical_status=saved['numerical_status'],
        gpu_execution='NOT_RUN_BY_PLANNER')
    if 'native_dependency' in cfg:result['native_dependency']=cfg['native_dependency']
    if p['plan_gate'].exists():require(read(p['plan_gate'])==result,'saved plan gate changed')
    else:write(p['plan_gate'],result)
    return result


def parse_proofs(local,phase,digest):
    proofs=[]
    for node in (0,1):
        stream=Path(local)/f'logs/{phase}-owner-node{node}.log'
        found=[readline.split('=',1)[1] for readline in stream.read_text(errors='replace').splitlines()
            if readline.startswith('H3_AR_OWNER=')]
        require(len(found)==1,'owner proof missing/ambiguous; recover existing files with CPU, never relaunch')
        row=json.loads(found[0]);require(row['node']==node and row['phase']==phase
            and row['config_sha256']==digest and row['cleanup_pass'] is True and row['status']=='PASS_NATIVE_OWNER',
            'owner proof failed/changed')
        proofs.append(row)
    return proofs


def launch(cfg,config_path,phase,digest,remote_config,plan_gate=None,admission_proof=None):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','root coordinator must hide CUDA')
    require(admission_proof is not None,'copy successful CPU asset-admission proof before any native start')
    admitted=read(admission_proof);admitted_sha=sha(admission_proof)
    require(admitted['status']=='PASS_ASSET_ADMISSION_CPU_ONLY' and admitted['config_sha256']==digest
        and admitted['source']['source_head']==cfg['expected_head'],'CPU asset admission source/config differs')
    local=Path(cfg['local_root']);(local/'artifacts').mkdir(parents=True,exist_ok=True);(local/'logs').mkdir(exist_ok=True)
    with (local/'artifacts/controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=local/'artifacts/native-starts.jsonl'
        used=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        eofs={r['phase']:read(local/f'artifacts/eof-{r["phase"]}.json') for r in used
            if (local/f'artifacts/eof-{r["phase"]}.json').exists()}
        check_sequence(cfg,phase,used,eofs)
        require(all(r['config_sha256']==digest for r in used),'campaign config changed after a native start')
        require(all(r['admission_sha256']==admitted_sha for r in used),'CPU asset admission changed after a native start')
        if phase in PHASES[2:]:
            require(plan_gate is not None,'copy the receipt-host CPU plan-gate proof first')
            gate=read(plan_gate);require(gate['status']=='PASS_REAL_COMMON_NATIVE_PLAN' and gate['config_sha256']==digest
                and gate['source']['source_head']==cfg['expected_head'],'actual measured plan gate missing/changed')
        row=dict(phase=phase,attempt=len(used)+1,config_sha256=digest,source_head=cfg['expected_head'],admission_sha256=admitted_sha,
            failure_counts=True,profiler=False,cycles=2,between_cycles_reset=False)
        with ledger.open('a') as stream:
            stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
        children=[];streams=[]
        try:
            for node in (0,1):
                stream=(local/f'logs/{phase}-owner-node{node}.log').open('x');streams.append(stream)
                children.append(subprocess.Popen(node_command(cfg,node,phase,digest,remote_config,admitted_sha),stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT))
            deadline=time.monotonic()+cfg['lease_seconds']+120;next_status=0
            while any(child.poll() is None for child in children):
                require(time.monotonic()<deadline,'OWNER_TERMINAL_UNKNOWN; preserve streams, no auxiliary SSH or retry')
                if time.monotonic()>=next_status:
                    print(json.dumps(dict(stage='LOCAL_STREAMS_ONLY',phase=phase,
                        active=sum(child.poll() is None for child in children))),flush=True);next_status=time.monotonic()+30
                time.sleep(2)
        finally:
            for stream in streams:stream.close()
        codes=[child.returncode for child in children]
        # EOF is persisted even if proof collection fails. Recovery is CPU-only.
        terminal=dict(status='EOF_NOT_YET_NATIVE_PROOF',phase=phase,config_sha256=digest,returncodes=codes)
        write(local/f'artifacts/transport-eof-{phase}.json',terminal)
        require(codes==[0,0],'failed owner/transport; preserve; no automatic native retry')
        proofs=parse_proofs(local,phase,digest)
        terminal.update(status='PASS_CLEAN_NATIVE_PHASE',node_proofs=proofs)
        write(local/f'artifacts/eof-{phase}.json',terminal);return terminal


def collect(cfg,digest):
    gate=read(paths(cfg)['plan_gate']);require(gate['config_sha256']==digest,'plan gate changed')
    arms={phase:validate_ranks(cfg,phase) for phase in PHASES[2:]}
    manual=[r for _,r in arms['manual-ap']];selected=[r for _,r in arms['selected']]
    require(len({r['input_identity']['sha256'] for r in manual})==16,'rank sample inventory differs')
    require(all(a['input_identity']==b['input_identity'] and a['identity']==b['identity']
        and a['runtime']==b['runtime'] and a['hardware']==b['hardware']
        and a['capacity_policy']==b['capacity_policy'] for a,b in zip(manual,selected)), 'manual/selected cohort mismatch')
    proofs=[]
    for phase in PHASES[2:]:
        for node in (0,1):
            proofs.append(verified_owner_proof(cfg,phase,node,digest))
    # Exactly the existing contract's max16 C2 duration, not a component estimate.
    duration={phase:max(row['cycles'][1]['duration_ns'] for _,row in rows) for phase,rows in arms.items()}
    return dict(status='MEASURED_SINGLE_PAIRED_C2_REQUIRES_NUMERICAL_REVIEW',config_sha256=digest,
        source=source_state(cfg),plan_gate=ref(paths(cfg)['plan_gate']),duration_ns_max16_cycle2=duration,
        selected_over_manual=duration['selected']/duration['manual-ap'],
        timing_match_le_1_03=duration['selected']<=duration['manual-ap']*1.03,
        numerical_status='REVIEW_REQUIRES_MATCHED_REFERENCE',quality_validation='NOT_RUN',
        claimable_speedup=False,repetitions=1,profiler=False,owner_proofs=proofs,
        rank_sources={phase:[ref(path) for path,_ in rows] for phase,rows in arms.items()})


def configure(args):
    prior=read(args.owner_config);manifest=read(args.native_manifest);source=Path(args.source).resolve(strict=True)
    nodes=[]
    for spec in prior['nodes']:
        nodes.append(dict(hostname=spec['hostname'],python=executable_ref(spec['python']),assigned_gpu_uuids=spec['assigned_gpu_uuids'],
            ssh_target=spec['host'],ssh_port=spec['ssh_port']))
    candidates=[Path(p) for p,h in manifest['file_sha256'].items() if Path(p).name=='native_gpu_info_cache_cpu_r1.py' and h==GENERATOR_SHA]
    require(len(candidates)==1,'original metadata generator pin missing/ambiguous')
    cfg=dict(schema_version=SCHEMA,source=str(source),expected_head=args.expected_head,
        native_manifest=ref(args.native_manifest),task=ref(source/'configs/distillschedule/original_h3_ar/w6_reference.json'),
        capacity_policy=ref(args.capacity_policy),nodes=nodes,owner_packet=prior['original_owner_packet'],owner_files=OWNER_FILES,
        startup_generator=ref(candidates[0]),scripts=dict(controller=ref(__file__),node=ref(Path(__file__).with_name('h3_ar_match_node.py'))),
        local_root=args.local_root,run_root=args.run_root,cache_root=args.cache_root,
        master_addr=manifest['master_addr'],master_ports={p:args.master_port+i for i,p in enumerate(PHASES)},
        cycles=2,max_native_starts=4,profiler=False,work_seconds=manifest['work_seconds'],lease_seconds=manifest['lease_seconds'])
    source=source_state(cfg);native_manifest(cfg)
    if getattr(args,'native_accelerate_overlay',None):
        cfg['schema_version']=DEPENDENCY_SCHEMA
        cfg['native_dependency']=native_accelerate_contract(args.native_accelerate_overlay,manifest)
        native_dependency_identity(cfg,create=True)
    importlib.import_module('distillschedule.strategies.h3_ar_planning').validate_policy(read(bound(cfg['capacity_policy'])))
    write(args.output,cfg);load_config(args.output)
    return dict(status='CONFIGURED_NOT_GPU_ADMITTED',config=ref(args.output),head=args.expected_head,source=source)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('intent','launch','freeze-assets','assemble-plan','collect','configure','recover-eof'),default='intent')
    parser.add_argument('--config',type=Path);parser.add_argument('--phase',choices=PHASES)
    parser.add_argument('--remote-config');parser.add_argument('--plan-gate',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--admission-proof',type=Path,help='byte-for-byte local copy of receipt-host assets-admission.json')
    parser.add_argument('--node-proofs',type=Path,nargs=2,help='CPU-only recovery of existing durable owner proofs after transport EOF')
    parser.add_argument('--execute',action='store_true')
    for name in ('source','expected-head','native-manifest','owner-config','capacity-policy','local-root','run-root','cache-root','native-accelerate-overlay'):
        parser.add_argument('--'+name)
    parser.add_argument('--master-port',type=int)
    args=parser.parse_args(argv)
    try:
        if args.stage=='configure':
            require(args.execute and args.output is not None,'configure requires explicit CPU --execute/output')
            result=configure(args)
        else:
            require(args.config is not None,'--config required');cfg=load_config(args.config);digest=sha(args.config)
            cfg['_config_path']=str(args.config.resolve())
            if not args.execute or args.stage=='intent':
                require(args.phase is not None,'--phase required for intent');result=intent(cfg,args.phase,digest,args.remote_config or str(args.config))
            elif args.stage=='launch':
                require(args.phase is not None and args.remote_config,'explicit phase/remote config required')
                result=launch(cfg,args.config,args.phase,digest,args.remote_config,args.plan_gate,args.admission_proof)
            elif args.stage=='recover-eof':
                require(args.phase is not None,'phase required');local=Path(cfg['local_root'])
                terminal=read(local/f'artifacts/transport-eof-{args.phase}.json')
                require(terminal['config_sha256']==digest,'existing transport EOF binding required')
                if args.node_proofs:
                    proofs=[read(path) for path in args.node_proofs]
                    require(all(row['node']==node and row['phase']==args.phase and row['config_sha256']==digest
                        and row['status']=='PASS_NATIVE_OWNER' and row['cleanup_pass'] is True
                        for node,row in enumerate(proofs)),'existing durable native proofs differ')
                    terminal['recovery_proof_sources']=[ref(path) for path in args.node_proofs]
                else:
                    require(terminal['returncodes']==[0,0],'interrupted transport needs explicit durable proof recovery')
                    proofs=parse_proofs(local,args.phase,digest)
                terminal.update(status='PASS_CLEAN_NATIVE_PHASE',node_proofs=proofs,
                    recovery_scope='CPU_EXISTING_PROOFS_ONLY_ORIGINAL_TRANSPORT_RETURNCODES_PRESERVED')
                write(local/f'artifacts/eof-{args.phase}.json',terminal);result=terminal
            else:
                require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU receipt-host coordinator must hide CUDA')
                source_state(cfg);native_manifest(cfg)
                result=freeze_assets(cfg,digest) if args.stage=='freeze-assets' else assemble_plan(cfg,digest) if args.stage=='assemble-plan' else collect(cfg,digest)
                if args.output:write(args.output,result)
        print('H3_AR_CONTROLLER='+json.dumps(result,sort_keys=True,allow_nan=False),flush=True);return 0
    except (ValueError,KeyError,TypeError,OSError,subprocess.SubprocessError) as error:
        print('ERROR: '+str(error),file=sys.stderr,flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
