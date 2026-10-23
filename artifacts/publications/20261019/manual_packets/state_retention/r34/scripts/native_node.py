"""Bounded node adapter. Reuse the original owner/FD/cancel code unchanged.

New experiment, not a relaxation or rewrite of the frozen R3 window-one packet.
All full runs require actual process-cumulative allocated peak within total-2GiB.
Manual is a direct native reference; selected must consume the real saved plan.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import socket
import subprocess
import sys
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--sha256', required=True)
    p.add_argument('--node', type=int, choices=(0, 1), required=True)
    p.add_argument('--phase', choices=('manual',), required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    require(args.execute, 'explicit execution required')
    require(sha(args.config) == args.sha256, 'config hash changed')
    cfg = json.loads(args.config.read_text())
    require(cfg['schema_version'] == 'dmd.wan_exact_ema_graph_manual_pair.v1', 'wrong experiment schema')
    require(cfg['GPU_reserve_bytes'] == 2**31 and cfg['host_reserve_bytes'] == 64 * 2**30,
            'frozen common reserves changed')
    require(cfg['max_native_starts'] == 2 and cfg['cycles'] == 2, 'bounded experiment changed')
    require(sha(__file__) == cfg['adapter_sha256'], 'node adapter changed')
    parent = Path(cfg['original_owner_packet'])
    for rel, digest in cfg['owner_files'].items():
        require(sha(parent / rel) == digest, 'original owner bytes changed: ' + rel)
    sys.path.insert(0, str(parent / 'scripts'))
    owner = importlib.import_module('node_stage')
    fd_module = importlib.import_module('fd_ownership')
    common = importlib.import_module('packet_common')
    require(Path(owner.__file__).resolve() == parent / 'scripts/node_stage.py', 'mixed owner import')
    source = Path(cfg['source'])
    sys.path.insert(0, str(source / 'src'))
    from distillschedule.strategies.wan_strategy import WanStrategy
    from distillschedule.strategies.wan_evidence import validate_build
    spec = cfg['nodes'][args.node]
    require(socket.gethostname() == spec['hostname'], 'assigned host changed')
    require(subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == cfg['HEAD']
            and not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip(),
            'source changed or dirty')
    root = Path(cfg['artifact_root'])
    require(args.phase == 'manual', 'manual-only experiment; planner intentionally not used')
    from distillschedule.runtime.wan13b_fa3_manual_experiment import validate_packet
    exp = common.read(cfg['experiment_packet'])
    require(sha(cfg['experiment_packet']) == cfg['experiment_packet_sha256'], 'experiment packet changed')
    validate_packet(exp, head=cfg['HEAD'], task=cfg['task'], assets=cfg['assets'],
                    implementation=cfg['strategies']['manual'])
    command = [spec['python'], '-m', 'torch.distributed.run', '--nnodes=2', '--nproc-per-node=8',
               '--max-restarts=0', f'--node-rank={args.node}', '--master-addr=' + cfg['master_addr'],
               '--master-port=' + str(cfg['master_port']), cfg['manual_entry'],
               '--ema-packet', cfg['ema_packet'], '--ema-packet-sha256', cfg['ema_packet_sha256'],
               '--stream-packet', cfg['stream_packet'], '--stream-packet-sha256', cfg['stream_packet_sha256'],
               '--task', cfg['task'], '--assets', cfg['assets'], '--implementation', cfg['strategies']['manual'],
               '--output', str(root / 'manual'), '--packet', cfg['experiment_packet'],
               '--packet-sha256', cfg['experiment_packet_sha256'], '--execute']
    env = {k: os.environ[k] for k in ('HOME', 'USER', 'LOGNAME', 'LANG') if k in os.environ}
    pyroot = str(Path(spec['python']).parents[1])
    env.update(PATH=pyroot + '/bin:/usr/local/bin:/usr/bin:/bin', LD_LIBRARY_PATH=pyroot + '/lib',
               PYTHONPATH=str(source / 'src') + ':' + str(source) + ':' + str(source / 'third_party/helios_compat/source'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', HF_HOME=cfg['hf_home'], HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8',
               CUDA_VISIBLE_DEVICES=','.join(spec['assigned_gpu_uuids']), TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='1',
               TRITON_CACHE_DIR=cfg['cache_root'] + f'/node{args.node}/triton',
               TMPDIR=cfg['cache_root'] + f'/node{args.node}/tmp')
    for key in ('TMPDIR', 'TRITON_CACHE_DIR'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    stage = root / 'lifecycle' / args.phase / f'node{args.node}'
    coordination = root / 'coordination' / args.phase
    coordination.mkdir(parents=True, exist_ok=True)
    before = {}; fd = []; audits = []; next_audit = [0.]; context = {}

    def preflight():
        require(sha(args.config) == args.sha256, 'config changed under lock')
        require(subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == cfg['HEAD']
                and not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip(),
                'source changed under lock')
        for path, digest in cfg['input_metadata_sha256'].items():
            require(sha(path) == digest, 'input metadata changed: ' + path)
        gpu = owner.gpu_audit(spec['assigned_gpu_uuids'], require_idle=True)
        fd.append(fd_module.FDOwnership(parent, 'PINNED_VERIFIED_SSH_ANCESTOR'))
        observed = fd[0].audit()
        before.update(owner.cgroup_events())
        used = int(Path('/sys/fs/cgroup/memory.current').read_text())
        maximum = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        require(maximum.isdigit() and int(maximum) - used >= cfg['host_reserve_bytes'], 'host reserve unavailable')
        if args.node == 0:
            sock = socket.socket()
            try:
                sock.bind(('0.0.0.0', cfg['master_port']))
            finally:
                sock.close()
        common.write_new(coordination / f'ready{args.node}.json', dict(config_sha256=args.sha256, node=args.node,
                         phase=args.phase, lock_held=True, owner_birth=owner.birth(os.getpid())))
        peer = coordination / f'ready{1-args.node}.json'; deadline = time.monotonic() + 90
        while not peer.exists():
            require(not (coordination / f'abort{1-args.node}.json').exists(), 'peer aborted')
            require(time.monotonic() < deadline, 'paired held-lock readiness timeout')
            time.sleep(.25)
        row = common.read(peer)
        require(row['config_sha256'] == args.sha256 and row['phase'] == args.phase and row['lock_held'],
                'peer config/phase/lock changed')
        return dict(gpu=gpu, fd=observed, cgroup_events=dict(before), host_reserve_bytes=cfg['host_reserve_bytes'])

    def heartbeat(known):
        require(not (coordination / f'abort{1-args.node}.json').exists(), 'peer abort')
        if time.monotonic() >= next_audit[0]:
            now = owner.cgroup_events()
            require(all(now[k] == before[k] for k in ('oom', 'oom_kill')), 'cgroup OOM increment')
            gpu = owner.gpu_audit(spec['assigned_gpu_uuids'], require_idle=False)
            observed = fd[0].audit(known, refresh=lambda: owner.discover(owner.birth(os.getpid()), known, owner_pid=os.getpid()),
                                  owner_context=context, fresh_gpu=lambda: owner.gpu_audit(spec['assigned_gpu_uuids'], require_idle=False),
                                  gpu_observation=gpu)
            audits.append(dict(monotonic=time.monotonic(), gpu=observed.get('fresh_gpu_observation', gpu), fd=observed))
            require(len(audits) <= 360, 'bounded audit inventory exceeded')
            next_audit[0] = time.monotonic() + 5

    def finalizer():
        if not before:
            return dict(status='PREFLIGHT_REJECTED_NO_NATIVE_ADMISSION')
        gpu = owner.gpu_audit(spec['assigned_gpu_uuids'], require_idle=True)
        require(all(owner.cgroup_events()[k] == before[k] for k in ('oom', 'oom_kill')), 'cgroup OOM at teardown')
        cleanup = dict(gpu=gpu, fd=fd[0].audit(), OOM_increment=0, audits=audits)
        common.write_new(stage / 'post-teardown-audit.json', cleanup)
        outputs = []
        for rank in range(8 * args.node, 8 * args.node + 8):
            path = root / args.phase / f'rank_{rank:02d}.json'; row = common.read(path)
            expected_mode = 'calibrate' if args.phase.startswith('calibrate-') else 'manual' if args.phase == 'manual' else 'run'
            require(row['status'] == 'COMPLETED_ORIGINAL_WAN_WORK' and row['mode'] == expected_mode and row['rank'] == rank,
                    'incomplete/mislabelled native rank')
            require(row['identity']['source_head'] == cfg['HEAD'] and row['task'] == common.read(cfg['task'])
                    and row['finite'] and len(row['cycles']) == 2
                    and row['initial_versions'] == dict(generator=2, critic=10, ema=10)
                    and row['final_versions'] == dict(generator=4, critic=20, ema=20)
                    and all(c['finite'] and c['call_census']['status'] == 'PASS_TRACE_DERIVED_CALL_CENSUS' for c in row['cycles']),
                    'native source/work/census/version/finite failed')
            memory = row['gpu_memory']
            require(memory['scope'].startswith('PYTORCH_HIGHWATER_ACROSS_OBSERVED_NATIVE_PYTHON_RESETS')
                    and memory['peak_allocated_bytes'] <= memory['total_bytes'] - cfg['GPU_reserve_bytes'],
                    'actual cold+warm cumulative allocated capacity exceeds common budget')
            validate_build(WanStrategy.from_dict(row['strategy']), row['build'], executed=True, cycles=2)
            runtime = row['runtime']
            require(runtime['torch'] == '2.10.0+cu128' and runtime['cuda'] == '12.8'
                    and runtime['tf32_matmul'] and runtime['tf32_cudnn']
                    and runtime['tf32_cublas_override'] == '1' and runtime['nvidia_tf32_override'] == 'NOT_SET'
                    and row['hardware']['host'] == spec['hostname'] and row['hardware']['local_rank'] == rank % 8,
                    'native runtime/precision/device mismatch')
            if args.phase == 'manual':
                require(row['plan_sha256'] is None and row['search_performed'] is False
                        and row['execution_kind'] == 'MANUAL_REFERENCE_DIRECT_NATIVE_ORDER'
                        and row['strategy'] == common.read(cfg['strategies']['manual']), 'manual reference changed')
            elif args.phase == 'selected':
                document = common.read(root / 'selected-plan.json')
                require(row['plan_sha256'] == document['plan_sha256'] and row['identity'] == document['identity']
                        and row['strategy'] == document['selected']['strategy'], 'saved plan not actually executed')
            evidence = row['build']['critic']['metadata']['self_attention_backend_experiment']
            require(evidence['packet_sha256'] == cfg['experiment_packet_sha256']
                    and evidence['source_head'] == cfg['HEAD'] and evidence['arm'] == exp['arm']
                    and evidence['enabled'] is (exp['arm'] == 'fa3-large-v1')
                    and evidence['kernel_files'] == exp['kernel_files']
                    and evidence['weights_merged'] is False and evidence['dtype_unchanged'] is True
                    and evidence['planner'] is False, 'actual backend receipt drift')
            if exp['arm'] == 'fa3-large-v1':
                require(evidence['fa3_calls'] > 0 and evidence['fa3_grad_calls'] > 0
                        and evidence['fa3_nograd_calls'] > 0 and evidence['native_fallback_calls'] > 0
                        and set(evidence['fa3_lengths']) <= {'2160','8640','11040'},
                        'planned large-shape backend not actually called or small/cross path changed')
            else:
                require(evidence['fa3_calls'] == evidence['fa3_grad_calls'] == evidence['fa3_nograd_calls'] == 0,
                        'control entered unrequested backend')
            stream = row['build']['critic']['metadata']['post_g_stream_overlap_experiment']
            require(stream['packet_sha256'] == cfg['stream_packet_sha256']
                    and stream['entry_sha256'] == cfg['stream_execution_sha256']
                    and stream['enabled'] is (cfg['arm'] == 'overlap')
                    and stream['contract_failures'] == 0 and stream['planner'] is False
                    and stream['weights_merged'] is False and stream['precision_changed'] is False
                    and stream['host_action_order_changed'] is (cfg['arm'] == 'overlap') and stream['cold_cycle_original_stream'] is True,
                    'actual stream execution source/arm/semantics drift')
            if cfg['arm'] == 'overlap':
                require(stream['streams_created'] == stream['G_commit_stream_fences'] == stream['completed_overlap_cycles'] == 1
                        and stream['rollout_ready_events'] == stream['critic_event_waits'] == 5
                        and stream['consumer_storage_records'] > 0
                        and stream['graph_cache_entries_before'] == stream['graph_cache_entries_after']
                        and stream['host_launch_order'] == ['g','r0','r1','c0','e0','r2','c1','e1','r3','c2','e2','r4','c3','e3','c4','e4'],
                        'stream dependency/lifetime/warm-graph execution census failed')
            else:
                require(all(stream[k] == 0 for k in ('streams_created','G_commit_stream_fences','completed_overlap_cycles',
                        'rollout_ready_events','critic_event_waits','consumer_storage_records')),
                        'single-stream control unexpectedly overlapped')
            ema = row['build']['critic']['metadata']['exact_ema_graph_experiment']
            require(ema['enabled'] is (cfg['ema_arm'] == 'graph')
                    and ema['source_sha256'] == cfg['ema_source_sha256']
                    and ema['packet_sha256'] == cfg['ema_packet_sha256']
                    and ema['contract_failures'] == 0 and ema['planner'] is False
                    and ema['weights_merged'] is False and ema['optimizer_changed'] is False
                    and ema['math_changed'] is False and ema['observed_steps'] == list(range(11,21)),
                    'actual exact EMA source/arm/steps/semantics drift')
            require(ema['graph_replays'] == (10 if cfg['ema_arm'] == 'graph' else 0)
                    and ema['reference_steps'] == (0 if cfg['ema_arm'] == 'graph' else 10)
                    and ema['captured_graphs'] == (10 if cfg['ema_arm'] == 'graph' else 0),
                    'wrong real EMA implementation/count executed')
            outputs.append(dict(path=str(path), sha256=sha(path), memory=memory))
        return dict(**cleanup, native_outputs=outputs)

    try:
        result = owner.run_owned(source, command, env, stage, timeout=900, preflight=preflight,
                                 heartbeat=heartbeat, finalizer=finalizer, fd_context=context,
                                 binding=dict(source_head=cfg['HEAD'], config_sha256=args.sha256,
                                              run_id=cfg['run_id'], node=args.node, phase=args.phase))
        print('MATCH_OWNER=' + json.dumps(dict(node=args.node, phase=args.phase, status=result['status'],
              cleanup_pass=result['cleanup_pass'], lifecycle=str(stage / 'lifecycle.json'), lifecycle_sha256=sha(stage / 'lifecycle.json'))), flush=True)
    except BaseException as error:
        abort = coordination / f'abort{args.node}.json'
        if not abort.exists():
            common.write_new(abort, dict(config_sha256=args.sha256, error=repr(error)))
        raise


if __name__ == '__main__':
    main()
