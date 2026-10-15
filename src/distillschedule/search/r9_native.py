"""CPU prepare/qualify/seal/collect beside assets for the opt-in R9 transport.

The reviewed external owner alone starts and owns GPU children. Receipts below
are explicitly R9 evidence, never a synthetic legacy launch_manifest/teardown.
"""
import copy
import contextlib
import json
from pathlib import Path
import subprocess
import sys

from .contracts import require
from .r9_contract import (SCHEMA, KIND, OWNER_SHA256, WORKER_WRAPPER, CONTRACT_WRAPPER,
                          bound, read, validate_plan, validate_config, validate_rank)
from distillschedule.workloads.best_known.registry import (file_sha256, fingerprint,
    repository_root, write_new_json)


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=file_sha256(Path(path)))


def validate_deployment(value, *, local_files=False):
    require(isinstance(value, dict) and set(value) ==
            {'schema_version', 'owner', 'template', 'dependency_provenance'}, 'R9 explicit deployment fields')
    require(value['schema_version'] == SCHEMA and value['owner']['sha256'] == OWNER_SHA256,
            'R9 reviewed external owner required')
    for key in ('owner', 'template', 'dependency_provenance'):
        ref = value[key]
        require(set(ref) == {'path', 'sha256'} and Path(ref['path']).is_absolute()
                and len(ref['sha256']) == 64 and all(c in '0123456789abcdef' for c in ref['sha256']),
                'R9 absolute SHA-bound deployment input')
        if local_files: bound(ref)
    return value


def rebind(plan, *, run_id, assets):
    """CPU-only replay lowering beside the complete raw cost evidence."""
    from .native import handle_request
    validate_plan(plan)
    require(run_id != plan['run_id'] and run_id != 'source-inspection', 'a fresh explicit run-id is required')
    require(assets == plan['asset_config'], 'R9 replay assets/gates changed')
    result = handle_request(dict(action='resolve', source_head=plan['integration_head'],
        workload=plan['profile_id'], candidate_id=plan['search_candidate']['id'],
        assets=assets, run_id=run_id, mode=plan['mode']))
    for key in ('model_plan', 'execution_transport'):
        if key in plan:
            result[key] = copy.deepcopy(plan[key])
    validate_plan(result)
    return result


def prepare(plan, deployment):
    from distillschedule.workloads.best_known.dispatch import validate_local
    validate_plan(plan)
    validate_local(plan, root=repository_root(), node_rank=0)
    validate_deployment(deployment, local_files=True)
    require(plan.get('execution_transport') == dict(schema_version=SCHEMA,
            deployment_sha256=fingerprint(deployment)), 'R9 plan transport identity')
    donor = read(deployment['template'])
    # A deployment template supplies audited assignment/safety/assets and
    # expected import versions, not correctness or timing authority.
    require(donor['owned_worker_oom_fail_fast'] is True and donor['verified_zombie_fd_rescan'] is True
            and donor['cycles'] == 2 and donor['profile'] is False, 'R9 template safety scope')
    run = Path(plan['asset_config']['run_root']) / plan['run_id']
    run.mkdir(parents=True, exist_ok=False)
    packet = run / 'transport/r9'
    (packet / 'scripts').mkdir(parents=True)
    (packet / 'scripts/selected_plan_owner.py').write_bytes(bound(deployment['owner']).read_bytes())
    (packet / 'scripts/selected_plan_worker.py').write_text(WORKER_WRAPPER)
    (packet / 'scripts/selected_plan_contract.py').write_text(CONTRACT_WRAPPER)
    selected = packet / 'selected-plan.json'
    write_new_json(selected, plan)
    final = packet / 'sealed-manifest.json'
    source = repository_root()
    cfg = copy.deepcopy(donor)
    # Historical packet metadata cannot describe a newly selected CPU arm.
    # Keep audited safety/assignment pins, not stale experiment authority.
    for key in ('cache_policy_scope', 'frozen_model_config', 'prefetch', 'comparison_scope',
                'nsys', 'source_provenance', 'selected_candidate', 'claim', 'assets_identity',
                'exact_cache_layout_fix', 'assignment_recovery', 'short_tmp_reason', 'tmp_path_recovery'):
        cfg.pop(key, None)
    cfg.update(kind=KIND, source_head=plan['integration_head'], source_root=str(source),
        run_id=plan['run_id'], run_root=str(run), candidate_id=plan['search_candidate']['id'],
        python=plan['nodes'][0]['python'], selected_plan=binding(selected),
        worker_script=str(packet / 'scripts/selected_plan_worker.py'), worker_flags=[],
        owner_sha256=OWNER_SHA256, packet_root=str(packet), remote_manifest=str(final),
        owner_pythonpath=':'.join((str(packet / 'scripts'), str(source / 'src'), str(source))),
        dependency_provenance=deployment['dependency_provenance'], deployment=deployment,
        port=plan['asset_config']['master_port'], master_addr=plan['asset_config']['master_addr'],
        transport_nodes=plan['asset_config']['nodes'], profile=False,
        selection='MEASURED_MODEL_SELECTED' if 'model_plan' in plan else 'REGISTERED_CANDIDATE')
    from .h3_implementation import H3Implementation
    impl = H3Implementation.parse(cfg['candidate_id'])
    cfg.update(cache_storage=impl.storage, resolution=impl.request.to_dict(),
        config_by_node=[n['environment']['H3_CONFIG'] for n in plan['nodes']],
        comparison_scope='MATCHED_CURRENT_HEAD_CPU_STAGED_TRIM_VS_CUDA_NO_TRIM; NO_HISTORICAL_COST_AUTHORITY')
    cfg['python_sha256'] = file_sha256(Path(cfg['python']))
    require(set(cfg['expected_runtime']) == {'torch', 'cuda', 'diffusers', 'peft', 'transformers'},
            'R9 explicit qualified runtime versions required in deployment template')
    cfg['environment_by_node'] = []
    for node, selected_node in enumerate(plan['nodes']):
        write_new_json(Path(selected_node['environment']['H3_CONFIG']), selected_node['resolved_model_config'])
        cfg['environment_by_node'].append(dict(selected_node['environment'],
            EXPECTED_INTEGRATION_HEAD=cfg['source_head'], DMD_SELECTED_PLAN_MANIFEST=str(final),
            TMPDIR=f'/tmp/distillschedule-r9-{fingerprint(plan["run_id"])[:16]}/n{node}'))
    cfg['environment'] = cfg['environment_by_node'][0]
    old_source, old_packet = Path(donor['source_root']), Path(donor['packet_root'])
    pins = {}
    for name, digest in donor['file_sha256'].items():
        path = Path(name)
        if path.is_relative_to(old_source) or path.is_relative_to(old_packet): continue
        require(file_sha256(path) == digest, 'R9 deployment safety/asset pin changed: ' + name)
        pins[name] = digest
    for path in sorted((source / 'src/distillschedule').rglob('*.py')):
        pins[str(path)] = file_sha256(path)
    for path in sorted((packet / 'scripts').glob('*.py')):
        pins[str(path)] = file_sha256(path)
    for path in [selected, bound(deployment['dependency_provenance']),
                 *[Path(n['environment']['H3_CONFIG']) for n in plan['nodes']]]:
        pins[str(path)] = file_sha256(path)
    cfg['file_sha256'] = pins
    cfg['qualification'] = {f'source_cpu_{n}': dict(path=str(packet / f'source_cpu_node{n}.json'),
        sha256='UNQUALIFIED_DO_NOT_EXECUTE') for n in (0, 1)}
    validate_config(cfg)
    draft = packet / 'draft-manifest.json'
    write_new_json(draft, cfg)
    return dict(status='R9_PREPARED_NOT_EXECUTABLE', draft=binding(draft))


def qualify(draft, node):
    cfg = read(draft)
    validate_config(cfg)
    for path, digest in cfg['file_sha256'].items():
        require(file_sha256(Path(path)) == digest, 'R9 qualification source pin')
    env = dict(cfg['environment_by_node'][node], CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([cfg['python'], cfg['worker_script'], '--validate-only',
        '--manifest', draft['path'], '--manifest-sha256', draft['sha256'], '--node', str(node)],
        cwd=cfg['source_root'], env=env, capture_output=True, text=True, timeout=240)
    log = Path(cfg['packet_root']) / f'source_cpu_node{node}.log'
    with log.open('x') as stream: stream.write(proc.stdout + proc.stderr)
    require(proc.returncode == 0, 'R9 fresh CPU import/toolchain/plan gate failed: ' + str(log))
    rows = [json.loads(line) for line in proc.stdout.splitlines()
            if line.startswith('{') and 'PASS_H3_SELECTED_PLAN_CPU_NO_GPU' in line]
    require(len(rows) == 1 and rows[0]['node'] == node and rows[0]['CUDA_initialized'] is False,
            'R9 fresh CPU qualification missing')
    row = dict(rows[0], log=binding(log), qualified_files_sha256=cfg['file_sha256'])
    path = Path(cfg['qualification'][f'source_cpu_{node}']['path'])
    write_new_json(path, row)
    return binding(path)


def seal(draft):
    from .r9_dependencies import validate_toolchain_receipt
    cfg = read(draft)
    validate_config(cfg)
    for node in (0, 1):
        ref = cfg['qualification'][f'source_cpu_{node}']
        row = json.loads(Path(ref['path']).read_text())
        require(row['status'] == 'PASS_H3_SELECTED_PLAN_CPU_NO_GPU' and row['node'] == node
                and row['source_head'] == cfg['source_head'] and row['candidate_id'] == cfg['candidate_id']
                and row['selected_plan_sha256'] == cfg['selected_plan']['sha256']
                and row['qualified_files_sha256'] == cfg['file_sha256'], 'R9 CPU qualification binding')
        validate_toolchain_receipt(row['toolchain'])
        ref['sha256'] = file_sha256(Path(ref['path']))
    final = Path(cfg['remote_manifest'])
    write_new_json(final, cfg)
    return dict(manifest=binding(final), configuration=cfg)


def owner_evidence(cfg, manifest):
    """Validate actual R9 argv/environment/terminal safety, not old shell flags."""
    validate_config(cfg)
    require(binding(Path(cfg['remote_manifest'])) == manifest, 'R9 sealed manifest changed')
    for name, digest in cfg['file_sha256'].items():
        require(file_sha256(Path(name)) == digest, 'R9 post-execution pinned file changed')
    run = Path(cfg['run_root'])
    sources = [manifest, cfg['selected_plan']]
    for node in (0, 1):
        path = run / f'node_{node}/lifecycle.json'
        row = json.loads(path.read_text())
        require(row['status'] == 'PASS_LOCAL_OWNED_RUN_PENDING_WORLD_GATE'
                and row['cleanup_pass'] is True and row['lock_held'] is True
                and row['node'] == node and row['run_id'] == cfg['run_id']
                and row['source_head'] == cfg['source_head'] and row['manifest_sha256'] == manifest['sha256']
                and row['oom_delta'] == dict(oom=0, oom_kill=0), 'R9 owner terminal gate')
        require(row['terminal_FD_audit']['handles'] == [] and row['initial_FD_audit']['handles'] == []
                and 0 < row['held_seconds'] <= cfg['lease_seconds'], 'R9 FD/lease gate')
        expected = [cfg['python'], '-m', 'torch.distributed.run', '--nnodes=2', '--nproc-per-node=8',
            '--max-restarts=0', f'--node-rank={node}', '--master-addr=' + cfg['master_addr'],
            '--master-port=' + str(cfg['port']), cfg['worker_script']]
        require(len(row['phases']) == 1 and row['phases'][0]['argv'] == expected
                and row['phases'][0]['worker_rc'] == 0, 'R9 actual unprofiled worker argv/exit')
        expected_env = dict(cfg['environment_by_node'][node],
            DISTILLSCHEDULE_EXPECTED_SOURCE_RECEIPT=cfg['source_head'],
            DMD_SELECTED_PLAN_MANIFEST_SHA256=manifest['sha256'],
            CUDA_VISIBLE_DEVICES=','.join(cfg['ordered_UUIDs_by_node'][node]))
        require(all(row['environment'].get(k) == v for k, v in expected_env.items()), 'R9 actual owner environment')
        require(not (set(row['environment']) - set(expected_env) -
                     {'HOME', 'USER', 'LOGNAME', 'LANG', 'NVIDIA_VISIBLE_DEVICES'}),
                'R9 undeclared actual owner environment')
        sources.append(binding(path))
    return sources


def _checked_run(manifest):
    from .h3_native import read_native_receipts, summarize_native_two_cycle
    cfg = read(manifest)
    plan = validate_config(cfg)
    sources = owner_evidence(cfg, manifest)
    root = Path(cfg['run_root'])
    rows, native = read_native_receipts(root)
    summary = summarize_native_two_cycle(rows, expected_head=cfg['source_head'], candidate_id=cfg['candidate_id'])
    for raw in rows:
        validate_rank(cfg, raw)
        path = root / 'selected-plan-binding' / f'rank_{raw["rank"]:03d}.json'
        row = json.loads(path.read_text())
        require(row['rank'] == raw['rank'] and row['source_head'] == cfg['source_head']
                and row['selected_plan'] == cfg['selected_plan'] and row['candidate_id'] == cfg['candidate_id']
                and row['manifest_sha256'] == manifest['sha256'] and row['mode'] == plan['mode']
                and row['selection'] == cfg['selection'] and row['entrypoint'] == 'ORIGINAL_INTEGRATED_ENTRYPOINT',
                'R9 actual selected rank binding')
        sources.append(binding(path))
    for node in (0, 1):
        resource = root / f'search-resources.node{node}.json'
        row = json.loads(resource.read_text())
        require(row['status'] == 'PASS' and row['run_id'] == cfg['run_id'] and row['node_rank'] == node
                and len(row['samples']) >= 2, 'R9 actual resource observer')
        for sample in row['samples']:
            require([g['uuid'] for g in sorted(sample['gpus'], key=lambda g:g['device'])] ==
                    cfg['ordered_UUIDs_by_node'][node], 'R9 resource UUID assignment changed')
        for event in ('oom', 'oom_kill'):
            counts = [s['cgroup_events'][event] for s in row['samples']]
            require(max(counts) == min(counts), 'R9 resource OOM increased')
        sources.append(binding(resource))
    result = dict(schema_version='distillschedule.r9_execution.v1', status='PASS_NATIVE_R9_EXECUTION',
        source_head=cfg['source_head'], run_id=cfg['run_id'], mode=plan['mode'], selection=cfg['selection'],
        plan_sha256=cfg['selected_plan']['sha256'], manifest=manifest, native_summary=summary,
        evidence=sources + native, profiled=False, actual_entrypoint='ORIGINAL_INTEGRATED_ENTRYPOINT',
        measured_search_closed=False, formal_timing_eligible=False, claimable_speedup=False)
    return cfg, plan, result


def collect(manifest):
    from distillschedule.workloads.best_known.evidence import collect as collect_native
    cfg, plan, result = _checked_run(manifest)
    path = Path(cfg['run_root']) / 'r9-execution.json'
    write_new_json(path, result)
    with contextlib.redirect_stdout(sys.stderr):
        require(collect_native(plan) == 0, 'R9 native bound summary failed')
    return dict(status=result['status'], execution_evidence=binding(path), selection=cfg['selection'])


def verify_execution(plan):
    """Recheck all persisted actual R9 evidence, fail closed without fallback."""
    validate_plan(plan)
    require(plan.get('execution_transport', {}).get('schema_version') == SCHEMA, 'R9 explicit transport marker')
    root = Path(plan['asset_config']['run_root']) / plan['run_id']
    path = root / 'r9-execution.json'
    row = json.loads(path.read_text())
    cfg = read(row['manifest'])
    require(read(cfg['selected_plan']) == plan and row['plan_sha256'] == cfg['selected_plan']['sha256'],
            'R9 executed another serialized plan')
    require(row['schema_version'] == 'distillschedule.r9_execution.v1' and row['status'] == 'PASS_NATIVE_R9_EXECUTION'
            and row['source_head'] == plan['integration_head'] and row['mode'] == plan['mode']
            and row['run_id'] == plan['run_id'] and row['profiled'] is False
            and row['selection'] == cfg['selection'], 'R9 execution receipt binding')
    _, _, actual = _checked_run(row['manifest'])
    require(row == actual, 'R9 evidence differs from actual owner/rank/resource receipts')
    for ref in row['evidence']: bound(ref)
    return [binding(path), *row['evidence']]
