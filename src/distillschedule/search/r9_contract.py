"""Narrow H3 CPU/CUDA-cache contract for the reviewed external R9 owner.

This module does not grant GPU ownership. The owner independently retains its
lock, assigned UUID, FD/birth, permit, reserve, OOM, peer and cleanup checks.
"""
from pathlib import Path
import hashlib
import json
import os
import socket

from .contracts import require
from .execution import validated_profile
from .h3_implementation import H3, H3Implementation, validate_cache_registration
from .h3_request import H3Request
from distillschedule.workloads.best_known.registry import (
    file_sha256, git_head, repository_root, resolve_configuration,
)

SCHEMA = 'distillschedule.r9_transport.v1'
KIND = 'native_h3_serialized_plan_v1'
OWNER_SHA256 = 'f97a3dca1c777e2ee4f925e99ad7b0626493336476f5bcd92a629297bfeced92'
MODES = {'correctness', 'calibration', 'diagnostic-timing'}
WORKER_WRAPPER = 'from distillschedule.search.r9_worker import main\nraise SystemExit(main())\n'
CONTRACT_WRAPPER = ('from distillschedule.search.r9_contract import '
                    'validate_owner_config, validate_rank\n')


def bound(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'R9 exact file binding required')
    path = Path(ref['path'])
    require(path.is_absolute() and path.is_file() and file_sha256(path) == ref['sha256'],
            'R9 bound file changed: ' + str(path))
    return path


def read(ref):
    value = json.loads(bound(ref).read_text())
    require(isinstance(value, dict), 'R9 JSON object required')
    return value


def _validate_scope(plan):
    require(plan['integration_head'] == git_head(repository_root()), 'R9 current clean source required')
    require(plan['profile_id'] == H3 and plan['mode'] in MODES
            and plan['native_mode'] == plan['mode'], 'R9 only explicit H3 native modes supported')
    impl = H3Implementation.parse(plan['search_candidate']['id'])
    require(impl.request == H3Request(height=480, width=832) and impl.rollout_window is None
            and impl.prefetch_depth == 2 and impl.lora_epilogue_block_n == 128
            and impl.cache_policy in {'cpu_staged_trim', 'cuda_no_trim'},
            'R9 adapter is only the bounded low-resolution CPU/CUDA cache subspace')
    require(len(plan['nodes']) == 2 and len({n['repo_root'] for n in plan['nodes']}) == 1
            and len({n['python'] for n in plan['nodes']}) == 1, 'R9 matched two-node source/Python paths')


def validate_controller_envelope(plan):
    """Portable source/runtime envelope ONLY, not remote calibration evidence.

    Raw cost receipts deliberately stay on the receipt host. The original
    model_plan remains in the transported object and is fully checked by
    prepare/rebind and every actual worker before model construction.
    """
    _validate_scope(plan)
    portable = {key: value for key, value in plan.items() if key != 'model_plan'}
    profile = validated_profile(portable, root=repository_root())
    expected = resolve_configuration(profile, root=repository_root(), asset_config=plan['asset_config'],
        run_id=plan['run_id'], mode=plan['mode'])
    require(plan['nodes'] == expected['nodes'], 'R9 serialized source/runtime lowering changed')
    require(plan['source_layout'] == expected['source_layout'], 'R9 source layout changed')
    return profile


def validate_plan(plan):
    """Receipt-host/worker validation, including ALL selected cost sources."""
    validate_controller_envelope(plan)
    return validated_profile(plan, root=repository_root())


def validate_config(cfg):
    plan = read(cfg['selected_plan'])
    validate_plan(plan)
    from distillschedule.workloads.best_known.registry import fingerprint
    require(plan.get('execution_transport') == dict(schema_version=SCHEMA,
            deployment_sha256=fingerprint(cfg['deployment'])), 'R9 serialized transport binding changed')
    require(cfg['kind'] == KIND and cfg['source_head'] == plan['integration_head']
            and cfg['candidate_id'] == plan['search_candidate']['id'], 'R9 selected source/candidate')
    require(cfg['owner_sha256'] == OWNER_SHA256, 'R9 unreviewed owner')
    packet = Path(cfg['packet_root'])
    require(cfg['worker_script'] == str(packet / 'scripts/selected_plan_worker.py'), 'R9 worker path')
    for name, code in [('selected_plan_worker.py', WORKER_WRAPPER), ('selected_plan_contract.py', CONTRACT_WRAPPER)]:
        path = packet / 'scripts' / name
        digest = hashlib.sha256(code.encode()).hexdigest()
        require(file_sha256(path) == digest and cfg['file_sha256'].get(str(path)) == digest,
                'R9 wrapper must call the current source adapter unchanged')
    require(cfg['cycles'] == 2 and cfg['profile'] is False and cfg['worker_flags'] == []
            and cfg['numerical_status'] == 'GRID1000_REVIEW_POLICY_ONLY', 'R9 native two-cycle scope')
    require(cfg['owned_worker_oom_fail_fast'] is True and cfg['verified_zombie_fd_rescan'] is True,
            'R9 required safety helpers')
    require(cfg['expected_final_versions'] == dict(generator=2, critic=10, ema=0), 'R9 final versions')
    require(cfg['run_id'] == plan['run_id'] and cfg['run_root'] ==
            str(Path(plan['asset_config']['run_root']) / plan['run_id']), 'R9 run binding')
    require(cfg['port'] == plan['asset_config']['master_port']
            and cfg['master_addr'] == plan['asset_config']['master_addr'], 'R9 rendezvous binding')
    require(len(cfg['hosts']) == len(set(cfg['hosts'])) == 2, 'R9 two distinct assigned hosts')
    require(len(cfg['ordered_UUIDs_by_node']) == 2
            and all(len(v) == 8 for v in cfg['ordered_UUIDs_by_node'])
            and len(set(sum(cfg['ordered_UUIDs_by_node'], []))) == 16, 'R9 unique world16 assignment')
    require(cfg['selection'] == ('MEASURED_MODEL_SELECTED' if 'model_plan' in plan else 'REGISTERED_CANDIDATE'),
            'R9 selection origin changed')
    from .r9_dependencies import validate_plan_dependencies
    validate_plan_dependencies(cfg, plan)
    impl = H3Implementation.parse(cfg['candidate_id'])
    for node, selected in enumerate(plan['nodes']):
        require(selected['repo_root'] == cfg['source_root'] and selected['python'] == cfg['python'],
                'R9 source/Python binding')
        env = cfg['environment_by_node'][node]
        require(all(env.get(k) == v for k, v in selected['environment'].items()), 'R9 selected environment changed')
        require(set(env) - set(selected['environment']) ==
                {'EXPECTED_INTEGRATION_HEAD', 'DMD_SELECTED_PLAN_MANIFEST', 'TMPDIR'}, 'R9 undeclared child environment')
        require(env['EXPECTED_INTEGRATION_HEAD'] == cfg['source_head']
                and env['DMD_SELECTED_PLAN_MANIFEST'] == cfg['remote_manifest'], 'R9 worker manifest binding')
        config = json.loads(Path(env['H3_CONFIG']).read_text())
        require(config == selected['resolved_model_config'], 'R9 selected config changed')
        impl.receipt_binding(config, env, mode=plan['mode'])
    return plan


def validate_owner_config(cfg, node):
    from .r9_dependencies import verify_node_dependencies
    plan = validate_config(cfg)
    verify_node_dependencies(cfg, node)
    return plan


def validate_worker(cfg, *, node, actual_rank=None, environment=None):
    from distillschedule.workloads.best_known.dispatch import validate_local
    env = os.environ if environment is None else environment
    plan = validate_owner_config(cfg, node)
    require(node in (0, 1) and socket.gethostname() == cfg['hosts'][node], 'R9 worker assigned node')
    require(str(repository_root()) == cfg['source_root'], 'R9 worker source import root')
    validate_local(plan, root=repository_root(), node_rank=node)
    require(all(env.get(k) == v for k, v in cfg['environment_by_node'][node].items()), 'R9 actual worker environment')
    require({k for k in env if k.startswith('H3_')} ==
            {k for k in cfg['environment_by_node'][node] if k.startswith('H3_')}, 'R9 undeclared H3 flag')
    if actual_rank is not None:
        require(type(actual_rank) is int and 0 <= actual_rank < 16 and actual_rank // 8 == node,
                'R9 rank/node mismatch')
        require(env.get('WORLD_SIZE') == '16' and env.get('LOCAL_WORLD_SIZE') == '8'
                and int(env.get('LOCAL_RANK', '-1')) == actual_rank % 8, 'R9 torchrun world16')
        require(env.get('CUDA_VISIBLE_DEVICES') == ','.join(cfg['ordered_UUIDs_by_node'][node]), 'R9 assigned UUIDs')
    return plan


def validate_rank(cfg, row):
    # Per-rank terminal proof, not another GPU admission API. Owner admission,
    # every worker and the world collector already validate_config in full.
    # Re-read the hash-bound plan here, without repeatedly walking all raw cost
    # files for each rank on the shared filesystem. No cross-call cache.
    plan = read(cfg['selected_plan'])
    _validate_scope(plan)
    require(plan['search_candidate']['id'] == cfg['candidate_id']
            and plan['integration_head'] == cfg['source_head'], 'R9 rank selected plan binding')
    impl = H3Implementation.parse(cfg['candidate_id'])
    require(type(row['rank']) is int and 0 <= row['rank'] < 16, 'R9 native rank')
    native = plan['nodes'][row['rank'] // 8]
    expected = impl.receipt_binding(native['resolved_model_config'], native['environment'], mode=plan['mode'])
    require(row.get('integration_head') == cfg['source_head'] and row.get('run_kind') == plan['mode']
            and row.get('calibration_only') is (plan['mode'] == 'calibration'), 'R9 actual source/mode')
    require(all(row.get(k) == v for k, v in expected.items()), 'R9 actual selected implementation')
    require(len(row['cycles']) == 2, 'R9 both native cycles')
    for cycle in row['cycles']:
        validate_cache_registration(cycle.get('fa3_replay_cache_registration'), impl)
