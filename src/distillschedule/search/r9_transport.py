"""Explicit two-session R9 transport; CPU RPCs only before/after both owners.

The default dispatcher is unchanged. Failed admission has immutable numbered
attempts; GPU execution has a single exclusive started record and no retry.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from .contracts import require
from .r9_contract import SCHEMA, KIND, OWNER_SHA256, validate_controller_envelope
from .r9_native import validate_deployment
from distillschedule.workloads.best_known.registry import (
    file_sha256, fingerprint, git_head, repository_root, write_new_json,
)


def launch_owners(cfg, manifest, *, output, execute, attempt=1):
    require(type(attempt) is int and 1 <= attempt <= 999, 'R9 admission attempt 1..999 required')
    require(cfg['source_head'] == git_head(repository_root()) and cfg['kind'] == KIND
            and cfg['owner_sha256'] == OWNER_SHA256, 'R9 current source/reviewed owner')
    require(cfg['remote_manifest'] == manifest['path'], 'R9 sealed manifest path')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for started in output.glob('owner-*-started.json'):
        eof = started.with_name(started.name.replace('-started.json', '-eof.json'))
        require(eof.is_file(), 'R9 prior owner terminal state unknown; no new SSH or retry')
    label = 'execute' if execute else f'plan-attempt{attempt:03d}'
    admission = None
    if execute:
        path = output / f'owner-plan-attempt{attempt:03d}-eof.json'
        row = json.loads(path.read_text())
        require(row['status'] == 'PASS_OWNER_PLAN_GPU_NOT_RUN' and row['manifest'] == manifest,
                'R9 exact two-node successful admission required')
        admission = dict(path=str(path), sha256=file_sha256(path))
    helper_dirs = []
    for name in ('native_worker_oom_guard_r1.py', 'verified_zombie_fd_rescan_r1.py', 'historical_ssh_read_only_r2.py'):
        matches = [str(Path(p).parent) for p in cfg['file_sha256'] if Path(p).name == name]
        require(len(matches) == 1, 'R9 pinned safety helper location: ' + name)
        helper_dirs.extend(matches)
    write_new_json(output / f'owner-{label}-started.json', dict(status='STARTED_NOT_RESULT',
        manifest=manifest, admission_attempt=attempt, admission=admission, execute=execute))
    children, streams = [], []
    try:
        for node, transport in enumerate(cfg['transport_nodes']):
            boot = (f'import os;os.environ["NVIDIA_VISIBLE_DEVICES"]="/var/run/nvidia-container-devices";'
                f'os.environ["PYTHONPATH"]={":".join(dict.fromkeys(helper_dirs))+":"+cfg["owner_pythonpath"]!r};'
                f'os.execvp("python3",["python3",{cfg["packet_root"]+"/scripts/selected_plan_owner.py"!r},'
                f'"--manifest",{manifest["path"]!r},"--manifest-sha256",{manifest["sha256"]!r},'
                f'"--node",{str(node)!r}]' + ('+["--execute"]' if execute else '') + ')')
            stream = (output / f'owner-node{node}-{label}.transport.log').open('x')
            streams.append(stream)
            command = ['ssh', '-n', '-tt', '-oBatchMode=yes', '-oControlMaster=no', '-oConnectTimeout=10',
                '-oServerAliveInterval=30', '-oServerAliveCountMax=20',
                '-p', str(transport.get('ssh_port', 22)), transport['host'],
                'python3 -c ' + shlex.quote(boot) + '; exit']
            children.append(subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT))
        codes = [child.wait(timeout=cfg['lease_seconds'] + 180) for child in children]
    except BaseException as error:
        write_new_json(output / f'owner-{label}-interrupted.json', dict(status='UNKNOWN_OWNER_TERMINAL_STATE',
            error=repr(error), instruction='NO_NEW_SSH_OR_RETRY_UNTIL_BOTH_EXISTING_OWNER_STREAMS_TERMINATE'))
        raise
    finally:
        for stream in streams: stream.close()
    result = dict(status=('OWNER_EOF_REQUIRES_WORLD_COLLECTION' if execute else 'PASS_OWNER_PLAN_GPU_NOT_RUN')
        if codes == [0, 0] else 'FAIL_OWNER_TRANSPORT_OR_GATE', manifest=manifest, node_returncodes=codes,
        admission_attempt=attempt, admission=admission, source_head=cfg['source_head'])
    write_new_json(output / f'owner-{label}-eof.json', result)
    require(codes == [0, 0], 'R9 owner transport/gate failed; preserved logs, no automatic retry')
    return result


def rebind_r9_plan(plan, *, run_id, coordinator_output, assets_path=None):
    """Full selected-cost validation and fresh lowering on the receipt host."""
    from .execution import native_rpc
    validate_controller_envelope(plan)
    require(run_id != plan['run_id'] and run_id != 'source-inspection', 'a fresh explicit run-id is required')
    assets = json.loads(Path(assets_path).read_text()) if assets_path is not None else plan['asset_config']
    require(assets == plan['asset_config'], 'R9 replay assets/gates changed')
    output = Path(coordinator_output)
    output.mkdir(parents=True, exist_ok=False)
    result = native_rpc(transport_assets=assets, source_head=plan['integration_head'], output=output,
        action='r9_rebind', receipt=plan, run_id=run_id, assets=assets)
    validate_controller_envelope(result)
    require(result.get('model_plan') == plan.get('model_plan'), 'R9 RPC changed the selected model plan')
    require(result['run_id'] == run_id and result['search_candidate'] == plan['search_candidate']
            and result['asset_config'] == assets and result['mode'] == plan['mode']
            and result.get('execution_transport') == plan.get('execution_transport'), 'R9 RPC replay identity changed')
    write_new_json(output / 'execution-plan.json', result)
    return result


def run_r9_plan(plan, deployment, *, output, dry_run=False):
    from .execution import native_rpc
    validate_controller_envelope(plan)
    validate_deployment(deployment)
    marker = dict(schema_version=SCHEMA, deployment_sha256=fingerprint(deployment))
    require('execution_transport' not in plan or plan['execution_transport'] == marker,
            'R9 plan transport changed')
    if dry_run:
        print(json.dumps(dict(status='R9_SOURCE_ONLY_NOT_RUN', transport=marker, gpu_execution='NOT_RUN',
            calibration_sources='NOT_CHECKED_ON_CONTROLLER; RECEIPT_HOST_VALIDATION_REQUIRED')))
        return 0
    require(output is not None, 'R9 fresh coordinator output required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan['execution_transport'] = marker
    write_new_json(output / 'selected-plan.json', plan)
    def rpc(action, **kwargs):
        return native_rpc(transport_assets=plan['asset_config'], source_head=plan['integration_head'],
            output=output, action=action, **kwargs)
    try:
        prepared = rpc('r9_prepare', receipt=plan, deployment=deployment)
        for node in (0, 1):
            rpc('r9_qualify', _node_rank=node, node_rank=node, draft=prepared['draft'])
        sealed = rpc('r9_seal', draft=prepared['draft'])
        write_new_json(output / 'sealed-manifest.json', sealed['configuration'])
        require(file_sha256(output / 'sealed-manifest.json') == sealed['manifest']['sha256'],
                'R9 returned sealed manifest bytes differ')
        launch_owners(sealed['configuration'], sealed['manifest'], output=output / 'owners', execute=False)
        launch_owners(sealed['configuration'], sealed['manifest'], output=output / 'owners', execute=True)
        # Both wait() calls completed. No metadata SSH is allowed before here.
        result = rpc('r9_collect', manifest=sealed['manifest'])
        write_new_json(output / 'result.json', result)
        return 0
    except BaseException as error:
        write_new_json(output / 'failure.json', dict(status='FAIL', error=repr(error),
            retry='NO_AUTOMATIC_GPU_RETRY; EXPLICIT_NUMBERED_READ_ONLY_ADMISSION_ONLY'))
        raise


def main(argv=None):
    """Explicit admission retry / one execution of an already sealed plan."""
    parser = argparse.ArgumentParser(prog='distillctl workloads r9-owner')
    parser.add_argument('--manifest', type=Path, required=True, help='Exact copied local sealed manifest')
    parser.add_argument('--output', type=Path, required=True, help='Same immutable owner journal directory')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--admission-attempt', type=int, required=True)
    args = parser.parse_args(argv)
    cfg = json.loads(args.manifest.read_text())
    manifest = dict(path=cfg['remote_manifest'], sha256=file_sha256(args.manifest))
    result = launch_owners(cfg, manifest, output=args.output, execute=args.execute, attempt=args.admission_attempt)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
