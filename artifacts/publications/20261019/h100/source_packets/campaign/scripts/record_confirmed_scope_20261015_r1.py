"""Preserve the current failed candidate and apply the user's narrower scope.

No worker is signalled. The idle, owned controller exits through its existing
STOP_AFTER_CURRENT protocol after the scoped failure disposition is recorded.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess

ROOT = Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
RUNS = ROOT / 'research_artifacts/five_workload_h100_20260930'
OUT = RUNS / 'h100_resident_baselines_20261014_r1'


def immutable(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    status = json.loads((OUT / 'campaign_r5_status.json').read_text())
    assert status['phase'] == '022r2_helios_retention_actual_launcher'
    assert status['status'] == 'FAIL_PRESERVED_WAITING_SCOPED_DISPOSITION'
    contexts = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'], text=True).strip()
    assert not contexts, 'Do not change controller state while GPU contexts exist'
    queue = OUT / 'queue_r5'
    fail = queue / '022r2_helios_retention_actual_launcher.failure.json'
    assert json.loads(fail.read_text())['status'] == 'FAIL_PRESERVED'
    pending = [str(p) for p in queue.glob('*.ready.json')
               if not p.with_name(p.name.removesuffix('.ready.json') + '.done.json').exists()
               and not p.with_name(p.name.removesuffix('.ready.json') + '.failure.json').exists()]
    assert not pending, pending
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = {
        'run_id': 'h100_confirmed_scope_reconciliation_20261015_r1',
        'utc': stamp,
        'scope_source': 'latest user confirmation table takes precedence over earlier expanded campaign',
        'order': ['W2', 'W3', 'W4', 'W1', 'W5/W6 preparation summary only'],
        'repetitions': 1,
        'new_W5_W6_training': 'NOT_AUTHORIZED_IN_THIS_ROUND_NOT_RUN',
        'completed_historical_W5_W6_artifacts': 'PRESERVE_DO_NOT_REWRITE',
        'new_optimization_GPU_runs': 'DEFER_TO_SUBSEQUENT_SCOPE',
        'extra_retention_failure': str(fail),
        'extra_retention_failure_sha256': hashlib.sha256(fail.read_bytes()).hexdigest(),
        'failure_class': 'LAUNCHER_POLICY_ADMISSION_ERROR_BEFORE_MODEL_CONSTRUCTION',
        'failure_signature': 'Generator no-grad persistent unshard requires GAN world16 FSDP2 B1',
        'frozen_control_runs': 'REUSE_COMPLETE_EVIDENCE_DO_NOT_REPEAT',
        'gpu_contexts_before': [],
        'worker_or_foreign_signals': 'NONE',
        'shared_environment_changes': 'NONE',
        'controller_action': 'existing STOP_AFTER_CURRENT protocol; release idle locks, no artificial occupancy',
        'next': 'CPU-only evidence reconciliation, H3 preparation inventory, acceptance report',
    }
    p = OUT / 'manifests/confirmed_scope_20261015_r1.json'
    immutable(p, manifest)
    with (queue / 'STOP_AFTER_CURRENT').open('x') as f:
        f.write('Latest user scope: no new H3 training or extra optimization in this round.\n')
    disposition = queue / '022r2_helios_retention_actual_launcher.disposition.json'
    temp = disposition.with_suffix('.partial')
    immutable(temp, {
        'utc': stamp, 'preserve_failed_artifacts': True,
        'action': 'CONTINUE_OTHER_ADMITTED_STEPS',
        'note': 'STOP_AFTER_CURRENT is already present: controller must exit before any new work',
        'scope_manifest': str(p), 'GPU_model_workload': 'NOT_RUN',
    })
    os.link(temp, disposition)
    temp.unlink()
    with (OUT / 'research_log.jsonl').open('a') as f:
        f.write(json.dumps({'utc': stamp, 'event': 'LATEST_USER_SCOPE_RECONCILED',
                            'manifest': str(p), 'W5_W6_new_training': 'NOT_RUN',
                            'failed_retention_candidate_preserved': True}) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
