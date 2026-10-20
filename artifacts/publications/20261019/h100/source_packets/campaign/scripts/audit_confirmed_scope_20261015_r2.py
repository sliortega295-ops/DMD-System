"""CPU-only audit of completed H100 runs; never imports a training package."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import subprocess

ROOT = Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
RUNS = ROOT / 'research_artifacts/five_workload_h100_20260930'
OUT = RUNS / 'h100_resident_baselines_20261014_r1'
REPORT = OUT / 'reports/confirmed_scope_acceptance_20261015_r2'


def load(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v):
    with p.open('x') as f:
        json.dump(v, f, indent=2)
        f.write('\n')


def receipt_evidence(path):
    p = Path(path)
    value = load(p)
    return {'path': str(p), 'sha256': sha(p),
            'recorded_status': value.get('status', value.get('run_id', 'SEE_RECEIPT'))}


def arm_evidence(workload, record, expected_head):
    path = Path(record['summary'])
    assert sha(path) == record['summary_sha256'], path
    s = load(path)
    assert s['status'] == 'PASS' and record['profiled'] is False
    assert s.get('expected_head', s.get('head')) == expected_head
    n = s.get('physical_rank_count', s.get('rank_count'))
    assert n == 8
    if workload == 'W4':
        receipts = [load(p) for p in sorted((Path(s['run_root']) / 'receipts').glob('rank_*.json'))]
    else:
        receipts = s.get('physical_receipts', s.get('ranks'))
    assert len(receipts) == 8
    assert sorted(r['rank'] for r in receipts) == list(range(8))
    assert all(r['status'] == 'PASS' for r in receipts)
    answer = {**record, 'head': expected_head, 'physical_receipts_verified': 8,
              'logical_batch': 16, 'strict_full_state_parity': 'NOT_RUN',
              'paper_equivalence_claimable': False,
              'timing_scope': s.get('formal_timer_scope', s.get('mode', s.get('run_kind')))}
    if workload in ('W2', 'W3'):
        ema = 5 if workload == 'W2' else 1
        expected = {'generator': 1, 'critic': 5, 'ema': ema}
        assert all(r['updates'] == expected for r in receipts)
        assert sorted(s['logical_slots']) == list(range(16))
        assert len(s['logical_slots']) == 16
        final = {'generator': 3, 'critic': 15, 'ema': 15} if workload == 'W2' else {
            'generator': 201, 'critic': 1005, 'ema': 801}
        assert s['logical_final_versions'] == final
        health = s['launcher_health']
        assert health['exit_code'] == 0 and health['cgroup_oom_delta'] == {'oom': 0, 'oom_kill': 0}
        assert health['gpu_processes_after'] == [] if 'gpu_processes_after' in health else True
        answer.update(updates_per_cycle=expected, logical_final_versions=final,
                      logical_initial_versions=s['logical_initial_versions'],
                      physical_initial_versions=s['physical_initial_versions'],
                      physical_final_versions=s['physical_final_versions'],
                      policy=s['policy'], asset_fingerprint=s['asset_fingerprint'],
                      cgroup_oom_delta=health['cgroup_oom_delta'],
                      peak_allocated_GiB=max(r['final']['memory']['max_allocated_gib'] for r in receipts),
                      peak_reserved_GiB=max(r['final']['memory']['max_reserved_gib'] for r in receipts),
                      min_driver_free_at_final_snapshot_GiB=min(r['final']['memory']['driver_free_gib'] for r in receipts))
    elif workload == 'W1':
        assert sorted(s['logical_slots']) == list(range(16))
        expected = {'generator_updates': 1, 'critic_updates': 5, 'ema_updates': 5}
        for r in receipts:
            assert len(r['cycle_census']) == 2
            assert all(all(c[k] == v for k, v in expected.items()) for c in r['cycle_census'])
        answer.update(updates_per_cycle=expected, final_versions=s['final_versions'],
                      memory_per_rank=[{'rank': r['rank'], 'memory': r['memory']} for r in receipts],
                      matched_contract=s['matched_contract'],
                      launcher_health_status=s['launcher_health']['status'])
    else:
        assert s['profiled'] is False and s['physical_world_size'] == 8 and s['logical_world_size'] == 16
        assert s['cycle_2']['seconds'] == record['seconds']
        answer.update(final_versions=s['final_versions'], cycle2=s['cycle_2'],
                      trace_sha256=s['trace_sha256'],
                      memory_per_rank=[{'rank': r['rank'], 'peak_memory': r['peak_memory'],
                                        'host_memory': r['host_memory'],
                                        'cgroup_before': r['cgroup_oom_before'],
                                        'cgroup_after': r['cgroup_oom_after']} for r in receipts],
                      matched_workload_fingerprint=s['matched_workload_fingerprint'])
    return answer


def main():
    rows = []
    for workload, name in [('W2', 'helios_no-gan_official_swap_matched_r1'),
                           ('W3', 'helios_gan_official_swap_matched_r1'),
                           ('W4', 'livewan_resident_matched_r3'), ('W1', 'wan_matched_r1')]:
        p = OUT / 'artifacts' / name / 'result.json'
        d = load(p)
        assert d['replications'] == 1 and d['profiled'] is False
        arms = [arm_evidence(workload, x, d['exact_head']) for x in d['records']]
        b, c = ({x['arm']: x for x in arms}[k] for k in ('baseline', 'candidate'))
        ratio = b['seconds'] / c['seconds']
        assert abs(ratio - d['single_pair_ratio']) < 1e-12
        rows.append({'workload': workload, 'status': 'PASS_EXECUTION_SINGLE_DIAGNOSTIC_PAIR',
                     'source_result': receipt_evidence(p), 'head': d['exact_head'],
                     'baseline_seconds': b['seconds'], 'candidate_seconds': c['seconds'],
                     'diagnostic_ratio': ratio, 'candidate_faster_single_trial': ratio > 1,
                     'replications': 1, 'CV': 'NOT_MEASURED', 'arms': arms,
                     'strict_equivalence': 'NOT_RUN', 'formal_speedup_claim': 'NOT_ADMITTED'})
    prep_paths = [
        RUNS / 'h3_grid_h100_native_table_20261014_r1/validator.json',
        RUNS / 'h3_grid_h100_native_table_20261014_r1/adaln_grid1000.json',
        RUNS / 'h3_three_arm_common_bootstrap_build_20261014_r2/bootstrap/manifest.json',
        OUT / 'artifacts/h3_w6_preparation/private_environment_cpu_gate_r6.json',
        OUT / 'artifacts/h3_w6_preparation/native_inputs_cpu_gate_r1.json',
        OUT / 'artifacts/h3_w6_preparation/h100_qualified_cache_manifest_r1.json',
        OUT / 'artifacts/w6_raw_model_payload_revalidation_r1/receipt.json',
    ]
    prep = [receipt_evidence(p) for p in prep_paths]
    scope = load(OUT / 'manifests/confirmed_scope_20261015_r1.json')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                   '--format=csv,noheader'], text=True).strip()
    controller = load(OUT / 'campaign_r5_status.json')
    assert controller['status'] == 'EXIT_0'
    assert not Path('/proc/516246').exists(), 'owned controller must be fully exited'
    result = {
        'run_id': 'h100_confirmed_scope_acceptance_20261015_r2',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'analysis': 'CPU_ONLY_EXISTING_RECEIPTS_NO_NEW_TRAINING_OR_TIMING',
        'order': scope['order'], 'rows': rows,
        'H3_preparation': {'status': 'EXISTING_ASSET_AND_ENVIRONMENT_RECEIPTS_VERIFIED',
                           'evidence': prep,
                           'new_training_this_round': 'NOT_RUN',
                           'historical_training': 'Already existing in the earlier expanded campaign; preserved, not rerun or relabelled as this round',
                           'W5_semantics': 'Grid1000 approximate; H100-only common bootstrap, not A100 starting-state replay',
                           'W6_semantics': 'private native AR8 G1/F1 continuous sigma; W5 Grid not reused',
                           'large_model_payload_rehashed_in_this_audit': False},
        'cleanup': {'owned_controller_exited': True, 'controller': controller,
                    'GPU_contexts_at_owned_teardown_0039_UTC': [],
                    'current_GPU_inventory_csv': gpu,
                    'current_context_note': 'Foreign StreamingAR gpu_lease reserve workers started at00:39:24 UTC after our controller exited. No signals or preemption. Host GPU PIDs are namespace-external.',
                    'process_signals': 'NONE', 'training_code_changes': 'NONE_IN_THIS_AUDIT'},
        'boundaries': [
            'All timings unprofiled, one trial per arm; no median/CV claim.',
            'Execution/update/version PASS is not strict cross-arm numerical parity PASS.',
            'W2/W3 official role-swap baseline uses GPU ZeRO1, BF16 full active backbone and CPU inactive role.',
            'W2/W3 optimized arms change admitted attention/operator/residency implementations, not placement alone.',
            'Original fully-resident OOM evidence retained; role-swap is a separately named baseline.',
            'No new W5/W6 training, retention experiment, profiler capture, push or shared-environment modification.',
        ],
    }
    REPORT.mkdir(exist_ok=False)
    write(REPORT / 'acceptance.json', result)
    with (REPORT / 'acceptance.csv').open('x') as f:
        writer = csv.DictWriter(f, fieldnames=['workload', 'head', 'baseline_seconds', 'candidate_seconds',
                                              'diagnostic_ratio', 'candidate_faster_single_trial', 'strict_equivalence'])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    lines = ['# H100 confirmed scope acceptance', '',
             'Run ID: `h100_confirmed_scope_acceptance_20261015_r2`', '',
             'CPU-only audit of preserved raw receipts. No new GPU workload was launched.', '',
             '| Workload | Baseline s | Candidate s | Diagnostic ratio | Receipt gate |',
             '|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['workload']} | {r['baseline_seconds']:.6f} | {r['candidate_seconds']:.6f} | {r['diagnostic_ratio']:.6f}x | 8 physical / 16 logical PASS |")
    lines += ['', 'W2 actual update receipts: G1/C5/EMA5; logical final G3/C15/EMA15.',
              'W3 actual update receipts: G1/C5/EMA1; logical final G201/C1005/EMA801.',
              'W4 reuses the completed baseline after a hardware-binding postprocessor fix. The pipeline did NOT speed it up.',
              '', '## Claim boundaries', *['- ' + x for x in result['boundaries']],
              '', '## H3 preparation only in this round',
              'W5 H100 table, immutable bootstrap, and W6 exact private source/environment/cache/model receipts exist and were checked. See JSON evidence paths and SHA256.',
              'Earlier expanded-campaign training artifacts already exist and are preserved separately; they are not new runs in this scope.',
              '', '## Deferred extra experiment',
              'The H100 W2 retained-unshard candidate failed validate-only at the legacy GAN/world16 policy guard before model loading. No capacity or timing result exists. Failure preserved; no guard bypass.',
              '', '## Cleanup', 'Owned controller exited at00:39:20 UTC through STOP_AFTER_CURRENT with EXIT_0; the preceding owned teardown inventory was empty. Foreign StreamingAR gpu_lease reserve workers started at00:39:24 UTC; the current inventory is recorded without intervention. No worker or foreign process was signalled.',
              '', 'The first CPU-only audit wrongly required globally idle GPUs even for parsing old JSON; it stopped before report creation when foreign reservations appeared. This R2 changes only that audit gate and preserves R1. No GPU workload was rerun.', '']
    (REPORT / 'acceptance.md').write_text('\n'.join(lines))
    write(REPORT / 'artifact_sha256.json', {str(p.relative_to(REPORT)): sha(p) for p in sorted(REPORT.iterdir()) if p.is_file()})
    with (OUT / 'research_log.jsonl').open('a') as f:
        f.write(json.dumps({'utc': result['utc'], 'event': 'CONFIRMED_SCOPE_RECEIPTS_RECONCILED',
                            'report': str(REPORT / 'acceptance.json'), 'new_GPU_workload': 'NOT_RUN',
                            'W4_speedup': 'NOT_OBSERVED', 'strict_equivalence': 'NOT_RUN'}) + '\n')
    print(json.dumps({'status': 'PASS_EVIDENCE_RECONCILIATION', 'report': str(REPORT),
                      'rows': [{k: r[k] for k in ('workload', 'baseline_seconds', 'candidate_seconds', 'diagnostic_ratio')} for r in rows]}, indent=2))


if __name__ == '__main__':
    main()
