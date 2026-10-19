#!/usr/bin/env python3
"""Fail-closed W1 R2 collector; raw C2 and mean of C2-C4 rank maxima only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from distillschedule.runtime import wan13b_factorial_r2 as r2

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_rows(rows: list[dict[str, Any]], cfg: dict[str, Any], *, run_id: str,
                   allow_synthetic_test_only: bool = False) -> dict[str, Any]:
    r2.validate_configuration(cfg, expected_head=cfg['source']['expected_head'])
    worker = _load('wan13b_r2_summary_worker', ROOT / 'scripts/distillschedule/probe_wan13b_factorial_r2_world16.py')
    legacy = worker.load_legacy()
    errors: list[str] = []
    maxima = {cycle: [] for cycle in range(1, 5)}
    payloads: list[str] = []
    hosts: dict[str, list[int]] = {}
    c1_digests = set()
    synthetic = any(row.get('evidence_kind') == 'SYNTHETIC_TEST_ONLY' for row in rows)
    expected_kind = 'SYNTHETIC_TEST_ONLY' if allow_synthetic_test_only else 'NATIVE_GPU_OBSERVED'
    expected_status = 'SYNTHETIC_TEST_ONLY' if allow_synthetic_test_only else 'PASS'
    if len(rows) != 16 or [row.get('rank') for row in rows] != list(range(16)):
        errors.append('expected all ordered rank0..15 receipts')
    for row in rows:
        rank = row.get('rank')
        try:
            required = {'schema_version': r2.SCHEMA + '.rank', 'status': expected_status,
                        'evidence_kind': expected_kind, 'run_id': run_id,
                        'expected_head': cfg['source']['expected_head'], 'world_size': 16,
                        'rank_runtime_seed': r2.MATCHED['seed'] + rank,
                        'placement': cfg['placement'], 'cell': cfg['cell'],
                        'configuration_fingerprint': r2.digest(cfg),
                        'matched_contract': r2.MATCHED, 'runtime_policy': cfg['runtime_policy'],
                        'initial_versions': r2.INITIAL, 'cycle_versions': [r2.expected_versions(i) for i in range(1, 5)],
                        'final_versions': r2.expected_versions(4), 'finite_status': [True] * 4,
                        'lowered_runtime_environment': r2.lower_environment(cfg),
                        'runtime_cache_created': cfg['runtime_policy']['runtime_constants_cache'],
                        'profiled': False, 'numerical_status': 'NOT_RUN', 'formal_timing_eligible': False,
                        'evidence_extraction': 'AFTER_C1_BEFORE_C2_AND_AFTER_C4'}
            for key, expected in required.items():
                if r2.digest(row.get(key)) != r2.digest(expected):
                    raise AssertionError(f'{key} source/arm/runtime/workload binding changed')
            if not cfg['runtime_policy']['runtime_constants_cache'] and row.get('runtime_cache_keys') != []:
                raise AssertionError('B=OFF runtime cache executed')
            contract = row['trace_rollout_contract']
            cycles, censuses = row['cycles'], row['cycle_census']
            if len(cycles) != 4 or len(censuses) != 4:
                raise AssertionError('expected exactly four full cycles')
            for i, (timer, census) in enumerate(zip(cycles, censuses, strict=True), 1):
                if timer.get('cycle') != i or timer.get('role') != ('WARMUP_QUALIFICATION' if i == 1 else 'MEASURED_DIAGNOSTIC'):
                    raise AssertionError('cycle timer identity changed')
                duration = timer['local_elapsed_seconds']
                if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                    raise AssertionError('invalid cycle duration')
                start, stop = timer['start_unix_ns'], timer['stop_unix_ns']
                if type(start) is not int or type(stop) is not int or start <= 0 or stop <= start:
                    raise AssertionError('invalid timer endpoints')
                if i > 1 and start < cycles[i - 2]['stop_unix_ns']:
                    raise AssertionError('cycle timers overlap')
                maxima[i].append(duration)
                r2.validate_census(census, cycle=i, rollout_contract=contract)
                if census['sample_payload_identities'] != censuses[0]['sample_payload_identities']:
                    raise AssertionError('immutable rank-slot payload mapping not replayed')
            payloads.extend(censuses[0]['sample_payload_identities'])
            window = row['c1_qualification_window']
            if not (cycles[0]['stop_unix_ns'] <= window['start_unix_ns'] < window['stop_unix_ns'] <= cycles[1]['start_unix_ns']):
                raise AssertionError('C1 qualification escaped the C1/C2 boundary')
            c1 = row['c1_binding']
            qualified = r2.qualify_c1(c1['observations'], cfg, run_id=run_id, rollout_contract=contract)
            if r2.digest(qualified) != r2.digest(c1):
                raise AssertionError('C1 qualification hash/scope changed')
            own = c1['observations'][rank]
            if r2.digest(own['census']) != r2.digest(censuses[0]):
                raise AssertionError('C1 binding not consumed by this rank/cycle')
            c1_proof = r2.validate_mechanisms(legacy, row['c1_build_evidence'], cfg, cycles=1)
            if r2.digest(c1_proof) != own['mechanism_fingerprint']:
                raise AssertionError('C1 actual mechanism binding changed')
            proof = r2.validate_mechanisms(legacy, row['build_evidence'], cfg, cycles=4)
            if r2.digest(proof) != r2.digest(row['mechanism_receipts']):
                raise AssertionError('final actual mechanism binding changed')
            if worker._cold_counts(row['build_evidence']) != worker._cold_counts(row['c1_build_evidence']):
                raise AssertionError('graph cold work escaped C1')
            runtime = legacy._validate_build_contract(row['build_evidence'], placement=cfg['placement'],
                        ema=SimpleNamespace(implementation='reference'), nccl_algo=row['runtime_contract'].get('nccl_algo'))
            if r2.digest(runtime) != r2.digest(row['runtime_contract']):
                raise AssertionError('actual runtime/build contract changed')
            if runtime['helios_transformer_sha256'] != cfg['expected_helios_transformer_sha256']:
                raise AssertionError('original external Helios runtime identity changed')
            c1_digests.add(c1['observations_sha256'])
            host, local_rank = row['hostname'], row['local_rank']
            if not isinstance(host, str) or type(local_rank) is not int or local_rank != rank % 8:
                raise AssertionError('host/local-rank identity changed')
            hosts.setdefault(host, []).append(local_rank)
        except (AssertionError, ValueError, KeyError, TypeError, IndexError) as error:
            errors.append(f'rank{rank}: {error}')
    if len(set(payloads)) != 96:
        errors.append('expected 96 unique consumed rank-slot payloads')
    if len(c1_digests) != 1:
        errors.append('world16 C1 bindings differ or are incomplete')
    if len(hosts) != 2 or any(sorted(ranks) != list(range(8)) for ranks in hosts.values()):
        errors.append('expected two hosts x local ranks0..7')
    if any(len(values) != 16 for values in maxima.values()):
        errors.append('missing rank/cycle timer evidence')
    values = {i: max(v) if len(v) == 16 else None for i, v in maxima.items()}
    if errors:
        values = dict.fromkeys(range(1, 5))
    status = 'FAIL' if errors else ('SYNTHETIC_TEST_ONLY' if synthetic else 'PASS')
    return {'schema_version': r2.SCHEMA + '.summary', 'status': status, 'errors': errors,
            'run_id': run_id, 'cell': cfg['cell'], 'expected_head': cfg['source']['expected_head'],
            'rank_count': len(rows), 'configuration_fingerprint': r2.digest(cfg),
            'cycle_world_max_seconds': values, 'C2_seconds': values[2],
            'C2_C4_mean_seconds': sum(values[i] for i in (2, 3, 4)) / 3 if not errors else None,
            'timing_aggregation': 'RAW_C2_RANK_MAX_AND_MEAN_OF_C2_C3_C4_RANK_MAX; N_INDEPENDENT_RUNS=1',
            'C1_excluded': True, 'native_data_status': 'NOT_RUN' if synthetic else ('OBSERVED_DIAGNOSTIC' if not errors else 'INVALID'),
            'numerical_status': 'NOT_RUN', 'formal_timing_eligible': False,
            'synthetic_test_only': synthetic, 'host_topology': hosts,
            'final_versions': r2.expected_versions(4) if not errors else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'factorial-config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--placement', required=True)
    parser.add_argument('--factorial-config-sha256', required=True)
    args = parser.parse_args()
    cfg = r2.load_configuration(args.factorial_config, expected_hash=args.factorial_config_sha256,
                                expected_head=args.expected_head, root=ROOT)
    if args.placement != cfg['placement']:
        raise ValueError('R2 collector placement binding changed')
    errors: list[str] = []
    old = _load('wan13b_r2_old_collector', ROOT / 'scripts/distillschedule/summarize_wan13b_two_cycle_timing_world16.py')
    manifests = []
    for node in range(2):
        path = args.run_root / 'launcher' / f'node_{node}' / 'launch_manifest.txt'
        try:
            manifest = dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)
            required = {'mode': 'four-cycle-factorial-r2', 'node_rank': str(node),
                        'dmd_system_head': args.expected_head, 'placement': cfg['placement'],
                        'factorial_config_sha256': args.factorial_config_sha256,
                        'factorial_configuration_fingerprint': r2.digest(cfg),
                        'factorial_cell': cfg['cell'], 'factorial_cycles': '4',
                        'factorial_measurement_cycles': '2,3,4', 'tf32_cublas_override': '1',
                        'world_size': '16', 'global_batch': '16', 'rollout_batch_size': '1',
                        'score_packing': 'false', 'optimizer_implementation': 'PYTORCH_DEFAULT',
                        'ema_implementation': 'reference', 'trace_hash': r2.MATCHED['trace_bundle_hash'],
                        'bootstrap_hash': r2.MATCHED['bootstrap_hash'],
                        'helios_transformer_sha256': cfg['expected_helios_transformer_sha256'],
                        'nccl_algo': 'Tree' if cfg['placement'] == r2.RESIDENT else 'AUTO'}
            required.update({name: str(int(flag)) for name, flag in cfg['operator_flags'].items()})
            for key, expected in required.items():
                if manifest.get(key) != expected:
                    errors.append(f'node{node}: launch {key} changed')
            manifests.append(manifest)
        except (OSError, ValueError) as error:
            errors.append(f'node{node}: invalid launch manifest: {error}')
    run_id = manifests[0].get('run_id', '') if manifests else ''
    if len(manifests) != 2 or any(m.get('run_id') != run_id for m in manifests):
        errors.append('launch run identities differ')
    rows = []
    for rank in range(16):
        try:
            row = json.loads((args.run_root / 'execution/ranks' / f'rank_{rank:02d}.json').read_text())
            if row.get('configuration_file_sha256') != args.factorial_config_sha256:
                errors.append(f'rank{rank}: configuration file hash changed')
            memory = row.get('memory', {})
            for key in ('allocated_bytes', 'reserved_bytes', 'max_allocated_bytes',
                        'max_reserved_bytes', 'driver_free_bytes', 'driver_total_bytes'):
                if type(memory.get(key)) is not int or memory[key] < 0:
                    errors.append(f'rank{rank}: memory receipt missing/invalid: {key}')
            construction = row.get('construction_seconds')
            if not isinstance(construction, (int, float)) or not math.isfinite(construction) or construction <= 0:
                errors.append(f'rank{rank}: construction timer missing/invalid')
            rows.append(row)
        except (OSError, ValueError) as error:
            errors.append(f'rank{rank}: missing/invalid receipt: {error}')
    summary = summarize_rows(rows, cfg, run_id=run_id)
    for node in range(2):
        old._cgroup_delta(args.run_root, node=node, errors=errors)
        old._teardown(args.run_root, node=node, expected_run_id=run_id, errors=errors)
    if any(args.run_root.rglob('*.nsys-rep')) or any(args.run_root.rglob('*.sqlite')):
        errors.append('R2 unprofiled run contains profiler artifacts')
    if errors:
        summary['errors'].extend(errors)
        summary.update(status='FAIL', C2_seconds=None, C2_C4_mean_seconds=None,
                       cycle_world_max_seconds=dict.fromkeys(range(1, 5)))
    old._atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
