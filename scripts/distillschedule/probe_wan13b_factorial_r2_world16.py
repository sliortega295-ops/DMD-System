#!/usr/bin/env python3
"""Opt-in W1 R2: one construction, C1 qualification, full original C2-C4.

This versioned semantic gate does not claim strict cross-arm numerical parity.
All low-level model builders, math handlers and cycle ordering are inherited.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from distillschedule.runtime import wan13b_factorial_r2 as r2

ROOT = Path(__file__).resolve().parents[2]


def load_legacy() -> Any:
    path = ROOT / 'scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py'
    spec = importlib.util.spec_from_file_location('wan13b_r2_legacy_cycle', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--run-id', required=True)
    value.add_argument('--output-dir', type=Path, required=True)
    value.add_argument('--placement', choices=(r2.FSDP, r2.RESIDENT), required=True)
    for name in ('bootstrap', 'model-root', 'trace-bundle', 'negative-prompt', 'factorial-config'):
        value.add_argument('--' + name, type=Path, required=True)
    for name in ('expected-bootstrap-hash', 'expected-trace-hash', 'expected-head', 'factorial-config-sha256'):
        value.add_argument('--' + name, required=True)
    value.add_argument('--validate-config-only', action='store_true')
    return value


def _finite(values: list[torch.Tensor]) -> bool:
    if not values:
        raise AssertionError('R2 deferred finite observations missing')
    return bool(torch.stack(values).all().detach().cpu())


def _cold_counts(build: dict[str, Any]) -> dict[str, Any]:
    g = build['generator']['metadata'].get('generator_rollout_cuda_graph', {})
    c = build['critic']['metadata'].get('critic_grad_cuda_graph', {})
    return {'generator': {k: g.get(k) for k in ('cache_entry_count', 'capture_model_call_count', 'warmup_model_call_count')},
            'critic': {k: c.get(k) for k in ('capture_entries', 'cold_original_forward_calls', 'cold_original_backward_calls')}}


def run_four_cycles(*, cfg: dict[str, Any], args: Any, legacy: Any,
                    rank: int, device: Any, generator_engine: Any, critic_engine: Any,
                    ema: Any, build: dict[str, Any], trace: Any, negative: Any,
                    barrier: Callable[[], None], synchronize: Callable[[], None],
                    gather_c1: Callable[[dict[str, Any]], list[dict[str, Any]]],
                    synthetic_test_only: bool = False) -> dict[str, Any]:
    """Production orchestration; CPU tests replace only device/model/math seams."""
    r2.validate_configuration(cfg, expected_head=args.expected_head)
    r2.assert_runtime_environment(cfg)
    if args.placement != cfg['placement'] or args.run_id == '':
        raise ValueError('R2 runtime source/arm binding changed')
    if torch.initial_seed() != r2.MATCHED['seed'] + rank:
        raise AssertionError('R2 observed rank runtime seed changed')
    versions = lambda: {'generator': int(generator_engine.global_steps),
                        'critic': int(critic_engine.global_steps), 'ema': int(ema.optimization_step)}
    if versions() != r2.INITIAL:
        raise AssertionError('R2 original bootstrap versions changed')
    rollout_contract = legacy._trace_rollout_contract(trace)
    if r2.digest(rollout_contract) != r2.MATCHED['trace_rollout_contract_sha256']:
        raise AssertionError('R2 frozen trace-derived rollout contract changed')
    runtime = legacy._validate_build_contract(build, placement=cfg['placement'], ema=ema)
    if runtime['helios_transformer_sha256'] != cfg['expected_helios_transformer_sha256']:
        raise AssertionError('R2 original external Helios runtime source changed')
    cache = {} if cfg['runtime_policy']['runtime_constants_cache'] else None
    clean = cfg['runtime_policy']['clean_runtime']
    policy = legacy.StaticFSDP2P0Policy(mesh_policy='world16-fsdp', rollout_batch_size=1,
                                      pack_real_score_pair=False, rollout_backend='native-fsdp2-batch')
    diagnostics = legacy.StaticProfileDiagnostics(enabled=False, rank=rank, output_path=None,
                                                  collect_cpu_diagnostics=False)
    rows, censuses, version_rows, finite_rows = [], [], [], []
    raw_cycles, deferred, finite_slices = [], [], []
    c1_binding = None
    for cycle in range(1, 5):
        barrier()
        synchronize()
        check_start = len(deferred)
        start_ns, started = time.time_ns(), time.perf_counter()
        raw = legacy._run_workload_cycle(rank=rank, device=device, generator_engine=generator_engine,
                critic_engine=critic_engine, ema=ema, trace=trace, negative=negative,
                diagnostics=diagnostics, policy=policy, clean_runtime=clean, runtime_cache=cache,
                deferred_finite_checks=deferred)
        synchronize()
        barrier()
        elapsed, stop_ns = time.perf_counter() - started, time.time_ns()
        rows.append({'cycle': cycle, 'role': 'WARMUP_QUALIFICATION' if cycle == 1 else 'MEASURED_DIAGNOSTIC',
                     'start_unix_ns': start_ns, 'stop_unix_ns': stop_ns,
                     'local_elapsed_seconds': elapsed})
        finite_slices.append((check_start, len(deferred)))
        raw_cycles.append(raw)
        version_rows.append(versions())
        if version_rows[-1] != r2.expected_versions(cycle):
            raise AssertionError(f'R2 observed cycle{cycle} versions changed')
        if cycle == 1:
            # Outside C1 timer and before C2: finite/census/mechanism proofs are
            # actually observed, bound and gathered, not filled from config.
            qualification_start = time.time_ns()
            finite_rows.append(_finite(deferred[check_start:]))
            census = legacy._cycle_census(raw, cycle=1, trace_rollout_contract=rollout_contract)
            mechanism = r2.validate_mechanisms(legacy, build, cfg, cycles=1)
            c1_build = copy.deepcopy(build)
            c1_cold = _cold_counts(build)
            observation = {'rank': rank, 'run_id': args.run_id, 'expected_head': args.expected_head,
                           'configuration_fingerprint': r2.digest(cfg), 'matched_contract': dict(r2.MATCHED),
                           'finite': finite_rows[0], 'versions': versions(), 'census': census,
                           'mechanism_status': 'OBSERVED_VALIDATED',
                           'mechanism_fingerprint': r2.digest(mechanism), 'numerical_status': 'NOT_RUN'}
            c1_binding = r2.qualify_c1(gather_c1(observation), cfg, run_id=args.run_id,
                                     rollout_contract=rollout_contract)
            qualification_stop = time.time_ns()
            barrier()
    # C2-C4 contain no host evidence extraction; all remaining checks follow C4.
    finite_rows.extend(_finite(deferred[start:stop]) for start, stop in finite_slices[1:])
    if finite_rows != [True] * 4 or any(stop - start != 6 for start, stop in finite_slices):
        raise FloatingPointError('R2 observed finite/update inventory failed')
    censuses = [legacy._cycle_census(raw, cycle=i, trace_rollout_contract=rollout_contract)
                for i, raw in enumerate(raw_cycles, 1)]
    for i, census in enumerate(censuses, 1):
        r2.validate_census(census, cycle=i, rollout_contract=rollout_contract)
        if census['sample_payload_identities'] != censuses[0]['sample_payload_identities']:
            raise AssertionError('R2 fixed rank-slot payload cohort changed')
    mechanism = r2.validate_mechanisms(legacy, build, cfg, cycles=4)
    if _cold_counts(build) != c1_cold:
        raise AssertionError('R2 CUDA Graph cold work escaped C1')
    return {'schema_version': r2.SCHEMA + '.rank',
            'status': 'SYNTHETIC_TEST_ONLY' if synthetic_test_only else 'PASS',
            'evidence_kind': 'SYNTHETIC_TEST_ONLY' if synthetic_test_only else 'NATIVE_GPU_OBSERVED',
            'run_id': args.run_id, 'expected_head': args.expected_head, 'rank': rank,
            'world_size': 16, 'placement': cfg['placement'], 'cell': cfg['cell'],
            'rank_runtime_seed': torch.initial_seed(),
            'configuration_fingerprint': r2.digest(cfg), 'matched_contract': dict(r2.MATCHED),
            'lowered_runtime_environment': {k: os.environ.get(k) for k in r2.lower_environment(cfg)},
            'runtime_policy': cfg['runtime_policy'], 'runtime_contract': runtime,
            'trace_rollout_contract': rollout_contract, 'initial_versions': dict(r2.INITIAL),
            'cycles': rows, 'cycle_census': censuses, 'cycle_versions': version_rows,
            'final_versions': version_rows[-1], 'finite_status': finite_rows,
            'c1_binding': c1_binding, 'c1_build_evidence': c1_build,
            'c1_qualification_window': {'start_unix_ns': qualification_start, 'stop_unix_ns': qualification_stop},
            'build_evidence': copy.deepcopy(build), 'mechanism_receipts': mechanism,
            'runtime_cache_created': cache is not None, 'runtime_cache_keys': sorted(cache) if cache is not None else [],
            'numerical_status': 'NOT_RUN', 'formal_timing_eligible': False,
            'profiled': False, 'evidence_extraction': 'AFTER_C1_BEFORE_C2_AND_AFTER_C4'}


def main() -> int:
    args = parser().parse_args()
    cfg = r2.load_configuration(args.factorial_config, expected_hash=args.factorial_config_sha256,
                                expected_head=args.expected_head, root=ROOT)
    r2.assert_runtime_environment(cfg)
    if (args.placement != cfg['placement'] or args.expected_bootstrap_hash != r2.MATCHED['bootstrap_hash']
            or args.expected_trace_hash != r2.MATCHED['trace_bundle_hash']):
        raise ValueError('R2 asset/placement arguments differ from frozen configuration')
    if not args.run_id.replace('-', '').replace('_', '').replace('.', '').isalnum():
        raise ValueError('unsafe run id')
    if args.validate_config_only:
        return 0  # No CUDA init/model imports/lock acquisition in preflight.
    legacy = load_legacy()
    legacy._load_native_runtime()
    args.factorial_configuration = cfg  # Consumed by the real shared model builder.
    legacy._configure_shared_mode_args(args)
    legacy._validate_arm_environment(args.placement)
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 16:
        raise ValueError('R2 requires world16')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    seed = r2.MATCHED['seed'] + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output = args.output_dir / 'ranks' / f'rank_{rank:02d}.json'
    try:
        base = legacy.base
        bootstrap = base.Wan13BBootstrap.load(args.bootstrap, expected_hash=args.expected_bootstrap_hash, verify_payloads=rank == 0)
        trace = legacy.StaticWorld16Trace(args.trace_bundle, variant='no-gan')
        negative = legacy.NegativePromptAsset.load(args.negative_prompt)
        observed = {**legacy.runtime_precision_contract(),
                    'bootstrap_hash': bootstrap.manifest_hash,
                    'model_identity_sha256': bootstrap.manifest['model_identity_sha256'],
                    'trace_bundle_hash': trace.bundle_hash, 'trace_binding_hash': trace.binding_hash,
                    'negative_prompt_manifest_hash': negative.manifest['manifest_hash'],
                    'trace_rollout_contract_sha256': r2.digest(legacy._trace_rollout_contract(trace))}
        if any(observed[k] != r2.MATCHED[k] for k in observed):
            raise ValueError('R2 actual source/asset/precision identity changed before construction')
        if Path(bootstrap.manifest['model_root']).resolve() != args.model_root.resolve():
            raise ValueError('R2 bootstrap/model-root identity changed')
        negative_tensor = negative.embedding.to(device=device, dtype=torch.bfloat16)
        started = time.perf_counter()
        g, c, ema, build = base._build_engines(args=args, bootstrap=bootstrap, device=device)
        torch.cuda.synchronize(device)
        construction_seconds = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats(device)

        def gather(observation: dict[str, Any]) -> list[dict[str, Any]]:
            observations: list[Any] = [None] * 16
            dist.all_gather_object(observations, observation)
            return observations

        result = run_four_cycles(cfg=cfg, args=args, legacy=legacy, rank=rank, device=device,
                    generator_engine=g, critic_engine=c, ema=ema, build=build, trace=trace,
                    negative=negative_tensor, barrier=dist.barrier,
                    synchronize=lambda: torch.cuda.synchronize(device), gather_c1=gather)
        result.update(hostname=socket.gethostname(), local_rank=local_rank,
                      construction_seconds=construction_seconds, memory=base._memory(device),
                      configuration_file_sha256=args.factorial_config_sha256)
        legacy._atomic_json(output, result)
        dist.barrier()
    except BaseException as error:
        legacy._atomic_json(output, {'schema_version': r2.SCHEMA + '.rank', 'status': 'FAIL',
                                    'rank': rank, 'run_id': args.run_id, 'error': str(error),
                                    'numerical_status': 'NOT_RUN', 'formal_timing_eligible': False})
        raise
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
