"""Opt-in C29 two-cycle diagnostic capture, never a native timing arm.

The original exact-head trajectory gate remains immutable. A separate profile
receipt binds that gate to an audited instrumentation-only descendant.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

NAME = "two_cycle_profile_c29_b16_depth1_v1"
VARIANT = "c29-continuous-b16-generator-rms-epilogue"
SCHEMA = "distillschedule.livewan_two_cycle_profile_gate.v1"


def enabled(run_kind: str) -> bool:
    flag = os.environ.get("LIVEWAN_PROFILE_TWO_CYCLES", "0")
    if flag not in ("0", "1"):
        raise RuntimeError("invalid two-cycle profile flag")
    if flag == "1" and run_kind != "profile":
        raise RuntimeError("two-cycle profiling cannot enter a native/trajectory arm")
    return flag == "1"


def apply_contract(plan: dict[str, Any], *, mode: str) -> None:
    from .livewan_crossscale_worker import (
        _apply_two_cycle_timing_contract, _generator_rms_epilogue_enabled)

    if mode != "pipeline" or not _generator_rms_epilogue_enabled(plan, mode=mode):
        raise RuntimeError("two-cycle profile is restricted to the validated C29 pipeline")
    _apply_two_cycle_timing_contract(plan, mode=mode)
    c = plan["metadata"]["continuous_pipeline"]
    c.update(name=NAME, profiled=True, formal_timing_eligible=False)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gate(*, path, expected_sha256, expected_head, mode, pipeline_variant,
              arm_id, seed, trace, bootstrap, source_receipt) -> dict[str, Any]:
    from .livewan_crossscale_worker import _load_two_cycle_timing_gate

    path = Path(path)
    if sha256(path) != expected_sha256:
        raise RuntimeError("profile gate SHA changed")
    v = json.loads(path.read_text())
    s = v.get("profile_source", {})
    if (v.get("schema_version") != SCHEMA or v.get("status") != "PASS_DIAGNOSTIC_ONLY"
        or v.get("profile_authorized") is not True or v.get("formal_timing_authorized") is not False
        or v.get("cycles") != 2 or mode != "pipeline" or pipeline_variant != VARIANT
        or s.get("profile_head") != expected_head or s.get("instrumentation_only") is not True
        or v.get("pipeline_variant") != VARIANT):
        raise RuntimeError("two-cycle diagnostic identity/claim boundary changed")
    native = v['native_gate']
    binding = _load_two_cycle_timing_gate(
        path=native['path'], expected_sha256=native['sha256'],
        expected_head=s['workload_head'], mode=mode, pipeline_variant=pipeline_variant,
        arm_id=arm_id, seed=seed, trace=trace, bootstrap=bootstrap, source_receipt=source_receipt)
    return dict(profile_gate_path=str(path), profile_gate_sha256=expected_sha256,
                profile_head=expected_head, workload_head=s['workload_head'],
                native_gate_binding=binding, status='PASS_DIAGNOSTIC_ONLY',
                formal_timing_eligible=False, profiled=True, cycles=2)


def summarize(*, run_root: Path, trace_path: Path, expected_head: str) -> dict[str, Any]:
    """Validate actual receipts; no conversion of profile walls into native time."""
    from .livewan_crossscale_worker import FormalTrace
    from .livewan_generator_rms_epilogue import runtime_receipt_errors as rms_errors
    from .livewan_rollout_cast_cache import runtime_receipt_errors as cast_errors
    from .livewan_score_rmsnorm import runtime_receipt_errors as score_errors

    trace = FormalTrace.load(trace_path, expected_cycles=2, expected_global_batch=16)
    expected_logical = {
        'student/nograd/plain': sum(1+r.new_blocks*5 for r in trace.rows),
        'student/grad/checkpoint': 2*16*3, 'teacher/nograd/plain': 2*16,
        'critic/nograd/plain': 2*16, 'critic/grad/checkpoint': 2*80}
    expected_physical = dict(expected_logical, **{'critic/grad/checkpoint': 2*40})
    rows, errors, nodes = [], [], []
    for rank in range(16):
        p = run_root/'receipts'/f'rank_{rank:03d}.json'
        if not p.is_file():
            errors.append(f'rank{rank}: missing receipt'); continue
        r = json.loads(p.read_text()); rows.append(r)
        c, binding = r.get('continuous_pipeline', {}), r.get('two_cycle_profile', {})
        if (r.get('status') != 'PASS' or r.get('rank') != rank
            or r.get('run_kind') != 'profile' or r.get('profiled') is not True
            or r.get('mode') != 'pipeline' or r.get('trace_sha256') != trace.sha256
            or r.get('two_cycle_timing') is not None
            or r.get('final_versions') != dict(generator=2,critic=10,ema=0)
            or r.get('logical_generator_updates') != 2 or r.get('logical_critic_updates') != 10
            or (r.get('physical_generator_commits'),r.get('physical_critic_commits'))
            != ((2,0) if rank < 8 else (0,10))
            or c.get('cycles') != 2 or c.get('execution_status') != 'PASS'
            or len(c.get('events',[])) != 2 or c.get('name') != NAME
            or binding.get('profile_head') != expected_head
            or binding.get('formal_timing_eligible') is not False):
            errors.append(f'rank{rank}: source/cycle/profile contract failed')
        for key in ('oom','oom_kill'):
            if r['cgroup_oom_before'].get(key,0) != r['cgroup_oom_after'].get(key,0):
                errors.append(f'rank{rank}: {key} delta')
        for fn,key in [(rms_errors,'generator_rms_epilogue'),(cast_errors,'generator_rollout_cast_cache')]:
            errors.extend(f'rank{rank}: {e}' for e in fn(r.get(key,{}),rank=rank,cycles=2))
        errors.extend(f'rank{rank}: {e}' for e in score_errors(r.get('score_rmsnorm_kernel',{}),rank=rank))
        if r.get('generator_runahead_transport',{}).get('execution_status') != 'PASS':
            errors.append(f'rank{rank}: transport failed')
        sync=r.get('critic_shadow_hot_sync',{})
        if sync.get('execution_status')!='PASS' or [x['version'] for x in sync.get('events',[])] != [5]:
            errors.append(f'rank{rank}: C5 hot-sync failed')
    for kind,expected in [('logical_call_inventory',expected_logical),('physical_call_inventory',expected_physical)]:
        actual={k:sum(int(r.get(kind,{}).get(k,0)) for r in rows) for k in expected}
        if actual != expected: errors.append(f'{kind}: {actual} != {expected}')
    for node in range(2):
        p=run_root/'nsight'/f'node_{node}.receipt.json'
        if not p.is_file(): errors.append(f'node{node}: missing Nsight container'); continue
        r=json.loads(p.read_text());nodes.append(r)
        if (r.get('status')!='PASS_CAPTURE_CONTAINER' or r.get('head')!=expected_head
            or r.get('kernel_devices')!=list(range(8))
            or r.get('nvtx_rank_ranges')!=list(range(node*8,node*8+8))):
            errors.append(f'node{node}: incomplete capture')
        for name in ('nsys-rep','sqlite'):
            artifact=r.get('artifacts',{}).get(name,{})
            if not artifact.get('bytes',0) or not Path(artifact.get('path','')).is_file():
                errors.append(f'node{node}: missing {name}')
        log=run_root/'launcher'/f'node_{node}'
        status=(log/'teardown_status.txt').read_text() if (log/'teardown_status.txt').is_file() else ''
        if 'status=PASS' not in status or 'gpu_process_count=0' not in status:
            errors.append(f'node{node}: teardown missing')
    return dict(status='PASS' if not errors else 'FAIL',schema_version='distillschedule.livewan_two_cycle_profile_summary.v1',
        head=expected_head,run_id=run_root.name,run_kind='profile',pipeline_variant=VARIANT,
        rank_count=len(rows),cycles=2,profiled=True,formal_timing_eligible=False,
        claim_boundary='DIAGNOSTIC_ONLY_NOT_NATIVE_TIMING',trace_sha256=trace.sha256,
        final_versions=dict(generator=2,critic=10,ema=0),expected_logical=expected_logical,
        expected_physical=expected_physical,nodes=nodes,errors=errors)
