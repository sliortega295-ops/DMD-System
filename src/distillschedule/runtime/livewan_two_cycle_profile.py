"""Opt-in validated C29/C34 diagnostic capture, never a native timing arm.

The original exact-head trajectory gate remains immutable. A separate profile
receipt binds that gate to an audited instrumentation-only descendant.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

NAME = "two_cycle_profile_b16_depth1_v2"
VARIANT = "c29-continuous-b16-generator-rms-epilogue"
C34_VARIANT = "c34-continuous-b16-residual-layout"
SUPPORTED_VARIANTS = frozenset({VARIANT, C34_VARIANT})
SCHEMA = "distillschedule.livewan_two_cycle_profile_gate.v1"


def enabled(run_kind: str) -> bool:
    flag = os.environ.get("LIVEWAN_PROFILE_TWO_CYCLES", "0")
    if flag not in ("0", "1"):
        raise RuntimeError("invalid two-cycle profile flag")
    if flag == "1" and run_kind != "profile":
        raise RuntimeError("two-cycle profiling cannot enter a native/trajectory arm")
    return flag == "1"


def plan_variant(plan: dict[str, Any], *, mode: str) -> str:
    from .livewan_crossscale_worker import (
        _generator_rms_epilogue_enabled, _generator_residual_layout_enabled)

    if mode != "pipeline" or not _generator_rms_epilogue_enabled(plan, mode=mode):
        raise RuntimeError("two-cycle profile requires a validated C29/C34 pipeline")
    if _generator_residual_layout_enabled(plan, mode=mode):
        return C34_VARIANT
    if plan['metadata'].get('checkpoint_composition') is not None:
        raise RuntimeError("two-cycle profile does not authorize intermediate compositions")
    return VARIANT


def apply_contract(plan: dict[str, Any], *, mode: str) -> None:
    from .livewan_crossscale_worker import _apply_two_cycle_timing_contract

    plan_variant(plan, mode=mode)
    _apply_two_cycle_timing_contract(plan, mode=mode)
    c = plan["metadata"]["continuous_pipeline"]
    c.update(name=NAME, profiled=True, formal_timing_eligible=False)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_prepare_script(*, repo_root: Path, receipt_path: Path, gate_path: Path) -> Path:
    """Keep the immutable native source receipt's canonical paths on relocation.

    Only an authorized diagnostic descendant may reuse its native parent's
    byte-identical source preparation script. Training workers still execute
    the new checkout, whose pinned upstream files are independently checked.
    No receipt is rewritten and no native gate is relaxed.
    """
    import subprocess

    gate = json.loads(gate_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    source = gate.get('profile_source', {})
    if (gate.get('status') != 'PASS_DIAGNOSTIC_ONLY'
        or source.get('instrumentation_only') is not True
        or gate.get('pipeline_variant') not in SUPPORTED_VARIANTS):
        raise RuntimeError('canonical source preparation requires audited profile gate')
    upstream = Path(receipt['pythonpath'][0])
    if upstream.name != 'livewan' or upstream.parent.name != 'third_party':
        raise RuntimeError('canonical LiveWan source root layout changed')
    native_root = upstream.parent.parent
    current = repo_root/'scripts/distillschedule/prepare_livewan_crossscale_source.sh'
    native = native_root/'scripts/distillschedule/prepare_livewan_crossscale_source.sh'
    for root, expected in ((repo_root, source['profile_head']),
                           (native_root, source['workload_head'])):
        head = subprocess.check_output(
            ['git','-c',f'safe.directory={root}','-C',str(root),'rev-parse','HEAD'],
            text=True).strip()
        if head != expected:
            raise RuntimeError('canonical source preparation checkout HEAD changed')
    if sha256(current) != sha256(native):
        raise RuntimeError('source preparation script is not byte-identical')
    for row in receipt['files'].values():
        original = Path(row['path'])
        if sha256(original) != row['sha256']:
            raise RuntimeError('registered source file changed')
        if original.is_relative_to(upstream):
            relocated = repo_root/'third_party/livewan'/original.relative_to(upstream)
            if sha256(relocated) != row['sha256']:
                raise RuntimeError('relocated upstream source differs from native')
    return native


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
        or v.get("cycles") != 2 or mode != "pipeline" or pipeline_variant not in SUPPORTED_VARIANTS
        or s.get("profile_head") != expected_head or s.get("instrumentation_only") is not True
        or v.get("pipeline_variant") != pipeline_variant):
        raise RuntimeError("two-cycle diagnostic identity/claim boundary changed")
    native = v['native_gate']
    binding = _load_two_cycle_timing_gate(
        path=native['path'], expected_sha256=native['sha256'],
        expected_head=s['workload_head'], mode=mode, pipeline_variant=pipeline_variant,
        arm_id=arm_id, seed=seed, trace=trace, bootstrap=bootstrap, source_receipt=source_receipt)
    return dict(profile_gate_path=str(path), profile_gate_sha256=expected_sha256,
                profile_head=expected_head, workload_head=s['workload_head'],
                native_gate_binding=binding, status='PASS_DIAGNOSTIC_ONLY',
                formal_timing_eligible=False, profiled=True, cycles=2,
                pipeline_variant=pipeline_variant)


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
    first = run_root/'receipts'/'rank_000.json'
    variant = (json.loads(first.read_text()).get('two_cycle_profile', {}).get('pipeline_variant')
               if first.is_file() else None)
    if variant not in SUPPORTED_VARIANTS:
        errors.append('missing/unsupported actual diagnostic pipeline variant')
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
            or binding.get('pipeline_variant') != variant
            or binding.get('formal_timing_eligible') is not False):
            errors.append(f'rank{rank}: source/cycle/profile contract failed')
        for key in ('oom','oom_kill'):
            if r['cgroup_oom_before'].get(key,0) != r['cgroup_oom_after'].get(key,0):
                errors.append(f'rank{rank}: {key} delta')
        errors.extend(f'rank{rank}: {e}' for e in rms_errors(
            r.get('generator_rms_epilogue',{}), rank=rank, cycles=2,
            fuse_cast_square=variant == C34_VARIANT))
        errors.extend(f'rank{rank}: {e}' for e in cast_errors(
            r.get('generator_rollout_cast_cache',{}), rank=rank, cycles=2))
        if variant == C34_VARIANT:
            from .livewan_generator_modulation import runtime_receipt_errors as mod_errors
            from .livewan_generator_gate import runtime_receipt_errors as gate_errors
            from .livewan_attention_checkpoint import runtime_receipt_errors as sac_errors
            from .livewan_c30_contract import C34_PLAN_CONTRACT
            if r.get('checkpoint_composition') != C34_PLAN_CONTRACT:
                errors.append(f'rank{rank}: C34 composition changed')
            errors.extend(f'rank{rank}: {e}' for e in mod_errors(
                r.get('generator_modulation',{}), rank=rank, cycles=2))
            errors.extend(f'rank{rank}: {e}' for e in gate_errors(
                r.get('generator_gate',{}), rank=rank, cycles=2, output_layout='row_major'))
            errors.extend(f'rank{rank}: {e}' for e in sac_errors(
                r.get('critic_attention_checkpoint',{}), rank=rank, expected_calls=300))
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
        head=expected_head,run_id=run_root.name,run_kind='profile',pipeline_variant=variant,
        rank_count=len(rows),cycles=2,profiled=True,formal_timing_eligible=False,
        claim_boundary='DIAGNOSTIC_ONLY_NOT_NATIVE_TIMING',trace_sha256=trace.sha256,
        final_versions=dict(generator=2,critic=10,ema=0),expected_logical=expected_logical,
        expected_physical=expected_physical,nodes=nodes,errors=errors)
