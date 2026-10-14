#!/usr/bin/env python3
"""Native same-process C1/C2 for a fresh, explicit Wan parameter-grid point.

Reuses the measured model/optimizer/EMA factories and unchanged P0 losses.
Only workload parameters named in the accepted grid differ. Frozen plans
must declare their applicability; neither silent retuning nor truncated F5
trace substitution is allowed. No source-only test is a GPU timing receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import torch
import torch.distributed as dist

from distillschedule.runtime.wan13b_parameter_grid import WanParameterGridSpec, WanGridDistributedExecution
from runtime.wan_parameter_grid_trace import WanParameterGridTrace


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_worker():
    p = ROOT / 'scripts/distillschedule/probe_wan13b_weakscale_two_cycle.py'
    s = importlib.util.spec_from_file_location('wan_grid_factory_admitted', p)
    m = importlib.util.module_from_spec(s); sys.modules[s.name] = m; s.loader.exec_module(m)
    return m, m.shared_worker()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run-id', 'expected-head', 'expected-bootstrap-hash', 'expected-trace-hash', 'expected-workload-hash', 'expected-trace-admission-hash'):
        p.add_argument('--' + name, required=True)
    for name in ('output-dir', 'bootstrap', 'model-root', 'trace-bundle', 'negative-prompt', 'workload', 'trace-admission'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--placement', choices=('world16-fsdp', 'resident-async128-tree-chunk2048'), required=True)
    p.add_argument('--validate-only', action='store_true')
    return p


def workload_spec(path):
    r = json.loads(path.read_text())
    if set(r) != {'height', 'width', 'fake_updates', 'pyramid_budget', 'seed', 'start_step'}:
        raise ValueError('unknown/missing workload fields')
    r['pyramid_budget'] = tuple(r['pyramid_budget'])
    return WanParameterGridSpec(**r)


def execute_cycle(native, spec, *, rank, device, generator, critic, ema, trace,
                  negative, diagnostics, policy, clean, cache, finite):
    entries = {}
    g = native.p0._run_nogan_generator_update_p0(
        rank=rank, device=device, generator_engine=generator, critic_engine=critic,
        trace=trace, negative=negative, collect_diagnostics=False, collect_scalars=False,
        diagnostics=diagnostics, policy=policy, score_pack_parity_check=False,
        clean_runtime=clean, runtime_cache=cache, deferred_finite_checks=finite)
    outputs, rollouts = native.p0._precompute_critic_rollouts(
        rank=rank, steps=spec.logical_steps, device=device, generator_engine=generator,
        trace=trace, policy=policy, collect_diagnostics=False, diagnostics=diagnostics,
        rollout_parity_check=False, clean_runtime=clean, runtime_cache=cache,
        entry_outputs=entries)
    critics = []
    for step in spec.logical_steps:
        critics.append(native.p0._run_nogan_critic_precomputed(
            rank=rank, step=step, x0=outputs[step], device=device, critic_engine=critic,
            trace=trace, collect_diagnostics=False, collect_scalars=False,
            diagnostics=diagnostics, entry=entries[step], runtime_cache=cache,
            deferred_finite_checks=finite))
        ema.step()
        del outputs[step]
    return dict(generator=g, rollouts=rollouts, critic=critics)


def check_census(native, raw, expected, spec):
    if raw['generator']['physical_score_forwards'] != 3:
        raise ValueError('original three DMD Score calls changed')
    if [r['logical_step'] for r in raw['critic']] != list(spec.logical_steps):
        raise ValueError('Fake update order/count changed')
    if any(r['score_forwards'] != 1 for r in raw['critic']):
        raise ValueError('Fake score call count changed')
    if len(raw['rollouts']) != spec.fake_updates or any(r.get('batch_size', 1) != 1 for r in raw['rollouts']):
        raise ValueError('physical B1 rollout changed')
    rows = [raw['generator']['rollout'], *raw['rollouts']]
    inventory = [native.validate_exact_rollout_call_inventory(r, label=f'grid.{i}') for i, r in enumerate(rows)]
    if inventory != [v['inventory'] for v in expected]:
        raise ValueError('source-derived DiT exit/call/physical inventory changed')
    return dict(generator_updates=1, critic_updates=spec.fake_updates, ema_updates=spec.fake_updates,
                generator_score_forwards=3, critic_score_forwards=spec.fake_updates,
                rollout_call_inventory=inventory, rollout_batch_size=1,
                logical_steps=list(spec.logical_steps))


def check_grid_graphs(build, arm, *, spec, inventory):
    # Counter expectations derive from this fresh workload, not canonical27/5.
    gen = build['generator']['metadata'].get('generator_rollout_cuda_graph', {})
    critic = build['critic']['metadata'].get('critic_grad_cuda_graph', {})
    if arm['generator_rollout_cuda_graph']:
        count = 2 * sum(r['inventory']['call_count'] for r in inventory[1:])
        required = dict(enabled=True, weights_merged=False, active=False,
                        model_call_count=count, graph_replay_model_call_count=count,
                        rollout_entry_count=2 * spec.fake_updates)
        if any(gen.get(k) != v for k, v in required.items()):
            raise ValueError('grid rollout graph execution/call count mismatch')
        entries = [dict(version=3+i, logical_steps=[s]) for i in range(2) for s in spec.logical_steps]
        if gen.get('entries') != entries:
            raise ValueError('grid graph consumed wrong step/version/input')
    elif gen.get('enabled'):
        raise ValueError('baseline unexpectedly enabled rollout graphs')
    if arm['critic_grad_cuda_graph']:
        if not critic.get('enabled'):
            raise ValueError('frozen plan requested missing Critic graph')
        # Detailed backend counters are checked by its typed validator below.
        from distillschedule.runtime.wan13b_critic_grad_graph import validate_build
        validate_build(build, cycles=2, reference_checks=False, updates_per_cycle=spec.fake_updates)
    elif critic.get('enabled'):
        raise ValueError('baseline unexpectedly enabled Critic graph')
    return dict(generator=gen, critic=critic)


def main():
    a = parser().parse_args()
    validate_only = a.validate_only
    if subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() != a.expected_head:
        raise ValueError('source HEAD mismatch')
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('source must be clean')
    if sha(a.workload) != a.expected_workload_hash or sha(a.trace_admission) != a.expected_trace_admission_hash:
        raise ValueError('workload/source-generation receipt changed')
    spec = workload_spec(a.workload); a.timing_seed = spec.seed
    a.world_size = 16; a.distributed_execution = WanGridDistributedExecution(fake_updates=spec.fake_updates)
    factory, native = load_worker(); native._configure_shared_mode_args(a)
    arm = native._validate_arm_environment(a.placement)
    if arm['generator_rollout_merged_shadow']:
        raise ValueError('systemically biased merged weights excluded')
    from distillschedule.runtime.wan13b_upper_bound import Wan13BBootstrap
    boot = Wan13BBootstrap.load(a.bootstrap, expected_hash=a.expected_bootstrap_hash, verify_payloads=validate_only)
    trace = WanParameterGridTrace(a.trace_bundle, workload=spec)
    if trace.bundle_hash != a.expected_trace_hash:
        raise ValueError('fresh trace identity mismatch')
    admission = json.loads(a.trace_admission.read_text())
    if admission['status'] != 'PASS_FRESH_NATIVE_PARAMETER_GRID_TRACE' or admission['bundle_hash'] != trace.bundle_hash or admission['workload_fingerprint'] != spec.receipt()['fingerprint']:
        raise ValueError('fresh native trace generation not proven')
    operations = [f'step/{spec.start_step}/generator/rollout', *(f'step/{s}/critic/rollout' for s in spec.logical_steps)]
    inventory = [dict(operation=op, inventory=native.expected_trace_rollout_call_inventory(
        trace.rank_control_validation, operation=op, expected_rank_count=16)) for op in operations]
    row = dict(schema='dmd.wan_parameter_grid_native_c2.v1', run_id=a.run_id,
               source_head=a.expected_head, hostname=socket.gethostname(), placement=a.placement,
               workload=spec.receipt(), bootstrap_hash=boot.manifest_hash,
               trace_hash=trace.bundle_hash, trace_admission_sha256=sha(a.trace_admission),
               profiler='NOT_RUN', numerical_status='DEFERRED_BY_USER_NOT_NUMERICAL_PASS',
               merged_lora_count=0, arm_contract=arm, trace_rollout_inventory=inventory)
    if validate_only:
        if torch.cuda.is_initialized(): raise ValueError('CPU admission initialized CUDA')
        print(json.dumps(dict(row, status='PASS_CPU_GRID_ADMISSION_GPU_NOT_RUN', cuda_initialized=False)))
        return 0
    dist.init_process_group('nccl');rank=dist.get_rank();local=int(os.environ['LOCAL_RANK'])
    if dist.get_world_size()!=16 or torch.cuda.device_count()!=8:raise ValueError('world16 owned eight GPUs/node required')
    device=torch.device('cuda',local);torch.cuda.set_device(device)
    random.seed(spec.seed+rank);torch.manual_seed(spec.seed+rank);torch.cuda.manual_seed_all(spec.seed+rank)
    row.update(rank=rank,local_rank=local);path=a.output_dir/'ranks'/f'rank_{rank:02d}.json'
    where='construction'
    try:
        negative=native.NegativePromptAsset.load(a.negative_prompt).embedding.to(device,dtype=torch.bfloat16)
        cold=time.perf_counter();g,c,ema,build=factory.build_engines(native,a,boot,device)
        torch.cuda.synchronize(device);row['construction_s']=time.perf_counter()-cold
        row['build']=build;row['runtime_contract']=native._validate_build_contract(build,placement=a.placement,ema=ema)
        clean=native._clean_runtime(a.placement);cache={}if clean else None
        policy=native.StaticFSDP2P0Policy(mesh_policy='world16-fsdp',rollout_batch_size=1,pack_real_score_pair=False,rollout_backend='native-fsdp2-batch')
        diagnostics=native.StaticProfileDiagnostics(enabled=False,rank=rank,output_path=None,collect_cpu_diagnostics=False)
        rows=[];raws=[];finite=[];slices=[]
        for i in(1,2):
            where=f'cycle{i}';dist.barrier();torch.cuda.synchronize(device);start=time.perf_counter();at=len(finite)
            raw=execute_cycle(native,spec,rank=rank,device=device,generator=g,critic=c,ema=ema,trace=trace,negative=negative,diagnostics=diagnostics,policy=policy,clean=clean,cache=cache,finite=finite)
            torch.cuda.synchronize(device);dist.barrier();wall=time.perf_counter()-start
            rows.append(dict(cycle=i,wall_s=wall,role='WARMUP'if i==1 else'STEADY_STATE',versions=dict(generator=g.global_steps,critic=c.global_steps,ema=ema.optimization_step)))
            raws.append(raw);slices.append((at,len(finite)))
        where='post-timer gates'
        fin=[bool(torch.stack(finite[s:e]).all().cpu())for s,e in slices]
        if fin!=[True,True]:raise ValueError('nonfinite losses')
        census=[check_census(native,r,inventory,spec)for r in raws]
        expected=dict(generator=4,critic=10+2*spec.fake_updates,ema=10+2*spec.fake_updates)
        if rows[-1]['versions']!=expected or g.step_calls!=2 or c.step_calls!=2*spec.fake_updates:raise ValueError('optimizer/version counts changed')
        graphs=check_grid_graphs(build,arm,spec=spec,inventory=inventory)
        row.update(status='PASS_NATIVE_GRID_RUNTIME_CONTRACTS',cycles=rows,cycle_census=census,finite_status=fin,final_versions=expected,graph_receipts=graphs,memory=native.base._memory(device))
        factory.atomic(path,row);dist.barrier();return 0
    except BaseException as e:
        row.update(status='CUDA_OOM'if isinstance(e,torch.cuda.OutOfMemoryError)else'FAIL',where=where,error=repr(e),traceback=traceback.format_exc())
        try:row['failure_memory']=native.base._memory(device)
        except BaseException:pass
        factory.atomic(path,row);raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    raise SystemExit(main())
