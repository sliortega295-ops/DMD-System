#!/usr/bin/env python3
"""C29 real Student/trace canary. Strict parity before standalone measurements."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--bootstrap', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    out = dict(status='NOT_RUN', candidate='C29_GENERATOR_RMS_EPILOGUE', parent='C28',
        input_kind='REAL_STUDENT_REAL_IMMUTABLE_TRACE', world16='NOT_RUN',
        production_installed=False, operator_cases=[], rollout_cases=[],
        thresholds=dict(assert_close='torch FP32 defaults', normalized_l2_max=1e-3,
                        cosine_min=.999999, real_rollout='BITWISE_ALL_RECORDED_LATENTS'))
    if args.plan_only:
        print(json.dumps(out)); return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(out, indent=2) + '\n'); tmp.replace(args.output)

    save()
    try:
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from wan.configs import WAN_CONFIGS
        from wan.modules.model import WanModel, WanRMSNorm
        from wanstreamer import blockcausal as bc
        from distillschedule.runtime import livewan_crossscale_worker as worker
        from distillschedule.runtime.livewan_cross_kv_cache import install_generator_cross_kv_cache
        from distillschedule.runtime.livewan_attention_backend import install_generator_flash_attn2
        from distillschedule.runtime.livewan_rollout_cast_cache import rollout_cast_cache_scope
        from distillschedule.runtime.livewan_generator_rms_epilogue import (
            build_nograd_generator_rms_epilogue, model_epilogue_canary_scope,
            REFERENCE_CLASS_SHA256)

        assert torch.cuda.device_count() == 1
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        trace = worker.FormalTrace.load(args.trace, expected_cycles=2)
        bootstrap = worker.Bootstrap.load(args.bootstrap, trace=trace)
        assert hashlib.sha256(inspect.getsource(WanRMSNorm).encode()).hexdigest() == REFERENCE_CLASS_SHA256
        install_generator_cross_kv_cache(torch)
        install_generator_flash_attn2(torch)
        torch.manual_seed(0)
        t0 = time.perf_counter()
        student = worker._wan_model(WanModel, size='1.3B', config=WAN_CONFIGS['t2v-1.3B'])
        state = worker._load_student_state(torch, student, bootstrap)
        student.to(device).train()
        streamer = worker._build_streamers(torch, student=student, count=1, device=device)[0]
        loader = worker.InputLoader(torch=torch, trace=trace)
        op, receipt = build_nograd_generator_rms_epilogue(torch)
        torch.cuda.synchronize()
        out.update(torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), student_state=state, trace_sha256=trace.sha256,
            bootstrap_sha256=bootstrap.sha256, startup_wall_s=time.perf_counter()-t0,
            reference_class_sha256=REFERENCE_CLASS_SHA256, kernel_receipt=receipt)
        save()
        versions = {n: p._version for n, p in student.named_parameters()}
        rows = sorted({r.new_blocks: r for r in reversed(trace.rows) if r.cycle_index == 0}.values(), key=lambda r:r.new_blocks)
        rows = [rows[0], rows[-1]]

        def run(row, enabled):
            with torch.no_grad(), rollout_cast_cache_scope(torch, enabled=True), \
                 model_epilogue_canary_scope(torch, student, op, enabled=enabled):
                return worker.prepare_rollout(torch, row=row, streamer=streamer, student=student,
                    loader=loader, device=device, elide_full_context_lens=True)

        def pack(prepared):
            return prepared.x0.detach().cpu(), [
                [{k: v.detach().cpu() if torch.is_tensor(v) else v for k,v in r.items()} for r in g]
                for g in prepared.recs]

        def compare_rollout(a, b):
            assert a[0].dtype == b[0].dtype and a[0].shape == b[0].shape == worker.LATENT_SHAPE
            assert torch.equal(a[0], b[0]), 'rollout final latent is not bitwise equal'
            assert len(a[1]) == len(b[1])
            for g1,g2 in zip(a[1],b[1]):
                assert len(g1) == len(g2) == worker.STEPS_PER_BLOCK
                for r1,r2 in zip(g1,g2):
                    assert r1.keys() == r2.keys()
                    for k in r1:
                        assert torch.equal(r1[k],r2[k]) if torch.is_tensor(r1[k]) else r1[k] == r2[k]

        samples, norm_counts, active, ops = {}, Counter(), [], Counter()
        # Attribute ops inside the actual norm invocation on the issuing Python thread.
        class NormOps(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, positional=(), kwargs=None):
                if active:
                    ops[str(func)] += 1
                return func(*positional, **(kwargs or {}))

        handles = []
        for name, module in student.named_modules():
            if type(module) is WanRMSNorm:
                def before(m, inputs, name=name):
                    x = inputs[0]
                    key = (name, str(x.dtype), tuple(x.shape))
                    norm_counts[str(key)] += 1
                    if len(samples) < 6 and key not in samples:
                        samples[key] = (m, x.detach().clone())
                    active.append(name)
                def after(m, inputs, output):
                    assert active
                    active.pop()
                handles.extend([module.register_forward_pre_hook(before),
                                module.register_forward_hook(after, always_call=True)])
        rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        try:
            with NormOps():
                prepared = run(rows[0], False)
            reference = pack(prepared); del prepared
        finally:
            for h in handles: h.remove()
        assert not active
        out['observed_reference'] = dict(norm_call_count=sum(norm_counts.values()),
            per_module_shape_dtype=dict(norm_counts), same_thread_norm_operator_counts=dict(ops))
        save()

        for key, (module, x) in samples.items():
            with torch.no_grad():
                ref, cand = module(x), op(x, module.weight, module.eps)
                assert ref.dtype == cand.dtype == torch.float32 and ref.shape == cand.shape
                a,b=ref.double().flatten(),cand.double().flatten()
                na,nb=torch.linalg.vector_norm(a),torch.linalg.vector_norm(b)
                metrics = dict(normalized_l2=float(torch.linalg.vector_norm(a-b)/na.clamp_min(1e-20)),
                    cosine=float(torch.dot(a,b)/(na*nb).clamp_min(1e-20)),
                    max_abs=float((a-b).abs().max()), bitwise_equal=torch.equal(ref,cand),
                    finite=bool(torch.isfinite(ref).all() and torch.isfinite(cand).all()))
                case = dict(name=key[0],shape=list(x.shape),input_dtype=str(x.dtype),
                    weight_dtype=str(module.weight.dtype),output_dtype=str(ref.dtype),metrics=metrics)
                out['operator_cases'].append(case)
                save()
                torch.testing.assert_close(cand, ref)  # FP32 defaults; never weakened
                assert metrics['finite'] and metrics['normalized_l2']<=1e-3 and metrics['cosine']>=.999999
                case['status']='PASS'
        samples.clear()
        torch.cuda.synchronize()
        save()
        for row in rows:
            case=dict(operation_key=row.operation_key,new_blocks=row.new_blocks,arms={})
            out['rollout_cases'].append(case)
            ref=pack(run(row,False))
            compare_rollout(ref,pack(run(row,True)))
            for repeat,order in enumerate(((False,True),(True,False))):
                for enabled in order:
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    t=time.perf_counter(); prepared=run(row,enabled); torch.cuda.synchronize()
                    seconds=time.perf_counter()-t
                    case['arms'].setdefault('c29' if enabled else 'c28',[]).append(dict(
                        repeat=repeat,wall_s=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),driver_free_bytes=torch.cuda.mem_get_info()[0]))
                    compare_rollout(ref,pack(prepared)); del prepared
            obs=[]
            for enabled in (False,True):
                calls=[]; original=bc.block_forward
                def counted(*a,**kw):
                    calls.append((tuple(a[1].shape),str(a[1].dtype),kw.get('grad_checkpoint',False)))
                    return original(*a,**kw)
                bc.block_forward=counted
                try:
                    prepared=run(row,enabled); compare_rollout(ref,pack(prepared)); del prepared
                finally: bc.block_forward=original
                obs.append(calls)
            assert obs[0]==obs[1]
            case.update(status='PASS',application_dit_calls=len(obs[0]),bitwise_rollout_parity=True,
                        median_s={k:statistics.median(v['wall_s'] for v in vs) for k,vs in case['arms'].items()})
            save()
        assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state())
        assert versions=={n:p._version for n,p in student.named_parameters()}
        assert all(p.grad is None for p in student.parameters())
        assert all('forward' not in m.__dict__ for m in student.modules() if type(m) is WanRMSNorm)
        out.update(status='PASS_CANARY_ONLY',parameter_versions_unchanged=True,rng_equal=True,
                   gradients_absent=True,scope_restored=True)
        save(); print(json.dumps(out['rollout_cases'])); return 0
    except Exception:
        import traceback
        out.update(status='FAIL_CANARY',error=traceback.format_exc());save();raise


if __name__=='__main__':
    raise SystemExit(main())
