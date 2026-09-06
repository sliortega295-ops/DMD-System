#!/usr/bin/env python3
"""C32 real checkpoint/trace, source-scoped operation evidence before promotion."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('trace', 'bootstrap', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    out = dict(status='NOT_RUN', candidate='C32_GENERATOR_FP32_MODULATION', parent='C31',
        production_installed=False, world16='NOT_RUN', kernel_cases=[], rollout_cases=[],
        correctness='BITWISE_ALL_RECORDED_LATENTS_AND_OPERATOR_OUTPUTS',
        factor='no-grad G modulation pointwise only; native norm preserved')
    if args.plan_only:
        print(json.dumps(out)); return 0
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(out, indent=2)+'\n'); tmp.replace(args.output)
    save()
    try:
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from wan.configs import WAN_CONFIGS
        from wan.modules.model import WanModel
        from wanstreamer import blockcausal as bc
        from distillschedule.runtime import livewan_crossscale_worker as worker
        from distillschedule.runtime.livewan_cross_kv_cache import install_generator_cross_kv_cache
        from distillschedule.runtime.livewan_attention_backend import install_generator_flash_attn2
        from distillschedule.runtime.livewan_rollout_cast_cache import rollout_cast_cache_scope
        from distillschedule.runtime.livewan_generator_rms_epilogue import (
            build_nograd_generator_rms_epilogue, model_epilogue_canary_scope)
        from distillschedule.runtime.livewan_generator_modulation import build, canary_scope
        assert torch.cuda.device_count() == 1
        torch.cuda.set_device(0); device = torch.device('cuda:0')
        trace = worker.FormalTrace.load(args.trace, expected_cycles=2)
        bootstrap = worker.Bootstrap.load(args.bootstrap, trace=trace)
        install_generator_cross_kv_cache(torch); install_generator_flash_attn2(torch)
        torch.manual_seed(0); start = time.perf_counter()
        student = worker._wan_model(WanModel, size='1.3B', config=WAN_CONFIGS['t2v-1.3B'])
        state = worker._load_student_state(torch, student, bootstrap)
        student.to(device).train()
        streamer = worker._build_streamers(torch, student=student, count=1, device=device)[0]
        loader = worker.InputLoader(torch=torch, trace=trace)
        rms_op, _ = build_nograd_generator_rms_epilogue(torch, fuse_cast_square=True)
        op, receipt = build(torch)
        out.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
            trace_sha256=trace.sha256, bootstrap_sha256=bootstrap.sha256, student_state=state,
            startup_wall_s=time.perf_counter()-start, kernel_receipt=receipt)
        versions = {n: p._version for n, p in student.named_parameters()}
        rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        original = bc._modulate
        rows = sorted({r.new_blocks:r for r in reversed(trace.rows) if r.cycle_index==0}.values(), key=lambda r:r.new_blocks)
        rows = [rows[0], rows[-1]]
        def run(row, enabled):
            with torch.no_grad(), rollout_cast_cache_scope(torch, enabled=True), \
                 model_epilogue_canary_scope(torch, student, rms_op, enabled=True), \
                 canary_scope(torch, student, op, enabled=enabled):
                return worker.prepare_rollout(torch, row=row, streamer=streamer, student=student,
                    loader=loader, device=device, elide_full_context_lens=True)
        def pack(v):
            return v.x0.detach().cpu(), [[{k:t.detach().cpu() if torch.is_tensor(t) else t
                for k,t in r.items()} for r in group] for group in v.recs]
        def equal(a,b):
            assert torch.equal(a[0],b[0]) and a[0].shape==b[0].shape==worker.LATENT_SHAPE
            assert a[0].dtype==b[0].dtype and len(a[1])==len(b[1])
            for ag,bg in zip(a[1],b[1]):
                assert len(ag)==len(bg)==worker.STEPS_PER_BLOCK
                for ar,br in zip(ag,bg):
                    assert ar.keys()==br.keys()
                    for k in ar:
                        assert torch.equal(ar[k],br[k]) if torch.is_tensor(ar[k]) else ar[k]==br[k]
        active=[]; ops=Counter(); cases={}; calls=[]
        class ModulationOps(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, positional=(), kwargs=None):
                if active: ops[str(func)] += 1
                return func(*positional, **(kwargs or {}))
        def observe(x, norm, shift, scale, per_frame, n_frames):
            active.append(1)
            try: result=original(x,norm,shift,scale,per_frame,n_frames)
            finally: active.pop()
            key=(tuple(x.shape),str(x.dtype),str(scale.dtype),tuple(result.stride()))
            calls.append(str(key))
            if key not in cases and len(cases)<12:
                with torch.no_grad():
                    normalized=norm(x)
                    cases[key]=(normalized.detach().clone(), scale.detach().clone(),
                                shift.detach().clone(), result.detach().clone())
            return result
        bc._modulate=observe
        try:
            with ModulationOps(): ref=pack(run(rows[0],False))
        finally: bc._modulate=original
        assert not active
        out['source_scoped_operator_evidence']=dict(calls=len(calls), shape_dtype_histogram=dict(Counter(calls)),
                                                   issuing_thread_ops=dict(ops))
        for key,(normalized,scale,shift,reference) in cases.items():
            with torch.no_grad(): actual=op(normalized,scale,shift)
            row=dict(shape=list(normalized.shape),normalized_dtype=str(normalized.dtype),
                scale_dtype=str(scale.dtype),shift_dtype=str(shift.dtype),output_dtype=str(actual.dtype),
                output_stride=list(actual.stride()),reference_stride=list(reference.stride()),
                bitwise_equal=torch.equal(actual,reference),max_abs=float((actual-reference).abs().max()))
            out['kernel_cases'].append(row);save()
            assert row['bitwise_equal'] and actual.dtype==reference.dtype==torch.float32
            assert actual.stride()==reference.stride()
        cases.clear()
        torch.cuda.synchronize()
        for row in rows:
            ref=pack(run(row,False)); equal(ref,pack(run(row,True)))
            result=dict(operation_key=row.operation_key,new_blocks=row.new_blocks,arms={})
            out['rollout_cases'].append(result)
            for repeat,order in enumerate(((False,True),(True,False))):
                for enabled in order:
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter();v=run(row,enabled);torch.cuda.synchronize()
                    dt=time.perf_counter()-start
                    result['arms'].setdefault('c32' if enabled else 'c31',[]).append(dict(
                        repeat=repeat,wall_s=dt,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),driver_free_bytes=torch.cuda.mem_get_info()[0]))
                    equal(ref,pack(v));del v
            inventories=[]
            for enabled in (False,True):
                cs=[];f=bc.block_forward
                def counted(*a,**kw):
                    cs.append((tuple(a[1].shape),str(a[1].dtype),kw.get('grad_checkpoint',False)))
                    return f(*a,**kw)
                bc.block_forward=counted
                try: v=run(row,enabled);equal(ref,pack(v));del v
                finally:bc.block_forward=f
                inventories.append(cs)
            assert inventories[0]==inventories[1]
            result.update(status='PASS',bitwise_rollout_parity=True,application_dit_calls=len(inventories[0]),
                median_s={k:statistics.median(x['wall_s'] for x in xs) for k,xs in result['arms'].items()})
            save()
        assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state())
        assert versions=={n:p._version for n,p in student.named_parameters()} and all(p.grad is None for p in student.parameters())
        assert bc._modulate is original
        out.update(status='PASS_CANARY_ONLY',parameter_versions_unchanged=True,rng_equal=True,
                   gradients_absent=True,scope_restored=True)
        save();print(json.dumps(out['rollout_cases']));return 0
    except Exception:
        import traceback
        out.update(status='FAIL_CANARY',error=traceback.format_exc());save();raise


if __name__=='__main__':
    raise SystemExit(main())
