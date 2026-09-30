#!/usr/bin/env python3
"""Standalone source-native BF16 Score gate + norm proxy; no model or workers."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import traceback


NORM_SHA = "17d4c9908c060202755a6f1bfd7830ec5aca655c28ff90a3e4278ee483c0282b"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--seed", type=int, default=20260930)
    args = p.parse_args()
    out = dict(status="NOT_RUN", candidate="C35_SCORE_BF16_ROW_MAJOR_GATE",
        parent="C34", production_installed=False, world16="NOT_RUN",
        kind="SYNTHETIC_REAL_SHAPE_SOURCE_NATIVE_CANARY_NOT_FULL5",
        gate_tolerance="BITWISE_OUTPUT_DX_DY", checkpoint_tolerance="BITWISE_OUTPUT_DX_DY",
        norm_chain_tolerance="UNCHANGED_DEFAULT_P0_PARITY", cases=[],
        benchmarks=[], norm_chain=[], command=sys.argv, seed=args.seed)
    if args.plan_only:
        print(json.dumps(out)); return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")
        tmp.replace(args.output)
    save()
    try:
        import torch
        from torch.utils.checkpoint import checkpoint
        from wanstreamer import blockcausal as bc
        from wan.modules.model import WanLayerNorm
        from distillschedule.runtime.livewan_score_gate import build, SOURCE_SHA256
        from distillschedule.runtime.p0_parity import require_tensor_parity

        gate_sha = hashlib.sha256(inspect.getsource(bc._gate).encode()).hexdigest()
        norm_sha = hashlib.sha256(inspect.getsource(WanLayerNorm).encode()).hexdigest()
        assert gate_sha == SOURCE_SHA256, gate_sha
        assert norm_sha == NORM_SHA, norm_sha
        assert torch.cuda.device_count() == 1
        torch.cuda.set_device(0)
        torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
        op, receipt = build(torch)
        original = bc._gate
        out.update(torch=torch.__version__, cuda=torch.version.cuda,
            triton=receipt["triton_version"], gpu=torch.cuda.get_device_name(0),
            gate_source_sha256=gate_sha, norm_source_sha256=norm_sha,
            gate_source_file=inspect.getfile(original), norm_source_file=inspect.getfile(WanLayerNorm),
            candidate_sha256=hashlib.sha256(Path(inspect.getfile(build)).read_bytes()).hexdigest(),
            kernel_receipt=receipt, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))

        def native(x,y,g):
            return original(x,y,g,False,7)

        def tensor(shape, layout):
            b,s,d=shape
            if layout == "channel":
                return torch.randn((b,d,s),device="cuda",dtype=torch.bfloat16).transpose(1,2)
            return torch.randn(shape,device="cuda",dtype=torch.bfloat16)

        def bitwise(a,b):
            assert a.shape==b.shape and a.dtype==b.dtype
            assert torch.equal(a.contiguous().view(torch.int16),b.contiguous().view(torch.int16))

        def measured(fn):
            for _ in range(4): fn()
            samples=[]
            for _ in range(9):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record(); fn(); b.record(); b.synchronize()
                samples.append(a.elapsed_time(b))
            return dict(median_ms=statistics.median(samples),samples_ms=samples,
                scope="CUDA_EVENTS_CANARY_ONLY_NOT_NATIVE_C2")

        cases=[((b,s,d), xl,yl,1) for d in (1536,5120) for b in (1,2)
            for s in (512,19320) for xl,yl in (("channel","row"),("row","row"),("row","channel"))]
        cases += [((2,512,d),"channel","row",2) for d in (1536,5120)]
        torch.cuda.reset_peak_memory_stats()
        for shape,xl,yl,gb in cases:
            x=tensor(shape,xl).requires_grad_(True)
            y=tensor(shape,yl).requires_grad_(True)
            # Native ec is a strided chunk of [B,6,D], not a cloned dense gate.
            g=torch.randn((gb,6,shape[2]),device="cuda",dtype=torch.bfloat16)[:,2:3]
            dy=tensor(shape,"row")
            versions=(x._version,y._version,g._version)
            rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
            a=native(x,y,g); b=op(x,y,g)
            bitwise(a,b); assert b.is_contiguous()
            da=torch.autograd.grad(a,(x,y),dy)
            db=torch.autograd.grad(b,(x,y),dy)
            for reference,candidate in zip(da,db): bitwise(reference,candidate)
            assert versions==(x._version,y._version,g._version)
            assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state())
            row=dict(shape=shape,x_stride=x.stride(),y_stride=y.stride(),gate_stride=g.stride(),
                gate_shape=g.shape,native_output_stride=a.stride(),candidate_output_stride=b.stride(),
                output_dx_dy="BITWISE_PASS",rng="PASS",input_versions="PASS")
            # Test real non-reentrant replay separately on bounded small shapes.
            if shape[1]==512 and xl=="channel" and yl=="row":
                counts=[0,0]
                def nfn(x,y,g):
                    counts[0]+=1; return native(x,y,g)
                def cfn(x,y,g):
                    counts[1]+=1; return op(x,y,g)
                a=checkpoint(nfn,x,y,g,use_reentrant=False)
                b=checkpoint(cfn,x,y,g,use_reentrant=False)
                bitwise(a,b)
                da=torch.autograd.grad(a,(x,y),dy)
                db=torch.autograd.grad(b,(x,y),dy)
                for reference,candidate in zip(da,db): bitwise(reference,candidate)
                assert counts==[2,2], counts
                row.update(checkpoint="BITWISE_PASS",execution_counts=counts)
            out["cases"].append(row); save()
            if shape[0]==2 and shape[1]==19320 and yl=="row":
                with torch.no_grad():
                    result=dict(shape=shape,x_layout=xl,y_layout=yl,
                        native=measured(lambda:native(x,y,g)),candidate=measured(lambda:op(x,y,g)))
                result["speedup"]=result["native"]["median_ms"]/result["candidate"]["median_ms"]
                out["benchmarks"].append(result); save()
            del x,y,g,dy,a,b,da,db
            gc.collect()

        # Operator provenance, not time-overlap inference. Do not replace norms.
        for d in (1536,5120):
            shape=(2,19320,d)
            x=tensor(shape,"channel").requires_grad_(True)
            y=tensor(shape,"row").requires_grad_(True)
            g=torch.randn((2,6,d),device="cuda",dtype=torch.bfloat16)[:,2:3]
            norm=WanLayerNorm(d).to(device="cuda",dtype=torch.bfloat16)
            dy=tensor(shape,"row")
            def chain(enabled):
                with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                    r=(op if enabled else native)(x,y,g)
                    return norm(r)
            a,b=chain(False),chain(True)
            parity=require_tensor_parity(a,b,label=f"Score gate norm forward D{d}")
            da=torch.autograd.grad(a,(x,y),dy)
            db=torch.autograd.grad(b,(x,y),dy)
            grads=[require_tensor_parity(r,c,label=f"Score gate norm gradient D{d}/{i}")
                for i,(r,c) in enumerate(zip(da,db))]
            evidence=dict(shape=shape,forward_parity=parity,
                gradient_parity=grads,profiles=[])
            with torch.no_grad():
                for enabled in (False,True):
                    arm="candidate" if enabled else "native"
                    evidence[arm]=measured(lambda:chain(enabled))
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                        with torch.profiler.record_function("C35_SCORE_GATE_NORM/"+arm):
                            chain(enabled)
                    path=args.output.parent / f"{arm}_norm_d{d}_operator_trace.json"
                    prof.export_chrome_trace(str(path))
                    copies=[]
                    for event in prof.events():
                        if event.name not in ("aten::contiguous","aten::clone","aten::copy_","aten::_to_copy"):
                            continue
                        parents=[]; parent=event.cpu_parent
                        while parent is not None:
                            parents.append(parent.name);parent=parent.cpu_parent
                        if "aten::native_layer_norm" in parents:
                            copies.append(dict(operator=event.name,parents=parents,
                                shapes=event.input_shapes,thread=event.thread,
                                cpu_us=event.cpu_time_total,device_us=event.device_time_total))
                    evidence["profiles"].append(dict(arm=arm,path=str(path),norm_nested_copies=copies))
            evidence["speedup"]=evidence["native"]["median_ms"]/evidence["candidate"]["median_ms"]
            nc=[v for v in evidence["profiles"][0]["norm_nested_copies"] if v["operator"]=="aten::contiguous"]
            cc=[v for v in evidence["profiles"][1]["norm_nested_copies"] if v["operator"]=="aten::contiguous"]
            evidence["contiguous_copy_elision"]="PASS" if nc and not cc else "UNPROVEN"
            out["norm_chain"].append(evidence);save()
            del x,y,g,dy,a,b,da,db,norm
            gc.collect()
        assert original is bc._gate
        out.update(status="PASS_CANARY_ONLY",candidate_promotion="NOT_RUN",
            native_patch_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),driver_free_bytes=torch.cuda.mem_get_info()[0])
        save(); print(json.dumps(dict(status=out["status"],cases=len(out["cases"]),
            benchmarks=out["benchmarks"],norm_chain=out["norm_chain"])))
        return 0
    except Exception:
        out.update(status="FAIL_CANARY",error=traceback.format_exc());save();raise


if __name__=="__main__":
    raise SystemExit(main())
