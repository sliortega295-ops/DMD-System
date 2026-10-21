#!/usr/bin/env python3
"""C28 real B1 Student rollout, operator attribution and exact-cache canary.

No distributed workers, optimizer or training timing. All value reads and
diagnostic hooks are outside the standalone latency measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--bootstrap', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    result = dict(status='NOT_RUN', world16='NOT_RUN', production_installed=False,
                  kind='REAL_STAGE1_STUDENT_REAL_B1_ROLLOUT', cases=[],
                  claim_boundary='SINGLE_GPU_CANARY_NOT_FULL5_SPEEDUP')
    if args.plan_only:
        print(json.dumps(result))
        return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        tmp.replace(args.output)

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

        assert torch.cuda.device_count() == 1
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        trace = worker.FormalTrace.load(args.trace, expected_cycles=2)
        bootstrap = worker.Bootstrap.load(args.bootstrap, trace=trace)
        # Reproduce already enabled C18 then C20, not a new attention backend.
        kv_receipt = install_generator_cross_kv_cache(torch)
        fa_receipt = install_generator_flash_attn2(torch)
        t0 = time.perf_counter()
        torch.manual_seed(0)
        student = worker._wan_model(WanModel, size='1.3B', config=WAN_CONFIGS['t2v-1.3B'])
        state_receipt = worker._load_student_state(torch, student, bootstrap)
        student.to(device).train()
        assert all(p.dtype == torch.float32 and p.requires_grad for p in student.parameters())
        streamer = worker._build_streamers(torch, student=student, count=1, device=device)[0]
        loader = worker.InputLoader(torch=torch, trace=trace)
        torch.cuda.synchronize()
        result.update(torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0), trace_sha256=trace.sha256,
                      bootstrap_sha256=bootstrap.sha256, student_state=state_receipt,
                      startup_wall_s=time.perf_counter()-t0,
                      initial_allocated=torch.cuda.memory_allocated(),
                      scope_source_sha256=hashlib.sha256(inspect.getsource(rollout_cast_cache_scope).encode()).hexdigest())
        save()
        pointers = {p.data_ptr(): (n, tuple(p.shape)) for n, p in student.named_parameters()}
        versions = {n: p._version for n, p in student.named_parameters()}

        class CastCensus(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.groups = Counter()
                self.bytes = Counter()
                self.examples = {}

            def __torch_dispatch__(self, func, types, positional=(), kwargs=None):
                kwargs = kwargs or {}
                if func == torch.ops.aten._to_copy.default:
                    value = positional[0]
                    source = pointers.get(value.data_ptr()) if value.is_cuda else None
                    parameter = source is not None and tuple(value.shape) == source[1]
                    kind = ('parameter' if parameter else 'other') + '/' + str(value.dtype) + '->' + str(kwargs.get('dtype', value.dtype))
                    self.groups[kind] += 1
                    self.bytes[kind] += value.numel() * value.element_size()
                    if parameter:
                        self.examples.setdefault(source[0], dict(shape=list(value.shape), source_dtype=str(value.dtype),
                                                                 target_dtype=str(kwargs.get('dtype')), count=0))['count'] += 1
                return func(*positional, **kwargs)

            def receipt(self):
                return dict(counts=dict(self.groups), source_bytes=dict(self.bytes), parameter_names=self.examples)

        def run(row, enabled):
            with torch.no_grad(), rollout_cast_cache_scope(torch, enabled=enabled):
                return worker.prepare_rollout(torch, row=row, streamer=streamer, student=student,
                                              loader=loader, device=device, elide_full_context_lens=True)

        def pack(prepared):
            # Canary correctness only, AFTER all timer endpoints.
            records = [[{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in rec.items()}
                        for rec in group] for group in prepared.recs]
            return prepared.x0.detach().cpu(), records

        def identical(a, b):
            assert torch.equal(a[0], b[0]), 'final rollout latent mismatch'
            assert a[0].shape == b[0].shape == worker.LATENT_SHAPE
            assert a[0].dtype == b[0].dtype == torch.float32
            assert len(a[1]) == len(b[1])
            for ga, gb in zip(a[1], b[1]):
                assert len(ga) == len(gb) == worker.STEPS_PER_BLOCK
                for ra, rb in zip(ga, gb):
                    assert ra.keys() == rb.keys()
                    for key in ra:
                        assert torch.equal(ra[key], rb[key]) if torch.is_tensor(ra[key]) else ra[key] == rb[key]

        rows = sorted({r.new_blocks: r for r in reversed(trace.rows) if r.cycle_index == 0}.values(),
                      key=lambda row: row.new_blocks)
        rows = [rows[0], rows[-1]]
        for row in rows:
            case = dict(operation_key=row.operation_key, new_blocks=row.new_blocks, arms={}, operator_attribution={})
            rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            reference = pack(run(row, False))
            identical(reference, pack(run(row, True)))
            assert torch.equal(rng[0], torch.get_rng_state()) and torch.equal(rng[1], torch.cuda.get_rng_state())
            case.update(warmup_parity='BITWISE_PASS_ALL_RECORDED_NOISE_AND_FINAL_LATENT', rng_equal=True,
                        latent_sha256=hashlib.sha256(reference[0].contiguous().numpy().tobytes()).hexdigest())
            # Paired unprofiled standalone measurements; fixed order alternates.
            for repeat, order in enumerate(((False, True), (True, False))):
                for enabled in order:
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    prepared = run(row, enabled)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    label = 'cache_lifetime' if enabled else 'reference'
                    case['arms'].setdefault(label, []).append(dict(repeat=repeat, wall_s=seconds,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        after_scope_allocated_bytes=torch.cuda.memory_allocated(),
                        driver_free_bytes=torch.cuda.mem_get_info()[0]))
                    identical(reference, pack(prepared))
                    del prepared
            # Separate metadata-only attribution, never part of latency.
            observations = []
            for enabled in (False, True):
                census = CastCensus()
                layer_metadata = Counter()
                handles = []
                for name, module in student.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        def hook(mod, inputs, name=name):
                            layer_metadata[str((name, tuple(inputs[0].shape), str(inputs[0].dtype),
                                str(mod.weight.dtype), torch.is_autocast_enabled('cuda'),
                                str(torch.get_autocast_dtype('cuda')), torch.is_grad_enabled()))] += 1
                        handles.append(module.register_forward_pre_hook(hook))
                calls = []
                original = bc.block_forward
                def counted(*a, **kw):
                    calls.append((tuple(a[1].shape), str(a[1].dtype), kw.get('grad_checkpoint', False)))
                    return original(*a, **kw)
                bc.block_forward = counted
                try:
                    with census:
                        prepared = run(row, enabled)
                    identical(reference, pack(prepared))
                    del prepared
                finally:
                    bc.block_forward = original
                    for handle in handles:
                        handle.remove()
                observations.append((calls, layer_metadata))
                label = 'cache_lifetime' if enabled else 'reference'
                case['operator_attribution'][label] = census.receipt()
            assert observations[0] == observations[1], 'call inventory/effective dtype changed'
            case.update(application_dit_calls=len(observations[0][0]), exact_call_dtype_inventory='PASS',
                        median_s={key: statistics.median(x['wall_s'] for x in trials) for key, trials in case['arms'].items()})
            result['cases'].append(case)
            save()
        assert versions == {n: p._version for n, p in student.named_parameters()}
        assert all(p.grad is None for p in student.parameters())
        assert not torch.is_autocast_enabled('cuda') and torch.is_grad_enabled()
        result.update(status='PASS_CANARY_ONLY', parameter_versions_unchanged=True, gradients_absent=True,
                      kv_receipt=kv_receipt, fa_receipt=fa_receipt)
        save()
        print(json.dumps(dict(status=result['status'], cases=[{k: c[k] for k in ('new_blocks','median_s','application_dit_calls')} for c in result['cases']])))
        return 0
    except Exception:
        import traceback
        result.update(status='FAIL_CANARY', error=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
