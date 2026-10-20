"""Owned-GPU diagnostic: actual native LoRA + BF16 autocast + checkpoint.

This probe never loads the MiniMax checkpoint and cannot prove full-model fit.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

import torch
from torch.utils.checkpoint import checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.repo / 'src'), str(args.repo)]
    import h3_dmd2 as native
    from runtime.h3_diffsynth_lora_unit_inplace import UnitScaleInplaceLoRA
    from distillschedule.runtime.p0_parity import tensor_parity_metrics
    from types import SimpleNamespace
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == '77744f5ec391be813d42ebaf6a2692c18ea58f5ca7fb41398fd287d2a177ca62'
    assert torch.cuda.get_device_name(0).startswith('NVIDIA H100')
    torch.manual_seed(20261015)
    torch.cuda.set_device(0)
    rows = []
    dtype_rows = []
    for width, hidden, tokens in ((192, 384, 37), (5376, 28672, 64)):
        reference = native.DualLoRALinear(torch.nn.Linear(width, hidden, dtype=torch.bfloat16), 64, 64)
        for pair in reference.adapters.values():
            torch.nn.init.normal_(pair['B'].weight, std=0.01)
        reference = reference.cuda()
        candidate = copy.deepcopy(reference)
        for role in ('student', 'fake_score', 'teacher'):
            reference.role = candidate.role = role
            for replay in (False, True):
                reference.zero_grad(set_to_none=True)
                candidate.zero_grad(set_to_none=True)
                x = torch.randn(tokens, width, device='cuda', dtype=torch.bfloat16)
                a = x.clone().requires_grad_()
                b = x.clone().requires_grad_()
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    expected = checkpoint(reference, a, use_reentrant=False) if replay else reference(a)
                    expected.float().square().mean().backward()
                context = UnitScaleInplaceLoRA(native, SimpleNamespace(layers=[('actual', candidate)]), enabled=True)
                try:
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        actual = checkpoint(candidate, b, use_reentrant=False) if replay else candidate(b)
                        actual.float().square().mean().backward()
                    for label, left, right in [('output', expected, actual), ('input_grad', a.grad, b.grad)]:
                        rows.append({'shape': [tokens, width, hidden], 'role': role,
                                     'checkpoint': replay, 'kind': label,
                                     'bitwise_equal': torch.equal(left, right),
                                     **tensor_parity_metrics(left, right)})
                    for (name, left), (_, right) in zip(reference.named_parameters(), candidate.named_parameters()):
                        assert (left.grad is None) == (right.grad is None)
                        if left.grad is not None:
                            rows.append({'shape': [tokens, width, hidden], 'role': role,
                                         'checkpoint': replay, 'kind': name + '.grad',
                                         'bitwise_equal': torch.equal(left.grad, right.grad),
                                         **tensor_parity_metrics(left.grad, right.grad)})
                    dtype_rows.append({'role': role, 'base': str(candidate.base.weight.dtype),
                                       'A': str(candidate.adapters['student']['A'].weight.dtype),
                                       'B': str(candidate.adapters['student']['B'].weight.dtype),
                                       'output': str(actual.dtype)})
                finally:
                    context.close()
        del reference, candidate, expected, actual, a, b, x
        torch.cuda.empty_cache()
    report = {'status': 'PASS_GPU_NATIVE_LORA_CHECKPOINT_PARITY' if all(r['status'] == 'PASS' for r in rows) else 'FAIL_GPU_PARITY',
              'GPU': torch.cuda.get_device_name(0), 'thresholds': 'unchanged DEFAULT_P0_PARITY',
              'rows': rows, 'runtime_dtype': dtype_rows,
              'actual_model_capacity': 'NOT_RUN_NOT_PROVEN_BY_PRIMITIVE',
              'max_allocated_bytes': torch.cuda.max_memory_allocated(),
              'max_reserved_bytes': torch.cuda.max_memory_reserved()}
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'status': report['status'], 'comparisons': len(rows),
                      'failures': sum(r['status'] != 'PASS' for r in rows),
                      'nonbitwise_checks': sum(not r['bitwise_equal'] for r in rows)}))
    return 0 if report['status'].startswith('PASS_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
