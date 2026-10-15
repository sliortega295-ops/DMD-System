#!/usr/bin/env python3
"""Process-local deterministic W6 C1 diagnosis; never a formal timing entry."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def validate_diagnostic_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--one-cycle-correctness', action='store_true')
    parser.add_argument('--export-parity-artifacts', action='store_true')
    parser.add_argument('--receipt-dir', required=True, type=Path)
    args, _ = parser.parse_known_args(argv)
    if args.mode != 'correctness' or not args.one_cycle_correctness or not args.export_parity_artifacts:
        raise ValueError('DETERMINISTIC_DIAGNOSTIC_REQUIRES_ONE_REAL_CORRECTNESS_CYCLE_AND_EVIDENCE')
    return args


def main():
    args = validate_diagnostic_arguments(sys.argv[1:])
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('CUBLAS_WORKSPACE_CONFIG_MUST_BE_SET_BEFORE_PROCESS_START')
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('DIAGNOSTIC_FLAGS_MUST_PRECEDE_CUDA_INITIALIZATION')
    torch.use_deterministic_algorithms(True)
    worker = Path(__file__).with_name('probe_h3_diffsynth_h100_world8.py')
    rank = int(os.environ['RANK'])
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'rank': rank,
        'scope': 'PROCESS_LOCAL_ONE_CYCLE_CORRECTNESS_DIAGNOSTIC_ONLY_NOT_FORMAL_TIMING',
        'torch': torch.__version__, 'CUDA_runtime': torch.version.cuda,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'CUDA_initialized_before_worker': torch.cuda.is_initialized(),
        'CUDA_matmul_allow_TF32': torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'worker_sha256': hashlib.sha256(worker.read_bytes()).hexdigest(),
        'entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'loss_precision_RNG_update_order_changed': False,
    }
    with (args.receipt_dir / f'rank_{rank:02d}_determinism.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
    sys.argv[0] = str(worker)
    runpy.run_path(str(worker), run_name='__main__')


if __name__ == '__main__':
    main()
