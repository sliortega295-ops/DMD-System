#!/usr/bin/env python3
"""Source/config/asset validation only, without constructing a CUDA model."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def main():
    root = Path(os.environ['DMD_SYSTEM_ROOT'])
    output = Path(os.environ['DISTILLSCHEDULE_OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=False)
    expected = '0997fbe636eb39b8b50d57e28f87f2446cf439e3'
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == expected
    assert not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    command = ['bash', str(root / 'scripts/distillschedule/prepare_livewan_crossscale_source.sh'),
               '--output', os.environ['LIVEWAN_WAN21_SOURCE'],
               '--receipt', str(output / 'source_receipt.generated.json')]
    with (output / 'source_prepare.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    from distillschedule.runtime.asset_relocation import verify_relocated_source_receipt
    registered = json.loads(Path(os.environ['LIVEWAN_SOURCE_RECEIPT']).read_text())
    generated = json.loads((output / 'source_receipt.generated.json').read_text())
    verify_relocated_source_receipt(registered, generated)
    import torch
    import diffusers
    import transformers
    import peft
    from torch.distributed.fsdp import fully_shard
    # Upstream wan package imports T5EncoderModel with a CUDA-default
    # argument. Its real imports belong inside the locked GPU worker gate.
    assert not torch.cuda.is_initialized(), 'CPU preflight created a CUDA context'
    from distillschedule.core.config import load_plan
    from distillschedule.runtime import livewan_crossscale_worker as worker
    source_files = {
        'baseline': 'livewan_wan13b_14b_sf_dmd_g1c5_world16_fsdp2.json',
        'pipeline': 'livewan_wan13b_14b_sf_dmd_c34_continuous_b16_residual_layout.json',
    }
    validations = {}
    for mode, filename in source_files.items():
        plan = load_plan(root / 'configs/distillschedule/plans' / filename).plan.to_dict()
        worker._apply_trajectory_contract(plan, cycles=2, mode=mode)
        if mode == 'baseline':
            from distillschedule.runtime.livewan_baseline_placement import apply_contract
            spec = apply_contract(plan,mode=mode,name='full-resident',teacher_offload='off')
            assert spec.receipt()['gradient_mean'] == 'SUM_THEN_DIVIDE_8'
        worker._validate_plan_identity(plan, mode=mode)
        trace = worker.FormalTrace.load(plan['evidence']['operation_trace'], expected_cycles=2)
        worker._validate_trace_balanced_pairing(plan, trace)
        bootstrap = worker.Bootstrap.load(plan['evidence']['checkpoint'], trace=trace)
        source = worker.validate_source_receipt(bootstrap, repo_root=root)
        worker.configure_python_sources(repo_root=root, source_receipt=source)
        started = time.monotonic()
        result = worker._validate_only(plan=plan, trace=trace, bootstrap=bootstrap,
                                       source_receipt=source, mode=mode)
        result['wall_seconds'] = time.monotonic() - started
        result['physical_role_ranks'] = worker._plan_role_ranks(plan, mode=mode)
        state = bootstrap.value['student_state']
        from distillschedule.runtime.asset_relocation import relocate_asset_path
        state_path = relocate_asset_path(state['path'])
        assert state_path.stat().st_size == state['bytes']
        assert worker.sha256_file(state_path) == state['sha256']
        result['checkpoint_state_status'] = 'PASS_FULL_SHA'
        validations[mode] = result
        (output / f'resolved_{mode}.json').write_text(json.dumps(plan, indent=2) + '\n')
    value = {
        'run_id': 'livewan_h100_resident_CPU_preflight_20261014_r1',
        'status': 'PASS_ENV_CONFIG_TRACE_CHECKPOINT_CONTENT',
        'gpu_workload': 'NOT_RUN',
        'student_state_capacity': 'NOT_RUN_MODEL_NOT_CONSTRUCTED',
        'wan_architecture_import': 'NOT_RUN_DEFERRED_TO_LOCKED_WORKER',
        'source_head': head, 'host': socket.gethostname(),
        'python': os.sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'diffusers': diffusers.__version__, 'transformers': transformers.__version__,
        'peft': peft.__version__, 'cuda_context_created': torch.cuda.is_initialized(),
        'relocation_sha256': os.environ['DISTILLSCHEDULE_ASSET_RELOCATION_SHA256'],
        'registered_source_receipt_sha256': registered['source_receipt_sha256'],
        'generated_source_receipt_sha256': generated['source_receipt_sha256'],
        'source_path_only_relocation': 'PASS', 'validations': validations,
        'claim_boundary': 'CPU environment/config/content preflight; no real model or performance result.',
    }
    assert not value['cuda_context_created']
    (output / 'receipt.json').write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(value, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
