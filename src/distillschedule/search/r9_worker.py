"""Permit-owned adapter: validated serialized H3 plan -> original entrypoint."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

from .contracts import require
from .r9_contract import validate_worker
from .r9_dependencies import import_provenance, qualify_toolchain
from distillschedule.workloads.best_known.registry import file_sha256, write_new_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--node', type=int, choices=(0, 1))
    args = parser.parse_args()
    path = args.manifest or Path(os.environ['DMD_SELECTED_PLAN_MANIFEST'])
    digest = args.manifest_sha256 or os.environ['DMD_SELECTED_PLAN_MANIFEST_SHA256']
    require(file_sha256(path) == digest, 'R9 worker manifest SHA')
    cfg = json.loads(path.read_text())
    rank = None if args.validate_only else int(os.environ['RANK'])
    node = args.node if args.validate_only else rank // 8
    if not args.validate_only:
        require(str(path) == cfg['remote_manifest'], 'R9 execution requires sealed manifest')
    plan = validate_worker(cfg, node=node, actual_rank=rank)
    toolchain = qualify_toolchain(os.environ) if args.validate_only else None
    import torch
    require(not torch.cuda.is_initialized(), 'R9 preflight created CUDA context')
    from distillschedule.backends.training.minimax_h3 import trainer, grid_entrypoint
    import diffusers, peft, transformers
    require(not torch.cuda.is_initialized(), 'R9 native imports created CUDA context')
    runtime = dict(torch=torch.__version__, cuda=torch.version.cuda, diffusers=diffusers.__version__,
                   peft=peft.__version__, transformers=transformers.__version__)
    require(runtime == cfg['expected_runtime'], 'R9 runtime changed')
    imports = import_provenance(cfg, node, torch=torch, transformers=transformers, diffusers=diffusers,
                                trainer=trainer, entrypoint=grid_entrypoint)
    evidence = dict(node=node, source_head=cfg['source_head'], candidate_id=cfg['candidate_id'],
                    selected_plan_sha256=cfg['selected_plan']['sha256'], runtime=runtime,
                    import_provenance=imports, mode=plan['mode'], selection=cfg['selection'])
    if args.validate_only:
        print(json.dumps(dict(evidence, status='PASS_H3_SELECTED_PLAN_CPU_NO_GPU',
                              CUDA_initialized=False, toolchain=toolchain)), flush=True)
        return 0
    write_new_json(Path(cfg['run_root']) / 'selected-plan-binding' / f'rank_{rank:03d}.json',
        dict(evidence, status='PASS_SELECTED_PLAN_PREFLIGHT_BEFORE_MODEL', rank=rank,
             selected_plan=cfg['selected_plan'], manifest_sha256=digest,
             transport='SEALED_R9_NODE_OWNER', entrypoint='ORIGINAL_INTEGRATED_ENTRYPOINT'))
    from .resources import ResourceObserver
    from distillschedule.backends.training.minimax_h3.integrated_entrypoint import main as native_main
    # One observer per node, under the already-permitted tracked worker. No
    # remote SSH, CUDA context, barrier, tensor read or training-code patch.
    monitor = (ResourceObserver(Path(cfg['run_root']) / f'search-resources.node{node}.json',
                               cfg['run_id'], node) if rank % 8 == 0 else nullcontext())
    sys.argv = ['integrated_entrypoint', '--config', plan['nodes'][node]['environment']['H3_CONFIG']]
    with monitor:
        native_main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
