"""The public ``distillctl data`` command family."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from distillschedule._cli_streams import redirect_diagnostics

from .dataset import PreparedDataset, inspect_prepared
from .preparation import import_preencoded, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(prog="distillctl data", description="Prepare hash-bound native features offline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "import", "inspect", "validate"):
        command = sub.add_parser(name)
        command.add_argument("manifest", type=Path)
        if name in ("prepare", "import"):
            command.add_argument("--output", type=Path, required=True)
        if name == 'prepare':
            command.add_argument('--preparer', help='explicit trusted module:callable native data adapter')
        if name == "validate":
            command.add_argument("--sources", action="store_true", help="also verify original media and encoder source")
    helios = sub.add_parser("prepare-helios-stream", help="prepare explicit original W2 feature cohorts on CPU")
    helios.add_argument("--sources", type=Path, required=True)
    helios.add_argument("--training-config", type=Path, required=True)
    helios.add_argument("--output", type=Path, required=True)
    gan = sub.add_parser("prepare-helios-gan-stream", help="prepare fresh original W3 world16 GAN cohorts on CPU")
    gan.add_argument("--sources", type=Path, required=True)
    gan.add_argument("--training-config", type=Path, required=True)
    gan.add_argument("--output", type=Path, required=True)
    reference = sub.add_parser("prepare-helios-gan-reference", help="prepare original GAN CPU reference inputs, not W2 traces")
    reference.add_argument("--sources", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    wan = sub.add_parser("prepare-wan-reference", help="prepare original captured W1 CPU1 per-sample reference; not world16 execution")
    wan.add_argument("--sources", type=Path, required=True)
    wan.add_argument("--output", type=Path, required=True)
    nogan = sub.add_parser("prepare-helios-nogan-reference", help="prepare independent original W2 CPU1/rank0 records")
    nogan.add_argument("--sources", type=Path, required=True)
    nogan.add_argument("--output", type=Path, required=True)
    cached = sub.add_parser("prepare-h3-cached-reference", help="prepare original W6 cached AV CPU reference; no decoding/download")
    cached.add_argument("--sources", type=Path, required=True)
    cached.add_argument("--output", type=Path, required=True)
    grid = sub.add_parser("prepare-h3-grid", help="inventory original W5 cached conditions; no data/model download")
    grid.add_argument("--native-config", type=Path, required=True)
    grid.add_argument("--output", type=Path, required=True)
    livewan = sub.add_parser("prepare-livewan-inputs", help="admit finite original W4 teacher inputs; not a training command")
    livewan.add_argument("--data-root", type=Path, required=True)
    livewan.add_argument("--prompts", type=Path, required=True)
    livewan.add_argument("--source-root", type=Path, required=True)
    livewan.add_argument("--cycles", type=int, required=True)
    livewan.add_argument("--seed", type=int, required=True)
    livewan.add_argument("--output", type=Path, required=True)
    livewan.add_argument("--evidence-kind", default="NATIVE_TEACHER_ARCHIVES_UNVERIFIED_QUALITY",
                         choices=("NATIVE_TEACHER_ARCHIVES_UNVERIFIED_QUALITY", "CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED"))
    sub.add_parser("prepare-livewan-c51-inputs", parents=[livewan], add_help=False,
                   help="explicit C51 finite fresh teacher stream (up to 100 cycles); no downloads")
    args = parser.parse_args(argv)
    try:
        # Preparation/import can load trusted encoders and noisy dependencies.
        # Restore stdout even on failure, before emitting the JSON receipt.
        with redirect_diagnostics():
            if args.command == "prepare-helios-stream":
                from distillschedule.search.helios_request import OriginalHeliosRequest
                from distillschedule.training.helios_inputs import HeliosCohortSources, prepare_stream
                values = json.loads(args.sources.read_text())
                if not isinstance(values, list):
                    raise ValueError("explicit source cohort list required")
                result = prepare_stream(OriginalHeliosRequest(), [HeliosCohortSources.from_dict(s) for s in values],
                                        args.training_config, args.output)
            elif args.command == "prepare-helios-gan-stream":
                from distillschedule.training.helios_gan_inputs import HeliosCohortSources, prepare_stream
                values = json.loads(args.sources.read_text())
                if not isinstance(values, list):
                    raise ValueError("explicit source cohort list required")
                result = prepare_stream([HeliosCohortSources.from_dict(s) for s in values], args.training_config, args.output)
            elif args.command == "prepare-helios-gan-reference":
                from .helios_native_inputs import prepare_inputs
                result = inspect_prepared(prepare_inputs(args.sources, args.output))
            elif args.command == "prepare-wan-reference":
                from .wan_reference_inputs import prepare_inputs
                result = inspect_prepared(prepare_inputs(args.sources, args.output))
            elif args.command == "prepare-helios-nogan-reference":
                from .helios_nogan_reference_inputs import prepare_inputs
                result = inspect_prepared(prepare_inputs(args.sources, args.output))
            elif args.command == "prepare-h3-grid":
                from distillschedule.training.h3_grid_inputs import prepare_inputs
                result = prepare_inputs(args.native_config, args.output)
            elif args.command == "prepare-h3-cached-reference":
                from .h3_cached_inputs import prepare_inputs
                result = inspect_prepared(prepare_inputs(args.sources, args.output))
            elif args.command in ("prepare-livewan-inputs", "prepare-livewan-c51-inputs"):
                from .livewan_inputs import prepare_inputs, prepare_c51_inputs
                prepare_inputs = prepare_c51_inputs if args.command == "prepare-livewan-c51-inputs" else prepare_inputs
                result = prepare_inputs(args.data_root, args.prompts, args.output, source_root=args.source_root,
                                        cycles=args.cycles, seed=args.seed, evidence_kind=args.evidence_kind)
            elif args.command == "prepare":
                if args.preparer:
                    from distillschedule.training.plugins import load_callable, resolve_reference
                    reference = resolve_reference(args.preparer, args.manifest.resolve().parent)
                    factory, source = load_callable(reference)
                    manifest = Path(factory(args.manifest, args.output)).resolve()
                    if not manifest.is_relative_to(args.output.resolve()):
                        raise ValueError('native preparer must return its manifest below the requested output')
                    result = {**inspect_prepared(manifest), 'preparer_source': source}
                else:
                    result = inspect_prepared(prepare(args.manifest, args.output))
            elif args.command == "import":
                result = inspect_prepared(import_preencoded(args.manifest, args.output))
            elif args.command == "validate":
                result = PreparedDataset(args.manifest).validate(sources=args.sources)
            else:
                result = inspect_prepared(args.manifest)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
