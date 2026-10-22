"""The public ``distillctl data`` command family."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
        if name == "validate":
            command.add_argument("--sources", action="store_true", help="also verify original media and encoder source")
    helios = sub.add_parser("prepare-helios-stream", help="prepare explicit original W2 feature cohorts on CPU")
    helios.add_argument("--sources", type=Path, required=True)
    helios.add_argument("--training-config", type=Path, required=True)
    helios.add_argument("--output", type=Path, required=True)
    grid = sub.add_parser("prepare-h3-grid", help="inventory original W5 cached conditions; no data/model download")
    grid.add_argument("--native-config", type=Path, required=True)
    grid.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-helios-stream":
            from distillschedule.search.helios_request import OriginalHeliosRequest
            from distillschedule.training.helios_inputs import HeliosCohortSources, prepare_stream
            values = json.loads(args.sources.read_text())
            if not isinstance(values, list):
                raise ValueError("explicit source cohort list required")
            result = prepare_stream(OriginalHeliosRequest(), [HeliosCohortSources.from_dict(s) for s in values],
                                    args.training_config, args.output)
        elif args.command == "prepare-h3-grid":
            from distillschedule.training.h3_grid_inputs import prepare_inputs
            result = prepare_inputs(args.native_config, args.output)
        elif args.command == "prepare":
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
