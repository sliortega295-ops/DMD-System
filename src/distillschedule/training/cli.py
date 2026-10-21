"""Install -> prepare -> train -> resume -> generate/export public commands."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from .config import TrainingConfig, load_training_config


def doctor(config=None):
    packages = {}
    for name in ("distillschedule", "torch", "numpy", "Pillow", "omegaconf", "diffusers", "deepspeed"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"status": "INSPECTED", "python": platform.python_version(),
              "executable": sys.executable, "packages": packages,
              "model_imported": False, "cuda_initialized": False, "gpu_execution": "NOT_RUN"}
    if config is not None:
        if config.execution.get("adapter", "program") == "native":
            from .native import describe_native
            result["training"] = describe_native(config)
        else:
            checks = {"prepared_manifest": Path(config.data["prepared"]).is_file()}
            for section in ("model", "algorithm"):
                reference = getattr(config, section)["factory"]
                path = reference.rpartition(":")[0]
                if path.endswith(".py"):
                    checks[section + "_source"] = Path(path).is_file()
            result["training"] = {"adapter": "program", "identity": config.identity,
                                  "mode": config.execution.get("mode", "reference"),
                                  "checks": checks, "model_factory": config.model["factory"],
                                  "algorithm_factory": config.algorithm["factory"],
                                  "factory_code": "NOT_IMPORTED", "resolved": config.to_dict()}
            result["status"] = "READY_FOR_EXECUTION_CHECKS" if all(checks.values()) else "MISSING_INPUTS"
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "data":
        from distillschedule.data.cli import main as data_main
        return data_main(argv[1:])
    parser = argparse.ArgumentParser(prog="distillctl",
        description="DMD-System data, continuous training, checkpoint and export lifecycle")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("doctor", help="inspect installed dependencies without loading models")
    check.add_argument("--config", type=Path)
    config_parser = sub.add_parser("config", help="validate or render a training configuration")
    config_parser.add_argument("action", choices=("validate", "show"))
    config_parser.add_argument("config", type=Path)
    for name in ("train", "evaluate", "export"):
        p = sub.add_parser(name)
        p.add_argument("config", type=Path)
        p.add_argument("--output", type=Path)
        if name == "train":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--resume", type=Path)
            p.add_argument("--cycles", type=int)
        else:
            p.add_argument("--checkpoint", type=Path, required=True)
        if name == "evaluate":
            p.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        config = load_training_config(args.config) if getattr(args, "config", None) else None
        if args.command == "doctor":
            result = doctor(config)
        elif args.command == "config":
            result = {"status": "VALID_CONFIG", "identity": config.identity,
                      "resolved": config.to_dict(), "gpu_execution": "NOT_RUN"}
        elif args.command == "train":
            raw = config.to_dict()
            for key, value in (("output_dir", args.output), ("resume_from", args.resume), ("cycles", args.cycles)):
                if value is not None:
                    raw["trainer"][key] = str(value.resolve()) if isinstance(value, Path) else value
            config = TrainingConfig.from_dict(raw, source=config.source)
            if config.execution.get("adapter", "program") == "native":
                from .native import run_native
                result = run_native(config, dry_run=args.dry_run)
            elif args.dry_run:
                result = doctor(config)
                result["training_execution"] = "NOT_RUN_DRY_RUN"
            else:
                from .session import train
                result = train(config)
        else:
            if not args.output:
                raise ValueError("generation/export requires --output")
            if config.execution.get("adapter", "program") == "native":
                raise ValueError("use the native recipe's documented checkpoint/generation/export capability")
            from .session import evaluate, export
            result = (evaluate(config, args.checkpoint, args.output, sample_index=args.sample_index)
                      if args.command == "evaluate" else export(config, args.checkpoint, args.output))
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 2 if result.get("status") in {"MISSING_INPUTS", "FAILED"} else 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
