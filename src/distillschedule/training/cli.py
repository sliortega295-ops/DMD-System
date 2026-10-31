"""Install -> prepare -> train -> resume -> generate/export public commands."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from distillschedule._cli_streams import redirect_diagnostics

from .config import TrainingConfig, load_training_config


def doctor(config=None):
    packages = {}
    for name in ("distillschedule", "torch", "numpy", "Pillow", "omegaconf", "diffusers", "deepspeed",
                 "transformers", "peft", "sentencepiece", "torchaudio", "safetensors",
                 "tensorflow", "tensorflow-cpu"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    controls = {name: os.environ.get(name) for name in ("USE_TF", "USE_FLAX")}
    warnings = []
    if any(packages[name] is not None for name in ("tensorflow", "tensorflow-cpu")) and controls["USE_TF"] != "0":
        warnings.append({"code": "OPTIONAL_TENSORFLOW_IMPORTS", "message":
            "For PyTorch-only recipes, set USE_TF=0 and USE_FLAX=0 before preparing assets and launching workers. "
            "Unused TensorFlow/Keras/OpenCV imports can change library paths during model setup."})
    result = {"status": "INSPECTED", "python": platform.python_version(),
              "executable": sys.executable, "packages": packages,
              "environment_controls": controls, "import_warnings": warnings,
              "model_imported": False, "cuda_initialized": False, "gpu_execution": "NOT_RUN"}
    if config is not None:
        if config.execution.get("adapter", "program") == "native":
            from .native import describe_native
            result["training"] = describe_native(config)
        else:
            checks = ({'prepared_' + name: Path(stream['prepared']).is_file()
                       for name, stream in config.data['streams'].items()} if 'streams' in config.data else
                      {"prepared_manifest": Path(config.data["prepared"]).is_file()})
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


def _generation_parameters_file(path):
    if path is None:
        return None
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate generation parameter: ' + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError('generation parameters must be a JSON object')
    from .session import _generation_parameters
    return _generation_parameters(value)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "data":
        from distillschedule.data.cli import main as data_main
        return data_main(argv[1:])
    parser = argparse.ArgumentParser(prog="distillctl",
        description="DMD-System data, continuous training, checkpoint and export lifecycle")
    sub = parser.add_subparsers(dest="command", required=True)
    for kind in ('models', 'algorithms', 'strategies', 'capabilities'):
        catalog = sub.add_parser(kind, help='discover adapters and capabilities without importing model code')
        catalog.add_argument('action', choices=('list', 'show'), nargs='?', default='list')
        catalog.add_argument('name', nargs='?')
    check = sub.add_parser("doctor", help="inspect installed dependencies without loading models")
    check.add_argument("--config", type=Path)
    config_parser = sub.add_parser("config", help="validate or render a training configuration")
    config_parser.add_argument("action", choices=("validate", "show"))
    config_parser.add_argument("config", type=Path)
    initialize = sub.add_parser('import-weights', help='initialize Generator weights and export; never resume/train')
    initialize.add_argument('request', type=Path)
    initialize.add_argument('--output', type=Path, required=True)
    native_output = sub.add_parser('native-output', help='inspect or explicitly reproduce R375 C50 through its original node owner')
    native_output.add_argument('request', type=Path)
    native_output.add_argument('--node', type=int, choices=(0, 1))
    native_output.add_argument('--execute', action='store_true', help='exec the bound ORIGINAL owner; never a direct rank launch')
    for name in ("train", "evaluate", "export", "generate"):
        p = sub.add_parser(name)
        p.add_argument("config", type=Path)
        p.add_argument("--output", type=Path)
        if name == "train":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--resume", type=Path)
            p.add_argument("--cycles", type=int)
        elif name == "generate":
            p.add_argument("--weights", type=Path, required=True)
            p.add_argument("--prepared", type=Path, required=True)
            p.add_argument("--output-kind", choices=('image', 'video', 'audio', 'audio_video', 'latents'), default="video",
                           help="video requires a real decoder; latents is an explicit native diagnostic")
        else:
            p.add_argument("--checkpoint", type=Path, required=True)
        if name in ("evaluate", "generate"):
            p.add_argument("--sample-index", type=int, default=0)
            p.add_argument('--parameters', type=Path, help='explicit finite JSON generation parameters')
        if name == 'evaluate':
            p.add_argument('--prepared', type=Path, help='explicit physical evaluation inputs')
            p.add_argument('--stream', help='physical training input stream (default generator when present)')
    decoder = sub.add_parser("decode", help="decode prepared latents with an explicit native codec")
    decoder.add_argument("config", type=Path)
    decoder.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = (load_training_config(args.config) if getattr(args, "config", None) and args.command != "decode" else None)
        if args.command == 'native-output':
            from .h3_native_output_reproduction import dispatch
            result = dispatch(args.request, execute=args.execute, node=args.node)
        elif args.command == "decode":
            with redirect_diagnostics():
                from .decoding import decode_prepared
                result = decode_prepared(args.config, args.output)
        elif args.command in ('models', 'algorithms'):
            from .catalog import describe, resolve
            if args.action == 'show':
                if not args.name:
                    raise ValueError('show requires a registered name')
                result = resolve('model' if args.command == 'models' else 'algorithm', args.name)
            else:
                result = describe()[args.command]
        elif args.command in ('strategies', 'capabilities'):
            from .strategy_catalog import describe, resolve
            if args.action == 'show':
                if not args.name:
                    raise ValueError('show requires a registered name')
                result = resolve('strategy' if args.command == 'strategies' else 'capability', args.name)
            else:
                result = describe()[args.command]
        elif args.command == "doctor":
            result = doctor(config)
        elif args.command == "config":
            result = {"status": "VALID_CONFIG", "identity": config.identity,
                      "resolved": config.to_dict(), "gpu_execution": "NOT_RUN"}
        elif args.command == 'import-weights':
            with redirect_diagnostics():
                from . import wan_generator_import
                request = wan_generator_import.read_request(args.request)
                schema = request.get('schema_version') if isinstance(request, dict) else None
                if schema == wan_generator_import.SCHEMA:
                    initialize = wan_generator_import.import_generator_weights
                elif schema == 'distillschedule.livewan_generator_weights_initialization.v1':
                    from .livewan_generator_import import import_generator_weights as initialize
                else:
                    raise ValueError('unsupported weights initialization schema')
                result = initialize(args.request, args.output)
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
                # Trusted adapters and their dependencies may log on import as
                # well as during execution. Only the final receipt owns stdout.
                with redirect_diagnostics():
                    from .session import train
                    result = train(config)
        else:
            if not args.output:
                raise ValueError("generation/export requires --output")
            if config.execution.get("adapter", "program") == "native":
                if args.command != "export":
                    raise ValueError("native outputs require export first, then the standalone Program inference config")
                with redirect_diagnostics():
                    from .native import export_native
                    result = export_native(config, args.checkpoint, args.output)
            else:
                with redirect_diagnostics():
                    from .session import evaluate, export, generate_from_export
                    if args.command == "generate":
                        result = generate_from_export(config, args.weights, args.output,
                                                      prepared=args.prepared, sample_index=args.sample_index,
                                                      parameters=_generation_parameters_file(args.parameters),
                                                      output_kind=args.output_kind)
                    elif args.command == "evaluate":
                        result = evaluate(config, args.checkpoint, args.output, sample_index=args.sample_index,
                                          parameters=_generation_parameters_file(args.parameters),
                                          prepared=args.prepared, stream=args.stream)
                    else:
                        result = export(config, args.checkpoint, args.output)
        if type(result) is int:
            # Native front doors already emit their own original receipts.
            return result
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 2 if result.get("status") in {"MISSING_INPUTS", "FAILED"} else 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
