"""Command-line interface for validating, inspecting, and launching typed plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from distillschedule.core.config import load_plan
from distillschedule.runtime import DistillExecutor


def _default_repo_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "train_helios.py").is_file() and (candidate / "runtime").is_dir():
            return candidate
    return current


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distillctl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate and compile a plan")
    validate.add_argument("plan")
    validate.add_argument("--repo-root", default=str(_default_repo_root()))
    validate.add_argument("--json", action="store_true", dest="as_json")

    show = subparsers.add_parser("show", help="show the compiled operation sequence")
    show.add_argument("plan")
    show.add_argument("--repo-root", default=str(_default_repo_root()))
    show.add_argument("--graph-json", action="store_true")

    render = subparsers.add_parser("render", help="write the fully resolved plan")
    render.add_argument("plan")
    render.add_argument("--output", required=True)

    run = subparsers.add_parser("run", help="execute a plan through its adapter")
    run.add_argument("plan")
    run.add_argument("--repo-root", default=str(_default_repo_root()))
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--no-check", action="store_true")

    return parser


def _validate(args: argparse.Namespace) -> int:
    loaded = load_plan(args.plan)
    compiled = DistillExecutor().compile(loaded, repo_root=args.repo_root)
    if args.as_json:
        print(json.dumps(compiled.to_dict(), indent=2, sort_keys=True))
    else:
        plan = loaded.plan
        print(f"PASS  plan={plan.plan_id}")
        print(f"      digest={loaded.digest}")
        print(f"      adapter={plan.workload.adapter}")
        print(f"      profile={plan.profile_key}")
        print(f"      range=[{plan.evidence.start_step}, {plan.evidence.start_step + plan.evidence.steps})")
        print(f"      source_files={len(loaded.source_files)}")
        for warning in compiled.adapter_plan.warnings:
            print(f"WARN  {warning}")
    return 0


def _show(args: argparse.Namespace) -> int:
    loaded = load_plan(args.plan)
    compiled = DistillExecutor().compile(loaded, repo_root=args.repo_root)
    protocol = compiled.adapter_plan.protocol
    if args.graph_json:
        print(json.dumps(protocol.graph.to_dict(), indent=2, sort_keys=True))
        return 0
    for step in protocol.steps:
        intent = step.intent
        flags = " ".join(
            name
            for name, enabled in (
                ("G", intent.generator_update),
                ("C", intent.critic_update),
                ("EMA", intent.ema_update),
            )
            if enabled
        )
        print(f"step {intent.logical_step:>4}  {intent.kind.value:<13} {flags}")
        for operation_id in step.operation_ids:
            operation = next(
                item for item in protocol.graph.operations if item.operation_id == operation_id
            )
            print(
                f"    {operation.kind.value:<28} role={operation.role:<10} "
                f"grad={operation.grad_mode.value:<13} state={operation.physical_state or '-'}"
            )
    print(f"final_versions={dict(protocol.final_versions)}")
    if compiled.adapter_plan.invocation is not None:
        print("command=" + " ".join(compiled.adapter_plan.invocation.command))
    return 0


def _render(args: argparse.Namespace) -> int:
    loaded = load_plan(args.plan)
    path = loaded.write_resolved(args.output)
    print(path)
    return 0


def _run(args: argparse.Namespace) -> int:
    loaded = load_plan(args.plan)
    result = DistillExecutor().run(
        loaded,
        repo_root=args.repo_root,
        dry_run=args.dry_run,
        check=not args.no_check,
    )
    print(f"plan={result.plan_id}")
    print(f"output_dir={result.output_dir}")
    if result.command:
        print("command=" + " ".join(result.command))
    print("return_code=dry-run" if result.return_code is None else f"return_code={result.return_code}")
    for warning in result.warnings:
        print(f"warning={warning}")
    return 0 if result.return_code in {None, 0} else int(result.return_code)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:2] == ["workloads", "native"]:
        from distillschedule.native.cli import main as native_main
        return native_main(argv[2:])
    if argv and argv[0] == "workloads":
        from distillschedule.workloads.best_known.cli import main as workloads_main
        return workloads_main(argv[1:])
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "validate": _validate,
        "show": _show,
        "render": _render,
        "run": _run,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
