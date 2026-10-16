"""CPU-only launch gate for exact explicit-request native plans, not templates."""
import argparse
from pathlib import Path

from distillschedule.core.plan import ExecutionPlan
from distillschedule.search.livewan_request import LiveWanRequest, request_from_plan
from distillschedule.workloads.best_known.registry import _object, fingerprint, write_new_json


def checked_launch_plan(path, *, sha256, request):
    raw = _object(Path(path))
    if fingerprint(raw) != sha256 or request_from_plan(raw) != request:
        raise ValueError("LiveWan selected plan/launcher digest or request shape mismatch")
    if request == LiveWanRequest():
        raise ValueError("explicit launch-plan gate requires the bounded 69-frame request")
    normalized = ExecutionPlan.from_mapping(raw).to_dict()
    if normalized != raw:
        raise ValueError("LiveWan launch plan is not the exact resolved native plan")
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    raw = checked_launch_plan(args.input, sha256=args.sha256, request=LiveWanRequest.parse(args.request))
    write_new_json(args.output, raw)


if __name__ == '__main__':
    main()
