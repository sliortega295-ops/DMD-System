"""One public entrypoint for existing native families, not a new trainer.

Inventory and assembly are CPU-only. Calibration is source-only unless the
caller explicitly requests native execution; the existing runner owns locks,
matched correctness, source/assets/runtime receipts and teardown.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from distillschedule.workloads.best_known.registry import (
    _object, fingerprint, git_head, load_assets, repository_root, write_new_json, asset_identity,
)
from .contracts import require, semantic_contract
from .h3_implementation import H3, H3Implementation
from .h3_request import H3Request
from .livewan_request import LIVEWAN, LiveWanRequest
from .model_adapters import SUPPORTED_WORKLOADS, adapter_for


def candidate_inventory(workload, *, request=None, include_helios_safe_incumbent=False,
                        include_helios_gan_manual=False):
    require(workload in SUPPORTED_WORKLOADS, "unknown native workload")
    from .wan_implementation import WAN
    from .helios_implementation import NO_GAN, GAN
    require(request is None or workload in (H3,NO_GAN,GAN,LIVEWAN),
            "explicit request currently supported only for H3, ORIGINAL Helios and ORIGINAL LiveWan")
    require(type(include_helios_safe_incumbent) is bool,'safe incumbent opt-in must be boolean')
    require(not include_helios_safe_incumbent or workload==NO_GAN,'safe incumbent opt-in is No-GAN only')
    require(type(include_helios_gan_manual) is bool and (not include_helios_gan_manual or workload==GAN),
            'GAN manual opt-in is GAN only')
    if workload == H3:
        from .h3_implementation import implementations
        values = implementations(request=request or H3Request(), explicit_cache=True)
    elif workload == WAN:
        from .wan_model_adapter import implementations
        values = tuple(replace(i, rollout_window=w) for i in implementations() for w in range(1, 6))
    elif workload in (NO_GAN, GAN):
        from .helios_model_adapter import implementations
        values = implementations(workload,include_safe_incumbent=include_helios_safe_incumbent,
                                  include_gan_manual=include_helios_gan_manual,request=request)
    else:
        from .livewan_implementation import implementations
        values = implementations(request=request)
    from .registry import candidate_profile
    candidates = []
    for impl in values:
        candidate = impl.candidate()
        profile = candidate_profile(candidate)
        candidates.append(dict(id=impl.id, assignment=impl.assignment,
            semantic_fingerprint=fingerprint(semantic_contract(profile)),
            cost_status="MATCHED_NATIVE_CALIBRATION_REQUIRED", gpu_validation="NOT_RUN_ON_THIS_HEAD"))
    adapter_options=({"h3_request":request} if workload==H3 else
                     {"include_safe_incumbent":True} if include_helios_safe_incumbent else {})
    if workload in (NO_GAN,GAN) and request is not None:adapter_options['request']=request
    if include_helios_gan_manual:adapter_options['include_gan_manual']=True
    if workload == LIVEWAN: adapter_options['livewan_request'] = request
    return dict(status="PASS_CPU_CAPABILITY_INVENTORY", workload=workload,
        source_head=git_head(repository_root()), candidates=candidates,
        workload_request=request.to_dict() if request is not None else None,
        native_adapter=adapter_for(workload).capability_report(**adapter_options),
        gpu_execution="NOT_RUN", historical_seconds_are_costs=False, claimable_speedup=False)


def _precision(workload, impl):
    from .wan_implementation import WAN, PRECISION
    from .helios_implementation import NO_GAN, GAN
    if workload == WAN:
        return PRECISION.digest
    if workload in (NO_GAN, GAN):
        from .helios_action_costs import PRECISION
        return PRECISION
    from .native_macro_adapter import precision_contract
    return fingerprint(precision_contract(workload, h3_request=impl.request if workload == H3 else None,
                                         livewan_request=impl.request if workload == LIVEWAN else None))


def calibrate(args):
    from .execution import NativeTrialRunner, resolve_candidate
    from .registry import get_candidate
    from .runtime_identity import observed_binding
    root = repository_root()
    head = git_head(root)
    require(head == args.expected_head, "calibration source HEAD changed")
    require(2 <= len(args.candidate) <= 4 and len(set(args.candidate)) == len(args.candidate),
            "calibration requires 2..4 distinct candidates; first is the matched control")
    adapter = adapter_for(args.workload)
    implementations = [adapter.implementation_type.parse(cid) for cid in args.candidate]
    plans = [resolve_candidate(args.workload, cid, mode="calibration", run_id=args.run_id)
             for cid in args.candidate]
    semantics = {p["workload_semantics_hash"] for p in plans}
    require(len(semantics) == 1, "calibration candidates must share the exact workload request/semantics")
    precisions = {_precision(args.workload, impl) for impl in implementations}
    require(len(precisions) == 1, "calibration precision contracts differ")
    if args.workload == H3:
        require(all(i.rollout_window is None for i in implementations),
                "unified H3 calibration currently uses native macro evidence, not action receipts")
        require(implementations[0].storage == "cpu_staged", "H3 matched control must retain CPU staged cache")
    noise_path = getattr(args, "h3_noise_policy", None)
    noise_policy = _object(noise_path) if noise_path else None
    if noise_policy is not None:
        from .h3_noise_admission import validate_policy
        validate_policy(noise_policy, expected_head=head)
        require(args.workload == H3 and args.candidate ==
                [noise_policy["reference_candidate_id"], noise_policy["candidate_id"]],
                "explicit H3 noise policy requires its exact two cache anchors")
    target = args.output.resolve()
    require(target != root and root not in target.parents, "calibration evidence must be outside source checkout")
    require(not args.execute_native or args.assets_config is not None, "execute-native requires assets-config")
    args.output.mkdir(parents=True, exist_ok=False)
    intent = dict(status="NOT_RUN", source_head=head, workload=args.workload,
        candidates=args.candidate, matched_control=args.candidate[0],
        source_only_plans=plans, calibration_runs=len(plans), correctness_runs="UP_TO_ONE_PER_CANDIDATE",
        formal_timing_attempts=0, calibration_only=True, gpu_execution="NOT_RUN", claimable_speedup=False)
    if noise_policy is not None:
        intent.update(diagnostic_noise_policy=noise_policy,
                      correctness_runs="TWO_CONTROL_PLUS_ONE_CANDIDATE", numerical_status="REVIEW")
        write_new_json(args.output / "diagnostic-noise-policy.json", noise_policy)
    write_new_json(args.output / "intent.json", intent)
    if not args.execute_native:
        return intent
    require(not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip(),
            "native calibration requires a clean committed checkout")
    assets = load_assets(args.assets_config, plans[0]["fully_resolved_profile"])
    identity = dict(source_head=head, asset_fingerprint=fingerprint(asset_identity(assets)),
                    semantic_fingerprint=next(iter(semantics)), precision_fingerprint=next(iter(precisions)))
    runner = NativeTrialRunner(workload=args.workload, identity=identity, assets=assets, output=args.output,
        run_prefix=args.run_id, reference_id=args.candidate[0], bootstrap_peer=args.candidate[1],
        admitted_candidate_ids=set(args.candidate),
        transport_config=_object(args.r9_transport) if getattr(args, 'r9_transport', None) else None,
        h3_noise_policy=noise_policy)
    try:
        probe = runner._rpc("resolve", workload=args.workload, candidate_id=args.candidate[0],
            assets=assets, run_id=args.run_id + "-identity", mode="correctness")
        nodes = [runner._rpc("runtime_identity", _node_rank=n, node_rank=n, receipt=probe) for n in (0, 1)]
        identity.update(observed_binding(nodes, source_head=head))
        write_new_json(args.output / "observed-binding.json", dict(binding=identity, nodes=nodes))
        results = []
        for index, cid in enumerate(args.candidate):
            directory = args.output / f"calibration-{index}"
            runner.calibrate(get_candidate(args.workload, cid), directory=directory)
            results.append(str((directory / "result.json").resolve()))
        return dict(status="PASS_NATIVE_CALIBRATION_COLLECTION", source_head=head,
                    results=results, next_gate="ASSEMBLE_ON_RECEIPT_HOST_THEN_SEARCH",
                    formal_timing_eligible=False, claimable_speedup=False)
    except BaseException as error:
        write_new_json(args.output / "failure.json", dict(status="FAIL", error_type=type(error).__name__,
            error=str(error), source_head=head, retry="PRESERVE_FAILURE; NO_AUTOMATIC_RETRY"))
        raise


def assemble(args):
    from .wan_implementation import WAN
    from .helios_implementation import NO_GAN, GAN, HeliosImplementation
    require(args.workload in SUPPORTED_WORKLOADS, "unknown native workload")
    # Resolve each calibration ID through the requested family before assembly.
    margin_mib = getattr(args, 'helios_allocator_margin_mib', None)
    if margin_mib is not None:
        require(args.workload in (NO_GAN, GAN), 'allocator margin is Helios only')
        require(type(margin_mib) is int and margin_mib >= 0,
                'Helios allocator margin MiB must be a nonnegative integer')
    from .registry import get_candidate
    helios_requests=[]
    for path in args.result:
        row = _object(path)
        get_candidate(args.workload, row["implementation_id"])
        if args.workload in (NO_GAN,GAN):
            helios_requests.append(HeliosImplementation.parse(row['implementation_id']).request)
    if args.workload == WAN:
        from .wan_action_model_adapter import assemble_calibrations
    elif args.workload in (NO_GAN, GAN):
        from .helios_model_adapter import assemble_calibrations
    else:
        from .native_macro_calibration import assemble_calibrations
    safe=getattr(args,'include_helios_safe_incumbent',False)
    gan_manual=getattr(args,'include_helios_gan_manual',False)
    require(not safe or args.workload==NO_GAN,'safe incumbent assembly is No-GAN only')
    options={'include_safe_incumbent':True} if safe else {}
    if margin_mib is not None:options['allocator_margin_bytes']=margin_mib * 2**20
    require(not gan_manual or args.workload==GAN,'GAN manual assembly is GAN only')
    if gan_manual:options['include_gan_manual']=True
    if helios_requests:
        require(all(r==helios_requests[0] for r in helios_requests),'cannot assemble mixed Helios request cohorts')
        if helios_requests[0] is not None:options['request']=helios_requests[0]
    assemble_calibrations(args.result, output=args.output, expected_head=args.expected_head,
                         **options)
    return dict(status="PASS_CPU_CALIBRATION_ASSEMBLY", workload=args.workload,
        problem=str((args.output / "problem.json").resolve()),
        cost_model=str((args.output / "cost-model.json").resolve()),
        gpu_execution="NOT_RUN", claimable_speedup=False)


def main(argv):
    parser = argparse.ArgumentParser(prog="distillctl workloads")
    sub = parser.add_subparsers(dest="action", required=True)
    inventory = sub.add_parser("candidates", help="List actual native implementation bindings, never guessed costs")
    inventory.add_argument("--workload", required=True)
    inventory.add_argument("--request", type=Path, help="Explicit bounded H3, ORIGINAL Helios or ORIGINAL LiveWan request JSON")
    inventory.add_argument("--output", type=Path)
    inventory.add_argument("--include-helios-safe-incumbent", action="store_true",
                           help="Opt-in exact No-GAN HSDP/C512/p0/w5 k-bundle; not GAN")
    inventory.add_argument('--include-helios-gan-manual', action='store_true',
                           help='Opt-in canonical GAN FA3/down40/unmerged manual cohort; fresh costs required')
    collect = sub.add_parser("calibrate", help="Default source-only; explicit native matched-pair calibration")
    collect.add_argument("--workload", required=True)
    collect.add_argument("--candidate", nargs="+", required=True)
    collect.add_argument("--expected-head", required=True)
    collect.add_argument("--run-id", required=True)
    collect.add_argument("--assets-config", type=Path)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--execute-native", action="store_true")
    collect.add_argument("--r9-transport", type=Path, help="Opt-in SHA-bound reviewed R9 deployment; H3 cache pair only")
    collect.add_argument("--h3-noise-policy", type=Path,
                         help="Opt-in source-bound cache-only planner diagnostic; requires a control repeat")
    join = sub.add_parser("assemble-calibration", help="Verify and join native calibration files beside their receipts")
    join.add_argument("--workload", required=True)
    join.add_argument("--result", nargs="+", type=Path, required=True)
    join.add_argument("--expected-head", required=True)
    join.add_argument("--output", type=Path, required=True)
    join.add_argument("--include-helios-safe-incumbent", action="store_true",
                      help="Bind measured FSDP C512 k0 baseline and HSDP C512 k1 incumbent seeds")
    join.add_argument('--include-helios-gan-manual', action='store_true',
                      help='Bind measured canonical GAN baseline and complete manual incumbent seeds')
    join.add_argument('--helios-allocator-margin-mib', type=int,
                      help='Explicit Helios allocator margin in MiB; omitted keeps exactly 2048 MiB. Independent of driver-free reserve.')
    args = parser.parse_args(argv)
    try:
        if args.action == "candidates":
            request=None
            if args.request:
                from .helios_implementation import NO_GAN,GAN
                from .helios_request import OriginalHeliosRequest
                require(args.workload in (H3,NO_GAN,GAN,LIVEWAN),'request JSON not supported for this workload')
                Request = H3Request if args.workload==H3 else LiveWanRequest if args.workload==LIVEWAN else OriginalHeliosRequest
                request=Request.from_dict(_object(args.request))
            result = candidate_inventory(args.workload,
                request=request,
                include_helios_safe_incumbent=args.include_helios_safe_incumbent,
                include_helios_gan_manual=args.include_helios_gan_manual)
            if args.output:
                write_new_json(args.output, result)
        else:
            result = calibrate(args) if args.action == "calibrate" else assemble(args)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
