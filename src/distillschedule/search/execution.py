"""Compile registered recipes through the existing dispatcher and native gates.

No generic shell-command executor and no exit-code -> PASS shortcut. Workload
adapters must provide source-bound native correctness/measurement receipts.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from distillschedule.workloads.best_known.registry import (
    fingerprint, git_head, load_profile, repository_root, resolve_configuration,
    write_new_json, file_sha256, asset_identity, validate_resolved_asset_binding, validate_request_configuration,
)
from .contracts import Candidate, require, semantic_contract, validate_measurement
from .registry import WAN, candidate_profile, get_candidate


def resolve_candidate(workload: str, candidate_id: str, *, root: Path | None = None,
                      assets_path: Path | None = None, run_id: str = "search-inspection",
                      mode: str = "correctness") -> dict[str, Any]:
    root = root or repository_root()
    c = get_candidate(workload, candidate_id)
    p = candidate_profile(c, root=root)
    receipt = resolve_configuration(p, root=root, assets_path=assets_path, run_id=run_id, mode=mode)
    receipt.update(canonical_status="REGISTERED_SEARCH_CANDIDATE_NOT_CANONICAL_WINNER",
                   canonical_scope="SEARCH_RECIPE_BOUND_TO_FROZEN_WORKLOAD",
                   search_candidate=c.to_dict(),
                   canonical_profile_sha256=fingerprint(load_profile(workload, root=root)),
                   workload_semantics_hash=fingerprint(semantic_contract(p)),
                   execution_fingerprint=fingerprint(c.to_dict()))
    if "planner_implementation" in p["runtime"]:
        receipt.update(canonical_status="GENERATED_IMPLEMENTATION_NOT_CANONICAL_WINNER",
                       canonical_scope="COMPOSED_PLACEMENT_KERNEL_PARAMETERS_BOUND_TO_FROZEN_WORKLOAD")
    return receipt


def validated_profile(receipt: dict[str, Any], *, root: Path) -> dict[str, Any]:
    if "model_plan" in receipt:
        require(receipt["model_plan"]["problem"].get("workload") == receipt["profile_id"],
                "selected planning problem and native dispatch workload differ")
        from .model_adapters import adapter_for
        adapter_for(receipt["profile_id"]).validate_lowered_plan(receipt)
    c = get_candidate(receipt["profile_id"], receipt["search_candidate"]["id"])
    require(fingerprint(receipt["search_candidate"]) == fingerprint(c.to_dict()), "search candidate changed")
    p = candidate_profile(c, root=root)
    require(receipt.get("canonical_profile_sha256") == fingerprint(load_profile(c.workload, root=root)), "canonical source profile changed")
    require(receipt.get("profile_sha256") == fingerprint(p), "search recipe/profile changed")
    require(receipt.get("workload_semantics_hash") == fingerprint(semantic_contract(p)), "search semantics changed")
    require(receipt.get("execution_fingerprint") == fingerprint(c.to_dict()), "search execution identity changed")
    require(fingerprint(receipt["fully_resolved_profile"]) == fingerprint(p), "serialized profile was altered")
    if "asset_config" in receipt:
        validate_resolved_asset_binding(receipt, p)
        validate_request_configuration(receipt, p, root=root)
    elif c.workload == "livewan_crossscale_pipeline_best" and "request" in p["contract"]:
        rebuilt = resolve_configuration(p, root=root, run_id=receipt["run_id"], mode=receipt["mode"])
        require(receipt.get("native_plan") == rebuilt["native_plan"], "LiveWan request native plan changed")
    return p


def run_plan(receipt: dict[str, Any], *, dry_run: bool = False,
             coordinator_output: Path | None = None,
             transport_config: dict[str, Any] | None = None) -> int:
    root = repository_root()
    if transport_config is not None:
        from .r9_transport import run_r9_plan
        return run_r9_plan(receipt, transport_config, output=coordinator_output, dry_run=dry_run)
    require('execution_transport' not in receipt, 'explicit transport plan cannot fall back to default dispatch')
    validated_profile(receipt, root=root)
    require(receipt.get("integration_head") == git_head(root), "search plan belongs to another HEAD")
    c = get_candidate(receipt["profile_id"], receipt["search_candidate"]["id"])
    if not dry_run:
        require(c.executable, "candidate is CPU-registered only")
        # Fail closed until the native two-cycle gate/launcher is integrated.
        # This also protects correctness runs from accidentally executing C50.
        if c.workload == WAN:
            require(receipt["fully_resolved_profile"]["runtime"].get("search_two_cycle_ready") is True,
                    "WAN_TWO_CYCLE_BRIDGE_NOT_READY: cannot launch the one-cycle worker as cycle2")
    from distillschedule.workloads.best_known.dispatch import run_resolved
    return run_resolved(receipt, dry_run=dry_run, coordinator_output=coordinator_output)


def rebind_plan(receipt: dict[str, Any], *, run_id: str,
                coordinator_output: Path, assets_path: Path | None = None) -> dict[str, Any]:
    """Replay the registered recipe with a fresh run ID, never an old output.

    Only run identity changes. Assets, native correctness gate, source and
    recipe must match the searched plan exactly.
    """
    from distillschedule.workloads.best_known.registry import load_assets
    require('execution_transport' not in receipt, 'explicit transport plan requires its matching replay adapter')
    root = repository_root()
    p = validated_profile(receipt, root=root)
    require(receipt["integration_head"] == git_head(root), "search plan belongs to another HEAD")
    require(run_id != receipt["run_id"] and run_id != "source-inspection", "a fresh explicit run-id is required")
    assets = load_assets(assets_path, p) if assets_path is not None else receipt["asset_config"]
    require(assets == receipt["asset_config"], "replay assets/gates changed")
    from .registry import candidates
    from .wan_implementation import is_generated, WanImplementation
    from .livewan_implementation import PREFIX as LIVEWAN_PREFIX, LiveWanImplementation
    from .h3_implementation import H3Implementation
    candidate_id = receipt["search_candidate"]["id"]
    if "model_plan" in receipt:
        from .model_adapters import adapter_for
        from .plan_ir import PlanningProblem
        # validated_profile has checked the serialized graph, calibration and
        # selected implementation. Preserve its calibrated anchor pair too.
        problem = PlanningProblem.from_dict(receipt["model_plan"]["problem"])
        Implementation = adapter_for(receipt["profile_id"]).implementation_type
        require(len(problem.seeds) == 2 and problem.seeds[0] != problem.seeds[1],
                "exactly two distinct baseline/incumbent seeds required")
        reference_id, peer = (Implementation.from_assignment(seed).id for seed in problem.seeds)
    elif is_generated(candidate_id):
        window = 5 if WanImplementation.parse(candidate_id).rollout_window is not None else None
        reference_id, peer = WanImplementation("fsdp", 512, False, window).id, WanImplementation("resident", 2048, True, window).id
    elif candidate_id.startswith(LIVEWAN_PREFIX):
        request = LiveWanImplementation.parse(candidate_id).request
        reference_id, peer = LiveWanImplementation("fsdp", request=request).id, LiveWanImplementation("pipeline", request=request).id
    elif candidate_id.startswith("joint-h3"):
        from dataclasses import replace
        impl = H3Implementation.parse(candidate_id)
        reference_id = replace(impl, prefetch_depth=2, lora_epilogue_block_n=128,
            cache_policy="cpu_staged_trim" if impl.cache_policy != "legacy_cpu_staged" else impl.cache_policy).id
        peer = (replace(impl, prefetch_depth=1, lora_epilogue_block_n=128).id
                if impl.cache_policy == "legacy_cpu_staged" or impl.id == reference_id else impl.id)
    elif candidate_id.startswith("joint-helios1-"):
        from .helios_action_costs import anchors
        from .helios_implementation import HeliosImplementation
        request = HeliosImplementation.parse(candidate_id).request
        reference_id, peer = (i.id for i in anchors(receipt["profile_id"], request=request))
    else:
        reference_id = "fsdp-reference"
        peer = next(c.id for c in candidates(receipt["profile_id"]) if c.executable and c.id != reference_id)
    coordinator_output.mkdir(parents=True, exist_ok=False)
    runner = NativeTrialRunner(workload=receipt["profile_id"], assets=assets,
        identity={"source_head": receipt["integration_head"], "asset_fingerprint": receipt["asset_fingerprint"]},
        output=coordinator_output, run_prefix=run_id, bootstrap_peer=peer,
        reference_id=reference_id)
    # Resolve beside the immutable source receipts; those remote asset files
    # are deliberately not downloaded onto the controller machine.
    result = runner._rpc("resolve", workload=receipt["profile_id"], candidate_id=receipt["search_candidate"]["id"],
                         assets=assets, run_id=run_id, mode=receipt["mode"])
    if "model_plan" in receipt:
        result["model_plan"] = receipt["model_plan"]
        validated_profile(result, root=root)
    write_new_json(coordinator_output / "execution-plan.json", result)
    return result


def native_rpc(*, transport_assets, source_head, output: Path, action: str, _node_rank: int = 0, **kwargs):
    import base64
    import shlex
    import subprocess
    import zlib
    from distillschedule.workloads.best_known.dispatch import ssh_command, ssh_process_environment
    require(type(_node_rank) is int and _node_rank in (0, 1), "invalid RPC node rank")
    node = transport_assets["nodes"][_node_rank]
    python = str(Path(node["python_env"]) / "bin/python")
    worker = str(Path(node["repo_root"]) / "scripts/distillschedule/run_best_known_workload.py")
    transport = node.get("rpc_transport", "stdin")
    require(transport in {"stdin", "argv-json-tty-v1"}, "unsupported metadata RPC transport")
    payload = {"action": action, "source_head": source_head, **kwargs}
    stdin_payload = json.dumps(payload)
    environment = ["env", "PYTHONDONTWRITEBYTECODE=1",
                   "LD_LIBRARY_PATH=" + str(Path(node["python_env"]) / "lib"), python]
    if transport == "argv-json-tty-v1":
        # Some allocated-node gateways keep forwarded stdin open. Never pipe
        # JSON into a PTY: carry this bounded CPU request in argv instead, and
        # isolate its result from banners/CRLF with a single framed response.
        encoded = base64.b64encode(zlib.compress(stdin_payload.encode())).decode()
        require(len(encoded) <= 100_000, "metadata RPC payload exceeds argv transport bound")
        boot = ("import base64,contextlib,io,runpy,sys,zlib\n"
                f"sys.argv=[{worker!r},'_search']\n"
                f"sys.stdin=io.StringIO(zlib.decompress(base64.b64decode({encoded!r})).decode())\n"
                "result=io.StringIO()\n"
                "with contextlib.redirect_stdout(result):\n"
                " try:\n"
                f"  runpy.run_path({worker!r},run_name='__main__')\n"
                " except SystemExit as error:\n"
                "  if error.code not in (None,0): raise\n"
                "print('DMD_NATIVE_RPC_JSON='+base64.b64encode(result.getvalue().encode()).decode())\n")
        ssh = ssh_command(node)
        command = [*ssh[:-1], "-n", "-tt", ssh[-1],
                   shlex.join([*environment, "-c", boot]) + "; exit"]
        stdin_payload = None
    else:
        command = [*ssh_command(node), shlex.join([*environment, worker, "_search"])]
    import time
    logs = output / "rpc"
    logs.mkdir(exist_ok=True)
    directory = logs / f"{time.time_ns()}-{action}"
    directory.mkdir(exist_ok=False)
    write_new_json(directory / "request.json", {"command": command, "payload": payload,
                                              "rpc_transport": transport})
    try:
        response = subprocess.run(command, input=stdin_payload, capture_output=True,
                                  text=True, timeout=900, env=ssh_process_environment())
    except subprocess.TimeoutExpired as error:
        write_new_json(directory / "failure.json", {"status": "FAIL", "failure_type": "TIMEOUT"})
        for name, data in (("stdout", error.stdout), ("stderr", error.stderr)):
            (directory / f"{name}.log").write_bytes(data.encode() if isinstance(data, str) else (data or b""))
        raise
    (directory / "stdout.log").write_text(response.stdout)
    (directory / "stderr.log").write_text(response.stderr)
    write_new_json(directory / "exit.json", {"returncode": response.returncode})
    if response.returncode:
        raise RuntimeError(f"native {action} failed ({response.returncode}); evidence: {directory}: {response.stderr[-6000:]}")
    if transport == "argv-json-tty-v1":
        frames = [line.split("=", 1)[1] for line in response.stdout.splitlines()
                  if line.startswith("DMD_NATIVE_RPC_JSON=")]
        require(len(frames) == 1, "metadata RPC must return exactly one framed result")
        return json.loads(base64.b64decode(frames[0], validate=True))
    return json.loads(response.stdout)


class NativeTrialRunner:
    """Serial, source-bound native correctness -> measurement controller.

    RPC is only CPU metadata work beside persistent remote assets; GPU workers
    still go through the existing two-node dispatcher and native node locks.
    Preparation is cached immutably. Interrupted run IDs are never retried.
    """
    def __init__(self, *, workload: str, identity: dict[str, str], assets: dict[str, Any],
                 output: Path, run_prefix: str, bootstrap_peer: str,
                 reference_id: str = "fsdp-reference",
                 admitted_candidate_ids: set[str] | None = None,
                 transport_config: dict[str, Any] | None = None,
                 h3_noise_policy: dict[str, Any] | None = None):
        self.transport_config = transport_config
        self.h3_noise_policy = copy.deepcopy(h3_noise_policy)
        self.workload, self.identity, self.assets = workload, identity, assets
        self.output, self.run_prefix = Path(output), run_prefix
        self.admitted_candidate_ids = None if admitted_candidate_ids is None else frozenset(admitted_candidate_ids)
        require(self.admitted_candidate_ids is None or reference_id in self.admitted_candidate_ids,
                "native baseline was not admitted by prediction")
        self.bootstrap_peer = get_candidate(workload, bootstrap_peer)
        self.reference = get_candidate(workload, reference_id)
        require(self.reference.placement in {"world16-fsdp", "world16-fsdp2"}, "reference must use world16 FSDP")
        require(self.bootstrap_peer.executable and bootstrap_peer != reference_id, "invalid bootstrap peer")
        from .wan_implementation import is_generated
        require(is_generated(self.reference.id) == is_generated(self.bootstrap_peer.id),
                "cannot mix generated precision contract and legacy search arms")
        for family in ("joint-livewan1-", "joint-h3", "joint-helios1-"):
            require(self.reference.id.startswith(family) == self.bootstrap_peer.id.startswith(family),
                    "cannot mix generated native family and legacy arms")
        if self.reference.id.startswith("joint-livewan1-"):
            from .livewan_implementation import LiveWanImplementation
            require(LiveWanImplementation.parse(self.reference.id).request ==
                    LiveWanImplementation.parse(self.bootstrap_peer.id).request,
                    "LiveWan anchor pair must share request/shape")
        if self.reference.id.startswith("joint-h3"):
            from .h3_implementation import H3Implementation
            ref, peer = (H3Implementation.parse(c.id) for c in (self.reference, self.bootstrap_peer))
            require(ref.request == peer.request and (ref.rollout_window is None) == (peer.rollout_window is None)
                    and (ref.cache_policy == "legacy_cpu_staged") == (peer.cache_policy == "legacy_cpu_staged"),
                    "H3 anchor pair must share request/shape and macro/action evidence contract")
        require(fingerprint(asset_identity(assets)) == identity["asset_fingerprint"], "asset identity mismatch")
        if self.h3_noise_policy is not None:
            from .h3_noise_admission import validate_policy
            validate_policy(self.h3_noise_policy, expected_head=identity["source_head"])
            require(workload == "minimax_h3_grid1000_best"
                    and self.h3_noise_policy["reference_candidate_id"] == reference_id
                    and self.h3_noise_policy["candidate_id"] == bootstrap_peer,
                    "H3 noise policy does not bind the declared anchor pair")
        self.prepared: dict[str, dict[str, Any]] = {}
        self.trajectories: dict[str, dict[str, Any]] = {}

    def _rpc(self, action: str, *, _node_rank: int = 0, **kwargs: Any) -> dict[str, Any]:
        return native_rpc(transport_assets=self.assets, source_head=self.identity["source_head"],
            output=self.output, action=action, _node_rank=_node_rank, **kwargs)

    def _require_admitted(self, candidate: Candidate) -> None:
        require(self.admitted_candidate_ids is None or candidate.id in self.admitted_candidate_ids,
                f"candidate {candidate.id} rejected by prediction; current source-bound preparation required")

    def _trajectory(self, c: Candidate, *, noise_repeat: bool = False) -> dict[str, Any]:
        key = c.id + ("-control-noise-repeat" if noise_repeat else "")
        if key in self.trajectories:
            return self.trajectories[key]
        directory = self.output / "preparation" / key
        completed = directory / "completed-plan.json"
        if completed.exists():
            plan = json.loads(completed.read_text())
            validated_profile(plan, root=repository_root())
            require(plan["integration_head"] == self.identity["source_head"], "old preparation source")
            require(plan["asset_fingerprint"] == self.identity["asset_fingerprint"], "old preparation assets")
        else:
            self._require_admitted(c)
            directory.mkdir(parents=True, exist_ok=False)
            plan = self._rpc("resolve", workload=self.workload, candidate_id=c.id,
                assets=self.assets, run_id=self.run_prefix + "-gate-" + key, mode="correctness")
            write_new_json(directory / "execution-plan.json", plan)
            run_plan(plan, coordinator_output=directory / "dispatch", transport_config=self.transport_config)
            write_new_json(completed, plan)
        self.trajectories[key] = plan
        return plan

    def _prepare(self, c: Candidate) -> dict[str, Any]:
        cache = self.output / "preparation" / (c.id + "-ready.json")
        if cache.exists():
            value = json.loads(cache.read_text())
            require(value["source_head"] == self.identity["source_head"] and value["candidate_id"] == c.id,
                    "preparation source/candidate changed")
            require(fingerprint(asset_identity(value["assets"])) == self.identity["asset_fingerprint"],
                    "preparation asset identity changed")
            self.prepared[c.id] = value
        if c.id in self.prepared:
            if c.id.startswith("joint-h3"):
                value = self.prepared[c.id]
                require(value.get("diagnostic_noise_policy") == self.h3_noise_policy,
                        "H3 cached preparation admission policy differs from explicit opt-in")
                self._rpc("validate_h3_preparation", preparation=value,
                          candidate_id=c.id, noise_policy=self.h3_noise_policy)
            return self.prepared[c.id]
        self._require_admitted(c)
        if c.id == self.reference.id:
            self._prepare(self.bootstrap_peer)
            return self.prepared[c.id]
        ref = self._trajectory(self.reference)
        candidate = self._trajectory(c)
        noise = {}
        if self.h3_noise_policy is not None:
            noise = dict(repeat=self._trajectory(self.reference, noise_repeat=True),
                         noise_policy=self.h3_noise_policy)
        ready = self._rpc("prepare_pair", reference=ref, candidate=candidate, **noise)
        for cid, value in ready.items():
            if cid not in self.prepared:
                path = self.output / "preparation" / (cid + "-ready.json")
                if path.exists():
                    value = json.loads(path.read_text())
                else:
                    write_new_json(path, value)
                self.prepared[cid] = value
        return self.prepared[c.id]

    def __call__(self, candidate: Candidate, phase: str, directory: Path) -> dict[str, Any]:
        self._require_admitted(candidate)
        model_based = "environment_fingerprint" in self.identity
        if model_based:
            # Check both allocations/environments before even a correctness
            # worker can run. Never accept a caller-supplied digest by echo.
            self._observe_model_runtime(candidate)
        prepared = self._prepare(candidate)
        receipt = self._rpc("resolve", workload=self.workload, candidate_id=candidate.id,
            assets=prepared["assets"], run_id=self.run_prefix + "-" + directory.name,
            mode="diagnostic-timing")
        before = self._observe_model_runtime(candidate, receipt=receipt) if model_based else None
        require(receipt["asset_fingerprint"] == self.identity["asset_fingerprint"], "prepared asset identity changed")
        write_new_json(directory / "execution-plan.json", receipt)
        run_plan(receipt, coordinator_output=directory / "dispatch", transport_config=self.transport_config)
        raw = self._rpc("measurement", receipt=receipt, preparation=prepared, identity=self.identity)
        if model_based:
            raw["runtime_identity_evidence"] = {
                "before": before, "after": self._observe_model_runtime(candidate, receipt=receipt)}
            from .runtime_identity import verify_measurement_runtime
            verify_measurement_runtime(raw, self.identity)
        validate_measurement(raw, self.identity, candidate)
        return raw

    def _observe_model_runtime(self, candidate: Candidate, *, receipt: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        from .runtime_identity import verify_observed_binding
        if receipt is None:
            receipt = self._rpc("resolve", workload=self.workload, candidate_id=candidate.id,
                assets=self.assets, run_id=self.run_prefix + "-identity-" + candidate.id, mode="correctness")
        nodes = [self._rpc("runtime_identity", _node_rank=i, node_rank=i, receipt=receipt) for i in (0, 1)]
        verify_observed_binding(nodes, self.identity)
        return nodes

    def calibrate(self, candidate: Candidate, *, directory: Path) -> dict[str, Any]:
        """One diagnostic collection, not a timing trial or a matrix search."""
        self._require_admitted(candidate)
        from .wan_implementation import is_generated
        require((candidate.workload == WAN and is_generated(candidate.id)) or
                candidate.id.startswith(("joint-livewan1-", "joint-h3", "joint-helios1-")), "native calibration adapter unavailable")
        if candidate.id.startswith("joint-h3"):
            from .h3_implementation import H3Implementation
            require(H3Implementation.parse(candidate.id).rollout_window is None,
                    "H3 action calibration requires the action-qualified collector/assembler")
        directory.mkdir(exist_ok=False)
        self._observe_model_runtime(candidate)
        prepared = self._prepare(candidate)
        receipt = self._rpc("resolve", workload=self.workload, candidate_id=candidate.id,
            assets=prepared["assets"], run_id=self.run_prefix + "-cal-" + candidate.id, mode="calibration")
        before = self._observe_model_runtime(candidate, receipt=receipt)
        write_new_json(directory / "execution-plan.json", receipt)
        run_plan(receipt, coordinator_output=directory / "dispatch", transport_config=self.transport_config)
        after = self._observe_model_runtime(candidate, receipt=receipt)
        action = ("wan_stage_calibration" if candidate.workload == WAN else
                  "helios_calibration" if candidate.id.startswith("joint-helios1-") else "macro_calibration")
        value = self._rpc(action,
            receipt=receipt, preparation=prepared,
            identity=self.identity, runtime_identity_evidence={"before": before, "after": after})
        write_new_json(directory / "result.json", value)
        return value
