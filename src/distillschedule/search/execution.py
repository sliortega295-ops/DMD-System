"""Compile registered recipes through the existing dispatcher and native gates.

No generic shell-command executor and no exit-code -> PASS shortcut. Workload
adapters must provide source-bound native correctness/measurement receipts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from distillschedule.workloads.best_known.registry import (
    fingerprint, git_head, load_profile, repository_root, resolve_configuration,
    write_new_json, file_sha256,
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
        from .wan_model_adapter import validate_lowered_plan
        validate_lowered_plan(receipt)
    c = get_candidate(receipt["profile_id"], receipt["search_candidate"]["id"])
    require(fingerprint(receipt["search_candidate"]) == fingerprint(c.to_dict()), "search candidate changed")
    p = candidate_profile(c, root=root)
    require(receipt.get("canonical_profile_sha256") == fingerprint(load_profile(c.workload, root=root)), "canonical source profile changed")
    require(receipt.get("profile_sha256") == fingerprint(p), "search recipe/profile changed")
    require(receipt.get("workload_semantics_hash") == fingerprint(semantic_contract(p)), "search semantics changed")
    require(receipt.get("execution_fingerprint") == fingerprint(c.to_dict()), "search execution identity changed")
    require(fingerprint(receipt["fully_resolved_profile"]) == fingerprint(p), "serialized profile was altered")
    return p


def run_plan(receipt: dict[str, Any], *, dry_run: bool = False,
             coordinator_output: Path | None = None) -> int:
    root = repository_root()
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
    root = repository_root()
    p = validated_profile(receipt, root=root)
    require(receipt["integration_head"] == git_head(root), "search plan belongs to another HEAD")
    require(run_id != receipt["run_id"] and run_id != "source-inspection", "a fresh explicit run-id is required")
    assets = load_assets(assets_path, p) if assets_path is not None else receipt["asset_config"]
    require(assets == receipt["asset_config"], "replay assets/gates changed")
    from .registry import candidates
    from .wan_implementation import PREFIX, WanImplementation
    is_joint = receipt["search_candidate"]["id"].startswith(PREFIX)
    peer = (WanImplementation("resident", 2048, True).id if is_joint else
            next(c.id for c in candidates(receipt["profile_id"]) if c.executable and c.id != "fsdp-reference"))
    coordinator_output.mkdir(parents=True, exist_ok=False)
    runner = NativeTrialRunner(workload=receipt["profile_id"], assets=assets,
        identity={"source_head": receipt["integration_head"], "asset_fingerprint": receipt["asset_fingerprint"]},
        output=coordinator_output, run_prefix=run_id, bootstrap_peer=peer,
        reference_id=WanImplementation("fsdp", 512, False).id if is_joint else "fsdp-reference")
    # Resolve beside the immutable source receipts; those remote asset files
    # are deliberately not downloaded onto the controller machine.
    result = runner._rpc("resolve", workload=receipt["profile_id"], candidate_id=receipt["search_candidate"]["id"],
                         assets=assets, run_id=run_id, mode=receipt["mode"])
    if "model_plan" in receipt:
        result["model_plan"] = receipt["model_plan"]
        validated_profile(result, root=root)
    write_new_json(coordinator_output / "execution-plan.json", result)
    return result


class NativeTrialRunner:
    """Serial, source-bound native correctness -> measurement controller.

    RPC is only CPU metadata work beside persistent remote assets; GPU workers
    still go through the existing two-node dispatcher and native node locks.
    Preparation is cached immutably. Interrupted run IDs are never retried.
    """
    def __init__(self, *, workload: str, identity: dict[str, str], assets: dict[str, Any],
                 output: Path, run_prefix: str, bootstrap_peer: str,
                 reference_id: str = "fsdp-reference"):
        self.workload, self.identity, self.assets = workload, identity, assets
        self.output, self.run_prefix = Path(output), run_prefix
        self.bootstrap_peer = get_candidate(workload, bootstrap_peer)
        self.reference = get_candidate(workload, reference_id)
        require(self.reference.placement == "world16-fsdp", "reference must use world16 FSDP")
        require(self.bootstrap_peer.executable and bootstrap_peer != reference_id, "invalid bootstrap peer")
        from .wan_implementation import PREFIX
        require(self.reference.id.startswith(PREFIX) == self.bootstrap_peer.id.startswith(PREFIX),
                "cannot mix generated precision contract and legacy search arms")
        require(fingerprint(assets["assets"]) == identity["asset_fingerprint"], "asset identity mismatch")
        self.prepared: dict[str, dict[str, Any]] = {}
        self.trajectories: dict[str, dict[str, Any]] = {}

    def _rpc(self, action: str, **kwargs: Any) -> dict[str, Any]:
        import shlex
        import subprocess
        from distillschedule.workloads.best_known.dispatch import ssh_command
        node = self.assets["nodes"][0]
        python = str(Path(node["python_env"]) / "bin/python")
        worker = str(Path(node["repo_root"]) / "scripts/distillschedule/run_best_known_workload.py")
        command = [*ssh_command(node), shlex.join(["env", "PYTHONDONTWRITEBYTECODE=1",
            "LD_LIBRARY_PATH=" + str(Path(node["python_env"]) / "lib"), python, worker, "_search"])]
        import time
        logs = self.output / "rpc"
        logs.mkdir(exist_ok=True)
        directory = logs / f"{time.time_ns()}-{action}"
        directory.mkdir(exist_ok=False)
        payload = {"action": action, "source_head": self.identity["source_head"], **kwargs}
        write_new_json(directory / "request.json", {"command": command, "payload": payload})
        try:
            response = subprocess.run(command, input=json.dumps(payload), capture_output=True,
                                      text=True, timeout=900)
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
        return json.loads(response.stdout)

    def _trajectory(self, c: Candidate) -> dict[str, Any]:
        if c.id in self.trajectories:
            return self.trajectories[c.id]
        directory = self.output / "preparation" / c.id
        completed = directory / "completed-plan.json"
        if completed.exists():
            plan = json.loads(completed.read_text())
            validated_profile(plan, root=repository_root())
            require(plan["integration_head"] == self.identity["source_head"], "old preparation source")
            require(plan["asset_fingerprint"] == self.identity["asset_fingerprint"], "old preparation assets")
        else:
            directory.mkdir(parents=True, exist_ok=False)
            plan = self._rpc("resolve", workload=self.workload, candidate_id=c.id,
                assets=self.assets, run_id=self.run_prefix + "-gate-" + c.id, mode="correctness")
            write_new_json(directory / "execution-plan.json", plan)
            run_plan(plan, coordinator_output=directory / "dispatch")
            write_new_json(completed, plan)
        self.trajectories[c.id] = plan
        return plan

    def _prepare(self, c: Candidate) -> dict[str, Any]:
        cache = self.output / "preparation" / (c.id + "-ready.json")
        if cache.exists():
            value = json.loads(cache.read_text())
            require(value["source_head"] == self.identity["source_head"] and value["candidate_id"] == c.id,
                    "preparation source/candidate changed")
            require(fingerprint(value["assets"]["assets"]) == self.identity["asset_fingerprint"],
                    "preparation asset identity changed")
            self.prepared[c.id] = value
        if c.id in self.prepared:
            return self.prepared[c.id]
        if c.id == self.reference.id:
            self._prepare(self.bootstrap_peer)
            return self.prepared[c.id]
        ref = self._trajectory(self.reference)
        candidate = self._trajectory(c)
        ready = self._rpc("prepare_pair", reference=ref, candidate=candidate)
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
        prepared = self._prepare(candidate)
        receipt = self._rpc("resolve", workload=self.workload, candidate_id=candidate.id,
            assets=prepared["assets"], run_id=self.run_prefix + "-" + directory.name,
            mode="diagnostic-timing")
        require(receipt["asset_fingerprint"] == self.identity["asset_fingerprint"], "prepared asset identity changed")
        write_new_json(directory / "execution-plan.json", receipt)
        run_plan(receipt, coordinator_output=directory / "dispatch")
        raw = self._rpc("measurement", receipt=receipt, preparation=prepared, identity=self.identity)
        validate_measurement(raw, self.identity, candidate)
        return raw
