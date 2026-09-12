"""Deterministic, resumable budgeted search; all timing comes from receipts.

The profile score is a heuristic in GPU-seconds, NOT a latency prediction. A
measured one-factor edge calibrates an effect only for its matching mechanism;
the search never subtracts summed NCCL/copy unions from native wall time.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import time
from typing import Any, Callable

from distillschedule.workloads.best_known.registry import fingerprint, write_new_json
from .contracts import Candidate, CAPACITY_FAILURES, FAILURES, PILOTS, SCHEMA, require, validate_measurement
from .profile import feature_weights, validate_profile
from .registry import candidates, get_candidate


class SearchSession:
    def __init__(self, *, workload: str, identity: dict[str, str], profile: dict[str, Any],
                 output: Path, max_trials: int = 8, resume: bool = False,
                 historical_prior: bool = False):
        require(workload in PILOTS, "workload is registered CPU-only; GPU search NOT_RUN")
        require(type(max_trials) is int and 1 <= max_trials <= 8, "v1 budget must be 1..8 candidates")
        validate_profile(profile)
        require(profile["identity"].get("workload") == workload == identity.get("workload"),
                "profile belongs to another workload")
        require(profile["identity"].get("hardware_fingerprint") == identity["hardware_fingerprint"],
                "profile hardware_fingerprint mismatch")
        mismatches = {key: {"profile": profile["identity"].get(key), "run": identity.get(key)}
                      for key in ("source_head", "asset_fingerprint", "semantic_fingerprint")
                      if profile["identity"].get(key) != identity.get(key)}
        require(historical_prior or (not mismatches and profile["cycle"] == 2 and profile.get("profile_scope") != "REPRESENTATIVE_PRIOR_ONLY"),
                "unmatched profile requires explicit historical-prior mode")
        self.workload, self.identity, self.profile = workload, identity, profile
        self.output, self.max_trials = Path(output), max_trials
        self.weights = feature_weights(profile)
        self.space = candidates(workload)
        self.started = time.monotonic()
        self.spec = {"schema_version": SCHEMA, "workload": workload, "identity": identity,
                     "profile_sha256": fingerprint(profile), "max_trials": max_trials,
                     "beam_width": 2, "objective": "native_cycle2_seconds",
                     "profile_role": "HISTORICAL_PRIOR_ONLY" if historical_prior else "MATCHED",
                     "profile_identity_differences": mismatches, "profile_cycle": profile["cycle"],
                     "profile_scope": profile.get("profile_scope", "ALL_16_RANKS"),
                     "sampled_profile_ranks": profile.get("sampled_ranks", list(range(16))),
                     "candidate_space": [c.to_dict() for c in self.space]}
        if resume:
            require((self.output / "search.json").is_file(), "search state missing")
            require(json.loads((self.output / "search.json").read_text()) == self.spec,
                    "cannot resume under different code/assets/profile/budget/candidates")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            write_new_json(self.output / "search.json", self.spec)
            write_new_json(self.output / "profile.json", profile)
        self.journal = self.output / "events.jsonl"

    def events(self) -> list[dict[str, Any]]:
        if not self.journal.exists():
            return []
        return [json.loads(line) for line in self.journal.read_text().splitlines() if line.strip()]

    def append(self, event: dict[str, Any]) -> None:
        with self.journal.open("a") as file:
            file.write(json.dumps({"time_unix_ns": time.time_ns(), **event}, sort_keys=True, allow_nan=False) + "\n")
            file.flush()
            import os
            os.fsync(file.fileno())

    def ranking(self) -> list[dict[str, Any]]:
        completed = [e for e in self.events() if e["type"] == "COMPLETE"]
        tried = {e["candidate_id"] for e in completed}
        measured = {e["candidate_id"]: e["result"] for e in completed if e.get("result")}
        beam = sorted(measured, key=lambda k: (measured[k]["cycle2_seconds"], k))[:2]
        effects: dict[str, list[float]] = {}
        for c in self.space:
            if c.parent in measured and c.id in measured:
                delta = measured[c.parent]["cycle2_seconds"] - measured[c.id]["cycle2_seconds"]
                effects.setdefault(c.mechanism, []).append(delta)
        ranked = []
        for c in self.space:
            if c.id in tried or not c.executable:
                continue
            nearby = c.parent in beam or any(get_candidate(self.workload, p).parent == c.id for p in beam)
            direct = c.parent in measured
            feature_priority = sum(self.weights.get(t, 0.0) for t in c.targets)
            calibration = effects.get(c.mechanism, [])
            ranked.append({"candidate_id": c.id, "parent": c.parent, "mechanism": c.mechanism,
                           "beam_neighbor": nearby, "parent_measured": direct,
                           "profile_priority_gpu_seconds_hint": feature_priority,
                           "measured_same_mechanism_delta_seconds": sum(calibration) / len(calibration) if calibration else None,
                           "estimated_cycle2_seconds": None,
                           "reason": "MATCHED_EDGE_THEN_PROFILE_TARGET_HINT; NOT_A_SPEEDUP_PREDICTION"})
        # Baseline is an explicit control, not a candidate selected by its name.
        return sorted(ranked, key=lambda r: (-int(r["beam_neighbor"]), -int(r["parent_measured"]),
            -(r["measured_same_mechanism_delta_seconds"] or 0.0),
            -r["profile_priority_gpu_seconds_hint"], r["candidate_id"]))

    def _trial(self, candidate: Candidate, phase: str,
               execute: Callable[[Candidate, str, Path], dict[str, Any]]) -> dict[str, Any]:
        index = sum(e["type"] == "START" for e in self.events())
        trial_id = f"{index:02d}-{phase}-{candidate.id}"
        directory = self.output / "trials" / trial_id
        directory.mkdir(parents=True, exist_ok=False)
        self.append({"type": "START", "trial_id": trial_id, "candidate_id": candidate.id, "phase": phase})
        before = time.monotonic()
        try:
            raw = execute(candidate, phase, directory)
        except Exception as error:
            # A lost controller/SSH or incomplete native receipt is not an OOM.
            # Preserve the START without COMPLETE so resume cannot retry it.
            failure = {"status": "ABORTED_REQUIRES_RECONCILIATION", "exception": type(error).__name__,
                       "message": str(error), "capacity_classification": "UNKNOWN_NOT_ASSUMED_OOM",
                       "trial_id": trial_id, "candidate_id": candidate.id}
            write_new_json(directory / "controller-failure.json", failure)
            self.append({"type": "ABORTED", **failure})
            raise
        write_new_json(directory / "executor-receipt.json", raw)
        result = None
        if raw.get("status") == "FAIL":
            failure = raw.get("failure_type")
            require(failure in FAILURES and bool(raw.get("native_evidence")), "unclassified failure without evidence")
        else:
            require(raw.get("status") == "PASS", "runner has not produced a completed native measurement")
            result = validate_measurement(raw, self.identity, candidate)
            failure = None
        event = {"type": "COMPLETE", "trial_id": trial_id, "candidate_id": candidate.id,
                 "phase": phase, "result": result, "failure_type": failure,
                 "elapsed_seconds_including_gates": time.monotonic() - before,
                 "receipt": str(directory / "executor-receipt.json")}
        self.append(event)
        if failure is not None and (failure not in CAPACITY_FAILURES or phase != "candidate"):
            raise RuntimeError(f"search stopped on {failure}; no contract change or automatic retry")
        return event

    def run(self, execute: Callable[[Candidate, str, Path], dict[str, Any]]) -> dict[str, Any]:
        with (self.output / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.output / "result.json").exists():
                return json.loads((self.output / "result.json").read_text())
            events = self.events()
            started = {e["trial_id"] for e in events if e["type"] == "START"}
            complete = {e["trial_id"] for e in events if e["type"] == "COMPLETE"}
            require(started == complete, "in-flight/incomplete trial preserved; reconcile receipt before resuming")
            for e in events:
                if e["type"] == "COMPLETE" and e.get("failure_type") not in (None, *CAPACITY_FAILURES):
                    raise RuntimeError("previous non-capacity failure must be resolved before a new search")
            if not any(e.get("phase") == "baseline" for e in events):
                self._trial(get_candidate(self.workload, "fsdp-reference"), "baseline", execute)
            while sum(e["type"] == "START" and e["phase"] == "candidate" for e in self.events()) < self.max_trials:
                ranking = self.ranking()
                if not ranking:
                    break
                self.append({"type": "SELECTION", "ranking": ranking, "selected": ranking[0]["candidate_id"]})
                self._trial(get_candidate(self.workload, ranking[0]["candidate_id"]), "candidate", execute)
            valid = [e for e in self.events() if e["type"] == "COMPLETE" and e.get("result") and e["phase"] != "confirmation"]
            winner = min(valid, key=lambda e: (e["result"]["cycle2_seconds"], e["candidate_id"]))
            confirmation = next((e for e in self.events() if e["type"] == "COMPLETE" and e["phase"] == "confirmation"), None)
            if confirmation is None:
                confirmation = self._trial(get_candidate(self.workload, winner["candidate_id"]), "confirmation", execute)
            baseline = next(e for e in valid if e["phase"] == "baseline")
            strict = [e for e in valid if e["result"]["strict_eligible"]]
            fastest_strict = min(strict, key=lambda e: e["result"]["cycle2_seconds"]) if strict else None
            final = confirmation["result"]
            require(final is not None, "winner confirmation failed")
            result = {"schema_version": SCHEMA, "status": "COMPLETE_MEASURED_SEARCH", "identity": self.identity,
                "workload": self.workload, "baseline": baseline["result"], "diagnostic_winner": final,
                "strict_winner": fastest_strict["result"] if fastest_strict else None,
                "winner_plan": get_candidate(self.workload, winner["candidate_id"]).to_dict(),
                "executable_plan_path": str(self.output / "trials" / confirmation["trial_id"] / "execution-plan.json"),
                "cycle2_speedup": baseline["result"]["cycle2_seconds"] / final["cycle2_seconds"],
                "improved_on_baseline": final["cycle2_seconds"] < baseline["result"]["cycle2_seconds"],
                "search_cost_seconds": sum(e.get("elapsed_seconds_including_gates", 0) for e in self.events()),
                "candidate_attempts": sum(e["type"] == "START" and e["phase"] == "candidate" for e in self.events()),
                "measurement_scope": "SINGLE_TWO_CYCLE_TRIALS_PLUS_ONE_INDEPENDENT_CONFIRMATION",
                "claim_boundary": "BEST_TESTED_WITHIN_BUDGET_NOT_GLOBAL_OPTIMUM; REVIEW_IS_NOT_STRICT_EQUIVALENCE"}
            write_new_json(self.output / "result.json", result)
            return result
