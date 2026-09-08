"""Budgeted native validation of calibrated, generated plans.

The search RNG/simulator chooses proposals; only workload-owned native receipts
choose measured winners. A budget includes both mandatory anchors, failed
attempts, and independent confirmation. Interrupted trials are never relaunched.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, write_new_json
from .contracts import CAPACITY_FAILURES, FAILURES, Candidate, require, validate_measurement
from .cost_model import CostModel
from .model_search import validate_saved_prediction
from .plan_ir import PlanningProblem
from .registry import get_candidate
from .simulator import simulate


class ModelValidationSession:
    """Source-owned lowerers only; unsupported adapters fail before GPU launch.

    ``execute`` is the existing NativeTrialRunner, not an arbitrary shell plan.
    Tests inject a receipt producer to exercise failures without GPU resources.
    The native runner still owns correctness, atomic node locks and teardown.
    """

    def __init__(self, *, problem: PlanningProblem, costs: CostModel,
                 prediction: Mapping[str, Any], output: Path,
                 max_trials: int = 8, resume: bool = False):
        from .model_adapters import adapter_for
        self.adapter = adapter_for(problem.workload)
        Implementation = self.adapter.implementation_type
        require(type(max_trials) is int and 3 <= max_trials <= 8,
                "total timing budget must be 3..8 including anchors and confirmation")
        self.adapter.validate_native_problem(problem)
        costs.verify_receipts()
        validate_saved_prediction(prediction, problem, costs)
        require(len(problem.seeds) == 2 and problem.seeds[0] != problem.seeds[1],
                "exactly two distinct baseline/incumbent seeds required")
        self.problem, self.costs, self.max_trials = problem, costs, max_trials
        self.output = Path(output)
        self.identity = {"workload": problem.workload, **problem.binding}
        self.predictions = {}
        self.candidates = {}
        self.rejected_anchors = {}
        assignments = [*problem.seeds, *(r["assignment"] for r in prediction["ranked"])]
        for assignment in assignments:
            actual = simulate(problem, assignment, costs)
            implementation = Implementation.from_assignment(assignment)
            if actual["status"] != "PREDICTED_FEASIBLE":
                require(assignment == problem.seeds[1] and actual["status"] == "PREDICTED_CAPACITY_FAIL",
                        "native baseline/proposal is not calibrated/feasible")
                # A historical incumbent may run yet fail the current safety
                # margin. Preserve the rejection, never launch it or silently
                # substitute its old timing. Baseline remains mandatory.
                self.rejected_anchors[implementation.id] = actual
                continue
            self.predictions[implementation.id] = actual
            self.candidates[implementation.id] = get_candidate(problem.workload, implementation.id)
        self.seeds = tuple(Implementation.from_assignment(s).id for s in problem.seeds)
        self.ranked = sorted((cid for cid in self.candidates if cid not in self.seeds),
            key=lambda cid: (self.predictions[cid]["predicted_cycle2_ns"], cid))
        self.spec = {"schema_version": "distillschedule.model_validation.v1",
            "identity": self.identity, "problem_sha256": problem.digest,
            "cost_model_sha256": costs.digest, "max_total_timing_attempts": max_trials,
            "max_new_candidates": max_trials - 3, "baseline": self.seeds[0],
            "incumbent": self.seeds[1], "ranked_proposals": self.ranked,
            "predictions": self.predictions,
            "rejected_anchors": self.rejected_anchors,
            "candidate_contracts": {cid: c.to_dict() for cid, c in self.candidates.items()},
            "objective": "NATIVE_UNPROFILED_CYCLE2_NOT_CYCLE_AVERAGE"}
        if resume:
            require((self.output / "validation.json").is_file(), "validation journal missing")
            require(json.loads((self.output / "validation.json").read_text()) == self.spec,
                    "resume source/assets/precision/environment/calibration/budget changed")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            write_new_json(self.output / "validation.json", self.spec)

    def events(self) -> list[dict[str, Any]]:
        path = self.output / "events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def _append(self, event: dict[str, Any]) -> None:
        with (self.output / "events.jsonl").open("a") as file:
            file.write(json.dumps({"time_unix_ns": time.time_ns(), **event}, sort_keys=True, allow_nan=False) + "\n")
            file.flush()
            os.fsync(file.fileno())

    def _validate_raw(self, raw: Mapping[str, Any], candidate: Candidate) -> dict[str, Any] | None:
        # The inherited native validator checks source/assets/hardware/math;
        # model-based execution also binds environment and explicit precision.
        for key, expected in self.identity.items():
            require(raw.get(key) == expected, f"measurement {key} mismatch")
        require(raw.get("candidate_id") == candidate.id, "measurement candidate changed")
        if raw.get("status") == "FAIL":
            require(raw.get("failure_type") in FAILURES and bool(raw.get("native_evidence")),
                    "unclassified failure without native evidence")
            return None
        require(raw.get("status") == "PASS", "native receipt not complete")
        from .runtime_identity import verify_measurement_runtime
        verify_measurement_runtime(raw, self.identity)
        return validate_measurement(raw, self.identity, candidate)

    def _verify_journal(self) -> None:
        events = self.events()
        starts = [e for e in events if e["type"] == "START"]
        complete = [e for e in events if e["type"] == "COMPLETE"]
        require(len(starts) <= self.max_trials, "timing budget exceeded")
        require(len({e["trial_id"] for e in starts}) == len(starts), "duplicate trial IDs")
        require({e["trial_id"] for e in starts} == {e["trial_id"] for e in complete}
                and len(starts) == len(complete), "incomplete trial preserved; reconcile before resuming")
        for event in complete:
            path = Path(event["receipt"])
            require(file_sha256(path) == event["receipt_sha256"], "completed receipt changed")
            checked = self._validate_raw(json.loads(path.read_text()), self.candidates[event["candidate_id"]])
            require(checked == event["result"], "journal result differs from native receipt")
            failure = event.get("failure_type")
            require(failure is None or (failure in CAPACITY_FAILURES and event["phase"] == "candidate"),
                    "previous non-capacity/anchor failure stops this session")

    def _trial(self, cid: str, phase: str, execute: Callable) -> dict[str, Any]:
        require(cid in self.candidates, "native proposal was rejected by prediction")
        index = sum(e["type"] == "START" for e in self.events())
        require(index < self.max_trials, "total timing budget exhausted")
        directory = self.output / "trials" / f"{index:02d}-{phase}-{cid}"
        directory.mkdir(parents=True, exist_ok=False)
        # This is the graph-to-native binding proof. NativeTrialRunner resolves
        # the same typed implementation with fresh run IDs and native gates;
        # calibration paths need not be copied to every GPU worker filesystem.
        selection = {"problem_sha256": self.problem.digest, "cost_model_sha256": self.costs.digest,
                     "predicted_best": self.predictions[cid]}
        plan = self.adapter.lower_prediction(selection, self.problem, self.costs, run_id=directory.name)
        require(plan["search_candidate"] == self.candidates[cid].to_dict(), "lowering changed candidate")
        write_new_json(directory / "model-lowering.json", plan)
        self._append({"type": "START", "trial_id": directory.name, "candidate_id": cid, "phase": phase})
        started = time.monotonic()
        try:
            raw = execute(self.candidates[cid], phase, directory)
            path = directory / "executor-receipt.json"
            write_new_json(path, raw)
            result = self._validate_raw(raw, self.candidates[cid])
        except Exception as error:
            failure = {"status": "ABORTED_REQUIRES_RECONCILIATION", "exception": type(error).__name__,
                       "message": str(error), "capacity_classification": "UNKNOWN_NOT_ASSUMED_OOM"}
            write_new_json(directory / "controller-failure.json", failure)
            self._append({"type": "ABORTED", "trial_id": directory.name, **failure})
            raise
        event = {"type": "COMPLETE", "trial_id": directory.name, "candidate_id": cid, "phase": phase,
                 "result": result, "failure_type": raw.get("failure_type") if result is None else None,
                 "receipt": str(path), "receipt_sha256": file_sha256(path),
                 "elapsed_seconds_including_gates": time.monotonic() - started}
        self._append(event)
        if result is None and (event["failure_type"] not in CAPACITY_FAILURES or phase != "candidate"):
            raise RuntimeError(f"stopped on {event['failure_type']}; no automatic retry")
        return event

    def _result(self) -> dict[str, Any]:
        done = [e for e in self.events() if e["type"] == "COMPLETE"]
        baseline = next(e for e in done if e["phase"] == "baseline")["result"]
        incumbent_event = next((e for e in done if e["phase"] == "incumbent"), None)
        incumbent = incumbent_event["result"] if incumbent_event else None
        require(incumbent is not None or self.seeds[1] in self.rejected_anchors,
                "missing native incumbent without a preserved prediction rejection")
        final = next(e for e in done if e["phase"] == "confirmation")["result"]
        assert baseline and final
        strict = baseline["strict_eligible"] and final["strict_eligible"]
        discovery = next(e["result"] for e in done if e["candidate_id"] == final["candidate_id"]
                         and e["phase"] != "confirmation")
        strict = strict and discovery["strict_eligible"]
        errors = [{"candidate_id": e["candidate_id"], "phase": e["phase"],
            "predicted_cycle2_seconds": self.predictions[e["candidate_id"]]["predicted_cycle2_ns"] / 1e9,
            "measured_cycle2_seconds": e["result"]["cycle2_seconds"],
            "relative_prediction_error": (
                self.predictions[e["candidate_id"]]["predicted_cycle2_ns"] / 1e9
                / e["result"]["cycle2_seconds"] - 1)} for e in done if e["result"]]
        return {"schema_version": "distillschedule.model_validation_result.v1",
            "status": "COMPLETE_NATIVE_VALIDATION", "identity": self.identity,
            "validation_sha256": fingerprint(self.spec), "baseline": baseline, "incumbent": incumbent,
            "confirmed_candidate": final,
            "strict_winner": final if strict and final["cycle2_seconds"] < baseline["cycle2_seconds"] else None,
            "cycle2_speedup_diagnostic": baseline["cycle2_seconds"] / final["cycle2_seconds"],
            "improvement_over_incumbent": incumbent["cycle2_seconds"] / final["cycle2_seconds"] if incumbent is not None else None,
            "rejected_anchors": {cid: {"status": value["status"], "gpu_execution": value["gpu_execution"],
                "memory": value["memory"], "capacity_failure_observed": False}
                for cid, value in self.rejected_anchors.items()},
            "timing_attempts": len(done), "new_candidate_attempts": sum(e["phase"] == "candidate" for e in done),
            "capacity_failures": sum(e["failure_type"] in CAPACITY_FAILURES for e in done),
            "prediction_errors": errors, "canonical_winner_updated": False,
            "claim_boundary": "BEST_TESTED_WITHIN_BUDGET; REVIEW_IS_NOT_STRICT_EQUIVALENCE; NO_GLOBAL_OPTIMUM_CLAIM"}

    def run(self, execute: Callable[[Candidate, str, Path], dict[str, Any]]) -> dict[str, Any]:
        with (self.output / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.adapter.validate_native_problem(self.problem)
            self.costs.verify_receipts()
            self._verify_journal()
            result_path = self.output / "result.json"
            if result_path.exists():
                value = json.loads(result_path.read_text())
                require(value == self._result(), "cached validation result changed")
                return value
            for cid, phase in zip(self.seeds, ("baseline", "incumbent")):
                if cid in self.rejected_anchors:
                    continue
                if not any(e["type"] == "COMPLETE" and e["phase"] == phase for e in self.events()):
                    self._trial(cid, phase, execute)
            for cid in self.ranked[:self.max_trials - 3]:
                if not any(e["type"] == "START" and e["candidate_id"] == cid for e in self.events()):
                    self._trial(cid, "candidate", execute)
            done = [e for e in self.events() if e["type"] == "COMPLETE"]
            if not any(e["phase"] == "confirmation" for e in done):
                selected = min((e for e in done if e["result"]),
                               key=lambda e: (e["result"]["cycle2_seconds"], e["candidate_id"]))
                self._trial(selected["candidate_id"], "confirmation", execute)
            result = self._result()
            write_new_json(result_path, result)
            return result
