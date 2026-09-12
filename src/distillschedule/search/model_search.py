"""Bounded Metropolis plan search; no workload-name -> winning recipe lookup."""
from __future__ import annotations

import itertools
import math
import random
import time
from typing import Any, Mapping

from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import require
from .cost_model import CostModel, MissingCalibration
from .plan_ir import PlanningProblem
from .simulator import simulate


def search_plans(problem: PlanningProblem, costs: CostModel, *, seed: int = 0,
                 max_proposals: int = 10000, timeout_seconds: float = 600,
                 exhaustive: bool = False) -> dict[str, Any]:
    require(type(max_proposals) is int and 1 <= max_proposals <= 100000, "proposal budget must be 1..100000")
    require(math.isfinite(timeout_seconds) and 0 < timeout_seconds <= 600, "search deadline must be <=600 seconds")
    costs.bind(problem.binding, problem_scope=problem.evidence_scope)
    require(problem.seeds, "baseline and known-best seeds must be supplied")
    names = sorted(problem.decisions)
    cache: dict[str, dict[str, Any]] = {}
    journal = []
    started = time.monotonic()
    rng = random.Random(seed)

    def evaluate(assignment: Mapping[str, str], reason: str) -> dict[str, Any]:
        key = fingerprint(dict(assignment))
        if key not in cache:
            try:
                value = simulate(problem, assignment, costs)
            except MissingCalibration as error:
                value = {"status": "UNCALIBRATED", "detail": str(error), "assignment": dict(assignment)}
            except ValueError as error:
                value = {"status": "REJECTED_CONTRACT", "detail": str(error), "assignment": dict(assignment)}
            cache[key] = value
        journal.append({"proposal": len(journal), "assignment_sha256": key,
                        "reason": reason, "status": cache[key]["status"]})
        return cache[key]

    # Mandatory seeds are evaluated before spending proposal budget. Measured
    # historical times are never imported as costs or current confirmation.
    initial = [evaluate(s, "MANDATORY_BASELINE_OR_INCUMBENT") for s in problem.seeds]
    feasible = [r for r in initial if r["status"] == "PREDICTED_FEASIBLE"]
    current = dict(min(feasible, key=lambda r: r["predicted_cycle2_ns"])["assignment"] if feasible else problem.seeds[0])
    current_value = evaluate(current, "INITIAL_CHAIN")
    temperature = max(1.0, 0.05 * min((r["predicted_cycle2_ns"] for r in feasible), default=1))
    space_size = math.prod(len(problem.decisions[k]) for k in names)
    if exhaustive:
        require(space_size <= max_proposals, "exhaustive oracle exceeds budget")
        proposals = (dict(zip(names, values)) for values in itertools.product(*(problem.decisions[k] for k in names)))
    else:
        proposals = None
    mutable = [k for k in names if len(problem.decisions[k]) > 1]
    for index in range(max_proposals):
        if time.monotonic() - started >= timeout_seconds:
            break
        if exhaustive:
            try:
                proposal = next(proposals)
            except StopIteration:
                break
        else:
            if not mutable:
                break
            # Fixed domains + rejection of illegal assignments make this a
            # symmetric proposal. No omitted Hastings correction for a
            # state-dependent, pre-filtered neighbor domain.
            if index and index % 100 == 0:
                current = dict(problem.seeds[rng.randrange(len(problem.seeds))])
                current_value = evaluate(current, "CHAIN_RESTART")
            proposal = dict(current)
            key = rng.choice(mutable)
            proposal[key] = rng.choice([v for v in problem.decisions[key] if v != proposal[key]])
        result = evaluate(proposal, "EXHAUSTIVE_ORACLE" if exhaustive else "METROPOLIS_PROPOSAL")
        if result["status"] != "PREDICTED_FEASIBLE":
            continue
        old = current_value.get("predicted_cycle2_ns", math.inf) if current_value["status"] == "PREDICTED_FEASIBLE" else math.inf
        delta = result["predicted_cycle2_ns"] - old
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current, current_value = proposal, result
    ranked = sorted((r for r in cache.values() if r["status"] == "PREDICTED_FEASIBLE"),
                    key=lambda r: (r["predicted_cycle2_ns"], r["predicted_total_ns"], fingerprint(r["assignment"])))
    missing = sorted({r["detail"] for r in cache.values() if r["status"] == "UNCALIBRATED"})
    return {"schema_version": "distillschedule.model_search_result.v1",
            "status": "PASS_CPU_SEARCH" if ranked else "NO_CALIBRATED_FEASIBLE_PLAN",
            "workload": problem.workload, "binding": dict(problem.binding),
            "problem_sha256": problem.digest, "cost_model_sha256": costs.digest,
            "seed": seed, "space_size": space_size, "unique_evaluated": len(cache),
            "search_seconds": time.monotonic() - started, "exhaustive": exhaustive,
            "evidence_scope": problem.evidence_scope, "gpu_execution": "NOT_RUN",
            "strict_winner": None, "claimable_speedup": False,
            "predicted_best": ranked[0] if ranked else None, "ranked": ranked,
            "calibration_requests": missing,
            "rejections": [r for r in cache.values() if r["status"] != "PREDICTED_FEASIBLE"],
            "journal": journal}


def validate_saved_prediction(result: Mapping[str, Any], problem: PlanningProblem, costs: CostModel) -> dict[str, Any]:
    """Recompute rather than trust serialized predicted timings or assignments."""
    require(result.get("problem_sha256") == problem.digest, "saved problem identity changed")
    require(result.get("cost_model_sha256") == costs.digest, "saved calibration identity changed")
    predicted = result.get("predicted_best")
    require(bool(predicted), "no predicted plan to validate")
    actual = simulate(problem, predicted["assignment"], costs)
    require(fingerprint(predicted) == fingerprint(actual), "serialized prediction was altered")
    return actual
