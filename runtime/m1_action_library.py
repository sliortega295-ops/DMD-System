"""Frozen M1.2 same-primitive action-library validation.

The manifest is the authority for bounded-oracle membership. This module only
maps already implemented actions to runtime settings and prevents a run label
from disagreeing with the environment that was actually launched.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

MIB = 1024 * 1024

_OWNERSHIP = {
    "all_private": ("HOST_PRIVATE", "", ""),
    "score_blocks_0_4_shared": ("HOST_PRIVATE", "", "0,1,2,3,4"),
    "all_score_blocks_shared": (
        "HOST_PRIVATE", "", ",".join(str(i) for i in range(40))
    ),
    "all_generator_and_score_shared": ("HOST_SHARED_IMMUTABLE", "", ""),
}
_ACTIVATION = {
    "full_offload_min_0": 0,
    "expert_selective_min_32mib": 32 * MIB,
    "selective_min_64mib": 64 * MIB,
}
_PIN = {
    "pageable": (0, "all"),
    "generator": (1, "generator"),
    "generator_and_score": (1, "all"),
}


def load_action_library(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != "distillschedule.m1_static_action_library.v1":
        raise ValueError("unsupported M1 action-library schema")
    plans = payload.get("fixed_plan_set")
    if not isinstance(plans, list) or not plans:
        raise ValueError("M1 action library has no fixed plans")
    identifiers = [plan.get("id") for plan in plans]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("M1 action library contains duplicate plan ids")
    return payload


def resolve_action(payload: Mapping[str, Any], action_id: str) -> dict[str, Any]:
    plans = [plan for plan in payload["fixed_plan_set"] if plan["id"] == action_id]
    if len(plans) != 1:
        raise ValueError(f"unknown M1 action id: {action_id}")
    plan = dict(plans[0])
    if plan.get("status") == "PREREGISTERED_IMPLEMENTATION_REQUIRED":
        raise ValueError(f"M1 action is not executable yet: {action_id}")
    try:
        host_policy, shared_generator, shared_score = _OWNERSHIP[plan["ownership"]]
        activation_min_bytes = _ACTIVATION[plan["activation"]]
        pin_host_state, pin_host_component = _PIN[plan["pin"]]
    except KeyError as error:
        raise ValueError(f"M1 action uses an unavailable primitive: {error.args[0]}") from error
    return {
        "action_id": action_id,
        "host_policy": host_policy,
        "shared_generator_indices": shared_generator,
        "shared_score_indices": shared_score,
        "streaming_window": int(plan["window"]),
        "activation_min_bytes": activation_min_bytes,
        "pin_host_state": pin_host_state,
        "pin_host_component": pin_host_component,
        "construction_wave": int(plan["construction_wave"]),
        "status": plan["status"],
        "track": plan["track"],
    }


def validate_runtime_settings(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    fields = (
        "host_policy",
        "shared_generator_indices",
        "shared_score_indices",
        "streaming_window",
        "activation_min_bytes",
        "pin_host_state",
        "pin_host_component",
        "construction_wave",
    )
    mismatches = {
        name: {"expected": expected[name], "actual": actual.get(name)}
        for name in fields
        if str(expected[name]) != str(actual.get(name))
    }
    if mismatches:
        raise ValueError(
            "M1 action label/runtime mismatch: " + json.dumps(mismatches, sort_keys=True)
        )
