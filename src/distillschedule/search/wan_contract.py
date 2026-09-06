"""Opt-in specialization of the already-audited native Wan two-cycle gates.

Default behavior still admits only the original FSDP and fused Resident arms.
The first search bridge admits only reference-Adam/reference-EMA variants of
the same async128/Tree reducer; other legacy recipes remain CPU-only.
"""
from __future__ import annotations
import os
from .contracts import require

SEARCH_IDS = frozenset({"fsdp-reference", "resident-tree", "resident-chunk2048", "resident-lora-epilogue"})


def search_enabled() -> bool:
    return bool(os.environ.get("DISTILLSCHEDULE_SEARCH_CANDIDATE_ID"))


def arm(placement: str) -> dict:
    candidate_id = os.environ.get("DISTILLSCHEDULE_SEARCH_CANDIDATE_ID")
    if not candidate_id:
        require(placement in {"world16-fsdp", "resident-async128-tree-chunk2048"}, "unregistered native Wan arm")
        resident = placement != "world16-fsdp"
        return {"name": "RESIDENT_ASYNC128_TREE_CHUNK2048_FUSED_EPILOGUE" if resident else "WORLD16_FSDP_BASELINE",
                "fused": resident, "chunk": 2048 if resident else None,
                "algo": "Tree" if resident else "AUTO", "clean": resident}
    require(candidate_id in SEARCH_IDS, "Wan native two-cycle gate not integrated for this candidate")
    from .registry import WAN, get_candidate
    from distillschedule.workloads.best_known.registry import fingerprint
    c = get_candidate(WAN, candidate_id)
    require(c.placement == placement, "search candidate placement mismatch")
    return {"name": "SEARCH_" + fingerprint(c.to_dict()),
            "fused": "lora_nograd_epilogue" in c.kernels,
            "chunk": None if placement == "world16-fsdp" else (2048 if "chunk2048" in placement else 512),
            "algo": "Tree" if "tree" in placement else "AUTO", "clean": placement != "world16-fsdp"}


def registered_placements() -> frozenset[str]:
    if not search_enabled():
        return frozenset({"world16-fsdp", "resident-async128-tree-chunk2048"})
    from .registry import WAN, get_candidate
    key = os.environ["DISTILLSCHEDULE_SEARCH_CANDIDATE_ID"]
    require(key in SEARCH_IDS, "unsupported Wan search gate")
    return frozenset({get_candidate(WAN, key).placement})
