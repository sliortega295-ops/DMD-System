"""First bounded LiveWan composition: placement and three existing G kernels.

The native 8+8 depth-one pipeline is indivisible here. This lowerer neither
invented a new pool split nor predicts independent transfer/compute overlap.
Source templates encode prerequisite implementations; decisions must match
their enabled metadata exactly, never a nearest-recipe fallback.
"""
from dataclasses import dataclass
import re
from typing import Mapping

from .contracts import Candidate, require

LIVEWAN = "livewan_crossscale_pipeline_best"
PREFIX = "joint-livewan1-"


@dataclass(frozen=True)
class LiveWanImplementation:
    placement: str
    modulation: bool = False
    gate: bool = False
    row_major: bool = False

    def __post_init__(self):
        require(self.placement in {"fsdp", "pipeline"}, "unsupported LiveWan placement")
        require(all(type(x) is bool for x in (self.modulation, self.gate, self.row_major)), "kernel flags must be boolean")
        require(not self.gate or self.modulation, "gate requires modulation")
        require(not self.row_major or self.gate, "row-major requires gate")
        require(self.placement != "fsdp" or not any((self.modulation, self.gate, self.row_major)),
                "FSDP kernel variants beyond the matched control are not admitted")

    @property
    def id(self):
        return f"{PREFIX}{self.placement}-m{int(self.modulation)}-g{int(self.gate)}-r{int(self.row_major)}"

    @property
    def assignment(self):
        return {"placement": self.placement, **{k: "on" if v else "off" for k, v in
                (("modulation", self.modulation), ("gate", self.gate), ("row_major", self.row_major))}}

    @classmethod
    def parse(cls, value):
        m = re.fullmatch(r"joint-livewan1-(fsdp|pipeline)-m([01])-g([01])-r([01])", value)
        require(m is not None, "malformed LiveWan implementation")
        return cls(m[1], *(bool(int(m[i])) for i in (2, 3, 4)))

    @classmethod
    def from_assignment(cls, value: Mapping):
        require(set(value) == {"placement", "modulation", "gate", "row_major"}, "LiveWan decisions changed")
        require(all(value[k] in {"on", "off"} for k in ("modulation", "gate", "row_major")), "invalid kernel selection")
        return cls(value["placement"], *(value[k] == "on" for k in ("modulation", "gate", "row_major")))

    @property
    def source_id(self):
        return "fsdp-reference" if self.placement == "fsdp" else f"c{31 + sum((self.modulation, self.gate, self.row_major))}"

    def candidate(self):
        from .registry import get_candidate
        source = get_candidate(LIVEWAN, self.source_id)
        return Candidate(self.id, LIVEWAN, source.placement, source.kernels, source.targets,
                         mechanism="native_placement_and_prerequisite_checked_kernel_composition", variant=source.variant)

    def lower_profile(self, *, root=None):
        from .registry import get_candidate, candidate_profile
        p = candidate_profile(get_candidate(LIVEWAN, self.source_id), root=root)
        runtime, env = p["runtime"], p["runtime"]["environment"]
        runtime["planner_implementation"] = self.assignment
        env.update(DISTILLSCHEDULE_SEARCH_CANDIDATE_ID=self.id,
                   LIVEWAN_TWO_CYCLE_ARM_ID="search-" + self.id, LIVEWAN_TRAJECTORY_ARM_ID="search-" + self.id)
        return p


def implementations():
    return (LiveWanImplementation("fsdp"), *(LiveWanImplementation("pipeline", *flags) for flags in
            ((False, False, False), (True, False, False), (True, True, False), (True, True, True))))
