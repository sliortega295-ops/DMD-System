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
from .livewan_request import LiveWanRequest

LIVEWAN = "livewan_crossscale_pipeline_best"
PREFIX = "joint-livewan1-"


@dataclass(frozen=True)
class LiveWanImplementation:
    placement: str
    modulation: bool = False
    gate: bool = False
    row_major: bool = False
    request: LiveWanRequest = LiveWanRequest()
    bundle: str = "legacy"

    def __post_init__(self):
        require(self.placement in {"fsdp", "pipeline"}, "unsupported LiveWan placement")
        require(isinstance(self.request, LiveWanRequest), "LiveWan implementation requires typed request")
        require(all(type(x) is bool for x in (self.modulation, self.gate, self.row_major)), "kernel flags must be boolean")
        require(not self.gate or self.modulation, "gate requires modulation")
        require(not self.row_major or self.gate, "row-major requires gate")
        require(self.placement != "fsdp" or not any((self.modulation, self.gate, self.row_major)),
                "FSDP kernel variants beyond the matched control are not admitted")
        require(self.bundle in {"legacy", "control_v1", "c51_full_v1"}, "unsupported LiveWan bundle version")
        require(self.bundle == "legacy" or self.request == LiveWanRequest(),
                "C51 bundle cohort supports only the measured ORIGINAL 81-frame request")
        require(self.bundle != "c51_full_v1" or
                (self.placement == "pipeline" and self.modulation and self.gate and self.row_major),
                "C51 full bundle requires the unchanged C34 prerequisites")

    @property
    def id(self):
        suffix = "" if self.request == LiveWanRequest() else "-" + self.request.id
        if self.bundle != "legacy":
            suffix += "-bundle1-" + ("c51" if self.bundle == "c51_full_v1" else "control")
        return f"{PREFIX}{self.placement}-m{int(self.modulation)}-g{int(self.gate)}-r{int(self.row_major)}{suffix}"

    @property
    def assignment(self):
        return {**({"request": self.request.id} if self.request != LiveWanRequest() else {}),
                **({"bundle": self.bundle} if self.bundle != "legacy" else {}),
                "placement": self.placement, **{k: "on" if v else "off" for k, v in
                (("modulation", self.modulation), ("gate", self.gate), ("row_major", self.row_major))}}

    @classmethod
    def parse(cls, value):
        m = re.fullmatch(r"joint-livewan1-(fsdp|pipeline)-m([01])-g([01])-r([01])(-f69)?(?:-bundle1-(control|c51))?", value)
        require(m is not None, "malformed LiveWan implementation")
        return cls(m[1], *(bool(int(m[i])) for i in (2, 3, 4)),
                   request=LiveWanRequest(frames=69) if m[5] else LiveWanRequest(),
                   bundle={None: "legacy", "control": "control_v1", "c51": "c51_full_v1"}[m[6]])

    @classmethod
    def from_assignment(cls, value: Mapping):
        fields = {"placement", "modulation", "gate", "row_major"}
        require(set(value) in (fields, fields | {"request"}, fields | {"bundle"}), "LiveWan decisions changed")
        require(all(value[k] in {"on", "off"} for k in ("modulation", "gate", "row_major")), "invalid kernel selection")
        request = LiveWanRequest.parse(value["request"]) if "request" in value else LiveWanRequest()
        require("request" not in value or request != LiveWanRequest(), "default LiveWan request uses legacy assignment")
        require("bundle" not in value or value["bundle"] in {"control_v1", "c51_full_v1"},
                "versioned LiveWan bundle requires an explicit policy")
        return cls(value["placement"], *(value[k] == "on" for k in ("modulation", "gate", "row_major")),
                   request=request, bundle=value.get("bundle", "legacy"))

    @property
    def source_id(self):
        if self.bundle == "c51_full_v1":
            return "c51"
        return "fsdp-reference" if self.placement == "fsdp" else f"c{31 + sum((self.modulation, self.gate, self.row_major))}"

    def candidate(self):
        from .registry import get_candidate
        source = get_candidate(LIVEWAN, self.source_id)
        return Candidate(self.id, LIVEWAN, source.placement, source.kernels, source.targets,
                         mechanism="native_placement_and_prerequisite_checked_kernel_composition", variant=source.variant)

    def lower_profile(self, *, root=None):
        from .registry import get_candidate, candidate_profile
        p = candidate_profile(get_candidate(LIVEWAN, self.source_id), root=root)
        if self.bundle != "legacy":
            p["historical"] = dict(formal_speedup_eligible=False,
                scope="NO_MATCHING_INTEGRATION_MEASUREMENT; HISTORICAL_SECONDS_ARE_NOT_COSTS")
            # The whole cohort binds the same asset identity, including the
            # C51-only kernel. A control must not silently use another asset set.
            p["required_assets"]["score_flash_attn3_kernel"] = dict(
                environment="DISTILLSCHEDULE_SCORE_FA3_KERNEL", kind="directory",
                identity_file="metadata.json", fingerprint_required=True)
        if self.request != LiveWanRequest():
            requested = self.request.profile(root=root)
            p["contract"].update({k: v for k, v in requested["contract"].items() if k != "variant"})
            p["historical"] = requested["historical"]
        runtime, env = p["runtime"], p["runtime"]["environment"]
        runtime["planner_implementation"] = self.assignment
        if self.bundle == "c51_full_v1":
            from distillschedule.runtime.livewan_c51_bundle import receipt_contract
            runtime["planner_c51_bundle"] = receipt_contract()
        env.update(DISTILLSCHEDULE_SEARCH_CANDIDATE_ID=self.id,
                   LIVEWAN_TWO_CYCLE_ARM_ID="search-" + self.id, LIVEWAN_TRAJECTORY_ARM_ID="search-" + self.id)
        if self.placement == 'pipeline':
            # Source/identity observations can accompany the existing two-cycle
            # trajectory gate; this does not authorize native performance timing.
            p['allowed_run_modes'] = [*p['allowed_run_modes'], 'action-observation']
            runtime['native_run_modes']['action-observation'] = 'trajectory'
        return p


def implementations(*, request=None, include_c51=False):
    from dataclasses import replace
    require(type(include_c51) is bool, "C51 opt-in must be boolean")
    request = request or LiveWanRequest()
    values = (LiveWanImplementation("fsdp", request=request), *(LiveWanImplementation("pipeline", *flags, request=request) for flags in
              ((False, False, False), (True, False, False), (True, True, False), (True, True, True))))
    if not include_c51:
        return values
    # Every member of the opt-in cohort has the same decision schema. The old
    # five-candidate IDs/assignments and 69-frame request remain byte-identical.
    return (*[replace(i, bundle="control_v1") for i in values],
            LiveWanImplementation("pipeline", True, True, True, bundle="c51_full_v1"))
