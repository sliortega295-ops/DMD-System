"""Explicit bounded shapes for the ORIGINAL Grid1000 H3 protocol.

This is not a sixth canonical winner. Every noncanonical request needs its own
assets, calibration, matched CPU-cache control and target-GPU acceptance.
"""
from dataclasses import asdict, dataclass
import copy
import re

from distillschedule.workloads.best_known.registry import fingerprint, load_profile
from .contracts import require

H3 = "minimax_h3_grid1000_best"
SCHEMA = "distillschedule.h3_request.v1"


@dataclass(frozen=True)
class H3Request:
    height: int = 768
    width: int = 1344
    frames: int = 124
    schema_version: str = SCHEMA

    def __post_init__(self):
        require(self.schema_version == SCHEMA, "unsupported H3 request schema")
        require(all(type(v) is int for v in (self.height, self.width, self.frames)),
                "H3 request dimensions must be integers")
        # Pinned LightX2V H3 packing uses 17*n+5, not Wan's frame lattice.
        # Admit only these bounded requests, not every native packing length.
        require((self.height, self.width) in ((768, 1344), (480, 832)) and self.frames in (56, 124),
                "H3 request supports only 768x1344 or 480x832 with 56 or 124 frames; other semantics are not admitted")

    @property
    def id(self):
        return f"h{self.height}-w{self.width}-f{self.frames}"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict) and set(value) == {"height", "width", "frames", "schema_version"},
                "H3 request requires exact schema/height/width/frames; unknown semantics are not ignored")
        return cls(**value)

    @classmethod
    def parse(cls, value):
        match = re.fullmatch(r"h([1-9][0-9]*)-w([1-9][0-9]*)-f([1-9][0-9]*)", value)
        require(match is not None, "malformed H3 request identity")
        return cls(*(int(v) for v in match.groups()))

    def profile(self, *, root=None):
        profile = copy.deepcopy(load_profile(H3, root=root))
        if self == H3Request():
            return profile
        profile["display_name"] = f"Unvalidated H3 Grid1000 request: {self.id}"
        profile["contract"]["resolution"] = [self.height, self.width]
        profile["contract"]["frames"] = self.frames
        profile["contract"]["request"] = self.to_dict()
        profile["runtime"]["model_config"]["training"]["dmd"].update(
            height=self.height, width=self.width, num_frames=self.frames)
        profile["historical"] = dict(formal_speedup_eligible=False,
            scope="NO_MATCHING_INTEGRATION_MEASUREMENT; PARENT_CANONICAL_SECONDS_NOT_TRANSFERABLE",
            parent_canonical_profile_sha256=fingerprint(load_profile(H3, root=root)))
        profile["memory"].pop("minimum_historical_driver_free_mib", None)
        return profile
