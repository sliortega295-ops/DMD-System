"""Composable native Wan implementations, not a table of measured winners.

Only controls already implemented by the audited worker are admitted. A new
combination still needs its own correctness gate and operation calibration.
Legacy canonical/registered paths are unchanged without a joint candidate ID.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Mapping

from .contracts import Candidate, require
from .precision import ExecutionPrecision
from .wan_candidate_bundle import WanCandidateBundle, OFF, INCUMBENT, STREAM_INCUMBENT

WAN = "wan13b_nogan_resident_best"
PREFIX = "joint-wan1-"
PATTERN = re.compile(r"joint-wan1-(fsdp|resident)-c(512|1024|2048)-e([01])$")
BUNDLE_PREFIX = "joint-wan3-"
BUNDLE_PATTERN = re.compile(r"joint-wan3-resident-c2048-e1-bincumbent21a-v1(?:-w([1-5]))?$")
STREAM_PATTERN = re.compile(r"joint-wan3-resident-c2048-e1-bincumbent21a-stream-v2(?:-w([1-5]))?$")
SCHEDULE_PREFIX = "joint-wan2-"
SCHEDULE_PATTERN = re.compile(r"joint-wan2-(fsdp|resident)-c(512|1024|2048)-e([01])-w([1-5])$")
PRECISION = ExecutionPrecision("bfloat16", "float32", "float32", True, True,
                               "flash_attention_2_source_bound_by_environment")


@dataclass(frozen=True)
class WanImplementation:
    placement: str
    chunk_tokens: int
    lora_epilogue: bool
    rollout_window: int | None = None
    candidate_bundle: WanCandidateBundle = OFF

    def __post_init__(self) -> None:
        require(isinstance(self.candidate_bundle, WanCandidateBundle), "typed Wan candidate bundle required")
        self.candidate_bundle.validate_backend(self.placement, self.chunk_tokens, self.lora_epilogue)
        require(self.placement in {"fsdp", "resident"}, "unsupported Wan placement")
        require(type(self.chunk_tokens) is int and self.chunk_tokens in {512, 1024, 2048}, "unsupported chunk size")
        require(type(self.lora_epilogue) is bool, "epilogue must be boolean")
        require(self.rollout_window is None or (type(self.rollout_window) is int and 1 <= self.rollout_window <= 5),
                "rollout window must be [1,5]")
        require(self.placement != "fsdp" or (self.chunk_tokens == 512 and not self.lora_epilogue),
                "FSDP native kernel combinations beyond control are not yet admitted")

    @property
    def id(self) -> str:
        if self.candidate_bundle.enabled:
            suffix = "" if self.rollout_window is None else f"-w{self.rollout_window}"
            return f"{BUNDLE_PREFIX}resident-c2048-e1-b{self.candidate_bundle.id}{suffix}"
        prefix = PREFIX if self.rollout_window is None else SCHEDULE_PREFIX
        suffix = "" if self.rollout_window is None else f"-w{self.rollout_window}"
        return f"{prefix}{self.placement}-c{self.chunk_tokens}-e{int(self.lora_epilogue)}{suffix}"

    @property
    def native_placement(self) -> str:
        if self.placement == "fsdp":
            return "world16-fsdp"
        return "resident-async128-tree-chunk2048" if self.chunk_tokens == 2048 else "resident-async128-tree"

    @property
    def assignment(self) -> dict[str, str]:
        return {**({"candidate_bundle": self.candidate_bundle.id} if self.candidate_bundle.enabled else {}),
                "placement": self.placement, "chunk_tokens": str(self.chunk_tokens),
                "lora_epilogue": "on" if self.lora_epilogue else "off",
                **({"rollout_window": str(self.rollout_window)} if self.rollout_window is not None else {})}

    @classmethod
    def parse(cls, key: str) -> "WanImplementation":
        stream = STREAM_PATTERN.fullmatch(key)
        if stream is not None:
            return cls("resident", 2048, True, int(stream[1]) if stream[1] else None, STREAM_INCUMBENT)
        bundle = BUNDLE_PATTERN.fullmatch(key)
        if bundle is not None:
            return cls("resident", 2048, True, int(bundle[1]) if bundle[1] else None, INCUMBENT)
        match = PATTERN.fullmatch(key) or SCHEDULE_PATTERN.fullmatch(key)
        require(match is not None, "malformed/unregistered joint Wan implementation")
        return cls(match[1], int(match[2]), bool(int(match[3])),
                   int(match[4]) if match.re == SCHEDULE_PATTERN else None)

    @classmethod
    def from_assignment(cls, assignment: Mapping[str, str]) -> "WanImplementation":
        assignment = dict(assignment)
        bundle = WanCandidateBundle.parse(assignment.pop("candidate_bundle", "none"))
        required = {"placement", "chunk_tokens", "lora_epilogue"}
        require(set(assignment) in (required, required | {"rollout_window"}), "Wan plan has extra/missing decisions")
        require(assignment["chunk_tokens"] in {"512", "1024", "2048"}
                and assignment["lora_epilogue"] in {"on", "off"}, "invalid Wan decision")
        require("rollout_window" not in assignment or assignment["rollout_window"] in {"1","2","3","4","5"},
                "invalid rollout window")
        return cls(assignment["placement"], int(assignment["chunk_tokens"]), assignment["lora_epilogue"] == "on",
                   int(assignment["rollout_window"]) if "rollout_window" in assignment else None, bundle)

    def candidate(self) -> Candidate:
        kernels = (f"modulation_chunk{self.chunk_tokens}",) + (("lora_nograd_epilogue",) if self.lora_epilogue else ())
        if self.candidate_bundle.enabled:
            kernels += ("incumbent21a_stream_bundle_v2" if self.candidate_bundle.continuous
                        else "incumbent21a_coupled_bundle_v1",)
        return Candidate(self.id, WAN, self.native_placement, kernels,
                         ("nccl", "pointwise"), mechanism="joint_placement_and_existing_kernel_parameters")

    def environment(self) -> dict[str, str]:
        return {**PRECISION.launch_environment(), **self.candidate_bundle.environment(),
                "DISTILLSCHEDULE_WAN13B_PLANNER_PRECISION": PRECISION.digest,
                **({"DISTILLSCHEDULE_WAN_ROLLOUT_WINDOW":str(self.rollout_window)}
                   if self.rollout_window is not None else {})}


def is_generated(key: str) -> bool:
    return key.startswith((PREFIX, SCHEDULE_PREFIX, BUNDLE_PREFIX))


def current_implementation(environment: Mapping[str, str] | None = None) -> WanImplementation | None:
    key = (os.environ if environment is None else environment).get("DISTILLSCHEDULE_SEARCH_CANDIDATE_ID", "")
    return WanImplementation.parse(key) if is_generated(key) else None


def current_rollout_window(environment: Mapping[str, str] | None = None) -> int | None:
    env = os.environ if environment is None else environment
    impl = current_implementation(env)
    expected = None if impl is None else impl.rollout_window
    raw = env.get("DISTILLSCHEDULE_WAN_ROLLOUT_WINDOW", "")
    require(raw == ("" if expected is None else str(expected)),
            "rollout window differs from generated implementation identity")
    return expected


def verify_execution_precision(torch: Any, environment: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Setup-only explicit precision policy; never touches legacy runs.

Both generated control and generated candidates use this policy. Set flags
before construction, then assert and record observed values. No model casts,
extra collective, CUDA synchronization or per-call instrumentation is added.
"""
    env = os.environ if environment is None else environment
    if current_implementation(env) is None:
        require(not env.get("DISTILLSCHEDULE_WAN13B_PLANNER_PRECISION"), "planner precision without planner implementation")
        return None
    current_implementation(env).candidate_bundle.verify_environment(env)
    require(env.get("DISTILLSCHEDULE_WAN13B_PLANNER_PRECISION") == PRECISION.digest, "planner precision identity changed")
    require(all(env.get(k) == v for k, v in PRECISION.launch_environment().items()), "planner precision launch environment changed")
    require(env.get("NVIDIA_TF32_OVERRIDE") in {None, ""}, "unregistered driver precision override")
    torch.backends.cuda.matmul.allow_tf32 = PRECISION.matmul_allow_tf32
    torch.backends.cudnn.allow_tf32 = PRECISION.cudnn_allow_tf32
    return PRECISION.assert_runtime(torch, env)
