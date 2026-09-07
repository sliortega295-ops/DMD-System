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

WAN = "wan13b_nogan_resident_best"
PREFIX = "joint-wan1-"
PATTERN = re.compile(r"joint-wan1-(fsdp|resident)-c(512|1024|2048)-e([01])$")
PRECISION = ExecutionPrecision("bfloat16", "float32", "float32", True, True,
                               "flash_attention_2_source_bound_by_environment")


@dataclass(frozen=True)
class WanImplementation:
    placement: str
    chunk_tokens: int
    lora_epilogue: bool

    def __post_init__(self) -> None:
        require(self.placement in {"fsdp", "resident"}, "unsupported Wan placement")
        require(type(self.chunk_tokens) is int and self.chunk_tokens in {512, 1024, 2048}, "unsupported chunk size")
        require(type(self.lora_epilogue) is bool, "epilogue must be boolean")
        require(self.placement != "fsdp" or (self.chunk_tokens == 512 and not self.lora_epilogue),
                "FSDP native kernel combinations beyond control are not yet admitted")

    @property
    def id(self) -> str:
        return f"{PREFIX}{self.placement}-c{self.chunk_tokens}-e{int(self.lora_epilogue)}"

    @property
    def native_placement(self) -> str:
        if self.placement == "fsdp":
            return "world16-fsdp"
        return "resident-async128-tree-chunk2048" if self.chunk_tokens == 2048 else "resident-async128-tree"

    @property
    def assignment(self) -> dict[str, str]:
        return {"placement": self.placement, "chunk_tokens": str(self.chunk_tokens),
                "lora_epilogue": "on" if self.lora_epilogue else "off"}

    @classmethod
    def parse(cls, key: str) -> "WanImplementation":
        match = PATTERN.fullmatch(key)
        require(match is not None, "malformed/unregistered joint Wan implementation")
        return cls(match[1], int(match[2]), bool(int(match[3])))

    @classmethod
    def from_assignment(cls, assignment: Mapping[str, str]) -> "WanImplementation":
        require(set(assignment) == {"placement", "chunk_tokens", "lora_epilogue"}, "Wan plan has extra/missing decisions")
        require(assignment["chunk_tokens"] in {"512", "1024", "2048"}
                and assignment["lora_epilogue"] in {"on", "off"}, "invalid Wan decision")
        return cls(assignment["placement"], int(assignment["chunk_tokens"]), assignment["lora_epilogue"] == "on")

    def candidate(self) -> Candidate:
        kernels = (f"modulation_chunk{self.chunk_tokens}",) + (("lora_nograd_epilogue",) if self.lora_epilogue else ())
        return Candidate(self.id, WAN, self.native_placement, kernels,
                         ("nccl", "pointwise"), mechanism="joint_placement_and_existing_kernel_parameters")

    def environment(self) -> dict[str, str]:
        return {**PRECISION.launch_environment(),
                "DISTILLSCHEDULE_WAN13B_PLANNER_PRECISION": PRECISION.digest}


def current_implementation(environment: Mapping[str, str] | None = None) -> WanImplementation | None:
    key = (os.environ if environment is None else environment).get("DISTILLSCHEDULE_SEARCH_CANDIDATE_ID", "")
    return WanImplementation.parse(key) if key.startswith(PREFIX) else None


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
    require(env.get("DISTILLSCHEDULE_WAN13B_PLANNER_PRECISION") == PRECISION.digest, "planner precision identity changed")
    require(all(env.get(k) == v for k, v in PRECISION.launch_environment().items()), "planner precision launch environment changed")
    require(env.get("NVIDIA_TF32_OVERRIDE") in {None, ""}, "unregistered driver precision override")
    torch.backends.cuda.matmul.allow_tf32 = PRECISION.matmul_allow_tf32
    torch.backends.cudnn.allow_tf32 = PRECISION.cudnn_allow_tf32
    return PRECISION.assert_runtime(torch, env)
