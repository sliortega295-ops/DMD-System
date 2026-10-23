"""Bounded H3 controls lowered from the locked canonical profile, not a recipe.

v1 varies existing CPU-staged FA3 inflight depth and LoRA epilogue tile.
v2 additionally binds an exact retained-rollout window to the common action
runner, correctness-only until native action-qualified evidence is admitted.
v3 binds an explicit bounded shape and CPU-staged/trim or CUDA/no-trim policy.
The canonical one-cycle depth2/N128 profile is never mutated.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import Candidate, require, semantic_contract
from .h3_request import H3Request

H3 = "minimax_h3_grid1000_best"
PREFIX = "joint-h3v1-"
PATTERN = re.compile(r"joint-h3v1-d([12])-n(64|128)")
ACTION_PATTERN = re.compile(r"joint-h3v2-d([12])-n(64|128)-w([1-5])")
CACHE_PATTERN = re.compile(r"joint-h3v3-(h[0-9]+-w[0-9]+-f[0-9]+)-(cpu_staged_trim|cuda_no_trim)-d([12])-n(64|128)(?:-r([1-5]))?")
MODES = ("correctness", "diagnostic-timing", "calibration")
_PATH_FIELDS = (
    ("data", "train", "data_path"),
    ("model", "pretrained_model_name_or_path"),
    ("training", "a100", "adaln_grid", "manifest"),
    ("training", "output_dir"),
)


def normalized_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize pinned loader representations, never training decisions.

    Assets remain separately hash-bound by the owning execution receipt. A
    different run directory is not a different model configuration.
    """
    result = copy.deepcopy(dict(config))
    if "config_path" in result:
        require(isinstance(result["config_path"], str) and bool(result["config_path"]),
                "invalid H3 loader config_path")
        # LightX2V d034a6b runtime/config.py appends this provenance field.
        # receipt_binding separately checks it against the launcher's H3_CONFIG.
        result.pop("config_path")
    for path in _PATH_FIELDS:
        value = result
        for key in path[:-1]:
            require(isinstance(value.get(key), dict), f"H3 model config missing {'.'.join(path)}")
            value = value[key]
        require(isinstance(value.get(path[-1]), str) and bool(value[path[-1]]),
                f"H3 model config missing {'.'.join(path)}")
        value[path[-1]] = "{path:" + ".".join(path) + "}"
    for role in ("student", "fake"):
        optimizer = result["training"].get(role, {}).get("optimizer")
        require(isinstance(optimizer, dict), f"missing H3 {role} optimizer")
        for key in ("learning_rate", "adam_epsilon"):
            # PyYAML 1.1 reads JSON's 1e-08 / 2e-05 as strings. These are
            # explicitly consumed via float() by the pinned optimizer builder.
            value = optimizer.get(key)
            require(type(value) in (str, int, float), f"invalid H3 {role} {key}")
            value = float(value)
            require(math.isfinite(value), f"nonfinite H3 {role} {key}")
            optimizer[key] = value
    return result


@dataclass(frozen=True)
class H3Implementation:
    prefetch_depth: int = 2
    lora_epilogue_block_n: int = 128
    rollout_window: int | None = None
    cache_policy: str = "legacy_cpu_staged"
    request: H3Request = H3Request()

    def __post_init__(self) -> None:
        require(type(self.prefetch_depth) is int and self.prefetch_depth in (1, 2),
                "H3 cpu_staged prefetch depth must be 1 or 2")
        require(type(self.lora_epilogue_block_n) is int and self.lora_epilogue_block_n in (64, 128),
                "H3 LoRA epilogue block N must be 64 or 128")
        require(self.rollout_window is None or (type(self.rollout_window) is int and 1 <= self.rollout_window <= 5),
                "H3 rollout window must be an integer in [1,5]")
        require(isinstance(self.request, H3Request), "H3 request must be explicitly validated")
        require(self.cache_policy in ("legacy_cpu_staged", "cpu_staged_trim", "cuda_no_trim"),
                "unsupported H3 cache policy; only CPU staged/trim and CUDA/no-trim are admitted")
        require(self.cache_policy != "legacy_cpu_staged" or self.request == H3Request(),
                "noncanonical H3 shape requires an explicit v3 cache policy")
        require(self.storage != "cuda" or self.prefetch_depth == 2,
                "CUDA cache has no D2H prefetch decision; retain canonical inactive depth2")

    @property
    def storage(self) -> str:
        return "cuda" if self.cache_policy == "cuda_no_trim" else "cpu_staged"

    @property
    def trim_before_backward(self) -> bool:
        return self.storage == "cpu_staged"

    @property
    def id(self) -> str:
        if self.cache_policy != "legacy_cpu_staged":
            suffix = "" if self.rollout_window is None else f"-r{self.rollout_window}"
            return f"joint-h3v3-{self.request.id}-{self.cache_policy}-d{self.prefetch_depth}-n{self.lora_epilogue_block_n}{suffix}"
        if self.rollout_window is not None:
            return f"joint-h3v2-d{self.prefetch_depth}-n{self.lora_epilogue_block_n}-w{self.rollout_window}"
        return f"{PREFIX}d{self.prefetch_depth}-n{self.lora_epilogue_block_n}"

    @property
    def assignment(self) -> dict[str, str]:
        return {"prefetch_depth": str(self.prefetch_depth),
                "lora_epilogue_block_n": str(self.lora_epilogue_block_n),
                **({"fa3_cache_policy": self.cache_policy, "workload_request": self.request.id}
                   if self.cache_policy != "legacy_cpu_staged" else {}),
                **({"rollout_window": str(self.rollout_window)} if self.rollout_window is not None else {})}

    @classmethod
    def parse(cls, key: str) -> "H3Implementation":
        cache = CACHE_PATTERN.fullmatch(key) if isinstance(key, str) else None
        if cache is not None:
            return cls(int(cache[3]), int(cache[4]), int(cache[5]) if cache[5] else None,
                       cache[2], H3Request.parse(cache[1]))
        action = ACTION_PATTERN.fullmatch(key) if isinstance(key, str) else None
        if action is not None:
            return cls(int(action[1]), int(action[2]), int(action[3]))
        match = PATTERN.fullmatch(key) if isinstance(key, str) else None
        require(match is not None, "malformed/unregistered joint H3 implementation")
        return cls(int(match[1]), int(match[2]))

    @classmethod
    def from_assignment(cls, assignment: Mapping[str, str]) -> "H3Implementation":
        base = {"prefetch_depth", "lora_epilogue_block_n"}
        extended = base | {"fa3_cache_policy", "workload_request"}
        require(set(assignment) in (base, base | {"rollout_window"}, extended, extended | {"rollout_window"}),
                "H3 plan has extra/missing decisions")
        require(assignment["prefetch_depth"] in ("1", "2")
                and assignment["lora_epilogue_block_n"] in ("64", "128"), "invalid H3 decision")
        require("rollout_window" not in assignment or assignment["rollout_window"] in ("1", "2", "3", "4", "5"),
                "invalid H3 rollout window")
        if "fa3_cache_policy" in assignment:
            require(assignment["fa3_cache_policy"] in ("cpu_staged_trim", "cuda_no_trim"),
                    "v3 requires an explicit cache policy")
        return cls(int(assignment["prefetch_depth"]), int(assignment["lora_epilogue_block_n"]),
                   int(assignment["rollout_window"]) if "rollout_window" in assignment else None,
                   assignment.get("fa3_cache_policy", "legacy_cpu_staged"),
                   H3Request.parse(assignment["workload_request"]) if "workload_request" in assignment else H3Request())

    def candidate(self) -> Candidate:
        return Candidate(self.id, H3, "world16-fsdp2",
                         (f"lora_epilogue_n{self.lora_epilogue_block_n}",), ("h2d", "d2h", "pointwise"),
                         mechanism=("native_cpu_staged_depth_and_existing_lora_tile" if self.cache_policy == "legacy_cpu_staged"
                                    else "native_fa3_cache_residency_and_existing_lora_tile"))

    def environment(self) -> dict[str, str]:
        return {"DISTILLSCHEDULE_SEARCH_CANDIDATE_ID": self.id,
                "H3_FA3_REPLAY_CACHE_STORAGE": self.storage,
                "H3_FA3_REPLAY_CACHE_TRIM_BEFORE_BACKWARD": str(int(self.trim_before_backward)),
                "H3_FA3_REPLAY_CACHE_MAX_D2H_INFLIGHT": str(self.prefetch_depth),
                "H3_LORA_EPILOGUE_BLOCK_N": str(self.lora_epilogue_block_n),
                "H3_MAX_ITERS": "2", "H3_PLANNER_TWO_CYCLE": "1",
                "H3_PLANNER_CLOCK": "host_fenced", "H3_PLANNER_CALIBRATION": "0",
                **({"H3_PLANNER_ROLLOUT_WINDOW": str(self.rollout_window)} if self.rollout_window is not None else {})}

    @property
    def run_modes(self) -> tuple[str, ...]:
        # Common dispatch requires a hash-bound prior action proof for v2
        # non-correctness runs. Legacy sampled-state/macro receipts cannot admit it.
        return MODES

    def lower_profile(self, *, root: Path | None = None) -> dict[str, Any]:
        p = self.request.profile(root=root)
        semantics = semantic_contract(p)
        p["display_name"] = f"Generated native H3 diagnostic: {self.id}"
        p["allowed_run_modes"] = list(self.run_modes)
        p["required_assets"]["diffusers_source"] = {
            "kind": "directory", "environment": "H3_DIFFUSERS_SOURCE", "fingerprint_required": True,
            "identity_file": "src/diffusers/models/transformers/transformer_minimax_h3.py"}
        p["memory"]["pinned_staging_depth"] = self.prefetch_depth if self.storage == "cpu_staged" else 0
        p["memory"].update(fa3_replay_storage=self.storage, pageable_reservoir=self.storage == "cpu_staged")
        runtime = p["runtime"]
        runtime["environment"].update(self.environment())
        runtime["model_config"]["training"]["max_train_iters"] = 2
        runtime["model_config"]["training"]["a100"]["fa3_replay_cache"].update(
            max_d2h_inflight=self.prefetch_depth, storage=self.storage, trim_before_backward=self.trim_before_backward)
        runtime.update(planner_implementation=self.assignment,
                       native_run_modes={mode: mode for mode in self.run_modes},
                       native_two_cycle_adapter_available=True, search_two_cycle_ready=False,
                       strict_pair_gate="NOT_RUN")
        p["contract"].update(continuous_cycles=2, final_versions={"generator": 2, "critic": 10, "ema": 0})
        require(semantic_contract(p) == semantics, "H3 lowering changed frozen workload semantics")
        return p

    def receipt_binding(self, config: Mapping[str, Any], environment: Mapping[str, str], *,
                        mode: str = "correctness", root: Path | None = None) -> dict[str, Any]:
        """Setup-only config/flag gate, safe before importing any H3 CUDA code.

        Call with the actual loaded config and process environment, not a
        reconstructed receipt. Calibration is explicit and mode-bound; it is
        excluded from the training-config digest but reported separately.
        """
        require(mode in self.run_modes, "unsupported H3 native mode (action evidence precedes timing/calibration)")
        require(environment.get("H3_PLANNER_ROLLOUT_WINDOW", "") ==
                ("" if self.rollout_window is None else str(self.rollout_window)),
                "H3 rollout window differs from generated identity")
        if "config_path" in config:
            path = config["config_path"]
            expected_path = environment.get("H3_CONFIG")
            require(isinstance(path, str) and bool(path)
                    and isinstance(expected_path, str) and bool(expected_path)
                    and Path(path).resolve() == Path(expected_path).resolve(),
                    "H3 loader config_path differs from launcher H3_CONFIG")
        runtime = self.lower_profile(root=root)["runtime"]
        wanted = runtime["environment"]
        for key, expected in wanted.items():
            if key.startswith("H3_") or key == "DISTILLSCHEDULE_SEARCH_CANDIDATE_ID":
                if key == "H3_PLANNER_CALIBRATION":
                    expected = "1" if mode == "calibration" else "0"
                require(environment.get(key) == expected,
                        f"H3 launch {key} mismatch (calibration must match mode)" if key == "H3_PLANNER_CALIBRATION"
                        else f"H3 launch {key} mismatch")
        actual_config = normalized_model_config(config)
        require(actual_config == normalized_model_config(runtime["model_config"]),
                "H3 model config differs from exact generated canonical copy")
        effective_environment = {key: environment[key] for key in wanted
                                 if key.startswith("H3_") and key != "H3_PLANNER_CALIBRATION"}
        return {"candidate_id": self.id, "planner_implementation": self.assignment,
                "effective_configuration_sha256": fingerprint({
                    "model_config": actual_config, "environment": effective_environment})}


def current_implementation(environment: Mapping[str, str] | None = None) -> H3Implementation | None:
    env = os.environ if environment is None else environment
    key = env.get("DISTILLSCHEDULE_SEARCH_CANDIDATE_ID", "")
    # Any H3-family generated ID must parse; a future version cannot silently
    # fall back to canonical behavior.
    return H3Implementation.parse(key) if key.startswith("joint-h3") else None


def implementations(*, request: H3Request = H3Request(), explicit_cache: bool = False,
                    rollout_windows=(None,)) -> tuple[H3Implementation, ...]:
    """Legal combinations only; presence in this inventory never supplies a cost."""
    policies = ("cpu_staged_trim", "cuda_no_trim") if explicit_cache else ("legacy_cpu_staged",)
    return tuple(H3Implementation(depth, tile, window, policy, request)
                 for policy in policies for depth in ((2,) if policy == "cuda_no_trim" else (1, 2))
                 for tile in (64, 128) for window in rollout_windows)


def validate_cache_registration(actual, impl):
    """Check the installed native object, not merely the selected environment."""
    expected = dict(enabled=True, block_indices=list(range(50)), storage=impl.storage,
                    max_d2h_inflight=impl.prefetch_depth, trim_before_backward=impl.trim_before_backward)
    require(isinstance(actual, dict) and all(type(actual.get(k)) is type(v) and actual[k] == v
                                           for k, v in expected.items()),
            "H3 actual FA3 registration differs from selected cache policy")
    stats = actual.get("stats", {})
    require(isinstance(stats, dict), "H3 native cache counters must be an object")
    required = ("compact_forward_impl_calls", "compact_backward_calls", "cpu_d2h_bytes", "cpu_h2d_bytes",
                "cpu_pool_allocated_bytes", "cpu_stage_allocated_bytes", "allocator_trim_calls")
    require(all(type(stats.get(k)) is int and stats[k] >= 0 for k in required),
            "H3 actual FA3 registration lacks native cache counters")
    require(stats["compact_forward_impl_calls"] > 0 and stats["compact_backward_calls"] > 0,
            "H3 selected FA3 cache was not executed")
    if impl.storage == "cuda":
        require(all(value == 0 for key, value in stats.items()
                    if key.startswith("cpu_") or key in ("allocator_trim_calls", "unexpected_cpu_storage_calls")),
                "H3 CUDA cache unexpectedly used CPU staging or allocator trim")
