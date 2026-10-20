"""Bounded Helios scheduling controls lowered from immutable canonical profiles.

Generated identities do not themselves authorize execution or timing. The
controller must set an explicit adapter mode and retain its source-bound gates.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from distillschedule.workloads.best_known.registry import fingerprint, load_profile
from .contracts import Candidate, require, semantic_contract
from .helios_safe_bundle import BUNDLE_ID, HeliosSafeBundle
from .helios_gan_bundle import GAN_BUNDLE_ID, GAN_BUNDLE_ENV, HeliosGANBundle
from .helios_request import (OriginalHeliosRequest, TRACE_KEY, TRACE_HASH_KEY,
                             REQUEST_ENV, REQUEST_HASH_ENV)

NO_GAN = "helios14b_nogan_best"
GAN = "helios14b_gan_best"
PREFIX = "joint-helios1-"
PATTERN = re.compile(r"joint-helios1-(nogan|gan)-(fsdp|hsdp)-c(512|2048)-p([01])-w([1-5])"
                     r"(?:-k(safe-nogan-ed8-v1|gan-fa3-down40-nomerge-v1))?(?:-qf(17|33)-s(111|222)(-b4)?)?")
MODES = ("correctness", "calibration", "native")
MODE_ENV = "DISTILLSCHEDULE_HELIOS_PLANNER_MODE"
ASSIGNMENT_ENV = "DISTILLSCHEDULE_HELIOS_PLANNER_ASSIGNMENT"


@dataclass(frozen=True)
class HeliosImplementation:
    variant: str
    mesh: str
    chunk_tokens: int
    rollout_window: int
    persistent_unshard: bool
    candidate_bundle: HeliosSafeBundle | HeliosGANBundle = HeliosSafeBundle()
    request: OriginalHeliosRequest | None = None

    def __post_init__(self) -> None:
        require(self.variant in ("no-gan", "gan"), "unsupported Helios variant")
        require(self.mesh in ("world16-fsdp", "node-hsdp"), "unsupported Helios mesh")
        require(type(self.chunk_tokens) is int and self.chunk_tokens in (512, 2048),
                "Helios chunk tokens must be 512 or 2048")
        require(type(self.rollout_window) is int and 1 <= self.rollout_window <= 5,
                "Helios rollout window must be an integer in [1,5]")
        require(type(self.persistent_unshard) is bool, "Helios persistence must be boolean")
        require(self.variant != "gan" or self.mesh == "world16-fsdp",
                "GAN initially supports only world16-fsdp")
        require(not self.persistent_unshard or (self.variant == "gan" and self.rollout_window == 5),
                "persistent unshard requires GAN and a single five-rollout scope")
        require(type(self.candidate_bundle) in (HeliosSafeBundle, HeliosGANBundle), "Helios candidate bundle must be typed")
        require(self.request is None or (type(self.request) is OriginalHeliosRequest
                and self.request.variant == self.variant), "Helios request must be typed and match variant")
        self.candidate_bundle.validate_scope(self)

    @property
    def id(self) -> str:
        variant = "nogan" if self.variant == "no-gan" else "gan"
        mesh = "fsdp" if self.mesh == "world16-fsdp" else "hsdp"
        base = f"{PREFIX}{variant}-{mesh}-c{self.chunk_tokens}-p{int(self.persistent_unshard)}-w{self.rollout_window}"
        return (base + ("-k" + self.candidate_bundle.id if self.candidate_bundle.enabled else "")
                + ("-q" + self.request.id if self.request is not None else ""))

    @property
    def workload(self) -> str:
        return NO_GAN if self.variant == "no-gan" else GAN

    @property
    def trace_hash_environment(self) -> str:
        return "DISTILLSCHEDULE_GAN_TRACE_HASH" if self.variant == "gan" else "DISTILLSCHEDULE_TRACE_BUNDLE_HASH"

    @property
    def assignment(self) -> dict[str, str]:
        result = dict(variant=self.variant, mesh=self.mesh, chunk_tokens=str(self.chunk_tokens),
                    rollout_window=str(self.rollout_window),
                    persistent_unshard="on" if self.persistent_unshard else "off")
        if self.candidate_bundle.enabled:
            result["candidate_bundle"] = self.candidate_bundle.id
        if self.request is not None:
            result["original_request"] = json.dumps(self.request.to_dict(), sort_keys=True, separators=(",", ":"))
        return result

    @classmethod
    def parse(cls, key: str) -> "HeliosImplementation":
        match = PATTERN.fullmatch(key) if isinstance(key, str) else None
        require(match is not None, "malformed/unregistered joint Helios implementation")
        return cls("no-gan" if match[1] == "nogan" else "gan",
                   "world16-fsdp" if match[2] == "fsdp" else "node-hsdp",
                   int(match[3]), int(match[5]), match[4] == "1",
                   _parse_bundle(match[6] or "none"),
                   OriginalHeliosRequest(variant="no-gan" if match[1] == "nogan" else "gan",
                       video_frames=int(match[7]), stage2_inference_steps=tuple(map(int, match[8])),
                       stage_schedule_policy="benchmark_v4_strict_global" if match[9] else "source_normal_only")
                   if match[7] is not None else None)

    @classmethod
    def from_assignment(cls, assignment: Mapping[str, str]) -> "HeliosImplementation":
        keys = {"variant", "mesh", "chunk_tokens", "rollout_window", "persistent_unshard"}
        require(keys <= set(assignment) <= keys | {"candidate_bundle", "original_request"},
                "Helios assignment has extra/missing decisions")
        require(assignment["chunk_tokens"] in ("512", "2048")
                and assignment["rollout_window"] in ("1", "2", "3", "4", "5")
                and assignment["persistent_unshard"] in ("on", "off"), "invalid Helios decision")
        request = None
        if "original_request" in assignment:
            try:
                request = OriginalHeliosRequest.from_dict(json.loads(assignment["original_request"]))
            except (TypeError, ValueError) as error:
                raise ValueError("malformed Helios original request assignment") from error
            require(assignment["original_request"] == json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":")),
                    "Helios original request assignment must be complete/canonical")
        return cls(assignment["variant"], assignment["mesh"], int(assignment["chunk_tokens"]),
                   int(assignment["rollout_window"]), assignment["persistent_unshard"] == "on",
                   _parse_bundle(assignment.get("candidate_bundle", "none")), request)

    @property
    def run_modes(self) -> tuple[str, ...]:
        return MODES

    def candidate(self) -> Candidate:
        kernels = (f"modulation_chunk{self.chunk_tokens}",)
        if self.candidate_bundle.enabled:
            kernels += (self.candidate_bundle.id,)
        return Candidate(self.id, self.workload, self.mesh, kernels,
                         ("nccl", "pointwise", "residency"), variant=self.variant,
                         mechanism="native_mesh_chunk_retained_rollout_window_and_bounded_unshard")

    def environment(self, *, root: Path | None = None) -> dict[str, str]:
        env = dict(load_profile(self.workload, root=root)["runtime"]["environment"])
        env.update({"DISTILLSCHEDULE_SEARCH_CANDIDATE_ID": self.id,
                    ASSIGNMENT_ENV: json.dumps(self.assignment, sort_keys=True, separators=(",", ":")),
                    MODE_ENV: "off",
                    "DISTILLSCHEDULE_HELIOS_MODULATION_CHUNK_TOKENS": str(self.chunk_tokens),
                    # The pinned stride2 replay path has measured systematic
                    # drift. Generated plans use native checkpoint recompute;
                    # historical canonical recipes remain unchanged.
                    "DISTILLSCHEDULE_HELIOS_SELECTIVE_FLASH_CHECKPOINT_STRIDE": "0",
                    "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_PERSISTENT_UNSHARD":
                        "on" if self.persistent_unshard else "off"})
        env.update(self.candidate_bundle.environment())
        if self.request is not None:
            env.update({REQUEST_ENV: self.assignment["original_request"], REQUEST_HASH_ENV: self.request.digest})
            # Replaced only by the compiler's exact bundle.json asset binding;
            # a raw environment cannot silently reuse the old canonical trace.
            env[self.trace_hash_environment] = "ORIGINAL_HELIOS_TRACE_ASSET_BINDING_REQUIRED"
        return env

    def lower_profile(self, *, root: Path | None = None) -> dict[str, Any]:
        profile = copy.deepcopy(load_profile(self.workload, root=root))
        semantics = semantic_contract(profile)
        if self.request is not None:
            profile["contract"].update(self.request.trace_contract())
            profile["required_assets"]["trace"].pop("logical_hash", None)
            profile["semantics"] = ("original_helios_benchmark_v4_strict_global_native_rank_grids_not_upstream_local_exit_distribution"
                if self.request.stage_schedule_policy == "benchmark_v4_strict_global" else
                "original_helios_request_native_helpers_requires_fresh_correctness")
            # This is an explicit NEW workload cohort; no canonical result or
            # baseline timing is inherited for another shape or active horizon.
            semantics = semantic_contract(profile)
        profile["display_name"] = f"Generated native Helios two-cycle diagnostic: {self.id}"
        profile["allowed_run_modes"] = [*self.run_modes, "diagnostic-timing"]
        profile["topology"].update(placement=self.mesh, replicate=2 if self.mesh == "node-hsdp" else 1,
                                   shard=8 if self.mesh == "node-hsdp" else 16)
        runtime = profile["runtime"]
        runtime["arguments"][7] = self.mesh
        runtime["environment"].update(self.environment(root=root))
        runtime.update(planner_implementation=self.assignment,
                       native_run_modes={mode: "timing" for mode in (*self.run_modes, "diagnostic-timing")},
                       native_two_cycle_adapter_available=True, search_two_cycle_ready=False,
                       strict_pair_gate="NOT_RUN")
        profile["historical"] = dict(formal_speedup_eligible=False,
            scope="GENERATED_NATIVE_RECOMPUTE_REQUIRES_FRESH_CALIBRATION",
            parent_canonical_profile_sha256=fingerprint(load_profile(self.workload, root=root)))
        initial = profile["contract"]["initial_versions"]
        updates = profile["contract"]["updates"]
        profile["contract"].update(continuous_cycles=2,
            final_versions={key: initial[key] + 2*updates[key] for key in initial})
        require(semantic_contract(profile) == semantics, "Helios lowering changed frozen workload semantics")
        return profile

    def version_mapping(self, contract: Mapping[str, Any]) -> dict[str, Any]:
        """Schedule labels are not Adam step counters in the GAN bootstrap."""
        logical = dict(contract["initial_versions"])
        physical = (dict(generator=2, critic=10, ema=800)
                    if self.variant == "gan" else dict(logical))
        return dict(physical_initial=physical, logical_initial=logical,
                    logical_minus_physical={key: logical[key]-physical[key] for key in logical})

    def receipt_binding(self, *, args: Any, policy: Any,
                        environment: Mapping[str, str] | None = None) -> dict[str, Any]:
        env = os.environ if environment is None else environment
        require(current_implementation(env) == self, "Helios runtime identity mismatch")
        wanted = self.environment()
        for key, value in wanted.items():
            if key != MODE_ENV and not (self.request is not None and key == self.trace_hash_environment):
                require(env.get(key) == value, f"Helios launch {key} differs from canonical generated copy")
        if self.request is not None:
            actual_trace_hash = env.get(self.trace_hash_environment, "")
            require(re.fullmatch(r"[0-9a-f]{64}", actual_trace_hash) is not None
                    and getattr(args, "expected_trace_hash", None) == actual_trace_hash,
                    "Helios effective expected trace hash differs from exact launch binding")
            wanted[self.trace_hash_environment] = actual_trace_hash
        require(args.variant == self.variant and policy.mesh_policy == self.mesh
                and policy.modulation_chunk_tokens == self.chunk_tokens
                and policy.generator_nograd_persistent_unshard == self.persistent_unshard,
                "Helios actual scheduling policy differs from generated identity")
        require(args.timing_mode and not args.profile_mode and not args.profile_diagnostics,
                "Helios planner requires unprofiled timing-mode helpers; correctness probes run outside cycles")
        require(policy.rollout_batch_size == 1 and not policy.pack_real_score_pair,
                "Helios generated plans preserve B1 and unpacked Score calls")
        require(policy.rollout_backend == "native-fsdp2-batch", "Helios rollout backend changed")
        stride = 2 if type(self.candidate_bundle) is HeliosGANBundle else 0
        require(type(policy.selective_flash_checkpoint_stride) is int
                and policy.selective_flash_checkpoint_stride == stride,
                "Helios effective checkpoint policy differs from the explicit bundle")
        require(args.ema_device == "cuda" and args.attention_backend == "local-fa2"
                and args.generator_residual_storage == args.critic_residual_storage == "cuda_out_of_place",
                "Helios generated plans cannot change frozen residency/attention/residual policies")
        require(args.critic_only_step is None and not args.operation_source,
                "Helios generated plan is not a single-Critic/profiler diagnostic")
        if not self.candidate_bundle.enabled:
            require(getattr(policy, "lora_b_gemm_epilogue", False) is False
                    and getattr(args, "lora_checkpoint_replay_scope", False) is False,
                    "Helios bundle-off cannot enable new LoRA-B/replay-scope mechanisms")
        # The launcher translates these canonical environment flags into args
        # and policy fields. Check the effective values as well as the env.
        for key, value in wanted.items():
            if not key.startswith("DISTILLSCHEDULE_") or value not in ("on", "off"):
                continue
            name = key.removeprefix("DISTILLSCHEDULE_").removeprefix("HELIOS_").lower()
            for owner in (args, policy):
                if hasattr(owner, name):
                    require(getattr(owner, name) is (value == "on"), f"Helios effective {name} mismatch")
        binding = dict(candidate_id=self.id, planner_implementation=self.assignment, mode=env[MODE_ENV],
                       effective_configuration_sha256=fingerprint({k: v for k,v in wanted.items() if k != MODE_ENV}))
        if self.request is not None:
            require(getattr(args, "original_helios_request", None) == self.request,
                    "Helios effective worker request differs from generated identity")
            binding.update({TRACE_KEY: self.request.to_dict(), TRACE_HASH_KEY: self.request.digest})
            binding["expected_trace_hash"] = actual_trace_hash
        if self.candidate_bundle.enabled:
            binding["candidate_bundle"] = self.candidate_bundle.bind_effective(args=args, policy=policy)
        return binding


def current_implementation(environment: Mapping[str, str] | None = None) -> HeliosImplementation | None:
    env = os.environ if environment is None else environment
    key = env.get("DISTILLSCHEDULE_SEARCH_CANDIDATE_ID", "")
    mode = env.get(MODE_ENV, "off")
    if not key.startswith("joint-helios"):
        require(mode == "off" and not env.get(ASSIGNMENT_ENV)
                and not env.get(REQUEST_ENV) and not env.get(REQUEST_HASH_ENV)
                and not env.get(GAN_BUNDLE_ENV),
                "Helios planner mode/assignment/request without generated identity")
        return None
    impl = HeliosImplementation.parse(key)
    require(env.get(GAN_BUNDLE_ENV, "") == (GAN_BUNDLE_ID if type(impl.candidate_bundle) is HeliosGANBundle else ""),
            "GAN manual runtime opt-in differs from generated identity")
    if impl.request is None:
        require(not env.get(REQUEST_ENV) and not env.get(REQUEST_HASH_ENV),
                "Helios original request requires explicit request-qualified identity")
    else:
        require(env.get(REQUEST_ENV) == impl.assignment["original_request"]
                and env.get(REQUEST_HASH_ENV) == impl.request.digest,
                "Helios original request environment/hash differs from identity")
    if not impl.candidate_bundle.enabled:
        require(all(env.get(name, "off") == "off" for name in (
            "DISTILLSCHEDULE_HELIOS_LORA_B_GEMM_EPILOGUE",
            "DISTILLSCHEDULE_HELIOS_LORA_CHECKPOINT_REPLAY_SCOPE")),
            "Helios bundle-off identity cannot enable new LoRA-B/replay-scope environment flags")
    require(mode in MODES, "Helios generated identity requires explicit correctness/calibration/native mode")
    try:
        assignment = json.loads(env.get(ASSIGNMENT_ENV, ""))
    except (ValueError, TypeError) as error:
        raise ValueError("Helios generated assignment is missing/malformed") from error
    require(assignment == impl.assignment, "Helios assignment differs from generated identity")
    return impl


def _parse_bundle(value):
    return HeliosGANBundle.manual() if value == GAN_BUNDLE_ID else HeliosSafeBundle.parse(value)
