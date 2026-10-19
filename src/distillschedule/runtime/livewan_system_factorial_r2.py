"""Bounded, opt-in full-system factorial integration; never a new trainer.

The original Cxx/two-cycle contracts remain immutable. Full only_B remains
blocked; its separately named compatible subset does not claim the complete B.
"""
from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import hashlib
import json
import math
import os
import re
import subprocess
import socket
from typing import Any, Mapping

KEY = "system_factorial_r2"
SCHEMA = "distillschedule.livewan_system_factorial_r2.v1"
RUN_KIND = "system-factorial-timing"
CONTINUOUS_NAME = "system_factorial_c51_depth1_c4_v1"
ARMS = ("baseline", "only_A", "full")
COMPATIBLE_ARM = "only_B_fsdp_compatible"
COMPATIBLE_KEY = "system_factorial_fsdp_compatible_B_v1"
BASELINE_ARMS = ("baseline", COMPATIBLE_ARM)
COMPATIBLE_OPERATORS = ("critic_lora_kernel", "score_modulation_kernel",
                        "score_rmsnorm_kernel", "score_rope_kernel")
COMPATIBLE_OFF = {
    "generator_rollout_cast_cache": "A_X_B_RESIDENT_NON_DTENSOR_INSTALLER",
    "generator_rms_epilogue": "A_X_B_RESIDENT_FP32_GENERATOR_INSTALLER",
    "critic_attention_checkpoint": "A_X_B_SCORE_B2_QUERY19320_FA3_REPLAY_CONTRACT",
    "generator_cross_attention_kv_cache": "A_X_B_UNADAPTED_FSDP_PARAMETER_LIFETIME",
    "generator_attention_backend": "A_X_B_UNADAPTED_CAUSAL_LAYER_SCOPE",
    "score_attention_backend": "A_X_B_PINNED_FA3_CHECKPOINT_COMPOSITION",
    "generator_modulation": "A_X_B_RESIDENT_GENERATOR_SCOPE",
    "generator_gate": "A_X_B_RESIDENT_GENERATOR_SCOPE",
    "generator_residual_layout": "A_X_B_RESIDENT_GENERATOR_SCOPE",
    "score_residual_gate": "A_X_B_PINNED_C34_COMPOSITION",
    "score_layernorm": "A_X_B_PINNED_C35_COMPOSITION",
    "generator_readonly_splitkv": "A_X_B_UNADAPTED_GENERATOR_BACKEND_SCOPE",
    "full_context_lens_elision": "ORIGINAL_512_LENGTHS_PRESERVED",
}
CAPACITY_ARM = "only_A_with_capacity_offload"
CAPACITY_KEY = "system_factorial_capacity_offload_v1"
ARMS = (*ARMS, CAPACITY_ARM, COMPATIBLE_ARM)
A_ARMS = ("only_A", CAPACITY_ARM)
CAPACITY_POLICY = dict(name="generator_cpu_foreach_adamw_v1",
    scope="GENERATOR_OPTIMIZER_ONLY", generator_optimizer_device="cpu",
    critic_optimizer_device="cuda", parameter_offload=False,
    activation_offload=False, generator_numel=1418996800,
    transfers_and_cpu_update_inside_timer=True)
CYCLES = 4
SEED = 20260830
BASE_PLAN = "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_fsdp2.json"
FULL_PLAN = "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c51_generator_fused_rms.json"
B_METADATA = frozenset({
    "full_context_lens_elision", "critic_lora_kernel", "generator_optimizer_impl",
    "capacity_recovery", "score_rope_kernel", "score_attention_backend",
    "generator_cross_attention_kv_cache", "generator_attention_backend",
    "score_modulation_kernel", "score_rmsnorm_kernel", "generator_rollout_cast_cache",
    "generator_rms_epilogue", "critic_attention_checkpoint", "checkpoint_composition",
    "generator_modulation", "generator_gate", "generator_residual_layout",
    "score_residual_gate", "score_layernorm", "generator_readonly_splitkv",
})


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("R2: " + message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _template(root: Path, arm: str) -> dict[str, Any]:
    from distillschedule.core.config import load_plan
    loaded = load_plan(root / (BASE_PLAN if arm in BASELINE_ARMS else FULL_PLAN))
    for path in loaded.source_files:
        rel = str(path.relative_to(root))
        require(rel in PINNED_PLANS and hashlib.sha256(path.read_bytes()).hexdigest() == PINNED_PLANS[rel],
                "frozen template/source drift: " + rel)
    return loaded.plan.to_dict()


def make_plan(repo_root: str | Path, *, arm: str, source_head: str,
              seed: int = SEED) -> dict[str, Any]:
    require(arm != "only_B", "only_B BLOCKED: full B has real resident/role/batch dependencies")
    require(arm in ARMS, "unknown arm")
    require(re.fullmatch(r"[0-9a-f]{40}", source_head) is not None, "full source HEAD required")
    require(type(seed) is int and seed == SEED, "seed differs from frozen cohort")
    root = Path(repo_root).resolve()
    plan = _template(root, arm)
    template_sha = digest(plan)
    if arm in BASELINE_ARMS:
        from distillschedule.runtime.livewan_baseline_placement import apply_contract
        apply_contract(plan, mode="baseline", name="world16-fsdp", teacher_offload="off",
                       optimizer_device="cuda", student_padding="max-noop")
        if arm == COMPATIBLE_ARM:
            parent = _template(root, "full")
            plan["metadata"][COMPATIBLE_KEY] = dict(
                name="fsdp_scoped_score_operators_v1",
                scope="explicit_critic_teacher_calls_all_ranks",
                operators={key: parent["metadata"][key] for key in COMPATIBLE_OPERATORS},
                disabled=dict(COMPATIBLE_OFF), generator_optimizer_impl="fused_adamw",
                checkpoint="ORIGINAL_RECOMPUTE_ORIGINAL_RNG", complete_B=False,
                parameter_placement="UNCHANGED_WORLD16_FULL_SHARD")
            plan["training_backends"]["generator"]["options"]["optimizer_impl"] = "fused_adamw"
    else:
        if arm in A_ARMS:
            for key in B_METADATA:
                plan["metadata"].pop(key, None)
            plan["metadata"]["full_context_lens_elision"] = False
            plan["training_backends"]["generator"]["options"]["optimizer_impl"] = "foreach_adamw"
        plan["metadata"]["continuous_pipeline"].update(
            name=CONTINUOUS_NAME, cycles=CYCLES, critic_shadow_sync_versions=[5, 10, 15])
    if arm == CAPACITY_ARM:
        plan["metadata"][CAPACITY_KEY] = dict(CAPACITY_POLICY)
    plan["id"] = f"livewan-full-system-factorial-r2-{arm}-c4"
    plan["evidence"]["steps"] = 20
    plan["metadata"][KEY] = dict(
        schema_version=SCHEMA, arm=arm, source_head=source_head, seed=seed,
        cycles=CYCLES, warmup_cycles=[1], measured_cycles=[2, 3, 4],
        template_sha256=template_sha, qualification="C1_OBSERVED_SEMANTICS_DEFERRED_EXTRACTION",
        numerical_status="REVIEW_NOT_STRICT_EQUIVALENCE", profiler=False,
        independent_initializations=1)
    return plan


def validate(plan: Mapping[str, Any], *, mode: str,
             repo_root: str | Path | None = None) -> dict[str, Any]:
    binding = plan.get("metadata", {}).get(KEY)
    require(isinstance(binding, dict), "missing versioned binding")
    root = Path(repo_root or os.environ.get("DMD_SYSTEM_ROOT", "")).resolve()
    require((root / FULL_PLAN).is_file(), "repository root missing")
    expected_head = os.environ.get("LIVEWAN_EXPECTED_HEAD", "")
    require(binding.get("source_head") == expected_head, "source HEAD binding changed")
    arm = binding.get("arm")
    expected = make_plan(root, arm=arm, source_head=expected_head, seed=binding.get("seed"))
    require(mode == mode_for_arm(arm), "wrong execution mode")
    require(dict(plan) == expected, "lowered plan/shape/flags/source binding changed")
    return binding


def active(plan: Mapping[str, Any]) -> bool:
    return KEY in plan.get("metadata", {})


def mode_for_arm(arm: str) -> str:
    return "baseline" if arm in BASELINE_ARMS else "pipeline"


def only_b_compatible(plan: Mapping[str, Any], *, mode: str) -> bool:
    return active(plan) and validate(plan, mode=mode)["arm"] == COMPATIBLE_ARM


def only_a(plan: Mapping[str, Any], *, mode: str) -> bool:
    if not active(plan):
        return False
    return validate(plan, mode=mode)["arm"] in A_ARMS


def generator_optimizer_cpu(plan: Mapping[str, Any], *, mode: str) -> bool:
    """Independent capacity fallback; global baseline placement stays unchanged."""
    if not active(plan):
        return False
    return validate(plan, mode=mode)["arm"] == CAPACITY_ARM


def attach_capacity_optimizer_receipt(bundle: Any, *, plan: Mapping[str, Any],
                                      mode: str, rank: int) -> None:
    """Read actual optimizer placement after all four timing endpoints."""
    if not generator_optimizer_cpu(plan, mode=mode):
        return
    import resource
    observed = dict(CAPACITY_POLICY)
    if rank < 8:
        from .livewan_cpu_optimizer import CPUAdamW
        opt = bundle.generator_optimizer
        require(isinstance(opt, CPUAdamW), "capacity arm did not execute CPUAdamW")
        proof = opt.receipt()
        proof.update(cpu_gradient_buffer_bytes=sum(p.numel() * p.element_size() for p in opt.grad_buffers),
            pinned_parameter_mirror_bytes=sum(p.numel() * p.element_size() for p in opt.mirrors if p.is_pinned()),
            pinned_gradient_buffer_bytes=sum(p.numel() * p.element_size() for p in opt.grad_buffers if p.is_pinned()))
        bundle.metadata["student"]["cpu_optimizer"] = proof
        observed.update(execution_status="PASS", process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    else:
        observed["execution_status"] = "NOT_APPLICABLE_SCORE_RANK"
    bundle.metadata["capacity_offload"] = observed


def lower_parent(parent: Mapping[str, Any], *, repo_root: str | Path,
                 arm: str, source_head: str, seed: int = SEED) -> dict[str, Any]:
    require(dict(parent) == _template(Path(repo_root).resolve(), arm), "rendered parent template drift")
    return make_plan(repo_root, arm=arm, source_head=source_head, seed=seed)


def validate_native_environment(*, repo_root: str | Path, mode: str) -> None:
    """CPU-only preconstruction checks for the explicit native experiment."""
    root = Path(repo_root).resolve()
    arm = os.environ.get("LIVEWAN_SYSTEM_FACTORIAL_ARM")
    require(arm in ARMS, "only_B BLOCKED or missing/unknown arm")
    require(mode == mode_for_arm(arm), "arm/mode mismatch")
    require(os.environ.get("LIVEWAN_SYSTEM_FACTORIAL_SEED") == str(SEED), "seed missing/changed")
    require(not any(os.environ.get(k) for k in ("DISTILLSCHEDULE_NSYS_BIN", "DISTILLSCHEDULE_NCU_BIN")),
            "native factorial forbids profilers")
    require(os.environ.get("LIVEWAN_BASELINE_PLACEMENT", "legacy-hybrid") ==
            ("world16-fsdp" if arm in BASELINE_ARMS else "legacy-hybrid"), "baseline placement changed")
    require(os.environ.get("LIVEWAN_STUDENT_ROLLOUT_PADDING", "off") ==
            ("max-noop" if arm in BASELINE_ARMS else "off"), "collective padding changed")
    require(os.environ.get("LIVEWAN_OPTIMIZER_DEVICE", "cuda") == "cuda"
            and os.environ.get("LIVEWAN_TEACHER_PARAMETER_OFFLOAD", "off") == "off", "offload changed")
    observed = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    require(observed == os.environ.get("LIVEWAN_EXPECTED_HEAD"), "runtime source HEAD mismatch")
    for flags in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        require(subprocess.call(["git", "-C", str(root), *flags]) == 0, "dirty runtime source")


def validate_runner(args: Any) -> None:
    arm = args.system_factorial_arm
    require(arm != "only_B", "only_B BLOCKED pending explicit user definition")
    require(arm in ARMS and args.system_factorial_seed == SEED, "arm/seed missing or changed")
    require(args.execution_mode == mode_for_arm(arm), "arm/mode mismatch")
    require(args.pipeline_variant == ("b1" if arm in BASELINE_ARMS
            else "c51-continuous-b16-generator-fused-rms"), "wrong canonical parent")
    require(args.baseline_placement == ("world16-fsdp" if arm in BASELINE_ARMS else "legacy-hybrid")
            and args.student_rollout_padding == ("max-noop" if arm in BASELINE_ARMS else "off"),
            "strict FSDP baseline or pipeline placement missing")
    require(args.teacher_parameter_offload == "off" and args.optimizer_device == "cuda",
            "offload contract changed")
    require(not args.profile_two_cycles and not args.nsys_bin and not args.ncu_bin,
            "R2 native factorial cannot profile")


def validate_native_topology(torch: Any, dist: Any, *, rank: int, local_rank: int) -> None:
    """Original model-free checks folded into the single native allocation."""
    require(str(torch.__version__) == "2.10.0+cu128", "target Torch identity changed")
    require(dist.get_world_size() == 16 and dist.get_rank() == rank,
            "runtime world/rank mismatch")
    require(torch.cuda.device_count() == 8 and local_rank == rank % 8,
            "runtime local device topology mismatch")
    value = dict(rank=rank, local_rank=local_rank, node=rank // 8, host=socket.gethostname())
    rows: list[Any] = [None] * 16
    dist.all_gather_object(rows, value)
    require(sorted(x["rank"] for x in rows) == list(range(16)), "runtime rank coverage")
    for node in (0, 1):
        subset = [x for x in rows if x["node"] == node]
        require(len(subset) == 8 and len({x["host"] for x in subset}) == 1
                and sorted(x["local_rank"] for x in subset) == list(range(8)), "runtime node coverage")
    require(rows[0]["host"] != rows[8]["host"], "two node owners resolved to one host")


def finish_finite(torch: Any, evidence: Any, *, rank: int, mode: str) -> dict[str, Any]:
    checks = getattr(evidence, "system_factorial_c1_checks", [])
    expected = 6 if mode == "baseline" else 1 if rank < 8 else 5
    require(len(checks) == expected, "observed C1 gradient-norm finite census changed")
    finite = bool(torch.stack(checks).all().detach().cpu())
    require(finite, "non-finite C1 gradient norm")
    return dict(scope="C1_OBSERVED_GRADIENT_NORMS", count=len(checks), finite=finite,
                extraction="AFTER_C4_ENDPOINT_NO_MIDPOINT_HOST_READ", numerical_parity="NOT_ESTABLISHED")


class RecomputeCheckpointAudit:
    """Delegate to the original checkpoint; count real function entry/re-entry.

    No context_fn, saved-tensor policy, RNG override, attention or gradient
    substitution. Non-reentrant early-stop still counts on function entry.
    """
    def __init__(self, checkpoint_module: Any):
        self.module = checkpoint_module
        self.calls = 0
        self.entries = 0

    def checkpoint(self, function: Any, *args: Any, **kwargs: Any):
        require(kwargs.get("use_reentrant") is False and "context_fn" not in kwargs
                and "preserve_rng_state" not in kwargs, "native checkpoint/RNG call site changed")
        self.calls += 1

        def original(*values: Any, **options: Any):
            self.entries += 1
            return function(*values, **options)

        return self.module.checkpoint(original, *args, **kwargs)

    def receipt(self) -> dict[str, Any]:
        return dict(name="native_checkpoint_recompute_entry_audit_v1", checkpoint_calls=self.calls,
                    function_entries=self.entries, replay_entries=self.entries - self.calls,
                    cache_reads=0, saved_tensor_policy="ORIGINAL", rng_policy="ORIGINAL",
                    counter_scope="CHECKPOINT_BODY_ENTRIES_NOT_ATTENTION_KERNEL_EVENTS")


class ScopedScoreKernels:
    """Dispatch existing kernels only inside entered Critic/Teacher calls.

    FSDP still owns unshard/reshard. No parameter or full-weight view is cached.
    The caller must keep the Critic scope open through checkpoint backward.
    Outside a scope (including every Student call), invoke the exact original.
    """
    def __init__(self, targets: Mapping[str, tuple[Any, str, Any, Any]]):
        self._role = ContextVar("livewan_r2_score_role", default=None)
        self.entries = dict(critic=0, teacher=0)
        self.failures = 0
        self.calls = {role: {key: 0 for key in targets} for role in self.entries}
        self.original_calls = {key: 0 for key in targets}
        for key, (owner, attribute, original, optimized) in targets.items():
            def dispatch(*args: Any, _key=key, _original=original,
                         _optimized=optimized, **kwargs: Any):
                role = self.current_role
                if role is None:
                    self.original_calls[_key] += 1
                    return _original(*args, **kwargs)
                self.calls[role][_key] += 1
                return _optimized(*args, **kwargs)
            setattr(owner, attribute, dispatch)

    @property
    def current_role(self):
        return self._role.get()

    @contextmanager
    def role(self, role: str):
        require(role in self.entries, "unsupported Score kernel scope")
        token = self._role.set(role)
        self.entries[role] += 1
        try:
            yield
        except Exception:
            self.failures += 1
            raise
        finally:
            self._role.reset(token)

    def receipt(self) -> dict[str, Any]:
        return dict(name="fsdp_scoped_score_operators_v1",
                    scope="explicit_critic_teacher_calls_all_ranks",
                    current_role=self.current_role, scope_entries=dict(self.entries),
                    scope_failures=self.failures,
                    role_kernel_calls={k: dict(v) for k, v in self.calls.items()},
                    counter_scope="DISPATCH_ATTEMPTS_INCLUDING_CHECKPOINT_EARLY_STOP",
                    original_outside_scope_calls=dict(self.original_calls),
                    parameter_views_cached=0, complete_B=False)


class ScopedRecomputeCheckpointAudit(RecomputeCheckpointAudit):
    def __init__(self, checkpoint_module: Any, scope: ScopedScoreKernels):
        super().__init__(checkpoint_module)
        self.scope = scope

    def checkpoint(self, function: Any, *args: Any, **kwargs: Any):
        if self.scope.current_role != "critic":
            return self.module.checkpoint(function, *args, **kwargs)
        return super().checkpoint(function, *args, **kwargs)


def score_scope(model: Any, *, role: str):
    scope = getattr(model, "_livewan_fsdp_score_scope", None)
    return scope.role(role) if scope is not None else nullcontext()


def install_fsdp_compatible_score_kernels(torch: Any, *, bundle: Any,
                                        plan: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """Reuse original guarded operators; isolate their global patches from G."""
    require(only_b_compatible(plan, mode=mode), "compatible installer needs its exact opt-in plan")
    from wanstreamer import blockcausal
    from wanstreamer.lora import LoRALinear
    from wan.modules.model import WanRMSNorm
    from .livewan_lora_kernel import install_scale_one_lora_epilogue
    from .livewan_modulation_kernel import install_score_modulation_kernel
    from .livewan_score_rmsnorm import install_frozen_score_rmsnorm
    from .livewan_rope_kernel import install_score_rope_kernel
    require(bundle.student is not None and bundle.critic is not None and bundle.teacher is not None,
            "compatible B needs all-role FSDP models")
    loras = [m for m in bundle.critic.modules() if isinstance(m, LoRALinear)]
    require(len(loras) == 180 and all(m.scale == 1.0 for m in loras),
            "compatible Critic LoRA count/scale changed")
    definitions = {
        "critic_lora_kernel": (LoRALinear, "forward", lambda: install_scale_one_lora_epilogue(
            torch, autocast_input_elision=True)),
        "score_modulation_kernel": (blockcausal, "_modulate", lambda: install_score_modulation_kernel(torch)),
        "score_rmsnorm_kernel": (WanRMSNorm, "forward", lambda: install_frozen_score_rmsnorm(torch)),
        "score_rope_kernel": (blockcausal, "apply_rope", lambda: install_score_rope_kernel(torch)),
    }
    targets, receipts = {}, {}
    for key, (owner, attribute, install) in definitions.items():
        original = getattr(owner, attribute)
        receipts[key] = install()  # Original source/dtype/shape guards remain intact.
        targets[key] = (owner, attribute, original, getattr(owner, attribute))
    receipts["critic_lora_kernel"].update(module_count=len(loras), all_scales_one=True,
                                          role="score", execution_status="PASS")
    scope = ScopedScoreKernels(targets)
    for model in (bundle.critic, bundle.teacher):
        object.__setattr__(model, "_livewan_fsdp_score_scope", scope)
    bundle.system_factorial_score_scope = scope
    return receipts


def compatible_mechanism_errors(row: Mapping[str, Any]) -> list[str]:
    """Require real scoped hits, native recompute and the actual optimizer flag."""
    from .livewan_score_rmsnorm import runtime_receipt_errors
    from .livewan_lora_kernel import EXPECTED_FORWARD_SHA256
    from .livewan_modulation_kernel import EXPECTED_MODULATE_SHA256
    from .livewan_rope_kernel import EXPECTED_APPLY_ROPE_SHA256
    from .livewan_baseline_placement import BaselinePlacement
    errors = runtime_receipt_errors(dict(row.get("score_rmsnorm_kernel", {})), rank=8)
    scope = row.get(COMPATIBLE_KEY, {})
    if (scope.get("name") != "fsdp_scoped_score_operators_v1"
            or scope.get("scope") != "explicit_critic_teacher_calls_all_ranks"
            or scope.get("scope_entries") != dict(critic=24, teacher=4)
            or scope.get("current_role") is not None or scope.get("parameter_views_cached") != 0
            or scope.get("scope_failures") != 0
            or scope.get("counter_scope") != "DISPATCH_ATTEMPTS_INCLUDING_CHECKPOINT_EARLY_STOP"
            or scope.get("complete_B") is not False):
        errors.append("compatible B explicit role scope census changed")
    for key, counter in (("critic_lora_kernel", "autocast_input_elision_invocation_count"),
                         ("score_modulation_kernel", "invocation_count"),
                         ("score_rmsnorm_kernel", "forward_calls"),
                         ("score_rope_kernel", "invocation_count")):
        receipt = row.get(key, {})
        calls = scope.get("role_kernel_calls", {})
        critic = calls.get("critic", {}).get(key, 0)
        teacher = calls.get("teacher", {}).get(key, 0)
        if (type(critic) is not int or critic <= 0 or type(teacher) is not int
                or (teacher != 0 if key == "critic_lora_kernel" else teacher <= 0)
                # Native checkpoint may stop after entering a wrapper but
                # before the optimized leaf (e.g. inside LoRA's base Linear).
                or type(receipt.get(counter)) is not int or not 0 < receipt[counter] <= critic + teacher
                or receipt.get("enabled") is not True or receipt.get("install_status") != "PASS"):
            errors.append("compatible B kernel not observed in exact role scope: " + key)
        if key != "critic_lora_kernel" and scope.get("original_outside_scope_calls", {}).get(key, 0) <= 0:
            errors.append("Student original operator dispatch not observed: " + key)
    lora = row.get("critic_lora_kernel", {})
    if (lora.get("autocast_contract_failure_count") != 0 or lora.get("module_count") != 180
            or lora.get("all_scales_one") is not True or lora.get("autocast_input_elision") is not True):
        errors.append("compatible B original LoRA guard/census failed")
    if row.get("score_rope_kernel", {}).get("contract_failure_count") != 0:
        errors.append("compatible B original RoPE guard failed")
    for key, name, source_key, source in (
        ("critic_lora_kernel", "scale_one_autocast_bf16_input_elision_v1", "source_forward_sha256", EXPECTED_FORWARD_SHA256),
        ("score_modulation_kernel", "score_bf16_modulation_fusion_v1", "source_modulate_sha256", EXPECTED_MODULATE_SHA256),
        ("score_rope_kernel", "score_bf16_rope_fusion_v1", "source_apply_rope_sha256", EXPECTED_APPLY_ROPE_SHA256),
    ):
        receipt = row.get(key, {})
        if receipt.get("name") != name or receipt.get(source_key) != source:
            errors.append("compatible B source-bound operator changed: " + key)
    for key in ("score_modulation_kernel", "score_rope_kernel"):
        if int(row.get(key, {}).get("backward_invocation_count", 0)) <= 0:
            errors.append("compatible B original autograd not observed: " + key)
    build = row.get("build", {})
    runtime = build.get("baseline_placement_runtime", {})
    if (build.get("baseline_placement") != BaselinePlacement("world16-fsdp", student_padding="max-noop").receipt()
            or runtime.get("status") != "PASS"
            or any(runtime.get("roles", {}).get(role, {}).get("sharded") is not True
                   or runtime.get("roles", {}).get(role, {}).get("storage_device") != "cuda"
                   for role in ("student", "critic", "teacher"))):
        errors.append("compatible B changed original all-role FULL_SHARD placement")
    if (row.get("build", {}).get("student", {}).get("optimizer_impl") != "fused_adamw"
            or scope.get("observed_generator_optimizer_fused") is not True
            or scope.get("observed_generator_optimizer_dtensor_parameters") is not True):
        errors.append("compatible B fused Generator AdamW not observed")
    calls = int(row.get("physical_call_inventory", {}).get("critic/grad/checkpoint", -1)) * 30
    audit = row.get("system_factorial_recompute", {})
    if (audit.get("checkpoint_calls") != calls or audit.get("function_entries") != 2 * calls
            or audit.get("replay_entries") != calls or audit.get("cache_reads") != 0
            or audit.get("saved_tensor_policy") != "ORIGINAL" or audit.get("rng_policy") != "ORIGINAL"):
        errors.append("compatible B native Critic checkpoint recompute census changed")
    return errors


def run_baseline_cycles(worker: Any, *, torch: Any, rank: int, trace: Any, loader: Any,
                        bundle: Any, streamers: Any, score_context: Any, evidence: Any,
                        versions: dict[str, int], device: Any, inventory: Any,
                        cycle_endpoint_records: list[dict[str, Any]]) -> None:
    require(trace.cycles == CYCLES, "baseline needs four complete cycles")
    for cycle in range(CYCLES):
        worker._run_baseline(torch=torch, rank=rank, trace=trace, loader=loader, bundle=bundle,
                             streamers=streamers, score_context=score_context, run_kind=RUN_KIND,
                             evidence=evidence, versions=versions, device=device, step_offset=cycle * 5)
        require(versions == dict(generator=cycle + 1, critic=5 * (cycle + 1), ema=0),
                "baseline observed commit versions changed")
        worker._record_two_cycle_endpoint(torch, records=cycle_endpoint_records,
                                         versions=versions, inventory=inventory, evidence=evidence)


def finish_timing(*, torch: Any, start_event: Any, records: list[dict[str, Any]],
                  plan: Mapping[str, Any], mode: str, rank: int,
                  evidence_kind: str = "NATIVE") -> dict[str, Any]:
    """Call only after the caller's single final device sync, never mid-cycle."""
    binding = validate(plan, mode=mode)
    require(len(records) == CYCLES, "missing cycle endpoints")
    offsets = [float(start_event.elapsed_time(row["event"])) / 1000 for row in records]
    require(all(math.isfinite(x) and x > 0 for x in offsets)
            and all(a < b for a, b in zip(offsets, offsets[1:])), "endpoint ordering/non-finite")
    cycles = []
    for index, (row, offset) in enumerate(zip(records, offsets)):
        expected = (dict(generator=index + 1, critic=5 * (index + 1), ema=0) if mode == "baseline"
                    else dict(generator=index + 1, critic=0, ema=0) if rank < 8
                    else dict(generator=0, critic=5 * (index + 1), ema=0))
        require(row["local_versions"] == expected, "role-local observed versions changed")
        before = records[index - 1] if index else {}
        deltas = {key: {k: int(v) - int(before.get(key, {}).get(k, 0))
                        for k, v in row[key].items()}
                  for key in ("physical_call_inventory", "logical_call_inventory")}
        cycles.append(dict(cycle=index + 1, completion_offset_seconds=offset,
                           local_endpoint_delta_seconds=offset - (offsets[index - 1] if index else 0),
                           role="WARMUP" if index == 0 else "STEADY_STATE",
                           local_versions=expected, observed_call_deltas=deltas))
    return dict(schema_version=SCHEMA, arm=binding["arm"], source_head=binding["source_head"],
                seed=binding["seed"], cycles=cycles, effective_plan_sha256=digest(plan),
                evidence_kind=evidence_kind, profiled=False, state_anchors="NOT_COLLECTED",
                midpoint_synchronize=False, midpoint_barrier=False,
                numerical_status="REVIEW_NOT_STRICT_EQUIVALENCE",
                extraction="AFTER_ALL_FOUR_CUDA_COMPLETION_ENDPOINTS")


# Digests of the unchanged, audited original 20-file C51 include chain.
# Filled from the R2 source audit, not from a mutable branch's defaults.
PINNED_PLANS: dict[str, str] = {'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c20_generator_flash_attn2.json': '31c22a0ee50674832c787630ef347cb72fba48c0ed6b516ec9bef6be70598f49',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c21_continuous_b16.json': '81aaeac7174d240856d8866ea842ade52a4886e780f162dbe453802817c28735',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c49_generator_readonly_splitkv.json': '9d4df2ffc6d6e49e4eb16a9109ad377ff9f04689cfa8b831fa1bd8161894bc2b',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c51_generator_fused_rms.json': '1d59a1cd873e4a7e2aad42307bdc42d3cb3a91b1c0e075757311d7b8a26a8a03',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_fsdp2.json': '24abf1f1cd10d798e807f8c5584dd84d24170e3e3fe8bab6f39f4b6e39b3d3b4',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8.json': 'ee49f488ee8cc7d9fe6e01165ca48819bb8aca83b640d7ce1fd9d2f6df38fee9',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead.json': '776fba8462168458ffad4ef6fe51e6654d354cb0e3515e780ac1c545c71ac45d',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2.json': '3775f2d136661864803a0f64cb5e0501bec2f44d7dfcc104e5abcf9121e63c0a',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context.json': '0dc43de42a1adf1b220f753c4cfeb27ba643135c931b75975934fea5119ec868',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32.json': '71783c8217f6791f4999a1b01d1ca4261a14b501552b60ad1ec5e8833f639400',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced.json': '4643f703dbbe5a1ff34b1b5a7ccd98178239372fb7294b3bd176898c5f6722a1',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context.json': '2cdf323f64b3649c6f26f460a29edfd58ae191c155f1b530ce894d38cdf996c3',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision.json': '302902f92debf761da9a97ed69e6abd1438cc1672f9caf9282d6f9e76029dbed',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow.json': 'e3ac7971409ecda5b3ceb34b5be8a97f4ebd4946d739030a8c8006e49cf307e4',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam.json': 'b2364db7145793519744c82bc9e15e11311d6d5cc1e6b0d120f57644e16d632b',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope.json': '81fabe3235f5f580d84cda5b6e6727f7a1ef3977ac313e13053be355f09367e1',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope_flash_attn2.json': '4b15cb7c137bc595d42024ce57b919c22ecd374f6a639344b7b4fb77d40a3e71',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope_flash_attn2_generator_cross_kv_cache.json': 'c1e9baec3604e325035bd78af1b778659aead23dfa7be976b243d7f252681c93',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope_flash_attn2_generator_cross_kv_cache_host_runahead.json': 'd1fdb2963e3e45fdf96fe89970810e2a057cf68da56a5078816b00daa768a9ed',
 'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_scale1.json': '9f7ce00a39db9abf87af0f5a09415eafac12feea0bb4734e0e02d348edf05f1f'}
