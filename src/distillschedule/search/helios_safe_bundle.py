"""Opt-in No-GAN incumbent mechanisms; no trainer, timing prior or GAN claim.

The all-or-none tuple comes from ed8e1cc, not the old canonical 94s recipe.
Source identities and actual call deltas are part of admission and cost keys.
"""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from distillschedule.workloads.best_known.registry import file_sha256, repository_root
from .contracts import require

BUNDLE_ID = "safe-nogan-ed8-v1"
SOURCE_HEAD = "ed8e1ccbee44e039c73b303d33179964795ea71b"
CONFIG_SHA256 = "4164c17e3743045a97510a849bbc00a258d310dc7fefc2c68f1816bf505ed323"
FLAG_ENV = {
    "exact_modulated_norm_epilogue": "DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_EPILOGUE",
    "exact_modulated_norm_materialization": "DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_MATERIALIZATION",
    "exact_modulated_norm_backward": "DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_BACKWARD",
    "exact_modulated_norm_layernorm": "DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_LAYERNORM",
    "exact_modulated_norm_fused_backward": "DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_FUSED_BACKWARD",
    "elide_explicit_rollout_sync": "DISTILLSCHEDULE_ELIDE_EXPLICIT_ROLLOUT_SYNC",
    "lora_b_gemm_epilogue": "DISTILLSCHEDULE_HELIOS_LORA_B_GEMM_EPILOGUE",
    "lora_checkpoint_replay_scope": "DISTILLSCHEDULE_HELIOS_LORA_CHECKPOINT_REPLAY_SCOPE",
}
# These unchanged helpers plus the two exact ed8 LoRA files are source pinned.
PINNED_HELPERS = {
    "helios_modulated_norm_fusion.py": "4c5325d564df54b1f9e024076a3176f5a4e10e9fd4e60c764e4231fe213d56e6",
    "helios_modulated_norm_kernel.py": "38f615574bd0031058c81d3ad8e13287625a944a39028093789f72f0616fe493",
    "helios_fused_layernorm_kernel.py": "b91dfce7f143497fcce7d1f89f5b30956ed0e8f4e5aa303205b6df300cb6bafa",
    "helios_fused_layernorm_backward_kernel.py": "c89fb0500061f1f420480821ca74d89ba6368e6f36df355c03e736cfe984d8a9",
    "helios_lora_epilogue.py": "9e2ac9a5e381fdea1d17598bb9970448e8a82c0ea66e0c60857fb8af3dc45151",
    "helios_lora_b_gemm_kernel.py": "700054fa857437023974352c0c589b8bc232e7a45443e5271bd102ba07539bc5",
}
SOURCE_FILES = tuple("src/distillschedule/runtime/" + p for p in (
    *PINNED_HELPERS, "fsdp2_static.py", "fsdp2_p0.py",
    "helios_planner_cycle.py",
)) + (
    "scripts/distillschedule/probe_static_fsdp2_p0_world16.py",
    "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh",
    "runtime/gst_pool_helios.py",
    "runtime/static_fsdp2_p0_helios.py",
    "src/distillschedule/runtime/helios_lora_epilogue_kernel.py",
    "src/distillschedule/search/helios_safe_bundle.py",
    "src/distillschedule/search/helios_implementation.py",
    "src/distillschedule/search/helios_action_costs.py",
    "src/distillschedule/search/helios_native.py",
    "src/distillschedule/search/helios_model_adapter.py",
    "src/distillschedule/search/workflow_cli.py",
)
NORM_COUNTERS = ("fused_call_count", "materialization_fused_call_count",
                 "layernorm_fused_call_count", "backward_call_count", "backward_kernel_call_count")
LORA_COUNTERS = ("model_call_count", "fused_grad_invocation_count", "fused_nograd_invocation_count",
                 "checkpoint_wrap_count", "checkpoint_scope_execution_count", "checkpoint_scope_restore_count",
                 "lora_b_gemm_kernel_call_count", "lora_b_gemm_completed_invocation_count",
                 "lora_b_gemm_selected_attempt_count", "lora_b_gemm_selected_invocation_count",
                 "lora_b_gemm_selected_grad_invocation_count", "lora_b_gemm_selected_nograd_invocation_count")
B_GEMM_IMPLEMENTATION = ("TRITON_BF16_RANK256_B_GEMM_RN_BF16_THEN_ADD_RN_F32_"
                         "BF16_STORE_REFERENCE_MM_BACKWARD")


@dataclass(frozen=True)
class HeliosSafeBundle:
    exact_modulated_norm_epilogue: bool = False
    exact_modulated_norm_materialization: bool = False
    exact_modulated_norm_backward: bool = False
    exact_modulated_norm_layernorm: bool = False
    exact_modulated_norm_fused_backward: bool = False
    elide_explicit_rollout_sync: bool = False
    lora_b_gemm_epilogue: bool = False
    lora_checkpoint_replay_scope: bool = False

    def __post_init__(self):
        values = tuple(asdict(self).values())
        require(all(type(v) is bool for v in values), "Helios bundle fields must be explicit booleans")
        require(not any(values) or all(values), "Helios safe bundle is all-or-none")

    @property
    def enabled(self):
        return self.lora_b_gemm_epilogue

    @property
    def id(self):
        return BUNDLE_ID if self.enabled else "none"

    @classmethod
    def safe(cls):
        return cls(**{f.name: True for f in fields(cls)})

    @classmethod
    def parse(cls, value):
        require(value in ("none", BUNDLE_ID), "unknown Helios candidate bundle")
        return cls.safe() if value == BUNDLE_ID else cls()

    def validate_scope(self, impl):
        if self.enabled:
            require((impl.variant, impl.mesh, impl.chunk_tokens, impl.rollout_window,
                     impl.persistent_unshard) == ("no-gan", "node-hsdp", 512, 5, False),
                    "safe No-GAN bundle requires exactly HSDP/C512/p0/w5; GAN is not migrated")

    def environment(self):
        # Off does not erase inherited canonical flags or change old profiles.
        return {key: "on" for key in FLAG_ENV.values()} if self.enabled else {}

    def source_identity(self, *, root: Path | None = None):
        require(self.enabled, "bundle-off has no new source contract")
        root = root or repository_root()
        for name, expected in PINNED_HELPERS.items():
            require(file_sha256(root / "src/distillschedule/runtime" / name) == expected,
                    f"safe No-GAN helper source differs: {name}")
        return dict(authoritative_source_head=SOURCE_HEAD, accepted_config_sha256=CONFIG_SHA256,
                    files={p: file_sha256(root / p) for p in SOURCE_FILES})

    def cost_identity(self, *, root=None):
        return dict(id=self.id, fields=asdict(self), source=self.source_identity(root=root))

    def bind_effective(self, *, args, policy):
        require(self.enabled, "bundle-off cannot claim safe effects")
        for name in FLAG_ENV:
            owner = args if name == "lora_checkpoint_replay_scope" else policy
            require(getattr(owner, name, None) is True, f"safe No-GAN effective {name} missing/off")
        require(getattr(policy, "selective_flash_checkpoint_stride", None) == 0,
                "safe No-GAN bundle requires native stride0 recompute")
        return dict(id=self.id, fields=asdict(self), source=self.source_identity())

    def effect_snapshot(self, build: Mapping[str, Any]):
        """Validate installation and copy counters, never invent executed calls."""
        require(self.enabled, "bundle-off cannot claim safe effects")
        result = {}
        for role in ("generator", "critic"):
            row = build.get(role, {})
            lora = row.get("all_role_lora_scale1_add", {})
            require(lora.get("enabled") is True and lora.get("scope") == "ALL_ROLE_SCALE1_MODEL_FORWARD"
                    and lora.get("checkpoint_replay_scope") is True
                    and lora.get("lora_b_gemm_epilogue_requested") is True
                    and lora.get("lora_b_gemm_epilogue_enabled") is True,
                    f"safe No-GAN {role} LoRA mechanisms not installed")
            require(lora.get("checkpoint_missing_active_scope_count") == 0,
                    "safe No-GAN checkpoint lost active LoRA scope")
            require(lora.get("lora_b_gemm_runtime_identity", {}).get("implementation")
                    == B_GEMM_IMPLEMENTATION, "missing/injected LoRA is not production evidence")
            require(row.get("checkpoint_policy", {}).get("lora_checkpoint_replay_scope") is True
                    and row.get("checkpoint_policy", {}).get("selective_flash_stride") == 0,
                    "safe No-GAN actual checkpoint function lacks captured scope/stride0")
            result[role] = {k: _counter(lora, k) for k in LORA_COUNTERS}
            require(result[role]["checkpoint_scope_execution_count"]
                    == result[role]["checkpoint_scope_restore_count"], "unrestored LoRA checkpoint scope")
            require(result[role]["lora_b_gemm_completed_invocation_count"]
                    <= result[role]["lora_b_gemm_kernel_call_count"]
                    == result[role]["lora_b_gemm_selected_invocation_count"]
                    <= result[role]["lora_b_gemm_selected_attempt_count"], "LoRA-B call census differs")
        norm = build.get("generator", {}).get("exact_modulated_norm_epilogue", {})
        require(all(norm.get(k) is True for k in ("enabled", "requested", "materialization_requested",
                    "backward_requested", "layernorm_requested", "backward_kernel_requested")),
                "safe No-GAN exact norm mechanisms not installed")
        require(norm.get("reference_fallback_count") == 0
                and norm.get("backward_kernel_reference_fallback_count") == 0,
                "safe No-GAN exact norm unexpectedly fell back")
        result["shared_norm"] = {k: _counter(norm, k) for k in NORM_COUNTERS}
        return result

    def executed_effects(self, build, before):
        return self.effects_between(before, self.effect_snapshot(build))

    def effects_between(self, before, after):
        require(set(before) == set(after), "safe No-GAN pre-cycle installation census missing")
        delta = {}
        for role, counts in after.items():
            require(set(before[role]) == set(counts), "safe No-GAN counter inventory differs")
            require(all(type(v) is int and v >= 0 for v in (*counts.values(), *before[role].values())),
                    "safe No-GAN counter values invalid")
            delta[role] = {k: value - before[role][k] for k, value in counts.items()}
            require(all(type(v) is int and v > 0 for v in delta[role].values()),
                    f"safe No-GAN {role} mechanisms did not execute in observed native scope")
        return dict(status="PASS_EXECUTED_SAFE_NOGAN_BUNDLE", **self.cost_identity(),
                    before=before, after=after, call_deltas=delta,
                    scope="NATIVE_SCOPE_OUTSIDE_TIMER_CENSUS; NOT_NUMERICAL_PARITY")


def _counter(row, key):
    value = row.get(key)
    require(type(value) is int and value >= 0, f"safe No-GAN actual {key} missing/invalid")
    return value
