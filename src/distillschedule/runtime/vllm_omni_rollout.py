"""Fail-closed contract for Helios rollouts served by vLLM-Omni.

The BAAI rollout service is pinned to a CUDA-12-compatible vLLM-Omni 0.19
source plus one audited Helios latent/no-VAE backport. The training runtime does
not trust a package version alone: an adapter must report the exact upstream
base, compatibility-patch identity, exact operation-trace replay, freshly
synchronized Generator LoRA, and a physically separate rollout pool.
"""
from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Mapping, Sequence
from typing import Any, Callable


VLLM_OMNI_BACKEND = "vllm-omni-latent"
NATIVE_ROLLOUT_BACKEND = "native-fsdp2-batch"
VLLM_OMNI_VERSION_FAMILY = "0.19"
VLLM_VERSION_FAMILY = "0.19"
VLLM_OMNI_HELIOS_BASE_TAG = "v0.19.0rc1"
VLLM_OMNI_HELIOS_BASE_COMMIT = "191b9a8dbf22fc494b7db34ea5ae04153f44f94d"
VLLM_OMNI_HELIOS_LATENT_PATCH_PATH = (
    "third_party/patches/vllm_omni_v0_19_helios_latent_no_vae.patch"
)
VLLM_OMNI_HELIOS_LATENT_PATCH_SHA256 = (
    "84c9a914184373a771a5839bd6198d05ba8f981154732fc6f7bfbf5feafb3d99"
)
# PR #3800 is provenance for the no-VAE idea only. It is closed/unmerged and
# is deliberately not the deployed source identity.
VLLM_OMNI_HELIOS_LATENT_REFERENCE_PR = 3800
VLLM_OMNI_HELIOS_LATENT_REFERENCE_COMMIT = VLLM_OMNI_HELIOS_BASE_COMMIT
OFFICIAL_X0_SHAPE = (1, 16, 9, 48, 80)
OFFICIAL_X0_DTYPE = "torch.bfloat16"
SUPPORTED_BATCH_SIZES = (1, 2, 5)


@dataclasses.dataclass(frozen=True)
class VLLMOmniRolloutCapabilities:
    """Capabilities a rollout adapter must prove before model work."""

    backend: str = VLLM_OMNI_BACKEND
    engine: str = "vllm-omni"
    model_family: str = "helios"
    upstream_revision: str = VLLM_OMNI_HELIOS_BASE_COMMIT
    compatibility_patch_sha256: str = VLLM_OMNI_HELIOS_LATENT_PATCH_SHA256
    vllm_omni_version_family: str = VLLM_OMNI_VERSION_FAMILY
    vllm_version_family: str = VLLM_VERSION_FAMILY
    output_type: str = "latent"
    skips_vae_decode: bool = True
    exact_operation_trace_replay: bool = True
    hot_generator_lora_sync: bool = True
    dedicated_rollout_pool: bool = True
    supported_batch_sizes: tuple[int, ...] = SUPPORTED_BATCH_SIZES
    output_shape: tuple[int, ...] = OFFICIAL_X0_SHAPE
    output_dtype: str = OFFICIAL_X0_DTYPE
    transport: str = "adapter-defined"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VLLMOmniRolloutCapabilities":
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(value) - fields)
        if unknown:
            raise ValueError(f"unknown vLLM-Omni capability fields: {unknown}")
        required_identity = {
            "upstream_revision",
            "compatibility_patch_sha256",
            "vllm_omni_version_family",
            "vllm_version_family",
        }
        missing = sorted(required_identity - set(value))
        if missing:
            raise ValueError(
                "vLLM-Omni service omitted source identity fields: "
                f"{missing}"
            )
        normalized = dict(value)
        for name in ("supported_batch_sizes", "output_shape"):
            if name in normalized:
                normalized[name] = tuple(int(item) for item in normalized[name])
        return cls(**normalized)

    def validate(
        self,
        *,
        requested_batch_size: int,
        expected_revision: str,
        require_hot_lora_sync: bool,
        expected_patch_sha256: str = VLLM_OMNI_HELIOS_LATENT_PATCH_SHA256,
    ) -> dict[str, Any]:
        errors: list[str] = []
        if self.backend != VLLM_OMNI_BACKEND:
            errors.append(f"backend={self.backend!r}")
        if self.engine != "vllm-omni" or self.model_family.lower() != "helios":
            errors.append("engine/model_family is not vLLM-Omni Helios")
        if self.vllm_omni_version_family != VLLM_OMNI_VERSION_FAMILY:
            errors.append(
                f"vLLM-Omni family={self.vllm_omni_version_family!r}, "
                f"expected={VLLM_OMNI_VERSION_FAMILY!r}"
            )
        if self.vllm_version_family != VLLM_VERSION_FAMILY:
            errors.append(
                f"vLLM family={self.vllm_version_family!r}, "
                f"expected={VLLM_VERSION_FAMILY!r}"
            )
        if self.upstream_revision != expected_revision:
            errors.append(
                "vLLM-Omni base revision mismatch: "
                f"{self.upstream_revision} != {expected_revision}"
            )
        if self.compatibility_patch_sha256 != expected_patch_sha256:
            errors.append(
                "Helios latent backport mismatch: "
                f"{self.compatibility_patch_sha256} != {expected_patch_sha256}"
            )
        if self.output_type != "latent" or not self.skips_vae_decode:
            errors.append("latent output without VAE decode is not proven")
        if not self.exact_operation_trace_replay:
            errors.append("exact operation-trace replay is unavailable")
        if require_hot_lora_sync and not self.hot_generator_lora_sync:
            errors.append("hot Generator LoRA synchronization is unavailable")
        if not self.dedicated_rollout_pool:
            errors.append("rollout engine is not on a dedicated GPU pool")
        if requested_batch_size not in self.supported_batch_sizes:
            errors.append(
                f"B{requested_batch_size} is not supported; "
                f"supported={self.supported_batch_sizes}"
            )
        if tuple(self.output_shape) != OFFICIAL_X0_SHAPE:
            errors.append(f"output_shape={self.output_shape}")
        if self.output_dtype != OFFICIAL_X0_DTYPE:
            errors.append(f"output_dtype={self.output_dtype}")
        if errors:
            raise ValueError("invalid vLLM-Omni rollout capability: " + "; ".join(errors))
        return {
            "status": "PASS",
            "backend": self.backend,
            "upstream_revision": self.upstream_revision,
            "compatibility_patch_sha256": self.compatibility_patch_sha256,
            "vllm_omni_version_family": self.vllm_omni_version_family,
            "vllm_version_family": self.vllm_version_family,
            "requested_batch_size": requested_batch_size,
            "require_hot_lora_sync": require_hot_lora_sync,
            "output_type": self.output_type,
            "vae_decode_calls": 0,
            "exact_operation_trace_replay": True,
            "dedicated_rollout_pool": True,
        }


def expected_generator_version(variant: str) -> int:
    if variant == "no-gan":
        return 3
    if variant == "gan":
        return 201
    raise ValueError(f"unsupported Full5 variant: {variant}")


def rollout_phase_matrix(variant: str) -> tuple[dict[str, Any], ...]:
    """Describe exactly where the external latent backend is legal."""

    version = expected_generator_version(variant)
    steps = list(range(10, 15)) if variant == "no-gan" else list(range(1000, 1005))
    isolated_steps = steps[1:]
    return (
        {
            "phase": "post_generator_critic_input_rollouts",
            "steps": steps,
            "generator_version": version,
            "allowed": True,
            "requires_hot_lora_sync": True,
            "reason": "no-grad x0 generation after the single Generator commit",
        },
        {
            "phase": "isolated_critic_only_frozen_generator_rollouts",
            "steps": isolated_steps,
            "generator_version": version,
            "allowed": True,
            "requires_hot_lora_sync": False,
            "reason": "Generator version is pinned before the isolated subgraph starts",
        },
        {
            "phase": "differentiable_generator_update_rollout",
            "steps": [steps[0]],
            "allowed": False,
            "reason": "the Generator update needs an autograd graph on training ranks",
        },
        {
            "phase": "score_fake_real_or_gan_g",
            "steps": steps,
            "allowed": False,
            "reason": "these are Score/GAN model calls, not Generator rollout serving",
        },
        {
            "phase": "critic_backward_optimizer_or_ema",
            "steps": steps,
            "allowed": False,
            "reason": "stateful training commits must remain in the training runtime",
        },
        {
            "phase": "native_b1_correctness_reference",
            "steps": steps,
            "allowed": False,
            "reason": "an external engine cannot validate itself; native B1 is authoritative",
        },
    )


def source_has_latent_no_vae_guard(source: str) -> bool:
    """Prove that VAE decode is dominated by the latent-output guard.

    vLLM-Omni 0.19 already returned ``real_history_latents`` when requested, but
    still executed VAE decode first. Merely finding both strings is therefore
    insufficient: the ``output_type != latent`` guard must occur immediately
    before the decode site, and the latent return must remain after it.
    """

    compact = " ".join(source.replace("'", '"').split())
    guard = compact.find('if output_type != "latent":')
    decode = compact.find(".vae.decode(")
    # The vLLM-Omni module also contains a small post-processing helper with
    # an earlier ``output_type == "latent"`` branch.  Use the final occurrence
    # so the guard is proven to dominate the pipeline's actual latent return.
    latent_return = compact.rfind('if output_type == "latent"')
    if min(guard, decode, latent_return) < 0:
        return False
    return guard < decode < latent_return and (decode - guard) < 1200


def parse_factory_spec(spec: str) -> tuple[str, str]:
    module, separator, attribute = spec.partition(":")
    if not separator or not module or not attribute:
        raise ValueError("factory must use module.path:callable_name")
    if any(part in {"", ".", ".."} for part in module.split(".")):
        raise ValueError("factory module path is invalid")
    return module, attribute


def load_factory(spec: str) -> Callable[..., Any]:
    module_name, attribute = parse_factory_spec(spec)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"rollout factory {spec!r} is not callable")
    return value


def normalize_capabilities(value: Any) -> VLLMOmniRolloutCapabilities:
    if isinstance(value, VLLMOmniRolloutCapabilities):
        return value
    if isinstance(value, Mapping):
        return VLLMOmniRolloutCapabilities.from_mapping(value)
    raise TypeError("adapter capabilities must be a mapping or capability dataclass")


def validate_rollout_receipt(
    receipt: Mapping[str, Any],
    *,
    logical_steps: Sequence[int],
    generator_version: int,
    expected_revision: str,
    expected_patch_sha256: str = VLLM_OMNI_HELIOS_LATENT_PATCH_SHA256,
) -> dict[str, Any]:
    expected_steps = [int(step) for step in logical_steps]
    errors: list[str] = []
    if receipt.get("status") != "PASS":
        errors.append(f"status={receipt.get('status')!r}")
    if receipt.get("backend") != VLLM_OMNI_BACKEND:
        errors.append("backend mismatch")
    if receipt.get("output_type") != "latent":
        errors.append("output_type is not latent")
    if int(receipt.get("vae_decode_calls", -1)) != 0:
        errors.append("VAE decode was observed")
    if [int(step) for step in receipt.get("logical_steps", [])] != expected_steps:
        errors.append("logical step inventory changed")
    if int(receipt.get("generator_version", -1)) != int(generator_version):
        errors.append("Generator version mismatch")
    if receipt.get("upstream_revision") != expected_revision:
        errors.append("upstream base revision mismatch")
    patch = receipt.get("compatibility_patch_sha256")
    if patch is not None and patch != expected_patch_sha256:
        errors.append("compatibility patch mismatch")
    if not bool(receipt.get("exact_operation_trace_replay")):
        errors.append("exact trace replay receipt is absent")
    if int(receipt.get("call_count", 0)) <= 0:
        errors.append("positive physical DiT call_count is absent")
    if errors:
        raise ValueError("invalid vLLM-Omni rollout receipt: " + "; ".join(errors))
    return dict(receipt)


__all__ = [
    "NATIVE_ROLLOUT_BACKEND",
    "OFFICIAL_X0_DTYPE",
    "OFFICIAL_X0_SHAPE",
    "SUPPORTED_BATCH_SIZES",
    "VLLM_OMNI_BACKEND",
    "VLLM_OMNI_HELIOS_BASE_COMMIT",
    "VLLM_OMNI_HELIOS_BASE_TAG",
    "VLLM_OMNI_HELIOS_LATENT_PATCH_PATH",
    "VLLM_OMNI_HELIOS_LATENT_PATCH_SHA256",
    "VLLM_OMNI_HELIOS_LATENT_REFERENCE_COMMIT",
    "VLLM_OMNI_HELIOS_LATENT_REFERENCE_PR",
    "VLLM_OMNI_VERSION_FAMILY",
    "VLLM_VERSION_FAMILY",
    "VLLMOmniRolloutCapabilities",
    "expected_generator_version",
    "load_factory",
    "normalize_capabilities",
    "parse_factory_spec",
    "rollout_phase_matrix",
    "source_has_latent_no_vae_guard",
    "validate_rollout_receipt",
]
