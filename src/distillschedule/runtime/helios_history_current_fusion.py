"""Default-off exact Helios guidance-cross-attention data-movement fusion."""
from __future__ import annotations

import functools
import hashlib
import inspect
import re
import textwrap
from pathlib import Path
from typing import Any


SUPPORTED_HELIOS_TRANSFORMER_SHA256 = (
    "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886"
)

_SPLIT_PATTERN = re.compile(
    r"(?P<indent> +)else:\n"
    r"(?P=indent)    history_hidden_states, hidden_states = \(\n"
    r"(?P=indent)        hidden_states\[:, :history_seq_len\],\n"
    r"(?P=indent)        hidden_states\[:, history_seq_len:\],\n"
    r"(?P=indent)    \)\n"
)
_FUSION_PATTERN = re.compile(
    r"(?P<indent> +)hidden_states = hidden_states \+ attn_output\n"
    r"(?P=indent)hidden_states = torch\.cat\("
    r"\[history_hidden_states, hidden_states\], dim=1\)\n"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256(value: Any) -> str:
    return hashlib.sha256(inspect.getsource(value).encode()).hexdigest()


def _rewrite_forward_source(source: str) -> str:
    source = textwrap.dedent(source)
    split_matches = list(_SPLIT_PATTERN.finditer(source))
    if len(split_matches) != 1:
        raise RuntimeError("pinned non-NaViT history/current split changed")
    fusion_matches = list(_FUSION_PATTERN.finditer(source))
    if len(fusion_matches) != 1:
        raise RuntimeError("pinned history-current add/cat sequence changed")
    source = _SPLIT_PATTERN.sub(
        lambda match: (
            f"{match.group('indent')}else:\n"
            f"{match.group('indent')}    distillschedule_full_hidden_states = hidden_states\n"
            f"{match.group('indent')}    history_hidden_states, hidden_states = (\n"
            f"{match.group('indent')}        hidden_states[:, :history_seq_len],\n"
            f"{match.group('indent')}        hidden_states[:, history_seq_len:],\n"
            f"{match.group('indent')}    )\n"
        ),
        source,
        count=1,
    )
    return _FUSION_PATTERN.sub(
        lambda match: (
            f"{match.group('indent')}hidden_states = "
            "_distillschedule_exact_history_current_add(\n"
            f"{match.group('indent')}    distillschedule_full_hidden_states,\n"
            f"{match.group('indent')}    attn_output,\n"
            f"{match.group('indent')}    history_seq_len,\n"
            f"{match.group('indent')})\n"
        ),
        source,
        count=1,
    )


def install_exact_history_current_fusion(
    *,
    enabled: bool,
    role: str,
    transformer_module: Any | None = None,
    fused_function: Any | None = None,
) -> dict[str, Any]:
    """Patch the pinned block method once and return its mutable receipt."""

    receipt: dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": False,
        "role": str(role),
        "scope": "ALL_ROLE_NON_NAVIT_GUIDANCE_CROSS_ATTENTION_BLOCKS",
        "semantic_contract": (
            "BF16_CURRENT_PLUS_ATTENTION_AND_UNCHANGED_HISTORY_PREFIX;"
            "DIRECT_FULL_HIDDEN_AND_ATTENTION_SUFFIX_GRADIENTS;"
            "ATTENTION_NORM_NAVIT_FSDP_CHECKPOINT_AND_CALLS_UNCHANGED"
        ),
        "call_count": 0,
        "grad_enabled_call_count": 0,
        "nograd_call_count": 0,
        "batch_counts": {},
        "shape_counts": {},
        "reference_fallback_count": 0,
        "shared_source_modified": False,
    }
    if not enabled:
        receipt["disabled_reason"] = "NOT_REQUESTED"
        return receipt
    if role != "generator":
        receipt["disabled_reason"] = "INSTALL_ONCE_FROM_GENERATOR_BUILD"
        return receipt
    if transformer_module is None:
        from helios.modules import transformer_helios as transformer_module

    source_path = Path(transformer_module.__file__).resolve()
    observed_sha = _sha256(source_path)
    receipt.update(
        transformer_source=str(source_path),
        transformer_sha256=observed_sha,
        expected_transformer_sha256=SUPPORTED_HELIOS_TRANSFORMER_SHA256,
    )
    test_injected = fused_function is not None
    if not test_injected and observed_sha != SUPPORTED_HELIOS_TRANSFORMER_SHA256:
        raise RuntimeError(
            f"Helios transformer source changed: {observed_sha} != "
            f"{SUPPORTED_HELIOS_TRANSFORMER_SHA256}"
        )
    if getattr(transformer_module, "_distillschedule_history_current_receipt", None):
        raise RuntimeError("Helios history/current fusion was already installed")
    if fused_function is None:
        from distillschedule.runtime.helios_history_current_kernel import (
            exact_history_current_add,
        )

        fused_function = exact_history_current_add

    block = transformer_module.HeliosTransformerBlock
    original = block.forward
    receipt["forward_source_sha256"] = _source_sha256(original)
    rewritten_source = _rewrite_forward_source(inspect.getsource(original))

    def counted(
        hidden_states: Any, attention_output: Any, history_tokens: int
    ) -> Any:
        import torch

        receipt["call_count"] += 1
        key = (
            "grad_enabled_call_count"
            if torch.is_grad_enabled()
            else "nograd_call_count"
        )
        receipt[key] += 1
        batch = int(hidden_states.shape[0])
        batches = receipt["batch_counts"]
        batches[str(batch)] = int(batches.get(str(batch), 0)) + 1
        shape = (
            f"B{batch}:H{int(history_tokens)}:"
            f"C{int(hidden_states.shape[1]) - int(history_tokens)}:"
            f"W{int(hidden_states.shape[2])}"
        )
        shapes = receipt["shape_counts"]
        shapes[shape] = int(shapes.get(shape, 0)) + 1
        return fused_function(hidden_states, attention_output, int(history_tokens))

    namespace = dict(transformer_module.__dict__)
    namespace["_distillschedule_exact_history_current_add"] = counted
    exec(
        compile(
            rewritten_source,
            inspect.getsourcefile(original) or "<helios_history_current_fusion>",
            "exec",
        ),
        namespace,
    )
    rewritten = namespace["forward"]
    functools.update_wrapper(rewritten, original)
    block.forward = rewritten
    transformer_module._distillschedule_history_current_receipt = receipt
    receipt["enabled"] = True
    return receipt


__all__ = [
    "SUPPORTED_HELIOS_TRANSFORMER_SHA256",
    "_rewrite_forward_source",
    "install_exact_history_current_fusion",
]
