"""C38 opt-in pinned Score FA3: dense BF16 attention and local exact SAC.

Generator remains FA2. No dropout, masks, FP8, output offload, split tuning,
parameter placement, sample batching, or optimizer changes are registered.
"""
from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from distillschedule.runtime.livewan_attention_backend import (
    EXPECTED_FLASH_ATTENTION_SHA256, _install_flash_attn2, _sha256_file,
)
from distillschedule.runtime.livewan_attention_checkpoint import (
    AttentionCheckpointContexts, CriticAttentionCheckpointProxy,
)

VARIANT = "c38-continuous-b16-score-fa3"
INTERFACE_SHA256 = "cb46c3d89606305817ea20eb7b7cc9376b013eeabfd7637a328e7967e1d008d4"
OP_NAME = "_flash_attn3_cuda_373a45c::_flash_attn_forward"
DEFAULT_ROOT = Path(
    "/share/project/shuchenweng/yongyan_liu/DMD-System_assets/"
    "lightx2v_minimax_dmd_20260817/hf_home/hub/"
    "models--kernels-community--flash-attn3/snapshots/"
    "acb0f42e6901d3e9effa8a83bd32bc2bf0cfdfc2/build/"
    "torch210-cxx11-cu128-x86_64-linux"
)
HASHES = {
    "metadata.json": "f1b4b0ef991c0966f7d848777154c0591c4580968fb6da33f6b2a2829a109309",
    "__init__.py": "29756640933e2a159cd94a9625eb2ea4268ffa6a80bd3c7bc86b9e1404e09471",
    "_ops.py": "7f392b365cc945a3b3a420e24802ef776a79316c068e6e99e4da682e8927a1f6",
    "flash_attn_interface.py": INTERFACE_SHA256,
    "_flash_attn3_cuda_373a45c.abi3.so": "8aa9bb49e174872be4e2cb8c2608f2d66de29b513f704fb8d91771f9cd39df56",
}
BACKEND_CONTRACT = {
    "enabled": True,
    "name": "score_flash_attn3_unmasked_bf16_split1_v1",
    "scope": "score_ranks_only_critic_and_teacher",
    "source_flash_attention_sha256": EXPECTED_FLASH_ATTENTION_SHA256,
    "artifact_metadata_sha256": HASHES["metadata.json"],
    "artifact_package_sha256": HASHES["__init__.py"],
    "artifact_extension_sha256": HASHES["_flash_attn3_cuda_373a45c.abi3.so"],
    "artifact_interface_sha256": INTERFACE_SHA256,
    "artifact_ops_sha256": HASHES["_ops.py"],
    "math_change": "pinned_dense_fa2_to_fa3_under_unchanged_parity_gate",
    "mask_contract": "full_length_unmasked_only",
    "input_output_dtype": "bfloat16",
    "head_dimension": 128,
    "num_splits": 1,
}
SAC_CONTRACT = {
    "enabled": True,
    "name": "critic_fa3_exact_gpu_checkpoint_outputs_v1",
    "scope": "score_rank_critic_grad_self_attention_only",
    "interface_sha256": INTERFACE_SHA256,
    "operator": OP_NAME,
    "cache_location": "cuda_per_checkpoint",
    "query_tokens": 19320,
    "key_tokens": 19320,
    "allow_cache_entry_mutation": False,
    "preserve_rng_state": True,
    "application_calls_unchanged": True,
    "cross_attention_recomputed": True,
}


def load_flash_attn3() -> tuple[Any, Path]:
    root = Path(os.environ.get("DISTILLSCHEDULE_SCORE_FA3_KERNEL", DEFAULT_ROOT)).resolve()
    for relative, expected in HASHES.items():
        if not (root / relative).is_file() or _sha256_file(root / relative) != expected:
            raise RuntimeError(f"C38 pinned FA3 artifact missing/changed: {root / relative}")
    # Do not leave the kernel's generic flash_attn_interface name on sys.path.
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("flash_attn3")
    finally:
        sys.path.remove(str(root))
    source = Path(inspect.getfile(module.flash_attn_func.__globals__["FlashAttnFunc"])).resolve()
    # Hugging Face snapshot entries resolve into the sibling content-addressed
    # blobs directory. Bind to the exact verified entry, not a directory prefix.
    if source != (root / "flash_attn_interface.py").resolve() or _sha256_file(source) != INTERFACE_SHA256:
        raise RuntimeError("C38 FA3 imported from an unregistered source")
    return module, root


def install_score_flash_attn3(torch: Any) -> dict[str, Any]:
    module, root = load_flash_attn3()

    def dense(q, k, v, *, dropout_p, softmax_scale, causal, window_size, deterministic):
        if dropout_p != 0.0 or causal or tuple(window_size) != (-1, -1):
            raise RuntimeError("C38 unregistered attention options")
        return module.flash_attn_func(q, k, v, softmax_scale=softmax_scale,
            causal=False, window_size=(-1, -1), softcap=0.0, num_splits=1,
            deterministic=deterministic, return_attn_probs=False)

    identity = {k: v for k, v in BACKEND_CONTRACT.items()
                if k.startswith("artifact_") or k == "num_splits"}
    return _install_flash_attn2(torch, requested_name=BACKEND_CONTRACT["name"],
        role_label="Score FA3", allowed_input_dtypes={(torch.bfloat16,) * 3},
        kernel_binding=(SimpleNamespace(flash_attn_func=dense), root, identity))


# Defaults from the SHA-pinned custom operator, not guessed from another release.
OPTION_DEFAULTS = {
    **dict.fromkeys(("k_new", "v_new", "qv", "out_", "cu_seqlens_q", "cu_seqlens_k",
        "cu_seqlens_k_new", "seqused_q", "seqused_k", "max_seqlen_q", "max_seqlen_k",
        "page_table", "kv_batch_idx", "leftpad_k", "rotary_cos", "rotary_sin",
        "seqlens_rotary", "q_descale", "k_descale", "v_descale", "scheduler_metadata",
        "pack_gqa")),
    "softmax_scale": 128 ** (-0.5), "causal": False,
    "window_size_left": -1, "window_size_right": -1, "attention_chunk": 0,
    "softcap": 0.0, "rotary_interleaved": True, "num_splits": 1, "sm_margin": 0,
}
# The schema order differs from OPTION_DEFAULTS insertion order.
ARGUMENT_NAMES = ("q", "k", "v", "k_new", "v_new", "qv", "out_", "cu_seqlens_q",
    "cu_seqlens_k", "cu_seqlens_k_new", "seqused_q", "seqused_k", "max_seqlen_q",
    "max_seqlen_k", "page_table", "kv_batch_idx", "leftpad_k", "rotary_cos", "rotary_sin",
    "seqlens_rotary", "q_descale", "k_descale", "v_descale", "softmax_scale", "causal",
    "window_size_left", "window_size_right", "attention_chunk", "softcap",
    "rotary_interleaved", "scheduler_metadata", "num_splits", "pack_gqa", "sm_margin")


def discover_flash_forward(module: Any) -> tuple[Any, dict[str, Any]]:
    cls = module.flash_attn_func.__globals__["FlashAttnFunc"]
    path = Path(inspect.getfile(cls)).resolve()
    sha = _sha256_file(path)
    op = cls.forward.__globals__["_flash_attn_forward"]._opoverload
    if (sha != INTERFACE_SHA256 or op._schema.name != OP_NAME or op._schema.is_mutable
        or tuple(a.name for a in op._schema.arguments) != ARGUMENT_NAMES):
        raise RuntimeError(f"C38 FA3 source/operator/schema changed: {sha}: {op._schema}")
    return op, dict(interface_path=str(path), interface_sha256=sha,
                    operator=str(op), schema=str(op._schema))


def attention_metadata(torch: Any, args: tuple, kwargs: dict) -> tuple:
    """No tensor-value reads, sync, GPU copies, or loss/RNG changes."""
    if len(args) > len(ARGUMENT_NAMES) or set(kwargs) - set(ARGUMENT_NAMES):
        raise RuntimeError("C38 FA3 argument schema changed")
    supplied = dict(zip(ARGUMENT_NAMES, args))
    if set(supplied) & set(kwargs):
        raise RuntimeError("C38 duplicate FA3 arguments")
    supplied.update(kwargs)
    if not {"q", "k", "v"} <= supplied.keys():
        raise RuntimeError("C38 FA3 QKV incomplete")
    fields = dict(OPTION_DEFAULTS, **supplied)
    for name, expected in OPTION_DEFAULTS.items():
        value = fields[name]
        # Optional tensors must remain None; never bool/equal a CUDA tensor.
        if (expected is None and value is not None) or (
            expected is not None and (type(value) is not type(expected) or value != expected)):
            raise RuntimeError(f"C38 FA3 unregistered option: {name}")
    q, k, v = (fields[key] for key in ("q", "k", "v"))
    if (any(t.ndim != 4 or not t.is_cuda or t.dtype != torch.bfloat16 for t in (q, k, v))
        or any(t.device != q.device for t in (k, v))
        or q.shape[0] not in (1, 2) or q.shape[2:] != (12, 128)
        or k.shape != v.shape or k.shape[0] != q.shape[0] or k.shape[2:] != q.shape[2:]
        or any(t.stride(-1) != 1 for t in (q, k, v))):
        raise RuntimeError("C38 Critic FA3 device/shape/dtype/layout changed")
    return tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))
                 for t in (q, k, v))


class CriticFA3CheckpointProxy(CriticAttentionCheckpointProxy):
    def __init__(self, checkpoint_module: Any, torch: Any, module: Any):
        self.module = checkpoint_module
        op, identity = discover_flash_forward(module)
        self.contexts = AttentionCheckpointContexts(torch, op, identity,
            self_attention_only=True, metadata_reader=attention_metadata,
            receipt_name=SAC_CONTRACT["name"])
        self.receipt_errors = runtime_receipt_errors


def runtime_receipt_errors(r: dict[str, Any], *, rank: int, expected_calls: int) -> list[str]:
    if rank < 8:
        return [] if r.get("execution_status") == "NOT_APPLICABLE" and r.get("checkpoint_calls") == 0 else ["C38 Generator SAC changed"]
    if (expected_calls <= 0 or r.get("name") != SAC_CONTRACT["name"]
        or r.get("source", {}).get("interface_sha256") != INTERFACE_SHA256
        or r.get("source", {}).get("operator") != OP_NAME.replace("::", ".") + ".default"
        or r.get("execution_status") != "PASS" or r.get("production_installed") is not True
        or r.get("self_attention_only") is not True or r.get("contract_failures") != 0
        or r.get("checkpoint_calls") != expected_calls
        or r.get("original_fa_calls") != expected_calls
        or r.get("replay_fa_cache_reads") != expected_calls
        or r.get("expected_checkpoint_calls") != expected_calls
        or r.get("allow_cache_entry_mutation") is not False or r.get("cpu_offload") is not False):
        return ["C38 FA3 source/replay/census contract failed"]
    return []


def backend_receipt_errors(r: dict, *, rank: int) -> list[str]:
    expected = {k: v for k, v in BACKEND_CONTRACT.items()
                if k.startswith("artifact_") or k in ("name", "enabled", "num_splits", "source_flash_attention_sha256")}
    if any(r.get(k) != v for k, v in expected.items()) or r.get("contract_failure_count") != 0:
        return ["C38 backend source/option receipt changed"]
    if rank < 8:
        return [] if (r.get("execution_status") == "NOT_APPLICABLE"
            and r.get("function_patched") is False and r.get("invocation_count") == 0) else ["C38 Generator backend changed"]
    return [] if (r.get("execution_status") == "PASS" and r.get("install_status") == "PASS"
        and r.get("function_patched") is True and int(r.get("invocation_count", 0)) > 0
        and int(r.get("backward_invocation_count", 0)) > 0 and r.get("artifact_root")) else ["C38 Score backend was not executed"]
