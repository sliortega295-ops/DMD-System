"""Plan-gated FlashAttention-2 backend for persistent LiveWan Score ranks."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
from pathlib import Path
import sys
import textwrap
from typing import Any


EXPECTED_FLASH_ATTENTION_SHA256 = (
    "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
)
EXPECTED_METADATA_SHA256 = (
    "7749dd68b81a544c1973d68c30be5aa316ee1a718ae0228e0368c2cd04365516"
)
EXPECTED_PACKAGE_SHA256 = (
    "0c560f96b857c188c4a8297ff27d1299619957c3459a5e433e132329fbfdf066"
)
EXPECTED_EXTENSION_SHA256 = (
    "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
)
DEFAULT_KERNEL_PATH = Path(
    "/share/project/shuchenweng/DistillSchedule_assets/"
    "hf_home_official_torch210/hub/"
    "models--kernels-community--flash-attn2/snapshots/"
    "6a810a7a010ba2aad6613f1f0f7b226fb8bdee3c/build/"
    "torch210-cxx11-cu128-x86_64-linux"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256(value: Any) -> str:
    source = textwrap.dedent(inspect.getsource(value)).strip()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _load_flash_attn2() -> tuple[Any, Path]:
    root = Path(
        os.environ.get("DISTILLSCHEDULE_FLASH_ATTN_KERNEL", DEFAULT_KERNEL_PATH)
    ).expanduser().resolve()
    required = {
        "metadata.json": EXPECTED_METADATA_SHA256,
        "flash_attn2/__init__.py": EXPECTED_PACKAGE_SHA256,
        "_flash_attn2_cuda_a511a4c.abi3.so": EXPECTED_EXTENSION_SHA256,
    }
    for relative, expected in required.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"pinned FlashAttention-2 artifact missing: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"pinned FlashAttention-2 artifact changed: {path}: "
                f"expected {expected}, observed {actual}"
            )

    # Wan has already imported its patched attention module before this point.
    # Add the external-kernel directory only while importing ``flash_attn2``;
    # leaving it on sys.path would make Wan's optional ``flash_attn_interface``
    # import resolve the kernel package as though it were FlashAttention-3.
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("flash_attn2")
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
    return module, root


def _install_flash_attn2(
    torch: Any,
    *,
    requested_name: str,
    role_label: str,
    allowed_input_dtypes: set[tuple[Any, Any, Any]],
) -> dict[str, Any]:
    """Install the pinned dense FA2 kernel under an exact per-role contract.

    The upstream helper casts non-half query/key/value inputs to ``dtype`` and
    finally casts query/key to the value dtype before invoking attention.  The
    Generator C18 text-K/V cache intentionally preserves normalized keys in
    FP32, so its registered input contract contains both all-BF16 self
    attention and FP32-query/key + BF16-value cross attention.  Reproducing
    those exact casts here is necessary to preserve the pinned helper's dtype
    semantics while bypassing its cuDNN-SDPA frontend.
    """

    from wan.modules import attention as attention_module
    from wan.modules import model as model_module
    from wanstreamer import blockcausal

    if getattr(blockcausal, "_distillschedule_attention_backend", False):
        installed = getattr(
            blockcausal, "_distillschedule_attention_backend_name", None
        )
        if installed != requested_name:
            raise RuntimeError(
                f"LiveWan Score attention already installed as {installed}, "
                f"requested {requested_name}"
            )
        return getattr(blockcausal, "_distillschedule_attention_receipt")

    originals = {
        "attention": attention_module.flash_attention,
        "model": model_module.flash_attention,
        "blockcausal": blockcausal.flash_attention,
    }
    source_hashes = {name: _source_sha256(value) for name, value in originals.items()}
    if set(source_hashes.values()) != {EXPECTED_FLASH_ATTENTION_SHA256}:
        raise RuntimeError(
            "LiveWan flash_attention source changed: "
            f"expected {EXPECTED_FLASH_ATTENTION_SHA256}, observed {source_hashes}"
        )

    flash_attn2, kernel_root = _load_flash_attn2()
    receipt: dict[str, Any] = {
        "enabled": True,
        "name": requested_name,
        "source_flash_attention_sha256": EXPECTED_FLASH_ATTENTION_SHA256,
        "artifact_metadata_sha256": EXPECTED_METADATA_SHA256,
        "artifact_package_sha256": EXPECTED_PACKAGE_SHA256,
        "artifact_extension_sha256": EXPECTED_EXTENSION_SHA256,
        "artifact_root": str(kernel_root),
        "function_patched": True,
        "install_status": "PASS",
        "invocation_count": 0,
        "backward_invocation_count": 0,
        "element_count": 0,
        "backward_element_count": 0,
        "shape_histogram": {},
        "raw_dtype_histogram": {},
        "input_cast_count": 0,
        "output_cast_count": 0,
        "contract_failure_count": 0,
    }

    def validate(
        q: Any,
        k: Any,
        v: Any,
        *,
        q_lens: Any,
        k_lens: Any,
        dropout_p: float,
        q_scale: Any,
        causal: bool,
        window_size: Any,
        dtype: Any,
        version: Any,
    ) -> None:
        try:
            if (
                not q.is_cuda
                or not k.is_cuda
                or not v.is_cuda
                or (q.dtype, k.dtype, v.dtype) not in allowed_input_dtypes
                or q.ndim != 4
                or k.ndim != 4
                or v.ndim != 4
                or not q.is_contiguous()
                or not k.is_contiguous()
                or not v.is_contiguous()
                or q.shape[0] != k.shape[0]
                or k.shape != v.shape
                or q.shape[2:] != k.shape[2:]
                or q.shape[-1] != 128
            ):
                raise RuntimeError("device, dtype, rank, shape, or contiguity mismatch")
            if q_lens is not None or k_lens is not None:
                raise RuntimeError("masked or variable-length attention is not registered")
            if (
                float(dropout_p) != 0.0
                or q_scale is not None
                or bool(causal)
                or tuple(window_size) != (-1, -1)
                or dtype != torch.bfloat16
                or version is not None
            ):
                raise RuntimeError("attention option contract changed")
        except Exception as error:
            receipt["contract_failure_count"] += 1
            raise RuntimeError(
                f"LiveWan {role_label} FlashAttention-2 contract changed: "
                f"q={q.device}/{q.dtype}/{tuple(q.shape)}/{tuple(q.stride())}, "
                f"k={k.device}/{k.dtype}/{tuple(k.shape)}/{tuple(k.stride())}, "
                f"v={v.device}/{v.dtype}/{tuple(v.shape)}/{tuple(v.stride())}, "
                f"q_lens={q_lens is not None}, k_lens={k_lens is not None}, "
                f"dropout={dropout_p}, q_scale={q_scale}, causal={causal}, "
                f"window={window_size}, dtype={dtype}, version={version}: {error}"
            ) from error

    def fused_attention(
        q: Any,
        k: Any,
        v: Any,
        q_lens: Any = None,
        k_lens: Any = None,
        dropout_p: float = 0.0,
        softmax_scale: Any = None,
        q_scale: Any = None,
        causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        deterministic: bool = False,
        dtype: Any = torch.bfloat16,
        version: Any = None,
    ) -> Any:
        validate(
            q,
            k,
            v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
            version=version,
        )
        raw_dtypes = (q.dtype, k.dtype, v.dtype)
        output_dtype = q.dtype
        # Match the pinned upstream helper: non-half inputs are first converted
        # to ``dtype`` and query/key are then converted to the value dtype.  The
        # registered contracts always use BF16 values, so FA2 receives one
        # homogeneous BF16 triplet.
        q_kernel = q if q.dtype == torch.bfloat16 else q.to(dtype)
        k_kernel = k if k.dtype == torch.bfloat16 else k.to(dtype)
        v_kernel = v if v.dtype == torch.bfloat16 else v.to(dtype)
        q_kernel = q_kernel.to(v_kernel.dtype)
        k_kernel = k_kernel.to(v_kernel.dtype)
        receipt["input_cast_count"] += int(q_kernel is not q)
        receipt["input_cast_count"] += int(k_kernel is not k)
        receipt["input_cast_count"] += int(v_kernel is not v)
        output = flash_attn2.flash_attn_func(
            q_kernel,
            k_kernel,
            v_kernel,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )
        if output.dtype != output_dtype:
            output = output.to(output_dtype)
            receipt["output_cast_count"] += 1
        receipt["invocation_count"] += 1
        receipt["element_count"] += int(output.numel())
        shape_key = (
            "q=" + "x".join(str(part) for part in q.shape)
            + "|k=" + "x".join(str(part) for part in k.shape)
        )
        receipt["shape_histogram"][shape_key] = (
            int(receipt["shape_histogram"].get(shape_key, 0)) + 1
        )
        dtype_key = "/".join(str(value).removeprefix("torch.") for value in raw_dtypes)
        receipt["raw_dtype_histogram"][dtype_key] = (
            int(receipt["raw_dtype_histogram"].get(dtype_key, 0)) + 1
        )
        if output.requires_grad:
            elements = int(output.numel())

            def record_backward(gradient: Any) -> Any:
                receipt["backward_invocation_count"] += 1
                receipt["backward_element_count"] += elements
                return gradient

            output.register_hook(record_backward)
        return output

    attention_module._distillschedule_original_flash_attention = originals["attention"]
    model_module._distillschedule_original_flash_attention = originals["model"]
    blockcausal._distillschedule_original_flash_attention = originals["blockcausal"]
    attention_module.flash_attention = fused_attention
    model_module.flash_attention = fused_attention
    blockcausal.flash_attention = fused_attention
    blockcausal._distillschedule_attention_backend = True
    blockcausal._distillschedule_attention_backend_name = requested_name
    blockcausal._distillschedule_attention_receipt = receipt
    return receipt


def install_score_flash_attn2(torch: Any) -> dict[str, Any]:
    """Replace only Score-rank unmasked BF16 attention with pinned FA2."""

    return _install_flash_attn2(
        torch,
        requested_name="score_flash_attn2_unmasked_bf16_v1",
        role_label="Score",
        allowed_input_dtypes={(torch.bfloat16, torch.bfloat16, torch.bfloat16)},
    )


def install_generator_flash_attn2(torch: Any) -> dict[str, Any]:
    """Replace Generator-rank dense attention after installing the C18 K/V cache.

    Self attention arrives as BF16/BF16/BF16.  Wan cross attention normalizes
    query/key in FP32 and projects value in BF16; the pinned upstream helper
    casts that triplet to BF16 for the kernel and casts its output back to the
    query dtype.  No other dtype combination is registered.
    """

    return _install_flash_attn2(
        torch,
        requested_name="generator_flash_attn2_unmasked_exact_cast_v1",
        role_label="Generator",
        allowed_input_dtypes={
            (torch.bfloat16, torch.bfloat16, torch.bfloat16),
            (torch.float32, torch.float32, torch.bfloat16),
        },
    )


def uninstall_score_flash_attn2() -> None:
    """Test-only restoration; production workers exit with the patch installed."""

    from wan.modules import attention as attention_module
    from wan.modules import model as model_module
    from wanstreamer import blockcausal

    targets = (
        (attention_module, "attention"),
        (model_module, "model"),
        (blockcausal, "blockcausal"),
    )
    for module, _ in targets:
        original = getattr(module, "_distillschedule_original_flash_attention", None)
        if original is not None:
            module.flash_attention = original
            delattr(module, "_distillschedule_original_flash_attention")
    for name in (
        "_distillschedule_attention_backend",
        "_distillschedule_attention_backend_name",
        "_distillschedule_attention_receipt",
    ):
        if hasattr(blockcausal, name):
            delattr(blockcausal, name)


def uninstall_generator_flash_attn2() -> None:
    """Test-only alias for restoring the globally patched Wan helper."""

    uninstall_score_flash_attn2()


__all__ = [
    "install_score_flash_attn2",
    "install_generator_flash_attn2",
    "uninstall_score_flash_attn2",
    "uninstall_generator_flash_attn2",
]
