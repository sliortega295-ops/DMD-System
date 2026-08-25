"""Fail-closed compatibility checks for Helios gated-residual storage."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any


def validate_chunked_residual_runtime(
    transformer_module: Any,
    *,
    requested_storage: str,
) -> dict[str, Any]:
    """Prove that an imported Helios source can honor a residual policy.

    Historical Helios sources always staged the gated residual through CPU.
    Setting ``_distillschedule_chunked_residual_storage`` on their blocks was a
    silent no-op, so a configuration receipt alone is not sufficient evidence.
    """

    if requested_storage not in {"cpu_inplace", "cuda_out_of_place"}:
        raise ValueError(f"unsupported chunked residual storage: {requested_storage}")

    helper = getattr(transformer_module, "_apply_chunked_gated_residual", None)
    autograd_function = getattr(transformer_module, "_TembAwareChunkedResidual", None)
    block_type = getattr(transformer_module, "HeliosTransformerBlock", None)
    if helper is None or autograd_function is None or block_type is None:
        raise RuntimeError("Helios runtime is missing chunked-residual primitives")

    helper_parameters = tuple(inspect.signature(helper).parameters)
    forward_parameters = tuple(inspect.signature(autograd_function.forward).parameters)
    try:
        block_forward_source = inspect.getsource(block_type.forward)
    except (OSError, TypeError) as error:
        raise RuntimeError("cannot audit Helios block forward source") from error

    storage_getter_count = block_forward_source.count(
        "_distillschedule_chunked_residual_storage"
    )
    storage_capable = (
        "residual_storage" in helper_parameters
        and "residual_storage" in forward_parameters
        and storage_getter_count == 2
    )
    if requested_storage == "cuda_out_of_place" and not storage_capable:
        raise RuntimeError(
            "imported Helios runtime cannot honor cuda_out_of_place residual "
            "storage; refusing the historical silent CPU-staging fallback"
        )

    source_file = inspect.getsourcefile(transformer_module)
    if not source_file:
        raise RuntimeError("cannot resolve Helios transformer source identity")
    source_path = Path(source_file).resolve()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return {
        "requested_storage": requested_storage,
        "storage_capable": storage_capable,
        "helper_has_storage_parameter": "residual_storage" in helper_parameters,
        "autograd_has_storage_parameter": "residual_storage" in forward_parameters,
        "block_storage_getter_count": storage_getter_count,
        "transformer_source_path": str(source_path),
        "transformer_source_sha256": source_sha256,
    }
