"""Correctness-first shared-file exchange for the N2 capacity search.

The exchange is intentionally not a performance primitive.  N2 uses it to keep
the block-offload capacity variable isolated.  Any future timing experiment
must replace or separately account for this transport.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from .gst_pool_contract import GLOBAL_BATCH, TARGET_EXECUTION_GRAPH, source_identity
from .operation_trace import tensor_sha256


INPUT_SCHEMA = "distillschedule.gst_pool_input.v1"
STUDENT_SCHEMA = "distillschedule.gst_pool_student.v1"
GRADIENT_SCHEMA = "distillschedule.gst_pool_gradient.v1"


def atomic_torch_save(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite exchange payload: {path}")
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def wait_for_file(path: str | Path, timeout_seconds: float) -> float:
    path = Path(path)
    started = time.monotonic()
    while not path.is_file():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.05)
    return time.monotonic() - started


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exchange_paths(root: str | Path, source_id: int) -> dict[str, Path]:
    source_identity(source_id)
    root = Path(root)
    return {
        "input": root / f"input_{source_id:02d}.pt",
        "student": root / f"student_{source_id:02d}.pt",
        "gradient": root / f"gradient_{source_id:02d}.pt",
    }


def lane_source_ids(lane: int, *, lane_count: int = 4) -> tuple[int, ...]:
    if lane_count <= 0 or GLOBAL_BATCH % lane_count:
        raise ValueError("lane count must divide the target global batch")
    if not 0 <= lane < lane_count:
        raise ValueError("lane is outside the Score pool")
    return tuple(range(lane, GLOBAL_BATCH, lane_count))


def make_input_payload(
    *,
    run_id: str,
    source_id: int,
    generator_version: int,
    trace_identity_sha256: str,
    x0: torch.Tensor,
) -> dict[str, Any]:
    if list(x0.shape) != [1, 16, 9, 48, 80] or x0.dtype != torch.bfloat16:
        raise ValueError("target G/S/T wire tensor must be BF16 [1,16,9,48,80]")
    cpu = x0.detach().to("cpu").contiguous()
    return {
        "schema_version": INPUT_SCHEMA,
        "execution_graph": TARGET_EXECUTION_GRAPH,
        "run_id": run_id,
        **source_identity(source_id),
        "logical_step": 15,
        "generator_version": int(generator_version),
        "trace_identity_sha256": trace_identity_sha256,
        "x0": cpu,
        "x0_sha256": tensor_sha256(cpu),
    }


def validate_input_payload(
    value: Mapping[str, Any],
    *,
    run_id: str,
    source_id: int,
    trace_identity_sha256: str,
) -> torch.Tensor:
    expected = {
        "schema_version": INPUT_SCHEMA,
        "execution_graph": TARGET_EXECUTION_GRAPH,
        "run_id": run_id,
        **source_identity(source_id),
        "logical_step": 15,
        "generator_version": 3,
        "trace_identity_sha256": trace_identity_sha256,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"input exchange identity mismatch for {key}")
    x0 = value.get("x0")
    if not isinstance(x0, torch.Tensor):
        raise TypeError("input exchange x0 is not a Tensor")
    if list(x0.shape) != [1, 16, 9, 48, 80] or x0.dtype != torch.bfloat16:
        raise ValueError("input exchange x0 metadata changed")
    if tensor_sha256(x0) != value.get("x0_sha256"):
        raise ValueError("input exchange x0 digest mismatch")
    return x0


def validate_gradient_payload(
    value: Mapping[str, Any],
    *,
    run_id: str,
    source_id: int,
    trace_identity_sha256: str,
    x0_sha256: str,
) -> torch.Tensor:
    expected = {
        "schema_version": GRADIENT_SCHEMA,
        "execution_graph": TARGET_EXECUTION_GRAPH,
        "run_id": run_id,
        **source_identity(source_id),
        "logical_step": 15,
        "generator_version": 3,
        "student_version": 11,
        "teacher_version": "official_frozen_base",
        "trace_identity_sha256": trace_identity_sha256,
        "x0_sha256": x0_sha256,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"DMD gradient identity mismatch for {key}")
    gradient = value.get("input_gradient")
    if not isinstance(gradient, torch.Tensor):
        raise TypeError("DMD gradient payload is not a Tensor")
    if list(gradient.shape) != [1, 16, 9, 48, 80]:
        raise ValueError("DMD gradient shape changed")
    if tensor_sha256(gradient) != value.get("input_gradient_sha256"):
        raise ValueError("DMD gradient tensor digest mismatch")
    if not torch.isfinite(gradient).all():
        raise ValueError("DMD gradient contains non-finite values")
    return gradient
