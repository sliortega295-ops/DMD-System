#!/usr/bin/env python3
"""Run the bounded No-GAN TP1x1 vLLM-Omni Stage-1 canary.

The service process is deliberately separate.  This client only performs a
health/capability gate, one warmup request, and the five immutable B1 rows
(steps 10..14 by default); it never starts training or a distributed worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


EXPECTED_SHAPE = (1, 16, 9, 48, 80)
EXPECTED_DTYPE = "torch.bfloat16"


def request(base: str, path: str, method: str = "GET", value: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if value is None else json.dumps(value, sort_keys=True).encode("utf-8")
    req = Request(base.rstrip("/") + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=3600) as response:
        result = json.loads(response.read())
    if result.get("status") == "FAIL":
        raise RuntimeError(result.get("error", result))
    return result


def tensor_sha256(value: Any) -> str:
    import torch

    value = value.detach().to("cpu").contiguous()
    raw = value.view(torch.uint16) if value.dtype == torch.bfloat16 else value
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--generator-version", type=int, default=3)
    parser.add_argument("--steps", nargs="+", type=int, default=[10, 11, 12, 13, 14])
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    latents = out / "latents"
    latents.mkdir(exist_ok=True)

    started = time.time()
    health_before = request(args.endpoint, "/health")
    capabilities = request(args.endpoint, "/capabilities")
    sync = request(
        args.endpoint,
        "/sync_generator",
        "POST",
        {"generator_version": int(args.generator_version)},
    )
    warmup_path = latents / "warmup_step10.pt"
    warmup_started = time.perf_counter()
    warmup = request(
        args.endpoint,
        "/rollout",
        "POST",
        {
            "rank": int(args.rank),
            "step": int(args.steps[0]),
            "generator_version": int(args.generator_version),
            "response_path": str(warmup_path),
        },
    )
    warmup_seconds = time.perf_counter() - warmup_started

    import torch

    warmup_value = torch.load(warmup_path, map_location="cpu", weights_only=True)
    warmup_shape = tuple(int(v) for v in warmup_value.shape)
    if warmup_shape != EXPECTED_SHAPE or str(warmup_value.dtype) != EXPECTED_DTYPE:
        raise AssertionError(f"warmup output contract mismatch: {warmup_shape} {warmup_value.dtype}")
    warmup_path.unlink(missing_ok=True)

    rows: list[dict[str, Any]] = []
    for step in args.steps:
        response_path = latents / f"step{int(step)}.pt"
        t0 = time.perf_counter()
        receipt = request(
            args.endpoint,
            "/rollout",
            "POST",
            {
                "rank": int(args.rank),
                "step": int(step),
                "generator_version": int(args.generator_version),
                "response_path": str(response_path),
            },
        )
        elapsed = time.perf_counter() - t0
        value = torch.load(response_path, map_location="cpu", weights_only=True)
        shape = tuple(int(v) for v in value.shape)
        finite = bool(torch.isfinite(value.float()).all().item())
        row = {
            **receipt,
            "elapsed_seconds": elapsed,
            "shape": list(shape),
            "dtype": str(value.dtype),
            "finite": finite,
            "latent_sha256_client": tensor_sha256(value),
            "identity_match": tensor_sha256(value) == receipt.get("latent_sha256"),
        }
        if shape != EXPECTED_SHAPE or str(value.dtype) != EXPECTED_DTYPE or not finite:
            raise AssertionError(f"step{step} output contract mismatch: {row}")
        rows.append(row)
        response_path.unlink(missing_ok=True)

    health_after = request(args.endpoint, "/health")
    result = {
        "status": "PASS",
        "run_id": "VLLM_OMNI_019_TP1X1_20260815_R2",
        "variant": "no-gan",
        "workload": "isolated B1 critic-input rollout",
        "rank": int(args.rank),
        "generator_version": int(args.generator_version),
        "logical_steps": [int(v) for v in args.steps],
        "warmup": {"receipt": warmup, "elapsed_seconds": warmup_seconds},
        "health_before": health_before,
        "capabilities": capabilities,
        "sync": sync,
        "rows": rows,
        "health_after": health_after,
        "elapsed_wall_seconds": time.time() - started,
        "vae_decode_calls": 0,
    }
    (out / "stage1_service_receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
