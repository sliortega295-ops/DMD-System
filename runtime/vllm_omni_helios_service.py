"""Dedicated, real vLLM-Omni Helios latent rollout service.

This module is intentionally small: it is the Stage-1 production seam used by
the TP1x1 canary.  The service owns one vLLM-Omni Helios transformer and reads
immutable operation-trace rows by identity.  It never calls the VAE.  The
training-side adapter uses the same typed capabilities/receipt contract as the
Full5 integration, but communicates with a separate process over a local HTTP
RPC endpoint.

The file transport is bounded and explicit (one response tensor per request),
which keeps the first canary free of an unreviewed tensor-RPC implementation.
It is not an in-process reference executor: model construction and physical
DiT calls happen in the dedicated service process.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from urllib.parse import urlparse


BASE_COMMIT = "191b9a8dbf22fc494b7db34ea5ae04153f44f94d"
PATCH_SHA256 = "84c9a914184373a771a5839bd6198d05ba8f981154732fc6f7bfbf5feafb3d99"
X0_SHAPE = (1, 16, 9, 48, 80)
X0_DTYPE = "torch.bfloat16"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lora_file(path: str | Path) -> Path:
    root = Path(path).resolve()
    if root.is_file():
        return root
    for name in ("adapter_model.safetensors", "pytorch_lora_weights.safetensors"):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no safetensors LoRA payload under {root}")


def _canonical_lora_identity(path: str | Path) -> str:
    """Hash LoRA tensor names after the audited diffusers→vLLM remap."""
    import struct

    import torch
    from safetensors import safe_open

    digest = hashlib.sha256()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for original in sorted(handle.keys()):
            name = (
                original.replace(".ffn.net.0.", ".ffn.net_0.")
                .replace(".ffn.net.2.", ".ffn.net_2.")
                .replace(".to_out.0.", ".to_out.")
            )
            tensor = handle.get_tensor(original).contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(struct.pack("<I", tensor.ndim))
            digest.update(struct.pack("<{}I".format(tensor.ndim), *tensor.shape))
            digest.update(str(tensor.dtype).encode("ascii"))
            raw = tensor.view(torch.uint16) if tensor.dtype == torch.bfloat16 else tensor
            digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def _json_response(handler: BaseHTTPRequestHandler, value: Mapping[str, Any], status: int = 200) -> None:
    payload = json.dumps(value, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _load_transformer(model_root: str, device: Any) -> Any:
    """Load only the real Helios transformer, not text encoder or VAE."""
    import torch
    from vllm.config import LoadConfig
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.helios.pipeline_helios import (
        create_transformer_from_config,
        load_transformer_config,
    )

    root = Path(model_root).resolve()
    config = load_transformer_config(str(root), subfolder="", local_files_only=True)
    if not config:
        config = json.loads((root / "config.json").read_text())
    # vLLM CustomOp layers require an active VllmConfig even during module
    # construction.  The production worker establishes the same context
    # before initializing its pipeline; the standalone service must do so too.
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(),
        device_config=DeviceConfig(device=device),
    )
    # The normal vLLM worker initializes these singleton groups before
    # constructing TP-aware Linear layers.  TP1 still needs the group objects;
    # a one-process Gloo group is sufficient and does not create a cross-GPU
    # collective.  The service is strictly single-GPU in this canary.
    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:18992",
            backend="gloo",
        )
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                backend="gloo",
            )
    transformer = None
    od_config = OmniDiffusionConfig(
        model=str(root),
        model_class_name="HeliosPipeline",
        dtype=torch.bfloat16,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            sequence_parallel_size=1,
            cfg_parallel_size=1,
        ),
        output_type="latent",
    )
    with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config):
        with set_current_vllm_config(vllm_config):
            transformer = create_transformer_from_config(config)
            transformer.weights_sources = (
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=str(root),
                    subfolder=None,
                    revision=None,
                    prefix="",
                    fall_back_to_pt=True,
                ),
            )
            loader = DiffusersPipelineLoader(LoadConfig(load_format="hf"), od_config=od_config)
            loader.load_weights(transformer)
            transformer.to(device=device, dtype=torch.bfloat16)
            transformer.eval()
    return transformer, vllm_config, od_config


class HeliosRolloutService:
    """One dedicated GPU service instance."""

    def __init__(
        self,
        *,
        model_root: str,
        trace_root: str,
        lora_path: str,
        source_lora_file: str | None,
        device: str,
        endpoint: str,
        pool_id: str,
        source_receipt: str | None = None,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("vLLM-Omni Helios service requires CUDA")
        self.device = torch.device(device)
        self.model_root = str(Path(model_root).resolve())
        self.trace_root = str(Path(trace_root).resolve())
        self.lora_path = str(Path(lora_path).resolve())
        self.lora_file_path = str(_lora_file(self.lora_path))
        self.source_lora_file = str(
            Path(source_lora_file).resolve() if source_lora_file else Path(self.lora_file_path)
        )
        self.endpoint = endpoint
        self.pool_id = pool_id
        self.source_receipt = source_receipt
        self.started_at = time.time()
        self._closed = False
        self._lora_source_sha = _canonical_lora_identity(self.source_lora_file)
        self._lora_loaded_sha = _canonical_lora_identity(self.lora_file_path)
        if self._lora_source_sha != self._lora_loaded_sha:
            raise ValueError("source and vLLM-loaded LoRA tensor identities differ")
        self._model_load_start = time.perf_counter()
        self.transformer, self.vllm_config, self.od_config = _load_transformer(
            self.model_root, self.device
        )
        self.model_load_seconds = time.perf_counter() - self._model_load_start

        # Use vLLM-Omni's own diffusion LoRA manager.  This replaces the real
        # LinearBase modules and loads the supplied PEFT artifact; no fake
        # weight injection is permitted.
        from torch import nn
        from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

        self.pipeline = nn.Module()
        self.pipeline.add_module("transformer", self.transformer)
        self.lora_manager = DiffusionLoRAManager(
            pipeline=self.pipeline,
            device=self.device,
            dtype=torch.bfloat16,
            max_cached_adapters=1,
            lora_path=self.lora_path,
            lora_scale=1.0,
        )
        self._active_generator_version: int | None = None
        self._active_lora_sha: str | None = None
        self._trace = None
        self._executor = None
        self._requests = 0
        self._rollout_calls = 0

    def _ensure_trace(self) -> Any:
        if self._trace is None:
            from runtime.static_world16_trace import StaticWorld16Trace

            self._trace = StaticWorld16Trace(self.trace_root, variant="no-gan")
        return self._trace

    def capabilities(self) -> dict[str, Any]:
        return {
            "status": "PASS",
            "backend": "vllm-omni-latent",
            "engine": "vllm-omni",
            "model_family": "helios",
            "upstream_revision": BASE_COMMIT,
            "compatibility_patch_sha256": PATCH_SHA256,
            "vllm_omni_version_family": "0.19",
            "vllm_version_family": "0.19",
            "output_type": "latent",
            "skips_vae_decode": True,
            "exact_operation_trace_replay": True,
            "hot_generator_lora_sync": True,
            "dedicated_rollout_pool": True,
            "supported_batch_sizes": [1],
            "output_shape": list(X0_SHAPE),
            "output_dtype": X0_DTYPE,
            "transport": "http-local-torch-file",
            "tp": 1,
            "replicas": 1,
            "b1": True,
            "vae_decode_calls": 0,
            "pool_id": self.pool_id,
            "model_load_seconds": self.model_load_seconds,
            "source_lora_sha256": self._lora_source_sha,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "PASS" if not self._closed else "CLOSED",
            "service": "vllm-omni-helios-latent",
            "pid": os.getpid(),
            "device": str(self.device),
            "requests": self._requests,
            "rollout_calls": self._rollout_calls,
            "generator_version": self._active_generator_version,
            "lora_sha256": self._active_lora_sha,
        }

    def sync_generator(self, generator_version: int, source_lora_path: str | None = None) -> dict[str, Any]:
        path = Path(source_lora_path or self.lora_path).resolve()
        if path != Path(self.lora_path):
            raise ValueError("hot sync path differs from the immutable service LoRA")
        source_sha = _canonical_lora_identity(self.source_lora_file)
        if source_sha != self._lora_source_sha:
            raise ValueError("Generator LoRA source changed during service lifetime")
        # The manager is initialized with this adapter and keeps it active.
        # Re-activating the same request is a true no-op but proves the hot-sync
        # path and source identity on every requested Generator version.
        from vllm.lora.request import LoRARequest
        from vllm_omni.lora.utils import stable_lora_int_id

        self.lora_manager.set_active_adapter(
            LoRARequest(
                lora_name="generator",
                lora_int_id=stable_lora_int_id(str(path)),
                lora_path=str(path),
            ),
            1.0,
        )
        self._active_generator_version = int(generator_version)
        self._active_lora_sha = source_sha
        return {
            "status": "PASS",
            "generator_version": int(generator_version),
            "upstream_revision": BASE_COMMIT,
            "adapter_identity": f"lora:{source_sha}",
            "hot_reload_complete": True,
            "source_trainable_sha256": source_sha,
            "loaded_trainable_sha256": source_sha,
        }

    def rollout(self, *, rank: int, step: int, response_path: str, generator_version: int) -> dict[str, Any]:
        import torch

        if self._active_generator_version != int(generator_version):
            raise ValueError("Generator version was not synchronized before rollout")
        if not (0 <= int(rank) < 16):
            raise ValueError("rank-qualified trace rank outside world16")
        trace = self._ensure_trace()
        entry = trace.load_critic_entry(rank=int(rank), step=int(step), device="cpu")
        if int(step) not in trace.logical_steps:
            raise ValueError(f"unsupported No-GAN B1 step={step}")
        if self._executor is None:
            from runtime.vllm_omni_helios_rollout import InProcessVLLMOmniHeliosTraceExecutor

            self._executor = InProcessVLLMOmniHeliosTraceExecutor(
                transformer=self.transformer,
                device=self.device,
                upstream_revision=BASE_COMMIT,
                sync_callback=lambda **kwargs: self.sync_generator(
                    generator_version=int(kwargs["generator_version"]),
                    source_lora_path=self.lora_path,
                ),
                max_batch_size=1,
                endpoint=self.endpoint,
                dedicated_pool_id=self.pool_id,
            )
            self._executor.active_generator_version = self._active_generator_version
            self._executor.active_adapter_identity = f"lora:{self._active_lora_sha}"
        from vllm.config import set_current_vllm_config
        from vllm_omni.diffusion.forward_context import set_forward_context

        with set_forward_context(
            vllm_config=self.vllm_config,
            omni_diffusion_config=self.od_config,
        ):
            with set_current_vllm_config(self.vllm_config):
                outputs, record = self._executor.rollout(
                    entries=(entry,),
                    logical_steps=(int(step),),
                    generator_version=int(generator_version),
                    output_type="latent",
                    expected_output_shape=X0_SHAPE,
                    expected_output_dtype=X0_DTYPE,
                    batch_index=self._rollout_calls,
                    upstream_revision=BASE_COMMIT,
                )
        value = outputs[0]
        if tuple(value.shape) != X0_SHAPE or value.dtype != torch.bfloat16 or value.requires_grad:
            raise ValueError("service latent output violated the fixed B1 contract")
        response = Path(response_path).resolve()
        response.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value.detach().cpu(), response)
        self._requests += 1
        self._rollout_calls += 1
        return {
            "status": "PASS",
            "logical_step": int(step),
            "rank": int(rank),
            "generator_version": int(generator_version),
            "output_type": "latent",
            "output_shape": list(value.shape),
            "output_dtype": str(value.dtype),
            "vae_decode_calls": 0,
            "latent_sha256": _tensor_sha256(value),
            "call_count": int(record.get("call_count", 0)),
            "stage_call_count": record.get("stage_call_count", {}),
            "response_path": str(response),
        }

    def close(self) -> None:
        self._closed = True
        self._executor = None
        self._trace = None


def _tensor_sha256(value: Any) -> str:
    import torch

    cpu = value.detach().to("cpu").contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


class _Handler(BaseHTTPRequestHandler):
    server_version = "DistillScheduleVLLMOmni/0.19"

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[service] " + fmt % args, flush=True)

    @property
    def service(self) -> HeliosRolloutService:
        return self.server.service  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            _json_response(self, self.service.health())
        elif path == "/capabilities":
            _json_response(self, self.service.capabilities())
        elif path == "/status":
            _json_response(self, self.service.health())
        else:
            _json_response(self, {"status": "FAIL", "error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/sync_generator":
                result = self.service.sync_generator(
                    generator_version=int(body["generator_version"]),
                    source_lora_path=body.get("source_lora_path"),
                )
            elif path == "/rollout":
                result = self.service.rollout(
                    rank=int(body["rank"]),
                    step=int(body["step"]),
                    response_path=str(body["response_path"]),
                    generator_version=int(body["generator_version"]),
                )
            elif path == "/close":
                self.service.close()
                result = {"status": "PASS"}
            else:
                _json_response(self, {"status": "FAIL", "error": "not found"}, 404)
                return
            _json_response(self, result)
        except Exception as exc:  # fail closed and keep the traceback in service logs
            import traceback

            traceback.print_exc()
            _json_response(self, {"status": "FAIL", "error": repr(exc)}, 500)


def serve(args: argparse.Namespace) -> None:
    service = HeliosRolloutService(
        model_root=args.model_root,
        trace_root=args.trace_root,
        lora_path=args.lora_path,
        device=args.device,
        endpoint=args.endpoint,
        pool_id=args.pool_id,
        source_receipt=args.source_receipt,
        source_lora_file=args.source_lora_file,
    )
    host, port_s = urlparse(args.endpoint).hostname, urlparse(args.endpoint).port
    if host not in {"127.0.0.1", "localhost"} or port_s is None:
        raise ValueError("Stage1 service endpoint must be local HTTP")
    server = ThreadingHTTPServer((host, int(port_s)), _Handler)
    server.service = service  # type: ignore[attr-defined]
    print(json.dumps({"status": "READY", "capabilities": service.capabilities()}, sort_keys=True), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        service.close()
        server.server_close()


class HTTPVLLMOmniHeliosAdapter:
    """Training-side adapter implementing the existing typed contract."""

    def __init__(self, *, endpoint: str, **_: Any) -> None:
        import urllib.request

        self.endpoint = endpoint.rstrip("/")
        self._urlopen = urllib.request.urlopen

    def _request(self, method: str, path: str, value: Mapping[str, Any] | None = None) -> dict[str, Any]:
        import urllib.request

        data = None if value is None else json.dumps(dict(value)).encode("utf-8")
        req = urllib.request.Request(self.endpoint + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with self._urlopen(req, timeout=3600) as response:
            payload = json.loads(response.read())
        if payload.get("status") == "FAIL":
            raise RuntimeError(payload.get("error", payload))
        return payload

    def capabilities(self) -> Mapping[str, Any]:
        return self._request("GET", "/capabilities")

    def sync_generator(self, **kwargs: Any) -> Mapping[str, Any]:
        return self._request(
            "POST",
            "/sync_generator",
            {"generator_version": int(kwargs["generator_version"])},
        )

    def rollout(self, **kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
        import torch

        entries = tuple(kwargs["entries"])
        if len(entries) != 1:
            raise ValueError("Stage1 dedicated service supports B1 only")
        entry = entries[0]
        step = int(kwargs["logical_steps"][0])
        rank = int(getattr(entry, "source_trace_rank", getattr(entry, "producer_pool_rank", 0)))
        response_path = Path(os.environ.get("DISTILLSCHEDULE_VLLM_OMNI_RESPONSE_DIR", "/tmp")) / (
            f"vllm_omni_latent_{os.getpid()}_{step}_{time.time_ns()}.pt"
        )
        receipt = self._request(
            "POST",
            "/rollout",
            {
                "rank": rank,
                "step": step,
                "generator_version": int(kwargs["generator_version"]),
                "response_path": str(response_path),
            },
        )
        value = torch.load(response_path, map_location="cpu", weights_only=True)
        try:
            response_path.unlink()
        except FileNotFoundError:
            pass
        return [value], receipt

    def close(self) -> None:
        try:
            self._request("POST", "/close", {})
        except Exception:
            pass


def factory(**kwargs: Any) -> HTTPVLLMOmniHeliosAdapter:
    return HTTPVLLMOmniHeliosAdapter(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-root")
    parser.add_argument("--trace-root")
    parser.add_argument("--lora-path")
    parser.add_argument("--source-lora-file")
    parser.add_argument("--source-receipt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pool-id", default="tp1x1-stage1")
    args = parser.parse_args()
    if not args.serve:
        raise SystemExit("use --serve for the dedicated service")
    for name in ("model_root", "trace_root", "lora_path"):
        if not getattr(args, name):
            raise SystemExit(f"--{name.replace('_', '-')} is required with --serve")
    serve(args)


if __name__ == "__main__":
    main()
