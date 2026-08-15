"""vLLM-Omni latent rollout adapter for the static Helios Full5 worker.

The training worker talks to an adapter factory rather than depending on a
particular RPC stack. A factory may return a remote client or the included
in-process exact-trace executor. Both must expose the same fail-closed
capability, LoRA-sync, and latent-receipt contract.
"""
from __future__ import annotations

import dataclasses
import os
import time
from types import SimpleNamespace
from typing import Any, Mapping, Protocol, Sequence

from distillschedule.runtime.p0_parity import tensor_parity_metrics
from distillschedule.runtime.vllm_omni_rollout import (
    OFFICIAL_X0_DTYPE,
    OFFICIAL_X0_SHAPE,
    VLLM_OMNI_BACKEND,
    VLLM_OMNI_HELIOS_LATENT_REFERENCE_COMMIT,
    VLLMOmniRolloutCapabilities,
    expected_generator_version,
    load_factory,
    normalize_capabilities,
    validate_rollout_receipt,
)


_LAST_RUN_RECEIPT: dict[str, Any] | None = None


class RolloutAdapter(Protocol):
    def capabilities(self) -> Mapping[str, Any] | VLLMOmniRolloutCapabilities: ...

    def sync_generator(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def rollout(self, **kwargs: Any) -> Any: ...

    def close(self) -> None: ...


@dataclasses.dataclass(frozen=True)
class VLLMOmniAdapterEnvironment:
    factory_spec: str
    endpoint: str
    expected_revision: str
    dedicated_pool_id: str

    @classmethod
    def from_environment(cls) -> "VLLMOmniAdapterEnvironment":
        factory = os.environ.get("DISTILLSCHEDULE_VLLM_OMNI_ADAPTER_FACTORY", "").strip()
        endpoint = os.environ.get("DISTILLSCHEDULE_VLLM_OMNI_ENDPOINT", "").strip()
        revision = os.environ.get(
            "DISTILLSCHEDULE_VLLM_OMNI_EXPECTED_REVISION",
            VLLM_OMNI_HELIOS_LATENT_REFERENCE_COMMIT,
        ).strip()
        pool_id = os.environ.get("DISTILLSCHEDULE_VLLM_OMNI_POOL_ID", "").strip()
        dedicated = os.environ.get("DISTILLSCHEDULE_VLLM_OMNI_DEDICATED_POOL", "0")
        missing = [
            name
            for name, value in (
                ("DISTILLSCHEDULE_VLLM_OMNI_ADAPTER_FACTORY", factory),
                ("DISTILLSCHEDULE_VLLM_OMNI_ENDPOINT", endpoint),
                ("DISTILLSCHEDULE_VLLM_OMNI_EXPECTED_REVISION", revision),
                ("DISTILLSCHEDULE_VLLM_OMNI_POOL_ID", pool_id),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"missing vLLM-Omni rollout environment: {missing}")
        if dedicated != "1":
            raise ValueError(
                "vLLM-Omni Full5 rollout requires a dedicated GPU pool; set "
                "DISTILLSCHEDULE_VLLM_OMNI_DEDICATED_POOL=1"
            )
        return cls(factory, endpoint, revision, pool_id)


class VLLMOmniRolloutClient:
    def __init__(
        self,
        *,
        adapter: RolloutAdapter,
        environment: VLLMOmniAdapterEnvironment,
        rank: int,
        world_size: int,
        requested_batch_size: int,
    ) -> None:
        self.adapter = adapter
        self.environment = environment
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.capabilities = normalize_capabilities(adapter.capabilities())
        self.capability_receipt = self.capabilities.validate(
            requested_batch_size=requested_batch_size,
            expected_revision=environment.expected_revision,
            require_hot_lora_sync=True,
        )
        self.sync_receipt: dict[str, Any] | None = None

    @classmethod
    def from_environment(
        cls,
        *,
        rank: int,
        world_size: int,
        requested_batch_size: int,
    ) -> "VLLMOmniRolloutClient":
        environment = VLLMOmniAdapterEnvironment.from_environment()
        factory = load_factory(environment.factory_spec)
        adapter = factory(
            endpoint=environment.endpoint,
            expected_revision=environment.expected_revision,
            dedicated_pool_id=environment.dedicated_pool_id,
            rank=int(rank),
            world_size=int(world_size),
        )
        for method in ("capabilities", "sync_generator", "rollout"):
            if not callable(getattr(adapter, method, None)):
                raise TypeError(f"vLLM-Omni adapter misses callable {method}()")
        return cls(
            adapter=adapter,
            environment=environment,
            rank=rank,
            world_size=world_size,
            requested_batch_size=requested_batch_size,
        )

    def sync_generator(
        self,
        *,
        generator_engine: Any,
        variant: str,
        generator_version: int,
    ) -> dict[str, Any]:
        import torch.distributed as dist

        receipt = dict(
            self.adapter.sync_generator(
                generator_engine=generator_engine,
                variant=variant,
                generator_version=int(generator_version),
                upstream_revision=self.environment.expected_revision,
                endpoint=self.environment.endpoint,
                dedicated_pool_id=self.environment.dedicated_pool_id,
                rank=self.rank,
                world_size=self.world_size,
            )
        )
        errors: list[str] = []
        if receipt.get("status") != "PASS":
            errors.append(f"status={receipt.get('status')!r}")
        if int(receipt.get("generator_version", -1)) != int(generator_version):
            errors.append("Generator version mismatch")
        if receipt.get("upstream_revision") != self.environment.expected_revision:
            errors.append("vLLM-Omni revision mismatch")
        if not receipt.get("adapter_identity"):
            errors.append("adapter_identity is absent")
        if not bool(receipt.get("hot_reload_complete")):
            errors.append("hot_reload_complete is false")
        if errors:
            raise ValueError("vLLM-Omni Generator sync failed: " + "; ".join(errors))

        identity = (
            int(receipt["generator_version"]),
            str(receipt["upstream_revision"]),
            str(receipt["adapter_identity"]),
        )
        if dist.is_initialized():
            gathered: list[Any] = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, identity)
            if any(item != identity for item in gathered):
                raise ValueError(f"rank-local vLLM-Omni sync identities differ: {gathered}")
            dist.barrier()
        self.sync_receipt = receipt
        return receipt

    @staticmethod
    def _normalize_outputs(value: Any) -> tuple[list[Any], dict[str, Any]]:
        if isinstance(value, tuple) and len(value) == 2:
            outputs, receipt = value
        elif isinstance(value, Mapping):
            outputs, receipt = value.get("outputs"), value.get("receipt")
        else:
            raise TypeError("adapter.rollout must return (outputs, receipt) or a mapping")
        if not isinstance(receipt, Mapping):
            raise TypeError("vLLM-Omni rollout receipt is not a mapping")
        try:
            import torch

            if torch.is_tensor(outputs):
                if outputs.ndim != 5:
                    raise ValueError(f"batched latent has shape={tuple(outputs.shape)}")
                outputs = list(outputs.split(1, dim=0))
        except ImportError:
            pass
        if not isinstance(outputs, Sequence):
            raise TypeError("vLLM-Omni rollout outputs are not a sequence")
        return list(outputs), dict(receipt)

    def rollout(
        self,
        *,
        entries: Sequence[Any],
        logical_steps: Sequence[int],
        generator_version: int,
        batch_index: int,
    ) -> tuple[list[Any], dict[str, Any]]:
        raw = self.adapter.rollout(
            entries=tuple(entries),
            logical_steps=tuple(int(step) for step in logical_steps),
            generator_version=int(generator_version),
            output_type="latent",
            expected_output_shape=OFFICIAL_X0_SHAPE,
            expected_output_dtype=OFFICIAL_X0_DTYPE,
            batch_index=int(batch_index),
            upstream_revision=self.environment.expected_revision,
        )
        outputs, receipt = self._normalize_outputs(raw)
        validate_rollout_receipt(
            receipt,
            logical_steps=logical_steps,
            generator_version=generator_version,
            expected_revision=self.environment.expected_revision,
        )
        if len(outputs) != len(entries):
            raise ValueError(
                f"vLLM-Omni returned {len(outputs)} rows for {len(entries)} entries"
            )
        return outputs, receipt

    def close(self) -> None:
        close = getattr(self.adapter, "close", None)
        if callable(close):
            close()


class _VLLMTransformerFacade:
    """Adapt a loaded vLLM-Omni Helios transformer to the trace executor."""

    def __init__(self, transformer: Any) -> None:
        self.transformer = transformer

    def enable_adapters(self) -> None:
        return None

    def train(self, mode: bool = True) -> "_VLLMTransformerFacade":
        self.transformer.eval()
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Helios native uses this flag to control its optional KV-cache path.
        # vLLM-Omni 0.19's patched transformer has no cache state and therefore
        # does not expose the keyword; drop only that unsupported, no-op flag
        # rather than silently changing any tensor or schedule input.
        if "is_first_denoising_step" not in self.transformer.forward.__code__.co_varnames:
            kwargs.pop("is_first_denoising_step", None)
        return self.transformer(*args, **kwargs)


class InProcessVLLMOmniHeliosTraceExecutor:
    """Reference exact-trace executor around a loaded vLLM-Omni transformer.

    It intentionally bypasses the VAE and high-level prompt pipeline. The
    immutable DistillSchedule tensors drive the same Helios scheduler while the
    physical DiT calls use vLLM-Omni's transformer implementation.
    """

    def __init__(
        self,
        *,
        transformer: Any,
        device: Any,
        upstream_revision: str,
        sync_callback: Any,
        max_batch_size: int = 5,
        endpoint: str = "inproc://vllm-omni-helios",
        dedicated_pool_id: str = "external",
    ) -> None:
        self.transformer = transformer
        self.facade = _VLLMTransformerFacade(transformer)
        self.device = device
        self.upstream_revision = str(upstream_revision)
        self.sync_callback = sync_callback
        self.max_batch_size = int(max_batch_size)
        self.endpoint = endpoint
        self.dedicated_pool_id = dedicated_pool_id
        self.active_generator_version: int | None = None
        self.active_adapter_identity: str | None = None

    def capabilities(self) -> Mapping[str, Any]:
        sizes = tuple(size for size in (1, 2, 5) if size <= self.max_batch_size)
        return dataclasses.asdict(
            VLLMOmniRolloutCapabilities(
                upstream_revision=self.upstream_revision,
                supported_batch_sizes=sizes,
                transport="in-process-reference-adapter",
            )
        )

    def sync_generator(self, **kwargs: Any) -> Mapping[str, Any]:
        if not callable(self.sync_callback):
            raise RuntimeError("in-process vLLM-Omni executor requires sync_callback")
        receipt = dict(self.sync_callback(**kwargs))
        self.active_generator_version = int(receipt["generator_version"])
        self.active_adapter_identity = str(receipt["adapter_identity"])
        return receipt

    @staticmethod
    def _entry_to_device(entry: Any, device: Any) -> Any:
        tensors = {
            name: value.to(device=device, non_blocking=False)
            for name, value in entry.tensors.items()
        }
        return SimpleNamespace(
            pool_entry=entry.pool_entry,
            operation_trace=entry.operation_trace,
            tensors=tensors,
        )

    def rollout(self, **kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
        if kwargs.get("output_type") != "latent":
            raise ValueError("training rollout adapter only supports output_type=latent")
        version = int(kwargs["generator_version"])
        if version != self.active_generator_version:
            raise ValueError(
                f"vLLM-Omni adapter is G{self.active_generator_version}, requested G{version}"
            )
        entries = tuple(
            self._entry_to_device(entry, self.device) for entry in kwargs["entries"]
        )
        if len(entries) > self.max_batch_size:
            raise ValueError("request exceeds vLLM-Omni trace executor max batch")
        if len(entries) == 1:
            # B1 is a single logical rollout.  The batched helper intentionally
            # requires a synthetic global_step field that is only present for
            # B2/B5 grouped traces; use the authoritative single-entry helper
            # here so the service does not invent metadata or alter the trace.
            from runtime.critic_pool_helios import run_generator_trace_rollout

            output, native_record = run_generator_trace_rollout(
                model=self.facade,
                loaded=entries[0],
                device=self.device,
                detailed_profile=False,
                record_output_sha256=False,
                empty_cache_before_rollout=False,
                collect_diagnostics=False,
            )
            outputs = [output]
        else:
            from runtime.static_fsdp2_p0_helios import run_generator_trace_rollout_batch

            outputs, native_record = run_generator_trace_rollout_batch(
                model=self.facade,
                loaded_entries=entries,
                device=self.device,
                detailed_profile=False,
                record_output_sha256=False,
                collect_diagnostics=False,
                diagnostics_scope=None,
            )
        receipt = {
            "status": "PASS",
            "backend": VLLM_OMNI_BACKEND,
            "engine": "vllm-omni",
            "upstream_revision": self.upstream_revision,
            "endpoint": self.endpoint,
            "dedicated_pool_id": self.dedicated_pool_id,
            "output_type": "latent",
            "vae_decode_calls": 0,
            "exact_operation_trace_replay": True,
            "logical_steps": [int(step) for step in kwargs["logical_steps"]],
            "generator_version": version,
            "adapter_identity": self.active_adapter_identity,
            "physical_batch_size": len(entries),
            "call_count": int(native_record["call_count"]),
            "stage_call_count": native_record["stage_call_count"],
        }
        return list(outputs), receipt

    def close(self) -> None:
        return None


def _validate_output_tensor(value: Any, *, logical_step: int) -> Any:
    import torch

    if not torch.is_tensor(value):
        raise TypeError(f"vLLM-Omni step{logical_step} output is not a tensor")
    if tuple(value.shape) != OFFICIAL_X0_SHAPE:
        raise ValueError(
            f"vLLM-Omni step{logical_step} output shape={tuple(value.shape)}, "
            f"expected={OFFICIAL_X0_SHAPE}"
        )
    if value.dtype != torch.bfloat16:
        raise ValueError(
            f"vLLM-Omni step{logical_step} dtype={value.dtype}, expected=torch.bfloat16"
        )
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError("external rollout output retained an autograd graph")
    if not bool(torch.isfinite(value.float()).all()):
        raise ValueError(f"vLLM-Omni step{logical_step} output is non-finite")
    return value.detach()


def precompute_vllm_omni_critic_rollouts(
    *,
    rank: int,
    steps: tuple[int, ...],
    device: Any,
    generator_engine: Any,
    trace: Any,
    policy: Any,
    collect_diagnostics: bool,
    diagnostics: Any,
    rollout_parity_check: bool,
    variant: str,
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    """Produce Full5 Critic inputs through a dedicated vLLM-Omni service."""

    import torch
    import torch.distributed as dist

    from distillschedule.runtime.fsdp2_p0 import chunked
    from runtime.critic_pool_helios import run_generator_trace_rollout

    del collect_diagnostics
    if policy.rollout_backend != VLLM_OMNI_BACKEND:
        raise ValueError(f"unexpected external rollout backend: {policy.rollout_backend}")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    generator_version = expected_generator_version(variant)
    client = VLLMOmniRolloutClient.from_environment(
        rank=rank,
        world_size=world_size,
        requested_batch_size=int(policy.rollout_batch_size),
    )
    started = time.perf_counter()
    sync_receipt = client.sync_generator(
        generator_engine=generator_engine,
        variant=variant,
        generator_version=generator_version,
    )
    outputs: dict[int, Any] = {}
    records: list[dict[str, Any]] = []
    try:
        for batch_index, step_group in enumerate(
            chunked(steps, int(policy.rollout_batch_size))
        ):
            entries = [
                trace.load_critic_entry(rank=rank, step=step, device="cpu")
                for step in step_group
            ]
            with diagnostics.scope(
                f"VLLM_OMNI_LATENT_ROLLOUT/batch{batch_index}/B{len(entries)}"
            ):
                rows, service_receipt = client.rollout(
                    entries=entries,
                    logical_steps=step_group,
                    generator_version=generator_version,
                    batch_index=batch_index,
                )
            candidates: list[Any] = []
            parity_rows: list[dict[str, Any]] = []
            for step, row in zip(step_group, rows, strict=True):
                candidate = _validate_output_tensor(row, logical_step=int(step))
                if candidate.device != device:
                    logical_bytes = int(candidate.numel() * candidate.element_size())
                    diagnostics.add_bytes(
                        f"vllm_omni_latent/step{step}",
                        logical_bytes,
                        direction="H2D",
                        source="external_rollout_return",
                    )
                    candidate = candidate.to(
                        device=device, dtype=torch.bfloat16, non_blocking=False
                    )
                candidate = _validate_output_tensor(candidate, logical_step=int(step))
                candidates.append(candidate)

                # Every external row, including B1, is compared with native B1.
                if rollout_parity_check:
                    native_entry = trace.load_critic_entry(
                        rank=rank, step=int(step), device=device
                    )
                    reference, _reference_record = run_generator_trace_rollout(
                        model=generator_engine.module,
                        loaded=native_entry,
                        device=device,
                        detailed_profile=False,
                        record_output_sha256=False,
                        empty_cache_before_rollout=False,
                        collect_diagnostics=False,
                    )
                    metrics = tensor_parity_metrics(reference, candidate)
                    parity_rows.append(
                        {
                            "logical_step": int(step),
                            "reference_backend": "native-fsdp2-b1",
                            "candidate_backend": VLLM_OMNI_BACKEND,
                            **metrics,
                        }
                    )
                    if metrics["status"] != "PASS":
                        raise AssertionError(
                            f"vLLM-Omni latent parity failed at step{step}: {metrics}"
                        )
                    del reference, _reference_record

            for step, candidate in zip(step_group, candidates, strict=True):
                outputs[int(step)] = candidate
            record = {
                **service_receipt,
                "batch_index": batch_index,
                "batch_size": len(candidates),
                "call_count": int(service_receipt.get("call_count", 0)),
                "batch_parity": {
                    "enabled": bool(rollout_parity_check),
                    "status": "PASS" if parity_rows else "NOT_RUN",
                    "policy": "external_vllm_omni_vs_native_b1_multi_metric",
                    "samples": parity_rows,
                },
            }
            records.append(record)
            del entries, rows, candidates
    finally:
        client.close()

    if set(outputs) != set(steps):
        raise AssertionError("vLLM-Omni rollout precompute missed logical steps")
    global _LAST_RUN_RECEIPT
    _LAST_RUN_RECEIPT = {
        "status": "PASS",
        "backend": VLLM_OMNI_BACKEND,
        "variant": variant,
        "generator_version": generator_version,
        "elapsed_seconds": time.perf_counter() - started,
        "capabilities": client.capability_receipt,
        "sync": sync_receipt,
        "batches": records,
        "native_reference_ran": bool(rollout_parity_check),
        "training_node_nsys_scope": "client_wait_and_return_transfer_only",
        "remote_service_nsys_required": True,
    }
    return outputs, records


def get_last_vllm_omni_receipt() -> dict[str, Any] | None:
    return None if _LAST_RUN_RECEIPT is None else dict(_LAST_RUN_RECEIPT)


__all__ = [
    "InProcessVLLMOmniHeliosTraceExecutor",
    "RolloutAdapter",
    "VLLMOmniAdapterEnvironment",
    "VLLMOmniRolloutClient",
    "get_last_vllm_omni_receipt",
    "precompute_vllm_omni_critic_rollouts",
]
