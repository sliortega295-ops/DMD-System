#!/usr/bin/env python3
"""Hardened entry point for PR #15 P0 world16 experiments.

This wrapper keeps the P0 implementation readable while enforcing the review
requirements that are awkward to express in the core worker:

* strict multi-metric B1-vs-packed numerical parity before optimizer work;
* one-use GPU retention of precomputed Critic trace entries (no duplicate H2D);
* one cached GAN Stage-3 source per rank;
* correctness-only per-step gradient plus final trainable/optimizer audits;
* profile-only named Score NVTX regions for operation-level NCCL attribution;
* conservative defaults when the low-level worker is invoked directly.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_IMPL_PATH = ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
_SPEC = importlib.util.spec_from_file_location("distillschedule_p0_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load P0 implementation from {_IMPL_PATH}")
impl = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = impl
_SPEC.loader.exec_module(impl)

from distillschedule.runtime.p0_parity import (
    require_recorded_parity,
    tensor_parity_metrics,
)
from distillschedule.profiling.fsdp2_operation_markers import install_fsdp2_operation_markers
import runtime.gan_score_packing as _gan_score_packing
import runtime.gst_pool_helios as _gst_pool_helios
from runtime.static_world16_trace import StaticWorld16Trace as _StaticWorld16Trace


_LAST_TRACE: Any | None = None
_ENGINES: dict[str, Any] = {}
_GRADIENT_AUDITS: dict[str, list[dict[str, Any]]] = {}
_OPERATION_MARKERS: Any | None = None
_BLOCK_MARKER_METADATA: dict[str, Any] = {}


class RetainedP0StaticWorld16Trace:
    """Reuse each GPU Critic entry once between rollout and its Critic update.

    The first load stages an immutable entry to the GPU and retains it. The
    second load for the same `(rank, step, device)` returns that object and
    evicts the cache entry, so the Critic update does not repeat input H2D and
    the entry can be released immediately afterwards. CPU validation loads are
    intentionally uncached.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        global _LAST_TRACE
        self._inner = _StaticWorld16Trace(*args, **kwargs)
        self._critic_gpu_entries: dict[tuple[int, int, str], Any] = {}
        self._source_gpu: dict[tuple[int, str], Any] = {}
        self.critic_entry_h2d_misses = 0
        self.critic_entry_reuse_hits = 0
        self.source_h2d_misses = 0
        self.source_reuse_hits = 0
        _LAST_TRACE = self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @staticmethod
    def _device_key(device: Any) -> str:
        return str(torch.device(device))

    def load_critic_entry(self, *, rank: int, step: int, device: Any) -> Any:
        device_key = self._device_key(device)
        if not device_key.startswith("cuda"):
            return self._inner.load_critic_entry(rank=rank, step=step, device=device)
        key = (int(rank), int(step), device_key)
        if key in self._critic_gpu_entries:
            self.critic_entry_reuse_hits += 1
            return self._critic_gpu_entries.pop(key)
        value = self._inner.load_critic_entry(rank=rank, step=step, device=device)
        self._critic_gpu_entries[key] = value
        self.critic_entry_h2d_misses += 1
        return value

    def source(self, *, rank: int, device: Any) -> Any:
        device_key = self._device_key(device)
        if not device_key.startswith("cuda"):
            return self._inner.source(rank=rank, device=device)
        key = (int(rank), device_key)
        if key in self._source_gpu:
            self.source_reuse_hits += 1
            return self._source_gpu[key]
        value = self._inner.source(rank=rank, device=device)
        self._source_gpu[key] = value
        self.source_h2d_misses += 1
        return value

    def receipt(self) -> dict[str, Any]:
        return {
            "policy": "retain-precomputed-entry-until-one-critic-consume",
            "critic_entry_h2d_misses": self.critic_entry_h2d_misses,
            "critic_entry_reuse_hits": self.critic_entry_reuse_hits,
            "critic_entries_still_cached": len(self._critic_gpu_entries),
            "source_h2d_misses": self.source_h2d_misses,
            "source_reuse_hits": self.source_reuse_hits,
            "source_entries_still_cached": len(self._source_gpu),
        }


def _nvtx_call(label: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    enabled = "--profile-mode" in sys.argv and torch.cuda.is_available()
    if enabled:
        torch.cuda.nvtx.range_push(label)
    try:
        return function(*args, **kwargs)
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def _strict_tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    metrics = tensor_parity_metrics(reference, candidate)
    # The legacy core worker calls this helper before the G backward/update and
    # before any C update. Raising here, rather than only validating the returned
    # receipt, guarantees that max-abs/L2/cosine/finite failures cannot mutate
    # training state even when mean-relative alone would have passed.
    require_recorded_parity(metrics, label="P0 packed tensor")
    return metrics


def _enforce_score_record(record: dict[str, Any]) -> None:
    parity = record.get("score_pack_parity", {})
    if not parity.get("enabled"):
        return
    for key in ("conditional", "unconditional", "dmd_gradient"):
        require_recorded_parity(parity[key], label=f"No-GAN packed {key}")
    parity["status"] = "PASS"
    parity["policy"] = "multi_metric"


def _enforce_rollout_records(records: list[dict[str, Any]]) -> None:
    for batch in records:
        parity = batch.get("batch_parity", {})
        for sample in parity.get("samples", []):
            require_recorded_parity(
                sample,
                label=f"batched rollout step{sample.get('logical_step')}",
            )
        if parity.get("samples"):
            parity["status"] = "PASS"
            parity["policy"] = "multi_metric_b1_integration_reference"
            parity["global_step_semantics"] = (
                "verified_by_each_row_against_original_step_b1_execution"
            )


def _local_tensor(value: Any) -> torch.Tensor:
    value = value.detach()
    if hasattr(value, "to_local"):
        value = value.to_local()
    return value.contiguous()


def _tensor_audit(name: str, value: Any) -> dict[str, Any]:
    local = _local_tensor(value)
    cpu = local.cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    numeric = cpu.float()
    l2 = torch.linalg.vector_norm(numeric.flatten().double())
    return {
        "name": name,
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
        "finite": bool(torch.isfinite(numeric).all()),
        "sum": float(numeric.double().sum()),
        "l2": float(l2),
        "max_abs": float(numeric.abs().max()) if cpu.numel() else 0.0,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _group_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["name"].encode())
        digest.update(row["sha256"].encode())
    return {
        "tensor_count": len(rows),
        "numel": sum(int(row["numel"]) for row in rows),
        "finite": all(bool(row["finite"]) for row in rows),
        "sha256": digest.hexdigest(),
        "tensors": rows,
    }


def _gradient_audit(component: str, engine: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, parameter in engine.module.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            rows.append(_tensor_audit(name, parameter.grad))
    record = {
        "physical_step_before": int(engine.global_steps),
        "gradient": _group_audit(rows),
    }
    if not rows:
        raise AssertionError(f"{component} correctness audit observed no trainable gradients")
    if not record["gradient"]["finite"]:
        raise AssertionError(f"{component} correctness audit observed non-finite gradients")
    return record


def _training_state_audit(component: str, engine: Any) -> dict[str, Any]:
    parameters: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    trainables = {
        name: parameter
        for name, parameter in engine.module.named_parameters()
        if parameter.requires_grad
    }
    for name, parameter in trainables.items():
        parameters.append(_tensor_audit(name, parameter))
        state = engine.optimizer.state.get(parameter, {})
        for state_name in ("exp_avg", "exp_avg_sq"):
            value = state.get(state_name)
            if value is not None and torch.is_tensor(value):
                optimizer_rows.append(_tensor_audit(f"{name}/{state_name}", value))
    return {
        "physical_step": int(engine.global_steps),
        "parameters": _group_audit(parameters),
        "optimizer": _group_audit(optimizer_rows),
        "gradient_steps": list(_GRADIENT_AUDITS.get(component, ())),
    }


def _correctness_audit_enabled() -> bool:
    if "--trajectory-mode" in sys.argv:
        # Trajectory mode has its own bounded scalar/census/state receipts.  Do
        # not produce the legacy full tensor/gradient dumps even for its
        # one-cycle instrumentation canary.
        return False
    if "--trajectory-cycles" in sys.argv:
        index = sys.argv.index("--trajectory-cycles")
        if index + 1 >= len(sys.argv):
            raise ValueError("--trajectory-cycles is missing its value")
        if int(sys.argv[index + 1]) > 1:
            # Full tensor/gradient dumps on all 300 updates would dominate the
            # 50-cycle diagnostic and can consume hundreds of GiB. The
            # trajectory worker records scalar losses every cycle and compact
            # state fingerprints only at registered anchors instead.
            return False
    return not any(
        flag in sys.argv
        for flag in ("--timing-mode", "--profile-mode", "--validate-only")
    )


# Patch only the P0 module's references. PR #13 remains untouched.
impl.StaticWorld16Trace = RetainedP0StaticWorld16Trace
impl._score_pack_error = _strict_tensor_error
impl._rollout_pack_error = _strict_tensor_error

# No-GAN Score calls flow through runtime.gst_pool_helios.score_forward. Label
# both the direct symbol imported by the P0 worker and late imports used by the
# packed Real pair. These NVTX ranges are profile-only annotations and do not
# synchronize or change model state.
_original_score_forward = _gst_pool_helios.score_forward


def _profiled_score_forward(
    model: Any,
    batch: dict[str, torch.Tensor],
    *,
    condition: torch.Tensor,
    adapter_enabled: bool,
) -> torch.Tensor:
    batch_size = int(batch["hidden_states"].shape[0])
    return _nvtx_call(
        f"P0_SCORE/nogan/B{batch_size}/adapter-{int(adapter_enabled)}",
        _original_score_forward,
        model,
        batch,
        condition=condition,
        adapter_enabled=adapter_enabled,
    )


_gst_pool_helios.score_forward = _profiled_score_forward
impl.score_forward = _profiled_score_forward

# GAN DMD fake/real Score calls use a private helper inside the scoped patch.
# Label those calls as well. GAN-G remains part of the outer official boundary
# and is intentionally not mislabelled as an ordinary Score call.
_original_gan_score_call = _gan_score_packing._score_call


def _profiled_gan_score_call(model: Any, **kwargs: Any) -> torch.Tensor:
    batch_size = int(kwargs["hidden_states"].shape[0])
    return _nvtx_call(
        f"P0_SCORE/gan_dmd/B{batch_size}",
        _original_gan_score_call,
        model,
        **kwargs,
    )


_gan_score_packing._score_call = _profiled_gan_score_call

_original_nogan_generator = impl._run_nogan_generator_update_p0


def _hardened_nogan_generator(*args: Any, **kwargs: Any) -> dict[str, Any]:
    record = _original_nogan_generator(*args, **kwargs)
    _enforce_score_record(record)
    return record


impl._run_nogan_generator_update_p0 = _hardened_nogan_generator
_original_precompute = impl._precompute_critic_rollouts


def _hardened_precompute(*args: Any, **kwargs: Any):
    outputs, records, persistence = _original_precompute(*args, **kwargs)
    _enforce_rollout_records(records)
    return outputs, records, persistence


impl._precompute_critic_rollouts = _hardened_precompute
_original_build_engine = impl.build_fsdp2_engine


def _capture_engine(*args: Any, **kwargs: Any):
    global _BLOCK_MARKER_METADATA
    engine, metadata = _original_build_engine(*args, **kwargs)
    component = str(kwargs.get("component", "unknown"))
    _ENGINES[component] = engine
    _GRADIENT_AUDITS.setdefault(component, [])
    original_step = engine.step

    def audited_step() -> Any:
        if _correctness_audit_enabled():
            _GRADIENT_AUDITS[component].append(_gradient_audit(component, engine))
        return original_step()

    engine.step = audited_step
    if _OPERATION_MARKERS is not None and _OPERATION_MARKERS.enabled:
        _BLOCK_MARKER_METADATA[component] = _OPERATION_MARKERS.install_module_markers(
            engine.module, role=component
        )
    return engine, metadata


impl.build_fsdp2_engine = _capture_engine
_original_atomic_json = impl.base._atomic_json


def _audited_atomic_json(path: Path, value: dict[str, Any]) -> None:
    hardening = value.setdefault("p0_hardening", {})
    hardening["tensor_parity_policy"] = "multi_metric_pre_update"
    hardening["global_step_contract"] = (
        "B2/B5 correctness compares every batched row against its original-step B1 execution"
    )
    hardening["score_nvtx_contract"] = (
        "profile-only: No-GAN Score and GAN DMD fake/real Score calls are labelled; GAN-G remains outer official boundary"
    )
    hardening["gradient_audit_scope"] = "correctness_only_before_each_optimizer_step"
    hardening["operation_source_markers"] = (
        _OPERATION_MARKERS.marker_status
        if _OPERATION_MARKERS is not None
        else {"status": "DISABLED"}
    )
    hardening["block_marker_metadata"] = _BLOCK_MARKER_METADATA
    if _LAST_TRACE is not None:
        hardening["input_reuse"] = _LAST_TRACE.receipt()
    if (
        value.get("status") == "PASS"
        and _correctness_audit_enabled()
        and _ENGINES
    ):
        final = value.setdefault("final", {})
        final["training_state_audit"] = {
            component: _training_state_audit(component, engine)
            for component, engine in sorted(_ENGINES.items())
        }
    _original_atomic_json(path, value)


impl.base._atomic_json = _audited_atomic_json


def _install_conservative_direct_defaults() -> None:
    # The central launcher already supplies mesh/batch explicitly. These fallbacks
    # protect direct worker invocation and match the documented experiment order.
    if "--mesh-policy" not in sys.argv:
        sys.argv.extend(("--mesh-policy", "world16-fsdp"))
    if "--rollout-batch-size" not in sys.argv:
        sys.argv.extend(("--rollout-batch-size", "1"))
    if "--pack-real-score-pair" not in sys.argv and "--no-pack-real-score-pair" not in sys.argv:
        sys.argv.append("--no-pack-real-score-pair")
    if "--all-role-lora-scale1-add" not in sys.argv and "--no-all-role-lora-scale1-add" not in sys.argv:
        sys.argv.append("--no-all-role-lora-scale1-add")
    if "--optimizer-dispatch-warmup" not in sys.argv and "--no-optimizer-dispatch-warmup" not in sys.argv:
        sys.argv.append("--no-optimizer-dispatch-warmup")
    if "--generator-nograd-residual-fusion" not in sys.argv and "--no-generator-nograd-residual-fusion" not in sys.argv:
        sys.argv.append("--no-generator-nograd-residual-fusion")
    if "--exact-grad-residual-fusion" not in sys.argv and "--no-exact-grad-residual-fusion" not in sys.argv:
        sys.argv.append("--no-exact-grad-residual-fusion")
    if "--exact-modulated-norm-epilogue" not in sys.argv and "--no-exact-modulated-norm-epilogue" not in sys.argv:
        sys.argv.append("--no-exact-modulated-norm-epilogue")
    if "--exact-modulated-norm-materialization" not in sys.argv and "--no-exact-modulated-norm-materialization" not in sys.argv:
        sys.argv.append("--no-exact-modulated-norm-materialization")
    if "--exact-modulated-norm-backward" not in sys.argv and "--no-exact-modulated-norm-backward" not in sys.argv:
        sys.argv.append("--no-exact-modulated-norm-backward")
    if "--exact-modulated-norm-layernorm" not in sys.argv and "--no-exact-modulated-norm-layernorm" not in sys.argv:
        sys.argv.append("--no-exact-modulated-norm-layernorm")
    if "--exact-modulated-norm-fused-backward" not in sys.argv and "--no-exact-modulated-norm-fused-backward" not in sys.argv:
        sys.argv.append("--no-exact-modulated-norm-fused-backward")
    if "--exact-history-current-fusion" not in sys.argv and "--no-exact-history-current-fusion" not in sys.argv:
        sys.argv.append("--no-exact-history-current-fusion")
    if "--generator-nograd-cross-kv-cache" not in sys.argv and "--no-generator-nograd-cross-kv-cache" not in sys.argv:
        sys.argv.append("--no-generator-nograd-cross-kv-cache")
    if "--generator-nograd-static-context-cache" not in sys.argv and "--no-generator-nograd-static-context-cache" not in sys.argv:
        sys.argv.append("--no-generator-nograd-static-context-cache")
    if "--generator-nograd-persistent-unshard" not in sys.argv and "--no-generator-nograd-persistent-unshard" not in sys.argv:
        sys.argv.append("--no-generator-nograd-persistent-unshard")
    if "--elide-explicit-rollout-sync" not in sys.argv and "--no-elide-explicit-rollout-sync" not in sys.argv:
        sys.argv.append("--no-elide-explicit-rollout-sync")
    if "--elide-critic-scalar-logs" not in sys.argv and "--no-elide-critic-scalar-logs" not in sys.argv:
        sys.argv.append("--no-elide-critic-scalar-logs")


def main() -> int:
    global _OPERATION_MARKERS
    _install_conservative_direct_defaults()
    enabled = "--profile-mode" in sys.argv
    with install_fsdp2_operation_markers(enabled=enabled) as session:
        _OPERATION_MARKERS = session
        return int(impl.main())


if __name__ == "__main__":
    raise SystemExit(main())
