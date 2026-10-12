"""Opt-in merged-weight noise study; evidence work stays outside cycle timers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ENV = "DISTILLSCHEDULE_WAN13B_SHADOW_TRAJECTORY_STUDY"
ROUNDING_ENV = "DISTILLSCHEDULE_WAN13B_SHADOW_STOCHASTIC_ROUNDING"


def flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def study_requested() -> bool:
    return flag(ENV)


def anchors(cycles: int) -> tuple[int, ...]:
    if cycles in {20, 30}:
        return tuple(c for c in (1, 5, 10, 20, 30) if c <= cycles)
    return tuple(sorted({c for c in (1, 10, 25, 50, cycles) if c <= cycles}))


def rounding_seed(*, seed: int, rank: int, version: int, layer: str) -> int:
    text = f"wan13b-shadow-sr-v1/{seed}/{rank}/{version}/{layer}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**63 - 1)


def stochastic_bf16(value: Any, *, generator: Any) -> Any:
    import torch

    if value.dtype != torch.float32:
        raise ValueError("stochastic BF16 conversion requires FP32 ideal weights")
    nearest_bf16 = value.to(torch.bfloat16)
    nearest = nearest_bf16.float()
    lo_neighbor = torch.nextafter(nearest_bf16, torch.full_like(nearest_bf16, -torch.inf)).float()
    hi_neighbor = torch.nextafter(nearest_bf16, torch.full_like(nearest_bf16, torch.inf)).float()
    lower = torch.where(nearest > value, lo_neighbor, nearest)
    upper = torch.where(nearest < value, hi_neighbor, nearest)
    span = upper - lower
    probability = torch.where(span > 0, (value - lower) / span, torch.zeros_like(span))
    uniform = torch.rand(value.shape, dtype=torch.float32, device=value.device, generator=generator)
    return torch.where(uniform < probability, upper, lower).to(torch.bfloat16)


def rounding_moments(base: Any, delta: Any, merged: Any) -> Any:
    """Return summable FP64 sufficient statistics; no host reads here."""
    import torch

    ideal = base.float() + delta
    error = merged.float() - ideal
    e, d = error.double(), delta.double()
    if base.dtype == torch.bfloat16:
        direction = torch.where(delta >= 0, torch.full_like(base, torch.inf), torch.full_like(base, -torch.inf))
        ulp = (torch.nextafter(base, direction).float() - base.float()).abs()
        small = (delta.abs() < 0.5 * ulp).double().sum()
        bf16_count = delta.numel()
    else:
        small = torch.zeros((), device=delta.device, dtype=torch.float64)
        bf16_count = 0
    return torch.stack((e.square().sum(), d.square().sum(), (e * -d).sum(),
                        small, torch.tensor(bf16_count, device=delta.device, dtype=torch.float64)))


def metrics(values: list[float]) -> dict[str, float | None]:
    e2, d2, dot, small, count = values
    return {"error_l2_squared": e2, "delta_l2_squared": d2,
            "lost_ratio": (e2 / d2) ** 0.5 if d2 > 0 else None,
            "bias_cos": dot / (e2 * d2) ** 0.5 if e2 > 0 and d2 > 0 else (0.0 if d2 > 0 else None),
            "sub_half_bf16_ulp_fraction": small / count if count else None,
            "bf16_element_count": int(count)}


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), path
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.rename(path)


class ShadowTrajectoryStudy:
    def __init__(self, *, rank: int, output: Path, generator_engine: Any, critic_engine: Any) -> None:
        self.rank, self.root = rank, output / "shadow_study"
        self.generator, self.critic = generator_engine, critic_engine
        self.receipt = {"enabled": True, "rank0_evidence_only": True,
                        "probe_model_call_count": 0, "completed_cycles": 0,
                        "rounding": "STOCHASTIC_BF16" if flag(ROUNDING_ENV) else "ROUND_TO_NEAREST",
                        "single_row_shadow_parity_gate": "NOT_APPLIED_TO_THIS_AUTHORIZED_STUDY",
                        "timing": "DIAGNOSTIC_ONLY_ALL_PROBES_SNAPSHOTS_AUDITS_OUTSIDE_TRAINING_WINDOW"}
        if rank == 0:
            self._snapshot(0)

    def _snapshot(self, cycle: int) -> None:
        import torch
        from safetensors.torch import save_file

        metadata = {}
        for role, engine in (("generator", self.generator), ("critic", self.critic)):
            values = {name: p.detach().cpu().contiguous() for name, p in engine.module.named_parameters() if p.requires_grad}
            assert values and all(t.dtype == torch.float32 for t in values.values())
            path = self.root / "snapshots" / f"cycle_{cycle:02d}_{role}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            assert not path.exists(), path
            save_file(values, str(path), metadata={"cycle": str(cycle), "role": role})
            metadata[role] = {"path": str(path), "bytes": path.stat().st_size,
                              "numel": sum(t.numel() for t in values.values()), "tensor_count": len(values)}
            del values
        _atomic(self.root / "snapshots" / f"cycle_{cycle:02d}.json", metadata)

    def after_cycle(self, *, cycle: int, trace: Any, device: Any, p0: Any,
                    clean_runtime: bool, runtime_cache: Any) -> dict[str, Any]:
        import torch
        import torch.distributed as dist
        import time
        from safetensors.torch import save_file

        dist.barrier()
        started = time.perf_counter()
        if self.rank == 0:
            before_cpu = torch.get_rng_state().clone()
            before_cuda = torch.cuda.get_rng_state(device).clone()
            shadow = getattr(self.generator.module, "_distillschedule_merged_rollout_shadow", None)
            with torch.no_grad():
                if shadow is not None:
                    assert shadow.current_source_version == int(self.generator.global_steps)
                    layers = []
                    layer_values = []
                    total = torch.zeros(5, dtype=torch.float64, device=device)
                    for name, source, destination, adapter in shadow.layers:
                        delta = source.get_delta_weight(adapter)
                        moments = rounding_moments(source.base_layer.weight, delta, destination.weight)
                        total += moments
                        layers.append({"layer": name, "base_dtype": str(source.base_layer.weight.dtype),
                                       "statistics_index": len(layer_values)})
                        layer_values.append(moments)
                        del delta, moments
                    host_values = torch.stack(layer_values).cpu().tolist()
                    for layer, values in zip(layers, host_values, strict=True):
                        layer.pop("statistics_index")
                        layer.update(metrics(values))
                    del layer_values
                    _atomic(self.root / "rounding" / f"cycle_{cycle:02d}.json",
                            {"cycle": cycle, "generator_version": int(self.generator.global_steps),
                             "rounding": self.receipt["rounding"], "global": metrics(total.cpu().tolist()),
                             "layers": layers, "training_rng_used": False, "outside_cycle_timer": True})
                # Fixed immutable B1 row, original unmerged G path, on every arm.
                entry = trace.load_critic_entry(rank=0, step=10, device=device)
                outputs, inventory = p0.run_generator_trace_rollout_batch(
                    model=self.generator.module, loaded_entries=[entry], device=device,
                    collect_diagnostics=False, clean_runtime=clean_runtime, runtime_cache=runtime_cache)
                assert inventory["call_count"] == 6
                self.receipt["probe_model_call_count"] += inventory["call_count"]
                path = self.root / "probes" / f"cycle_{cycle:02d}.safetensors"
                path.parent.mkdir(parents=True, exist_ok=True)
                assert not path.exists(), path
                save_file({"latent": outputs[0].detach().cpu().contiguous()}, str(path))
                _atomic(self.root / "probes" / f"cycle_{cycle:02d}.json",
                        {"cycle": cycle, "generator_version": int(self.generator.global_steps),
                         "trace_identity": inventory.get("member_trace_identities"),
                         "call_count": 6, "execution": "ORIGINAL_UNMERGED_G", "outside_cycle_timer": True})
                del entry, outputs, inventory
            assert torch.equal(before_cpu, torch.get_rng_state()), "study consumed CPU training RNG"
            assert torch.equal(before_cuda, torch.cuda.get_rng_state(device)), "study consumed CUDA training RNG"
            if cycle in {5, 10, 20, 30}:
                self._snapshot(cycle)
            print(f"SHADOW_STUDY_CYCLE {cycle} G{self.generator.global_steps} evidence_wall={time.perf_counter()-started:.3f}", flush=True)
        self.receipt["completed_cycles"] = cycle
        dist.barrier()
        return self.receipt


def accepted_timing_cycles(comparison: dict[str, Any], *, expected_head: str,
                           arm_id: str) -> int:
    """Legacy C2 stays unchanged; only a hashed study PASS permits a longer source receipt."""
    gate = comparison.get("shadow_noise_acceptance")
    if gate is None:
        return 2
    if not study_requested():
        raise ValueError("shadow noise timing requires explicit study opt-in")
    if gate.get("schema_version") != "wan13b.shadow_noise_acceptance.v1" or gate.get("status") != "PASS":
        raise ValueError("merged shadow trajectory noise gate is not PASS")
    if gate.get("expected_head") != expected_head or gate.get("cycles") not in {20, 30}:
        raise ValueError("shadow noise acceptance source/cycles changed")
    if arm_id not in gate.get("accepted_arm_ids", []):
        raise ValueError("timing arm did not pass the mechanism noise gate")
    manifest_path = Path(gate["preregistration_path"])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != gate.get("preregistration_sha256"):
        raise ValueError("noise study preregistration hash changed")
    return int(gate["cycles"])


def noise_ratio(candidate: float | None, repeat: float | None) -> float | None:
    if candidate is None or repeat is None:
        return None
    if repeat == 0:
        return 0.0 if candidate == 0 else None
    return candidate / repeat


def within_noise(candidate: float | None, repeat: float | None, multiplier: float = 2.0) -> bool:
    return candidate is not None and repeat is not None and candidate <= multiplier * repeat


def sustained_growth(values: list[float]) -> bool:
    """Preregistered anchor criterion: all three transitions strictly grow, final >1.25*first."""
    return (len(values) >= 4 and all(b > a for a, b in zip(values, values[1:]))
            and values[-1] > 1.25 * values[0])
