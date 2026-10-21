"""Explicit, bounded tasks for the ORIGINAL Helios 14B helpers.

This is a workload request, not a scheduling decision or a Wan model alias.
Legacy generated recipes have no request and retain their exact old contracts.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping

from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import require

SCHEMA = "distillschedule.original_helios_request.v1"
TRACE_KEY = "distillschedule_original_helios_request"
TRACE_HASH_KEY = "distillschedule_original_helios_request_sha256"
REQUEST_ENV = "DISTILLSCHEDULE_HELIOS_REQUEST"
REQUEST_HASH_ENV = "DISTILLSCHEDULE_HELIOS_REQUEST_SHA256"


@dataclass(frozen=True)
class OriginalHeliosRequest:
    schema_version: str = SCHEMA
    model_family: str = "helios14b"
    variant: str = "no-gan"
    video_height: int = 384
    video_width: int = 640
    video_frames: int = 33
    fake_updates: int = 5
    stage2_inference_steps: tuple[int, ...] = (2, 2, 2)
    denoising_step_list: tuple[int, ...] = (1000, 750, 500, 250)
    world_size: int = 16
    physical_batch: int = 1
    global_batch: int = 16
    stage_schedule_policy: str = "source_normal_only"

    def __post_init__(self):
        require(self.schema_version == SCHEMA and self.model_family == "helios14b",
                "unsupported ORIGINAL Helios model/protocol")
        require(self.variant in ("no-gan", "gan"), "unsupported Helios variant")
        for name in ("video_height", "video_width", "video_frames", "fake_updates",
                     "world_size", "physical_batch", "global_batch"):
            require(type(getattr(self, name)) is int, f"Helios {name} must be an integer")
        require((self.video_height, self.video_width) == (384, 640) and self.video_frames in (17, 33),
                "Helios request supports 384x640 and frames17/33 only")
        require(self.fake_updates == 5, "Helios currently preserves G1/C5; n1 is not implemented")
        require((self.world_size, self.physical_batch, self.global_batch) == (16, 1, 16),
                "Helios currently supports world16/B1/global16 only; physical8 is not implemented")
        require(isinstance(self.stage2_inference_steps, tuple)
                and all(type(n) is int for n in self.stage2_inference_steps)
                and self.stage2_inference_steps in ((1, 1, 1), (2, 2, 2)),
                "Helios active pyramid horizons must be (1,1,1) or (2,2,2)")
        require(self.denoising_step_list == (1000, 750, 500, 250),
                "inactive ordinary denoising list is not a Helios request dimension")
        require(self.stage_schedule_policy in ("source_normal_only", "benchmark_v4_strict_global"),
                "unsupported Helios stage-schedule scope")

    @property
    def source_steps(self):
        return tuple(range(1000, 1005)) if self.variant == "gan" else tuple(range(10, 15))

    @property
    def ema_after(self):
        return (0,) if self.variant == "gan" else (0, 1, 2, 3, 4)

    @property
    def latent_shape(self):
        return (1, 16, (self.video_frames-1)//4+1, self.video_height//8, self.video_width//8)

    @property
    def gan_d_shape(self):
        return (3, *self.latent_shape[1:3], self.latent_shape[3]//2, self.latent_shape[4]//2)

    @property
    def stage_spatial_shapes(self):
        h, w = self.latent_shape[-2:]
        return ((h//4, w//4), (h//2, w//2), (h, w))

    @property
    def id(self):
        scope = "-b4" if self.stage_schedule_policy == "benchmark_v4_strict_global" else ""
        return f"f{self.video_frames}-s{''.join(map(str, self.stage2_inference_steps))}{scope}"

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        require(isinstance(value, Mapping) and set(value) <= {f.name for f in fields(cls)},
                "unknown Helios request fields are not ignored")
        data = dict(value)
        for key in ("denoising_step_list", "stage2_inference_steps"):
            if key in data:
                require(isinstance(data[key], (list, tuple)), f"Helios {key} must be a sequence")
                data[key] = tuple(data[key])
        return cls(**data)

    def trace_contract(self):
        return {TRACE_KEY: self.to_dict(), TRACE_HASH_KEY: self.digest,
                "resolution": [self.video_height, self.video_width],
                "stage_spatial_shapes": [list(s) for s in self.stage_spatial_shapes],
                "full_latent_shape": list(self.latent_shape), "dtype": "bfloat16",
                "dfake_gen_update_ratio": 5}

    def validate_trace_metadata(self, trace):
        contract = trace.manifest.get("contract", {})
        require(contract.get(TRACE_KEY) == self.to_dict() and contract.get(TRACE_HASH_KEY) == self.digest,
                "Helios trace request/hash mismatch; capture fresh inputs, do not relabel")
        for key, value in self.trace_contract().items():
            require(contract.get(key) == value, f"Helios trace {key} shape/contract mismatch")

    def validate_rank_controls(self, controls):
        """Exact metadata gate before models; the existing rank-call validator stays active."""
        from runtime.operation_trace import validate_official_rank_stage_schedule_semantics
        from runtime.static_world16_trace import _generator_keys, _critic_keys
        require(set(controls) == set(range(16)), "Helios request requires all world16 rank controls")
        for trace in controls.values():
            self.validate_trace_metadata(trace)
            keys = [o["key"] for o in trace.manifest.get("operations", [])]
            require(len(keys) == len(set(keys)), "duplicate Helios trace operation")
            require({k for k in keys if k.startswith("step/") and k.endswith("/branch")} ==
                    {f"step/{s}/branch" for s in self.source_steps}, "Helios source cohort/frequency changed")
            for step in self.source_steps:
                branch = trace.operation(f"step/{step}/branch")["control"]
                expected = dict(global_step=step, dfake_gen_update_ratio=5,
                    train_generator=step == self.source_steps[0], train_critic=True,
                    gan=self.variant == "gan", reward=False, vae_decode=False)
                require(all(branch.get(k) == v for k, v in expected.items()), "Helios native branch changed")
                phases = [("critic", _critic_keys(step, gan=self.variant == "gan"))]
                trace.operation(f"step/{step}/critic/train")
                if self.variant == "gan": trace.operation(f"step/{step}/critic/gan_d")
                if step == self.source_steps[0]:
                    phases.append(("generator", _generator_keys(step, gan=self.variant == "gan")))
                    trace.operation(f"step/{step}/generator/dmd")
                    if self.variant == "gan": trace.operation(f"step/{step}/generator/gan_g")
                for phase, tensor_keys in phases:
                    descriptors = trace.manifest.get("tensors", {})
                    require(set(tensor_keys.values()) <= set(descriptors), "Helios trace missing operation tensors")
                    for role in ("entry_noise", "rollout_noise", "critic_noise" if phase == "critic" else "dmd_noise"):
                        desc = descriptors[tensor_keys[role]]
                        require(desc.get("shape") == list(self.latent_shape) and desc.get("dtype") == "bfloat16",
                                "Helios trace latent tensor shape/dtype mismatch")
                    control = trace.operation(f"step/{step}/{phase}/rollout")["control"]
                    schedule = control.get("pyramid_stage_call_sequence", [])
                    require(len(schedule) == 3 and [s.get("stage") for s in schedule] == [0, 1, 2],
                            "Helios requires three ordered pyramid stages")
                    benchmark = self.stage_schedule_policy == "benchmark_v4_strict_global"
                    if benchmark:
                        require(control.get("stage_schedule_policy") == self.stage_schedule_policy
                                and type(control.get("all_history_zero")) is bool,
                                "Helios benchmark scope/history predicate missing")
                    else:
                        require(not control.get("all_history_zero", False),
                                "normal Helios request forbids amplified all-zero-history scope")
                    for stage, item in enumerate(schedule):
                        budget = self.stage2_inference_steps[stage] * (2 if benchmark and control["all_history_zero"] else 1)
                        require(item.get("latent_shape") == [*self.latent_shape[:3], *self.stage_spatial_shapes[stage]],
                                "Helios stage geometry mismatch")
                        count = item.get("call_count")
                        require(type(count) is int and 1 <= count <= self.stage2_inference_steps[stage],
                                "Helios call count exceeds synchronized request horizon")
                        require(item.get("denoising_step_budget") == budget,
                                "Helios per-rank native timestep budget differs")
                        grid = item.get("timestep_grid")
                        require(isinstance(grid, list) and len(grid) == budget
                                and all(type(v) in (float, int) and math.isfinite(v) for v in grid)
                                and item.get("timestep") == grid[:count],
                                "Helios executed timesteps differ from declared native grid/budget")
        result = validate_official_rank_stage_schedule_semantics(controls)
        if self.stage_schedule_policy == "benchmark_v4_strict_global":
            result = {**result, "semantics": "helios_benchmark_v4_strict_global_native_rank_grids",
                "stage_schedule_policy": self.stage_schedule_policy,
                "upstream_local_amplified_exit_distribution": "NOT_CLAIMED",
                "native_per_rank_timestep_budget": "VALIDATED_REQUEST_QUALIFIED"}
        return result

    def bind_official_config(self, config):
        require(self.variant == "gan", "official GAN config requires a GAN request")
        bound = copy.deepcopy(config)
        bound.training_config.num_critic_input_frames = self.latent_shape[2]
        bound.validation_config.stage2_simulated_inference_steps = list(self.stage2_inference_steps)
        self.validate_official_config(bound)
        return bound

    def validate_official_config(self, config):
        training, validation = config.training_config, config.validation_config
        require(training.num_critic_input_frames == self.latent_shape[2]
                and training.is_enable_stage2 is True and training.stage2_num_stages == 3
                and tuple(validation.stage2_simulated_inference_steps) == self.stage2_inference_steps,
                "Helios effective official GAN geometry/horizon differs from request")

    def validate_loaded_history(self, trace):
        """Check the native sum predicate against hash-verified CPU payloads."""
        self.validate_trace_metadata(trace)
        for operation in trace.manifest.get("operations", []):
            if not operation["key"].endswith("/rollout"):
                continue
            prefix = operation["key"].removesuffix("/rollout")
            zero = all(bool(trace.tensor(prefix+"/"+role).sum() == 0)
                       for role in ("history_short", "history_mid", "history_long"))
            control = operation["control"]
            if self.stage_schedule_policy == "source_normal_only":
                require(not zero, "normal Helios trace contains amplified all-zero history")
            else:
                require(control.get("all_history_zero") is zero,
                        "Helios native history predicate differs from hash-verified payload")

    def validate_native_stage_grids(self, trace):
        """Worker-only, model-free recomputation with the actual original source.

        Metadata planning needs no native package imports. Before constructing
        models, the worker proves each rank's full grid (including the native
        doubled zero-history grid) rather than trusting a recorded budget.
        """
        import importlib
        import torch
        from distillschedule.workloads.best_known.registry import repository_root, file_sha256
        self.validate_trace_metadata(trace)
        root = repository_root()/"third_party/helios_compat/source"
        scheduler_module = importlib.import_module("helios.scheduler.scheduling_helios")
        base = importlib.import_module("helios.utils.utils_base")
        expected_sources = ((scheduler_module, root/"helios/scheduler/scheduling_helios.py"),
                            (base, root/"helios/utils/utils_base.py"))
        for module, path in expected_sources:
            require(Path(module.__file__).resolve() == path.resolve(),
                    "Helios native grid import resolved outside pinned original source")
        operations = []
        for operation in trace.manifest.get("operations", []):
            control = operation.get("control", {})
            if "pyramid_stage_call_sequence" not in control:
                continue
            zero = (self.stage_schedule_policy == "benchmark_v4_strict_global"
                    and control.get("all_history_zero") is True)
            scheduler = scheduler_module.HeliosScheduler(shift=1., stages=3,
                stage_range=[0, 1/3, 2/3, 1], gamma=1/3)
            for index, item in enumerate(control["pyramid_stage_call_sequence"]):
                budget = self.stage2_inference_steps[index] * (2 if zero else 1)
                scheduler.set_timesteps(budget+1, index, device="cpu")
                sigmas = torch.cat([scheduler.sigmas[:-2], scheduler.sigmas[-1:]])
                shifted, _ = base.apply_schedule_shift(sigmas,
                    torch.empty((*self.latent_shape[:3], *self.stage_spatial_shapes[index]), device="meta"),
                    sigmas_two=scheduler.sigmas_per_stage[index], base_seq_len=256, max_seq_len=4096,
                    base_shift=.5, max_shift=1.15, time_shift_type="linear")
                times = scheduler.timesteps_per_stage[index]
                grid = times.min()+shifted[:-1]*(times.max()-times.min())
                recorded = item.get("timestep_grid", [])
                require(len(recorded) == budget and torch.allclose(grid.double(),
                    torch.tensor(recorded, dtype=torch.float64), rtol=1e-6, atol=1e-4),
                    "Helios recorded stage grid differs from original native scheduler grid")
                require(item.get("denoising_step_budget") == budget
                        and item.get("timestep") == recorded[:item["call_count"]],
                        "Helios native timestep prefix/budget differs")
            operations.append(operation["key"])
        require(len(operations) == 6, "Helios native grid preflight needs G1/C5 rollout operations")
        return dict(status="PASS", request_sha256=self.digest, operation_count=len(operations),
                    native_sources={str(path.relative_to(repository_root())): file_sha256(path)
                                    for _, path in expected_sources},
                    scope="CPU_ORIGINAL_NATIVE_GRID_RECOMPUTATION_NO_MODEL_OR_GPU_EXECUTION")


def trace_request(operation_trace):
    """A hash-qualified Helios identity, never a fallback from malformed metadata."""
    contract = operation_trace.manifest.get("contract", {})
    if TRACE_KEY not in contract and TRACE_HASH_KEY not in contract:
        return None
    request = OriginalHeliosRequest.from_dict(contract.get(TRACE_KEY, {}))
    request.validate_trace_metadata(operation_trace)
    return request
