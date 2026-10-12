"""Parameterized ORIGINAL Wan/Helios-pyramid protocol and existing strategies.

No replacement sampler, loss, reduced update count or synthetic performance.
The physical backend is still world16/B1; new tensor shapes require matching
native trace payloads. Geometry acceptance is not a GPU validation receipt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from distillschedule.search.contracts import require
from distillschedule.runtime.parameterized_cycle import build_parameterized_cycle_graph, parameterized_window_order
from distillschedule.search.wan_implementation import WanImplementation, PRECISION
from distillschedule.workloads.best_known.registry import fingerprint

SCHEMA = "distillschedule.original_wan_task.v1"
PROTOCOL = "wan_native_nogan_post_g_independent_rollouts"
TRACE_KEY = "parameterized_wan_task"


@dataclass(frozen=True)
class OriginalWanTask:
    schema_version: str = SCHEMA
    protocol: str = PROTOCOL
    resolution: tuple[int, int] = (384, 640)
    video_frames: int = 33
    fake_updates: int = 5
    ema_after: tuple[int, ...] | None = None
    denoising_step_list: tuple[int, ...] = (1000, 750, 500, 250)
    stage2_steps: tuple[int, int, int] = (2, 2, 2)
    lora_rank: int = 256
    world_size: int = 16
    physical_batch: int = 1

    def __post_init__(self):
        require(self.schema_version == SCHEMA and self.protocol == PROTOCOL, "unsupported original Wan protocol")
        require(len(self.resolution) == 2 and all(type(x) is int and x >= 64 and x % 64 == 0
                for x in self.resolution), "resolution must fit VAE/pyramid/patch lattice (multiples of 64)")
        require(type(self.video_frames) is int and self.video_frames > 0 and (self.video_frames-1) % 4 == 0,
                "video frame count must be 4k+1; no rounding")
        require(type(self.fake_updates) is int and 1 <= self.fake_updates <= 64, "Fake count must be 1..64")
        if self.ema_after is None:
            object.__setattr__(self, "ema_after", tuple(range(self.fake_updates)))
        require(tuple(sorted(set(self.ema_after))) == self.ema_after and
                all(type(x) is int and 0 <= x < self.fake_updates for x in self.ema_after), "invalid EMA positions")
        require(self.denoising_step_list == (1000, 750, 500, 250),
                "stage2 uses stage2_steps; do not advertise the overwritten stage1 timestep list as a step control")
        require(len(self.stage2_steps) == 3 and all(type(x) is int and 1 <= x <= 128 for x in self.stage2_steps),
                "exactly three positive pyramid-stage step budgets required")
        require(type(self.lora_rank) is int and self.lora_rank == 256,
                "original bootstrap/model binder currently supports LoRA rank256 only")
        require(type(self.world_size) is int and self.world_size == 16 and
                type(self.physical_batch) is int and self.physical_batch == 1,
                "existing original Wan engines require world16/B1")

    @property
    def source_steps(self):
        return tuple(range(10, 10+self.fake_updates))

    @property
    def latent_shape(self):
        h,w = self.resolution
        return (1,16,(self.video_frames-1)//4+1,h//8,w//8)

    @property
    def stage_shapes(self):
        h,w = self.latent_shape[-2:]
        return ((h//4,w//4),(h//2,w//2),(h,w))

    def to_dict(self):
        import json
        return json.loads(json.dumps(asdict(self)))

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        require(isinstance(value, Mapping) and set(value) <= {f.name for f in fields(cls)},
                "unknown original Wan task fields; GAN/history/precision are not silently ignored")
        value = dict(value)
        for key in ("resolution","ema_after","denoising_step_list","stage2_steps"):
            if key in value and value[key] is not None:
                require(isinstance(value[key],(list,tuple)), f"{key} must be a sequence")
                value[key] = tuple(value[key])
        return cls(**value)

    def graph(self, versions: Mapping[str,int]):
        return build_parameterized_cycle_graph(source_steps=self.source_steps,
                    initial_versions=versions,ema_after=self.ema_after)

    def bridge_contract(self):
        return {"schema_version":"distillschedule.wan_prepared_trace_contract.v1", "protocol":self.protocol,
                "source_steps":list(self.source_steps),"generator_source_step":10,
                "ema_after":list(self.ema_after),"physical_batch":1,"world_size":16,
                "input_binding":"validated_operation_qualified_trace"}

    def trace_contract(self):
        return {"resolution":list(self.resolution),"stage_spatial_shapes":[list(x) for x in self.stage_shapes],
                "full_latent_shape":list(self.latent_shape),"dtype":"bfloat16",
                "dfake_gen_update_ratio":self.fake_updates,TRACE_KEY:self.to_dict()}


@dataclass(frozen=True)
class OriginalWanStrategy:
    placement: str = "fsdp"
    chunk_tokens: int = 512
    lora_epilogue: bool = False
    rollout_window: int = 1

    def __post_init__(self):
        self.implementation  # Reuse, not relax, the existing audited capability predicate.
        require(type(self.rollout_window) is int and 1 <= self.rollout_window <= 64, "invalid window")

    @property
    def implementation(self):
        return WanImplementation(self.placement,self.chunk_tokens,self.lora_epilogue)

    def validate(self, task):
        require(self.rollout_window <= task.fake_updates, "window exceeds this task's rollout count")
        return self

    def to_dict(self):
        return asdict(self)

    def order(self, task, versions):
        self.validate(task)
        return parameterized_window_order(task.graph(versions),max_live_rollouts=self.rollout_window)

    def environment(self, task):
        self.validate(task)
        impl = self.implementation
        return {**impl.environment(),"DISTILLSCHEDULE_SEARCH_CANDIDATE_ID":impl.id,
                "DISTILLSCHEDULE_ORIGINAL_WAN_TASK_SHA256":task.digest,
                "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE":str(int(self.lora_epilogue)),
                "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE":"0",
                "DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION":"reference",
                "DISTILLSCHEDULE_WAN_ROLLOUT_WINDOW":"",
                "NCCL_ALGO":"Tree" if self.placement == "resident" else "",
                "NVIDIA_TF32_OVERRIDE":""}


def candidates(task):
    """No winner lookup or estimated speedup. All variants reach old native code."""
    return tuple(OriginalWanStrategy(p,c,e,w) for p,c,e in
            [("fsdp",512,False),*(("resident",c,e) for c in (512,1024,2048) for e in (False,True))]
            for w in range(1,task.fake_updates+1))


def capabilities(task):
    return {"task":task.to_dict(),"task_sha256":task.digest,"latent_shape":list(task.latent_shape),
            "strategies":[x.to_dict() for x in candidates(task)],
            "native_engines":["FSDP2Engine","GradReadyAsyncResidentEngine"],
            "shared_score_backbone":"ORIGINAL_REAL_FAKE_ADAPTER_SWITCH_PRESERVED",
            "gradient_reduction":"FP32_SUM_POST_DIVIDE_16; resident async128MiB",
            "protocol_substitution":False,"gpu_execution":"NOT_RUN",
            "status":"SUPPORTED_CODE_PATHS_REQUIRE_MATCHED_TRACE_CALIBRATION_AND_NATIVE_VALIDATION",
            "limitations":["world16/B1/rank256", "Wan1.3B original three-stage no-GAN protocol only",
                           "no automatic construction of new data/trace payloads", "no arbitrary pool or backbone-sharing search"]}


def rollout_parameters(operation_trace):
    """Read native invocation constants from the already-bound trace, not global state.

    Legacy calls have exactly their former arguments. This function is shared by
    the original grad and no-grad rollout helpers; it introduces no sampler math.
    """
    contract = operation_trace.manifest.get("contract",{})
    task = OriginalWanTask.from_dict(contract[TRACE_KEY]) if TRACE_KEY in contract else OriginalWanTask()
    if TRACE_KEY in contract:
        require(all(contract.get(k) == v for k,v in task.trace_contract().items()),
                "native rollout trace/task constants disagree")
    return {"base_height":task.latent_shape[-2],"base_width":task.latent_shape[-1],
            "latent_shape":task.latent_shape,"num_critic_input_frames":task.latent_shape[2],
            "stage2_num_inference_steps_list":list(task.stage2_steps),
            "denoising_step_list":list(task.denoising_step_list),"identity":task.digest}
