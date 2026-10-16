"""Only the bounded ORIGINAL LiveWan 81/69-pixel-frame requests.

No optimizer, batch, world-prefix, sampler or training-step policy is tunable.
The historical default keeps its byte-identical canonical profile and IDs.
"""
from dataclasses import asdict, dataclass
import copy
import re

from distillschedule.workloads.best_known.registry import fingerprint, load_profile
from .contracts import require

LIVEWAN = "livewan_crossscale_pipeline_best"
SCHEMA = "distillschedule.livewan_request.v1"


@dataclass(frozen=True)
class LiveWanRequest:
    frames: int = 81
    schema_version: str = SCHEMA

    def __post_init__(self):
        require(self.schema_version == SCHEMA, "unsupported LiveWan request schema")
        require(type(self.frames) is int and self.frames in (69, 81),
                "LiveWan supports only ORIGINAL 69 or 81 pixel frames; other semantics are not admitted")

    @property
    def id(self):
        return f"f{self.frames}"

    @property
    def latent_shape(self):
        return (1, 16, (self.frames - 1) // 4 + 1, 46, 80)

    @property
    def attention_tokens(self):
        return self.latent_shape[2] * 23 * 40

    def new_blocks(self, world_frames):
        require(type(world_frames) is int and world_frames in (3, 6, 9), "LiveWan world choices changed")
        return (self.latent_shape[2] - world_frames) // 3

    def source_nograd_calls(self, world_frames):
        # set_world plus four denoise calls AND the clean KV commit per block.
        return 1 + (4 + 1) * self.new_blocks(world_frames)

    def kernel_scope_calls(self, kind):
        # Unchanged 30-block native helpers: three RMS sites + head norm,
        # two residual gates per block, two modulation sites + head modulation.
        factors = {"rms": (90, 30), "gate": (60, 0), "modulation": (61, 0)}
        require(kind in factors, "unknown LiveWan native kernel census")
        factor, tail = factors[kind]
        return tuple(self.source_nograd_calls(w) * factor + tail for w in (3, 6, 9))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict) and set(value) == {"frames", "schema_version"},
                "LiveWan request requires exact schema/frames; unknown semantics are not ignored")
        return cls(**value)

    @classmethod
    def parse(cls, value):
        match = re.fullmatch(r"f(69|81)", value)
        require(match is not None, "malformed LiveWan request identity")
        return cls(frames=int(match[1]))

    def profile(self, *, root=None):
        profile = copy.deepcopy(load_profile(LIVEWAN, root=root))
        if self == LiveWanRequest():
            return profile
        profile["display_name"] = f"Unvalidated ORIGINAL LiveWan request: {self.id}"
        profile["contract"].update(request=self.to_dict(), resolution=[368, 640], frames=self.frames,
            latent_shape=list(self.latent_shape), world_choices=[3, 6, 9], block_frames=3,
            steps_per_block=4, grad_blocks=3)
        profile["historical"] = dict(formal_speedup_eligible=False,
            scope="NO_MATCHING_REQUEST_MEASUREMENT; PARENT_CANONICAL_SECONDS_NOT_TRANSFERABLE",
            parent_canonical_profile_sha256=fingerprint(load_profile(LIVEWAN, root=root)))
        return profile


def request_from_plan(plan):
    value = plan.get("workload", {}).get("metadata", {}).get("workload_request")
    request = LiveWanRequest.from_dict(value) if value is not None else LiveWanRequest()
    shape = plan["workload"]["shape"]
    require(shape.get("frames") == request.latent_shape[2]
            and shape.get("latent") == dict(zip(("channels", "frames", "height", "width"), request.latent_shape[1:]))
            and shape.get("resolution") in ([368, 640], {"height": 368, "width": 640}),
            "LiveWan plan shape differs from request")
    if value is not None:
        require(plan["workload"]["metadata"].get("workload_request_sha256") == fingerprint(request.to_dict()),
                "LiveWan plan request SHA mismatch")
    return request


def validate_teacher_sample(value, request):
    """Host tensor metadata from the real pinned gen_teacher output dictionary."""
    if request == LiveWanRequest():
        return
    if (type(value.get("frames")) is not int or value["frames"] != request.frames
            or value.get("steps") != 40 or value.get("size") != "640*368"):
        raise RuntimeError("sample frames/gen_teacher metadata differs from LiveWan request")
    if tuple(value["latents"].shape) != request.latent_shape[1:]:
        raise RuntimeError("sample latent shape differs from LiveWan request; prefix cropping is forbidden")


def lower_native_plan(template, request, *, trace=None):
    """Expand a copy; source templates, cache/max frames and protocol stay fixed."""
    plan = copy.deepcopy(template)
    if request == LiveWanRequest():
        return plan
    plan["workload"]["shape"].update(frames=request.latent_shape[2],
        latent=dict(zip(("channels", "frames", "height", "width"), request.latent_shape[1:])))
    plan["workload"]["metadata"].update(workload_request=request.to_dict(),
        workload_request_sha256=fingerprint(request.to_dict()))
    metadata = plan["metadata"]
    if "generator_runahead_transport" in metadata:
        metadata["generator_runahead_transport"]["payload_shape"] = list(request.latent_shape)
    if "critic_attention_checkpoint" in metadata:
        from distillschedule.runtime.livewan_attention_checkpoint import plan_contract
        metadata["critic_attention_checkpoint"] = plan_contract(request.attention_tokens)
    if "slot_pairs_by_lane" in metadata:
        # Never copy old 21-frame load claims. A runnable plan must derive both
        # vectors from its own request-qualified immutable trace.
        if trace is None:
            metadata.pop("current_step_max_new_blocks", None)
            metadata.pop("candidate_step_max_new_blocks", None)
        else:
            groups = metadata["slot_pairs_by_lane"]
            def loads(pairs, offset):
                return [max(sum(request.new_blocks(trace.row(offset+s, slot).world_frames)
                                for slot in pair) for pair in pairs) for s in range(5)]
            current = [(i, i+8) for i in range(8)]
            a, b = loads(current, 0), loads(groups, 0)
            require(all(loads(current, c*5) == a and loads(groups, c*5) == b for c in range(trace.cycles)),
                    "LiveWan trace changes per-cycle routing load")
            require(sum(b) < sum(a), "LiveWan registered routing does not balance this new trace")
            metadata.update(current_step_max_new_blocks=a, candidate_step_max_new_blocks=b,
                            request_trace_sha256=trace.sha256)
    request_from_plan(plan)
    return plan
