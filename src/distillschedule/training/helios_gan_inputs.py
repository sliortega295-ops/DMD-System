"""Versioned W3 native GAN input stream, with genuinely step-qualified sources.

The original dataset/history/scheduler/noise and GAN crop distributions produce
the payloads. No loss, model, optimizer or alternative training math lives here.
The finite stream is the horizon; native P0 uses its unchanged constant LR.
"""
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path

from distillschedule.core.contracts import require
from distillschedule.search.helios_request import OriginalHeliosRequest
from distillschedule.workloads.best_known.registry import fingerprint, file_sha256, repository_root, write_new_json
from .helios_inputs import (HeliosCohortSources, HeliosTrainingTrace, _inventory,
                            _require_fresh_cohorts, source_fingerprint, validate_cohort_request,
                            history_control_options, ALIGNED_SCHEDULE_POLICY)

KIND = "original_helios_gan_cohort_stream.v1"
SCHEMA = "distillschedule." + KIND
NATIVE_KIND = "original_helios_gan_native_history_cohort_stream.v1"
NATIVE_SCHEMA = "distillschedule." + NATIVE_KIND
from distillschedule.strategies.wan_task import NATIVE_HISTORY_POLICY
TRACE_KEY = "distillschedule_helios_gan_training_cohort"
REQUEST = OriginalHeliosRequest(variant="gan")
_UNUSED_SYSTEM_LOCATORS = {
    "model_config": {"pretrained_model_name_or_path", "transformer_model_name_or_path",
                     "real_score_model_name_or_path", "reward_model_name_or_path"},
    "data_config": {"gan_data_root"},
}


@dataclass(frozen=True)
class GANCohortTask:
    index: int
    request: OriginalHeliosRequest = REQUEST
    native_policy: str | None = None

    def __post_init__(self):
        require(type(self.index) is int and self.index >= 0,
                "nonnegative fresh GAN33/222/world16/B1 cohort required", path="helios.gan.inputs")
        validate_cohort_request(self.request, variant="gan")
        require(self.native_policy is None or self.native_policy == NATIVE_HISTORY_POLICY,
                "unsupported explicit GAN native history policy", path="helios.gan.inputs")
        require(self.native_policy is None or self.request.stage_schedule_policy == "source_normal_only",
                "cannot combine aligned benchmark and native rank-local history", path="helios.gan.inputs")

    def __getattr__(self, name):
        return getattr(self.request, name)

    @property
    def stage_schedule_policy(self):
        return self.native_policy or self.request.stage_schedule_policy

    @property
    def first_step(self):
        return 1000 + 5 * self.index

    @property
    def source_steps(self):
        return tuple(range(self.first_step, self.first_step + 5))

    def to_dict(self):
        value = {"schema_version": "distillschedule.helios_gan_training_cohort.v1", "index": self.index,
                 "request": self.request.to_dict(), "source_steps": list(self.source_steps)}
        if self.native_policy is not None:
            value.update(schema_version="distillschedule.helios_gan_native_history_cohort.v1", native_policy=self.native_policy)
        return value

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict), "explicit GAN cohort required", path="helios.gan.inputs")
        native = value.get("schema_version") == "distillschedule.helios_gan_native_history_cohort.v1"
        require(set(value) == {"schema_version", "index", "request", "source_steps"} | ({"native_policy"} if native else set()),
                "complete explicit GAN cohort required", path="helios.gan.inputs")
        task = cls(value["index"], OriginalHeliosRequest.from_dict(value["request"]), value["native_policy"] if native else None)
        require(value == task.to_dict(), "GAN cohort changed", path="helios.gan.inputs")
        return task

    @property
    def digest(self):
        return fingerprint(self.to_dict())


def task_for_stream(stream, index):
    return GANCohortTask(index, request=OriginalHeliosRequest.from_dict(stream["identity"]["request"]),
                         native_policy=stream["identity"].get("native_history_policy"))


def keys(step, phase):
    from runtime.static_world16_trace import _generator_keys, _critic_keys
    selected = _generator_keys(step, gan=True) if phase == "generator" else _critic_keys(step, gan=True)
    return {**selected, "text_embedding": f"step/{step}/sample/text_embedding",
            "attention_mask": f"step/{step}/sample/attention_mask"}


def _canonical_config():
    import yaml
    return yaml.safe_load((repository_root() /
        "configs/distillschedule/templates/helios_stage3_gan_384x640_systems.yaml").read_text())


def portable_config():
    """Original settings without the five unused historical system locators.

    Native P0 loads model/negative/kernel assets and fresh prepared inputs from
    the explicit public asset/stream contracts, never these legacy YAML fields.
    The caller serializes this mapping once and uses that same SHA-bound file
    for preparation and execution. No loss/model/sampling setting is replaced.
    """
    value = _canonical_config()
    for section, names in _UNUSED_SYSTEM_LOCATORS.items():
        for name in names:
            value[section].pop(name)
    return value


def read_config(path):
    import yaml
    value = yaml.safe_load(Path(path).read_text())
    reference = _canonical_config()
    require(isinstance(value, dict) and set(value) == set(reference), "complete native GAN config required",
            path="helios.gan.inputs")
    # Only the input producer's documented random-drop/corruption knobs may
    # vary. GAN math, heads, optimizer, precision and native schedules do not.
    allowed = {"is_random_drop", "random_drop_i2v_ratio", "random_drop_v2v_ratio", "random_drop_t2v_ratio",
               "noise_corrupt_clean_prob_history"}
    for section in reference:
        actual, expected = value[section], reference[section]
        if section == "training_config":
            require(isinstance(actual, dict) and set(actual) <= set(expected) | allowed
                    and set(expected) <= set(actual)
                    and all(actual[k] == v for k, v in expected.items() if k not in allowed),
                    "native GAN algorithm/optimizer/config changed", path="helios.gan.inputs")
        elif section in _UNUSED_SYSTEM_LOCATORS:
            unused = _UNUSED_SYSTEM_LOCATORS[section]
            require(isinstance(actual, dict) and set(expected) - unused <= set(actual) <= set(expected)
                    and {k:v for k,v in actual.items() if k not in unused} ==
                        {k:v for k,v in expected.items() if k not in unused},
                    "native GAN non-input config changed", path="helios.gan.inputs")
            for name in unused & actual.keys():
                locator = actual[name]
                valid = locator is None or (isinstance(locator, list) and all(isinstance(p, str) for p in locator)
                                            if name == "gan_data_root" else isinstance(locator, str))
                require(valid, "invalid unused native GAN system locator", path="helios.gan.inputs")
        else:
            require(actual == expected, "native GAN non-input config changed", path="helios.gan.inputs")
    require(value["training_config"]["lr_scheduler"] == "constant"
            and value["training_config"]["is_separate_gan_grad"] is False,
            "unchanged native constant scheduler and joint GAN gradients required", path="helios.gan.inputs")
    return value["training_config"]


def capture_rank(task, sources, inventory, config, *, rank, output, provenance):
    import torch
    from runtime.operation_trace import OperationTraceWriter
    from runtime.official_baseline import stable_block_noise
    from distillschedule.strategies.wan_capture import native_primitives, input_stream, prepare_history, stage_control
    from distillschedule.strategies.wan_inputs import load_feature
    require(type(rank) is int and 0 <= rank < 16 and not Path(output).exists(),
            "fresh rank output required", path="helios.gan.inputs")
    native = native_primitives()
    rows = [r for r in inventory if r["rank"] == rank]
    require([r["step"] for r in rows] == list(task.source_steps), "rank source steps changed", path="helios.gan.inputs")
    writer = OperationTraceWriter(classification="NEW_ORIGINAL_HELIOS_GAN_STEP_QUALIFIED_INPUTS",
        source={**provenance, "rank": rank, "samples": rows, "seed": sources.seed, "cohort": task.index,
                "rng_scope": "new_native_CPU_streams_not_prior_training_rng_continuation"})
    dataset = native.dataset.__new__(native.dataset)
    dataset.base_seed, dataset._epoch = sources.seed, task.index
    for offset, row in enumerate(rows):
        step = row["step"]
        latent, text, mask = load_feature(task, row)
        primary, secondary = dataset.prepare_stage1_latent(latent, rank * 5 + offset, return_secondary=True)
        for source, selected in (("primary", primary), ("secondary", secondary)):
            for role, tensor in zip(("x0", "history", "target"), selected, strict=True):
                writer.add_tensor(f"step/{step}/sample/{source}/{role}", tensor.to(torch.bfloat16), role=role)
        writer.add_tensor(f"step/{step}/sample/text_embedding", text.to(torch.bfloat16), role="text_embedding")
        writer.add_tensor(f"step/{step}/sample/attention_mask", mask, role="attention_mask")
        writer.add_operation(f"step/{step}/branch", component="training", operation="stage3_branch", control={
            "global_step": step, "dfake_gen_update_ratio": 5, "train_generator": offset == 0,
            "train_critic": True, "gan": True, "reward": False, "vae_decode": False})
        for phase in (("generator", "critic") if offset == 0 else ("critic",)):
            prefix = f"step/{step}/{phase}"
            with input_stream(sources.seed, task.index, rank, step, phase, "history"):
                tensors = prepare_history(native, task, primary if phase == "generator" else secondary, config)
            noises = ("entry_noise", "rollout_noise", "dmd_noise", "gan_g_noise") if phase == "generator" else (
                      "entry_noise", "rollout_noise", "critic_noise", "r1_epsilon")
            for stream in noises:
                with input_stream(sources.seed, task.index, rank, step, phase, stream):
                    tensors[stream] = torch.randn(task.latent_shape, dtype=torch.bfloat16)
            for stage in (1, 2):
                h, w = task.stage_spatial_shapes[stage]
                with input_stream(sources.seed, task.index, rank, step, phase, "block", stage):
                    noise = stable_block_noise(16 * task.latent_shape[2] * (h // 2) * (w // 2))
                tensors[f"block_noise/{stage}"] = noise.view(1,16,9,h//2,w//2,2,2).permute(
                    0,1,2,3,5,4,6).reshape(1,16,9,h,w).to(torch.bfloat16)
            with input_stream(sources.seed, task.index, rank, step, phase, "score-time"):
                _, mu = native.base.apply_schedule_shift(torch.tensor([.5], dtype=torch.float64),
                    torch.empty(task.latent_shape, device="meta"), return_mu=True)
                tensors["dmd_timestep" if phase == "generator" else "critic_timestep"] = native.post.sample_dynamic_timestep(
                    B=1, timestep_shift=mu, dynamic_alpha=config[phase + "_dynamic_alpha"],
                    dynamic_beta=config[phase + "_dynamic_beta"], dynamic_sample_type=config[phase + "_dynamic_sample_type"],
                    global_step=step, dynamic_step=config[phase + "_dynamic_step"], device="cpu")
            crops = ("gan_g_crop",) if phase == "generator" else ("gan_d_fake_crop", "gan_d_real_crop", "gan_d_r1_crop")
            for crop in crops:
                # Same two native randint draws as gan_crop_video_spatial;
                # offsets are materialized once into an explicitly new stream.
                with input_stream(sources.seed, task.index, rank, step, phase, crop):
                    tensors[crop] = torch.stack((torch.randint(0, 25, (1,)), torch.randint(0, 41, (1,))), dim=1)
            for name, tensor in tensors.items():
                writer.add_tensor(prefix + "/" + name, tensor, role=name)
            options = history_control_options(task, zero=all(bool(tensors[name].sum() == 0)
                for name in ("history_short", "history_mid", "history_long")), rank=rank)
            controls = stage_control(native, task, sources.seed, task.index, step, phase, **options)
            writer.add_operation(prefix + "/rollout", component=phase, operation="native_pyramid_rollout",
                                 tensor_keys=list(keys(step, phase).values()), control=controls)
            writer.add_operation(prefix + ("/dmd" if phase == "generator" else "/train"),
                                 component=phase, operation="native_score_inputs", control={"global_step": step})
            writer.add_operation(prefix + ("/gan_g" if phase == "generator" else "/gan_d"),
                                 component=phase, operation="native_gan_inputs", control={"global_step": step})
    return writer.write(output, contract={TRACE_KEY: task.to_dict(), "resolution": [384,640],
        "full_latent_shape": list(task.latent_shape), "stage_spatial_shapes": [list(s) for s in task.stage_spatial_shapes],
        "dtype": "bfloat16", "dfake_gen_update_ratio": 5})


def prepare_stream(sources, training_config, output, *, input_policy=None, native_policy=None, request=REQUEST):
    from runtime.operation_trace import write_operation_trace_bundle_manifest
    from distillschedule.data.helios_epoch_inputs import build_cohorts, cohort_identity
    task_type = partial(GANCohortTask, request=request, native_policy=native_policy)
    task_type(0)  # Fail closed before reading input tensors or creating output.
    if input_policy is None:
        require(isinstance(sources, (list, tuple)) and len(sources) >= 2
                and all(type(s) is HeliosCohortSources for s in sources), "at least two explicit native GAN cohorts required",
                path="helios.gan.inputs")
        config = read_config(training_config)
        tasks = [task_type(i) for i in range(len(sources))]
        _require_fresh_cohorts(sources)
        inventories = [_inventory(t, s) for t, s in zip(tasks, sources, strict=True)]
        _require_fresh_cohorts(sources, inventories)
        epoch_policy = None
    else:
        sources, inventories, epoch_policy = build_cohorts(sources, task_type=task_type, input_policy=input_policy)
        tasks = [task_type(i) for i in range(len(sources))]
        config = read_config(training_config)
    identity = {"request": request.to_dict(), "sources": [s.to_dict() for s in sources], "inventory": inventories,
                "training_config_sha256": file_sha256(Path(training_config)), "source_fingerprint": source_fingerprint()}
    if epoch_policy is not None:
        identity["input_policy"] = epoch_policy
    if native_policy is not None:
        identity["native_history_policy"] = native_policy
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    write_new_json(root / "inputs.json", identity)
    entries = []
    for task, source, inventory in zip(tasks, sources, inventories, strict=True):
        directory = root / f"cohort-{task.index:04d}"
        ranks = []
        for rank in range(16):
            trace = capture_rank(task, source, inventory, config, rank=rank, output=directory / f"rank_{rank:02d}",
                provenance={**{k: identity[k] for k in ("source_fingerprint", "training_config_sha256")},
                            **cohort_identity(identity, task.index)})
            ranks.append({"rank": rank, "path": f"rank_{rank:02d}", "trace_hash": trace.trace_hash})
            del trace
        bundle = write_operation_trace_bundle_manifest(directory, ranks)
        entries.append({"index": task.index, "task": task.to_dict(), "trace_bundle": str(directory), "trace_hash": bundle["bundle_hash"],
                        **cohort_identity(identity, task.index)})
    require(identity["source_fingerprint"] == source_fingerprint()
            and identity["training_config_sha256"] == file_sha256(Path(training_config))
            and all(file_sha256(Path(row["path"])) == row["sha256"] for rows in inventories for row in rows),
            "native GAN source/input changed during preparation", path="helios.gan.inputs")
    result = {"schema_version": NATIVE_SCHEMA if native_policy is not None else SCHEMA,
              "status": "READY_FRESH_COHORT_STREAM", "identity": identity,
              "cohorts": entries, "gpu_execution": "NOT_RUN", "claimable_speedup": False}
    result["stream_sha256"] = fingerprint(result)
    write_new_json(root / "stream.json", result)
    return result


def validate_controls(task, controls):
    from runtime.operation_trace import validate_official_rank_stage_schedule_semantics
    require(set(controls) == set(range(16)), "complete rank16 GAN cohort required", path="helios.gan.inputs")
    conditioning = set()
    for rank, trace in controls.items():
        contract = trace.manifest["contract"]
        require(contract.get(TRACE_KEY) == task.to_dict()
                and not any(k in contract for k in ("distillschedule_original_helios_request",
                    "distillschedule_helios_training_cohort", "distillschedule_original_wan_task")),
                "new GAN cohort required, not a relabelled trace", path="helios.gan.inputs")
        if task.native_policy is not None:
            from distillschedule.runtime.helios_gan_native_history import validate_contract
            validate_contract(task, trace)
        descriptors = trace.manifest["tensors"]
        operations = []
        for step in task.source_steps:
            operations.append(f"step/{step}/branch")
            for phase in (("generator", "critic") if step == task.first_step else ("critic",)):
                prefix = f"step/{step}/{phase}"
                operations.extend((prefix + "/rollout", prefix + ("/dmd" if phase == "generator" else "/train"),
                                   prefix + ("/gan_g" if phase == "generator" else "/gan_d")))
                for role, key in keys(step, phase).items():
                    row = descriptors[key]
                    if role == "text_embedding":
                        shape = row["shape"]
                        require(len(shape) == 2 and type(shape[0]) is int and shape[0] > 0 and shape[1] == 4096
                                and row["dtype"] == "bfloat16", "GAN conditioning changed", path="helios.gan.inputs")
                        conditioning.add(tuple(shape)); continue
                    if role == "attention_mask":
                        require(row["shape"] == descriptors[keys(step, phase)["text_embedding"]]["shape"][:1],
                                "GAN conditioning mask changed", path="helios.gan.inputs"); continue
                    shape, dtype = list(task.latent_shape), "bfloat16"
                    if role.startswith("indices_"):
                        shape, dtype = [1, {"indices_hidden_states":9, "indices_history_short":2,
                            "indices_history_mid":2, "indices_history_long":16}[role]], "int64"
                    elif role.endswith("_timestep"):
                        shape, dtype = [1], "int64"
                    elif role.endswith("_crop"):
                        shape, dtype = [1, 2], "int64"
                    elif role == "block_noise_stage1":
                        shape[-2:] = [24,40]
                    elif role in ("history_short", "history_mid", "history_long", "continuous_history"):
                        shape[2] = {"history_short":2, "history_mid":2, "history_long":16, "continuous_history":19}[role]
                    require(row["shape"] == shape and row["dtype"] == dtype,
                            "native GAN tensor geometry/precision changed: " + role, path="helios.gan.inputs")
                control = trace.operation(prefix + "/rollout")["control"]
                schedule = control["pyramid_stage_call_sequence"]
                budget = 4 if (task.native_policy is not None or task.stage_schedule_policy == ALIGNED_SCHEDULE_POLICY) \
                    and control.get("all_history_zero") is True else 2
                if task.stage_schedule_policy == ALIGNED_SCHEDULE_POLICY:
                    require(control.get("stage_schedule_policy") == ALIGNED_SCHEDULE_POLICY
                            and type(control.get("all_history_zero")) is bool,
                            "aligned GAN history/schedule scope missing", path="helios.gan.inputs")
                require(len(schedule) == 3 and [s["stage"] for s in schedule] == [0,1,2]
                        and all(type(s["call_count"]) is int and 1 <= s["call_count"] <= (2 if task.native_policy is None else budget)
                                and s["denoising_step_budget"] == budget for s in schedule),
                        "native GAN222 stage scope changed", path="helios.gan.inputs")
            for source in ("primary", "secondary"):
                for role in ("x0", "history", "target"):
                    row = descriptors[f"step/{step}/sample/{source}/{role}"]
                    shape = row["shape"]
                    # Original prepare_stage1_latent keeps one x0 frame,
                    # exactly 16+2+1 history frames, and the 9-frame target.
                    require(shape == [16, {"x0":1, "history":19, "target":9}[role], 48,80]
                            and row["dtype"] == "bfloat16",
                            "step-qualified original GAN source missing/changed", path="helios.gan.inputs")
        observed = [op["key"] for op in trace.manifest["operations"]]
        require(len(observed) == len(set(observed)) == len(operations) and set(observed) == set(operations),
                "GAN cohort work inventory differs", path="helios.gan.inputs")
    require(len(conditioning) == 1, "rank/step GAN conditioning geometry differs", path="helios.gan.inputs")
    if task.native_policy is not None:
        from distillschedule.runtime.helios_gan_native_history import validate_schedules
        return validate_schedules(task, controls)
    return validate_official_rank_stage_schedule_semantics(controls)


def load_cohort(task, root, *, expected_hash):
    from runtime.operation_trace import OperationTrace, load_operation_trace_bundle_manifest, resolve_rank_trace_root
    from runtime.static_world16_trace import StaticWorld16Trace
    root = Path(root).resolve()
    bundle = load_operation_trace_bundle_manifest(root, verify=True)
    require(bundle is not None and bundle["bundle_hash"] == expected_hash and len(bundle["ranks"]) == 16,
            "GAN cohort rank/bundle identity changed", path="helios.gan.inputs")
    controls = {}
    for rank in range(16):
        directory = resolve_rank_trace_root(root, rank).resolve()
        require(root in directory.parents, "GAN rank path escaped cohort", path="helios.gan.inputs")
        controls[rank] = OperationTrace(root=directory, manifest=json.loads((directory / "trace.json").read_text()), tensors={})
    obj = GANTrainingTrace.__new__(GANTrainingTrace)
    obj.task, obj.root, obj.variant, obj.gan, obj.request = task, root, "gan", True, None
    obj.logical_steps, obj.bundle_hash = task.source_steps, expected_hash
    obj.rank_control_validation = validate_controls(task, controls)
    obj.binding_hash = fingerprint({"schema": "distillschedule.helios_gan_training_trace.v1", "task": task.to_dict(),
                                   "bundle_hash": expected_hash, "rank_slots": list(range(16))})
    obj._control, obj._loaded, obj._selected_step, obj._entries, obj._uses = controls, {}, None, {}, {}
    obj._selected_phase, obj._source_cache, obj._source_uses = None, {}, {}
    # The original fixed ABI remains unchanged; the independent native GAN
    # validator above checks exact GAN branches/geometry and rank-local calls.
    if task.native_policy is None:
        StaticWorld16Trace._validate(obj)
    return obj


def validate_rank_payload(trace, rank, source):
    import torch
    from distillschedule.strategies.wan_capture import native_primitives, stage_control
    value = trace._load(rank)
    require(all(bool(torch.isfinite(t).all()) for t in value.tensors.values() if t.is_floating_point()),
            "non-finite original GAN payload", path="helios.gan.inputs")
    native = native_primitives()
    for step in trace.logical_steps:
        for phase in (("generator", "critic") if step == trace.logical_steps[0] else ("critic",)):
            prefix = f"step/{step}/{phase}"
            timestep = value.tensor(prefix + ("/dmd_timestep" if phase == "generator" else "/critic_timestep"))
            require(bool(((timestep >= 20) & (timestep <= 980)).all()), "GAN timestep outside native clamp", path="helios.gan.inputs")
            zero = all(bool(value.tensor(prefix + "/" + role).sum() == 0)
                       for role in ("history_short", "history_mid", "history_long"))
            options = history_control_options(trace.task, zero=zero, rank=rank)
            require(value.operation(prefix + "/rollout")["control"] ==
                    stage_control(native, trace.task, source.seed, trace.task.index, step, phase, **options),
                    "GAN original stage grid/RNG exits changed", path="helios.gan.inputs")
            for role, key in keys(step, phase).items():
                if role.endswith("_crop"):
                    crop = value.tensor(key)
                    require(bool(((crop >= 0) & (crop <= torch.tensor([[24,40]]))).all()),
                            "native GAN crop outside spatial range", path="helios.gan.inputs")
        mask = value.tensor(f"step/{step}/sample/attention_mask")
        require(bool(((mask == 0) | (mask == 1)).all()), "invalid GAN condition mask", path="helios.gan.inputs")


def load_stream(path, *, cycles, verify_rank=None, expected_kind=KIND):
    path = Path(path).resolve()
    doc = json.loads(path.read_text()); unsigned = dict(doc); sha = unsigned.pop("stream_sha256", None)
    require(expected_kind in (KIND, NATIVE_KIND) and doc.get("schema_version") == "distillschedule." + expected_kind
            and doc.get("status") == "READY_FRESH_COHORT_STREAM"
            and sha == fingerprint(unsigned), "GAN stream changed/incomplete", path="helios.gan.inputs")
    identity = doc["identity"]
    native_policy = identity.get("native_history_policy")
    require((expected_kind == NATIVE_KIND and native_policy == NATIVE_HISTORY_POLICY)
            or (expected_kind == KIND and "native_history_policy" not in identity),
            "GAN stream history policy/schema differs", path="helios.gan.inputs")
    request = OriginalHeliosRequest.from_dict(identity["request"])
    task_type = partial(GANCohortTask, request=request, native_policy=native_policy)
    task_type(0)
    sources = [HeliosCohortSources.from_dict(s) for s in identity["sources"]]
    require(type(cycles) is int and cycles >= 2 and len(doc["cohorts"]) == len(sources) == len(identity["inventory"]) == cycles
            and identity["source_fingerprint"] == source_fingerprint(),
            "GAN stream cycles/source/request changed", path="helios.gan.inputs")
    from distillschedule.data.helios_epoch_inputs import validate_identity, cohort_identity
    if "input_policy" in identity:
        validate_identity(identity, task_type=task_type)
    else:
        _require_fresh_cohorts(sources, identity["inventory"])
    for index, entry in enumerate(doc["cohorts"]):
        task = task_type(index)
        require(entry["index"] == index and entry["task"] == task.to_dict()
                and Path(entry["trace_bundle"]).resolve() == path.parent / f"cohort-{index:04d}",
                "GAN cohort order/path/task changed", path="helios.gan.inputs")
        expected_policy = cohort_identity(identity, index)
        require({k: entry[k] for k in ("input_policy",) if k in entry} == expected_policy,
                "cohort policy/draw identity changed", path="helios.inputs")
        trace = load_cohort(task, entry["trace_bundle"], expected_hash=entry["trace_hash"])
        rows = identity["inventory"][index]
        require(len(rows) == 80 and [(r["step"], r["rank"]) for r in rows] ==
                [(s, r) for s in task.source_steps for r in range(16)], "incomplete GAN source inventory", path="helios.gan.inputs")
        require([r["path"] for r in rows] == [name for row in sources[index].to_dict()["samples"] for name in row],
                "GAN source inventory differs", path="helios.gan.inputs")
        for rank, control in trace._control.items():
            require(control.manifest["source"] == {"source_fingerprint": identity["source_fingerprint"],
                "training_config_sha256": identity["training_config_sha256"], "rank": rank,
                "samples": [r for r in rows if r["rank"] == rank], "seed": sources[index].seed, "cohort": index,
                "rng_scope": "new_native_CPU_streams_not_prior_training_rng_continuation", **expected_policy},
                "GAN source/RNG provenance differs", path="helios.gan.inputs")
        if verify_rank is not None:
            validate_rank_payload(trace, verify_rank, sources[index])
    return doc


class GANTrainingTrace(HeliosTrainingTrace):
    def _entry(self, rank, step, phase, device):
        from runtime.static_world16_trace import StaticWorld16Trace, _descriptor_hashes
        trace = self._load(rank); selected = keys(step, phase)
        tensors = StaticWorld16Trace._tensors(self, trace, selected, device)
        identity = fingerprint({"task": self.task.to_dict(), "phase": phase, "rank": rank, "step": step,
                                "trace_hash": trace.trace_hash, "tensors": _descriptor_hashes(trace, selected)})
        self._selected_step = step
        self._selected_phase = phase
        return trace, tensors, identity, selected

    def load_critic_entry(self, *, rank, step, device):
        # Keep the original hardened one-use H2D retention, but never retain
        # an entry across a cohort boundary. CPU validation is uncached.
        import torch
        self._selected_step = step
        self._selected_phase = "critic"
        if torch.device(device).type != "cuda":
            return super().load_critic_entry(rank=rank, step=step, device=device)
        key = (rank, step, str(torch.device(device)))
        require(self._uses.get(key, 0) < 2, "GAN entry already consumed twice", path="helios.gan.inputs")
        self._uses[key] = self._uses.get(key, 0) + 1
        if key in self._entries:
            return self._entries.pop(key)
        value = super().load_critic_entry(rank=rank, step=step, device=device)
        self._entries[key] = value
        return value

    def source(self, *, rank, device):
        import torch
        step = self._selected_step
        require(step in self.logical_steps, "GAN source requires an action-selected step", path="helios.gan.inputs")
        if torch.device(device).type != "cuda":
            return self._materialize_source(rank, step, device)
        key = (rank, step, str(torch.device(device)))
        use = (*key, self._selected_phase)
        require(self._selected_phase in ("generator", "critic") and use not in self._source_uses,
                "GAN source already consumed in this action", path="helios.gan.inputs")
        self._source_uses[use] = 1
        if key in self._source_cache:
            require(self._selected_phase == "critic", "GAN source cache must be consumed by Critic", path="helios.gan.inputs")
            return self._source_cache.pop(key)
        value = self._materialize_source(rank, step, device)
        if self._selected_phase == "generator":
            # Same first-step source is consumed by G and C. Other steps are
            # Critic-only and need no source retention after their one use.
            self._source_cache[key] = value
        return value

    def _materialize_source(self, rank, step, device):
        import torch
        from runtime.operation_trace import STAGE3_SOURCE_KEYS
        trace = self._load(rank)
        return {name: trace.tensor(key.replace("sample/0", f"step/{step}/sample"), clone=False).to(
                device=device, dtype=torch.bfloat16, copy=True).unsqueeze(0) for name, key in STAGE3_SOURCE_KEYS.items()}

    def validate_boundary(self):
        require(not self._entries and not self._source_cache and all(n == 2 for n in self._uses.values()),
                "GAN one-use input cache not drained", path="helios.gan.inputs")

    def close(self):
        self._entries.clear(); self._loaded.clear(); self._uses.clear()
        self._source_cache.clear(); self._source_uses.clear()
        self._selected_step = self._selected_phase = None
