"""Fresh W2 inputs from the licensed Helios dataset/history/scheduler owners.

This versioned reference seam is deliberately narrower than a general data
loader: explicit pre-encoded, step-qualified features, normal globally synced
33-frame/222 No-GAN, world16/B1/G1-C5-EMA5. It never relabels an old trace.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace

from distillschedule.core.contracts import require
from distillschedule.search.helios_request import OriginalHeliosRequest
from distillschedule.strategies.wan_task import NATIVE_HISTORY_POLICY
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, write_new_json

KIND = "original_helios_cohort_stream.v1"
SCHEMA = "distillschedule." + KIND
NATIVE_KIND = "original_helios_native_history_cohort_stream.v1"
NATIVE_SCHEMA = "distillschedule." + NATIVE_KIND
TRACE_KEY = "distillschedule_helios_training_cohort"
MAX_COHORTS = 100
ALIGNED_SCHEDULE_POLICY = "benchmark_v4_strict_global"


def validate_cohort_request(request, *, variant):
    """Admit the existing synchronized scope without widening its workload."""
    require(type(request) is OriginalHeliosRequest
            and replace(request, stage_schedule_policy="source_normal_only") == OriginalHeliosRequest(variant=variant),
            "fresh reference requires original33/222/world16/B1/G1-C5", path="helios.inputs")
    return request


def history_control_options(task, *, zero, rank):
    """Keep sampled history and its native grid; only the explicit scope owns exits."""
    require(type(zero) is bool, "explicit original history predicate required", path="helios.inputs")
    if task.native_policy is not None or task.stage_schedule_policy == ALIGNED_SCHEDULE_POLICY:
        return {"all_history_zero":zero, "rank":rank}
    require(not zero, "normal reference forbids rank-local amplified zero history", path="helios.inputs")
    return {}


@dataclass(frozen=True)
class HeliosCohortTask:
    index: int
    request: OriginalHeliosRequest = OriginalHeliosRequest()
    native_policy: str | None = None

    def __post_init__(self):
        require(type(self.index) is int and 0 <= self.index < MAX_COHORTS,
                f"cohort index must be in [0,{MAX_COHORTS-1}]", path="helios.inputs")
        validate_cohort_request(self.request, variant="no-gan")
        require(self.native_policy is None or self.native_policy == NATIVE_HISTORY_POLICY,
                "unsupported explicit W2 native history policy", path="helios.inputs")
        require(self.native_policy is None or self.request.stage_schedule_policy == "source_normal_only",
                "cannot combine aligned benchmark and native rank-local history", path="helios.inputs")

    def __getattr__(self, name):
        return getattr(self.request, name)

    @property
    def stage_schedule_policy(self):
        return self.native_policy or self.request.stage_schedule_policy

    @property
    def first_step(self):
        return 10 + 5 * self.index

    @property
    def source_steps(self):
        return tuple(range(self.first_step, self.first_step + 5))

    def to_dict(self):
        value = {"schema_version": "distillschedule.helios_training_cohort.v1", "index": self.index,
                "request": self.request.to_dict(), "source_steps": list(self.source_steps)}
        if self.native_policy is not None:
            value.update(schema_version="distillschedule.helios_native_history_cohort.v1",
                         native_policy=self.native_policy)
        return value

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict), "explicit Helios cohort required", path="helios.inputs")
        native = value.get("schema_version") == "distillschedule.helios_native_history_cohort.v1"
        require(set(value) == {"schema_version", "index", "request", "source_steps"} | ({"native_policy"} if native else set()),
                "complete explicit Helios cohort required", path="helios.inputs")
        task = cls(value["index"], OriginalHeliosRequest.from_dict(value["request"]),
                   value["native_policy"] if native else None)
        require(value == task.to_dict(), "cohort task changed", path="helios.inputs")
        return task

    @property
    def digest(self):
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class HeliosCohortSources:
    seed: int
    samples: tuple[tuple[str, ...], ...]

    def __post_init__(self):
        require(type(self.seed) is int and 0 <= self.seed < 2**63,
                "explicit 63-bit seed required", path="helios.inputs")
        require(len(self.samples) == 5 and all(len(row) == 16 for row in self.samples),
                "five actual per-rank source sets required", path="helios.inputs")
        require(all(isinstance(name, str) and Path(name).is_absolute() for row in self.samples for name in row),
                "absolute native feature paths required", path="helios.inputs")
        require(len({str(Path(name).resolve()) for row in self.samples for name in row}) == 80,
                "source steps/ranks must not repeat a feature set", path="helios.inputs")

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict) and set(value) == {"seed", "samples"}
                and isinstance(value["samples"], list)
                and all(isinstance(row, list) for row in value["samples"]),
                "explicit seed and five sample sets required", path="helios.inputs")
        return cls(value["seed"], tuple(tuple(row) for row in value["samples"]))

    def to_dict(self):
        return {"seed": self.seed, "samples": [[str(Path(name).resolve()) for name in row] for row in self.samples]}


def source_fingerprint():
    # This existing code-inventory helper is model-independent, including all
    # current runtime and original Helios source bytes, not a Wan model pin.
    from distillschedule.strategies.wan_planning import source_fingerprint as inventory
    return inventory()


def _inventory(task, sources):
    from distillschedule.strategies.wan_inputs import load_feature
    rows = []
    for offset, names in enumerate(sources.samples):
        for rank, name in enumerate(names):
            path = Path(name).resolve()
            require(path.is_file() and path.suffix in {".pt", ".safetensors"},
                    "local encoded feature required", path="helios.inputs")
            row = {"rank": rank, "step": task.source_steps[offset], "path": str(path), "sha256": file_sha256(path)}
            # Same original vae_latent/prompt_embed/attention_mask format as
            # BucketedFeatureDataset, not Wan model construction or loss.
            latent, text, _ = load_feature(task, row)
            row.update(latent_shape=list(latent.shape), text_shape=list(text.shape))
            rows.append(row)
    require(len({tuple(row["text_shape"]) for row in rows}) == 1,
            "rank/step conditioning geometry must match", path="helios.inputs")
    return rows


def _require_fresh_cohorts(sources, inventories=None):
    # Seeds control native sampling/augmentation, never feature identity.
    paths = {str(Path(name).resolve()) for source in sources for row in source.samples for name in row}
    require(len(paths) == 80 * len(sources),
            "globally distinct feature paths required across cohorts; changing seed is not fresh input",
            path="helios.inputs")
    if inventories is not None:
        hashes = [row["sha256"] for rows in inventories for row in rows]
        require(len(set(hashes)) == len(hashes) == 80 * len(sources),
                "globally distinct feature payload hashes required; copied bytes are not fresh input",
                path="helios.inputs")


def _keys(step, phase):
    from runtime.static_world16_trace import _generator_keys, _critic_keys
    keys = _generator_keys(step, gan=False) if phase == "generator" else _critic_keys(step, gan=False)
    # Unlike the historical sample/0 trace, the real trainer consumes a new
    # batch each step. Conditioning belongs to that exact step's source file.
    return {**keys, "text_embedding": f"step/{step}/sample/text_embedding",
            "attention_mask": f"step/{step}/sample/attention_mask"}


def capture_rank(task, sources, inventory, config, *, rank, output, provenance):
    """One rank's actual CPU-native producer, shared by preparation and tests."""
    import torch
    from runtime.operation_trace import OperationTraceWriter
    from runtime.official_baseline import stable_block_noise
    from distillschedule.strategies.wan_capture import native_primitives, input_stream, prepare_history, stage_control
    from distillschedule.strategies.wan_inputs import load_feature
    require(type(rank) is int and 0 <= rank < 16, "invalid rank", path="helios.inputs")
    require(not Path(output).exists(), "existing/incomplete rank output is preserved", path="helios.inputs")
    native = native_primitives()
    rows = [row for row in inventory if row["rank"] == rank]
    require([row["step"] for row in rows] == list(task.source_steps),
            "rank source steps missing/reordered", path="helios.inputs")
    writer = OperationTraceWriter(classification="NEW_ORIGINAL_HELIOS_STEP_QUALIFIED_INPUTS",
        source={**provenance, "rank": rank, "samples": rows, "seed": sources.seed, "cohort": task.index,
                "rng_scope": "new_native_CPU_streams_not_prior_training_rng_continuation"})
    dataset = native.dataset.__new__(native.dataset)
    dataset.base_seed, dataset._epoch = sources.seed, task.index
    for offset, row in enumerate(rows):
        step = row["step"]
        latent, text, mask = load_feature(task, row)
        primary, secondary = dataset.prepare_stage1_latent(latent, rank * 5 + offset, return_secondary=True)
        writer.add_tensor(f"step/{step}/sample/text_embedding", text.to(torch.bfloat16), role="text_embedding")
        writer.add_tensor(f"step/{step}/sample/attention_mask", mask, role="attention_mask")
        writer.add_operation(f"step/{step}/branch", component="training", operation="stage3_branch", control={
            "global_step": step, "dfake_gen_update_ratio": 5, "train_generator": offset == 0,
            "train_critic": True, "gan": False, "reward": False, "vae_decode": False})
        for phase in (("generator", "critic") if offset == 0 else ("critic",)):
            prefix = f"step/{step}/{phase}"
            with input_stream(sources.seed, task.index, rank, step, phase, "history"):
                tensors = prepare_history(native, task, primary if phase == "generator" else secondary, config)
            for stream in ("entry_noise", "rollout_noise", "dmd_noise" if phase == "generator" else "critic_noise"):
                with input_stream(sources.seed, task.index, rank, step, phase, stream):
                    tensors[stream] = torch.randn(task.latent_shape, dtype=torch.bfloat16)
            for stage in (1, 2):
                h, w = task.stage_spatial_shapes[stage]
                with input_stream(sources.seed, task.index, rank, step, phase, "block", stage):
                    noise = stable_block_noise(16 * task.latent_shape[2] * (h // 2) * (w // 2))
                tensors[f"block_noise/{stage}"] = noise.view(1,16,task.latent_shape[2],h//2,w//2,2,2).permute(
                    0,1,2,3,5,4,6).reshape(1,16,task.latent_shape[2],h,w).to(torch.bfloat16)
            with input_stream(sources.seed, task.index, rank, step, phase, "score-time"):
                _, mu = native.base.apply_schedule_shift(torch.tensor([.5], dtype=torch.float64),
                    torch.empty(task.latent_shape, device="meta"), return_mu=True)
                tensors["dmd_timestep" if phase == "generator" else "critic_timestep"] = native.post.sample_dynamic_timestep(
                    B=1, timestep_shift=mu, dynamic_alpha=config[phase + "_dynamic_alpha"],
                    dynamic_beta=config[phase + "_dynamic_beta"], dynamic_sample_type=config[phase + "_dynamic_sample_type"],
                    global_step=step, dynamic_step=config[phase + "_dynamic_step"], device="cpu")
            for name, tensor in tensors.items():
                writer.add_tensor(prefix + "/" + name, tensor, role=name)
            keys = list(_keys(step, phase).values())
            options = history_control_options(task, zero=all(bool(tensors[name].sum() == 0)
                for name in ("history_short", "history_mid", "history_long")), rank=rank)
            control = stage_control(native, task, sources.seed, task.index, step, phase, **options)
            writer.add_operation(prefix + "/rollout", component=phase, operation="native_pyramid_rollout",
                                 tensor_keys=keys, control=control)
            writer.add_operation(prefix + ("/dmd" if phase == "generator" else "/train"), component=phase,
                                 operation="native_score_inputs", tensor_keys=keys, control={"global_step": step})
    return writer.write(output, contract={TRACE_KEY: task.to_dict(), "resolution": [384,640],
        "full_latent_shape": list(task.latent_shape), "stage_spatial_shapes": [list(s) for s in task.stage_spatial_shapes],
        "dtype": "bfloat16", "dfake_gen_update_ratio": 5})


def prepare_stream(request, sources, training_config, output, *, input_policy=None, native_policy=None):
    """Immutable explicit finite stream. Failed/incomplete output is preserved."""
    from runtime.operation_trace import write_operation_trace_bundle_manifest
    from distillschedule.strategies.wan_capture import read_training_config
    from distillschedule.data.helios_epoch_inputs import build_cohorts, cohort_identity
    from functools import partial
    task_type = partial(HeliosCohortTask, request=request, native_policy=native_policy)
    task_type(0)  # Reject an invalid policy before reading inputs or creating output.
    if input_policy is None:
        require(isinstance(sources, (list, tuple)) and 2 <= len(sources) <= MAX_COHORTS
                and all(type(s) is HeliosCohortSources for s in sources), f"2..{MAX_COHORTS} typed cohorts required", path="helios.inputs")
        tasks = [task_type(i) for i in range(len(sources))]
        _require_fresh_cohorts(sources)
        config = read_training_config(training_config)
        inventories = [_inventory(t, s) for t, s in zip(tasks, sources, strict=True)]
        _require_fresh_cohorts(sources, inventories)
        epoch_policy = None
    else:
        # The task validator still owns the exact native W2 request.
        sources, inventories, epoch_policy = build_cohorts(sources, task_type=task_type, input_policy=input_policy)
        tasks = [task_type(i) for i in range(len(sources))]
        config = read_training_config(training_config)
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
            trace = capture_rank(task, source, inventory, config, rank=rank, output=directory/f"rank_{rank:02d}",
                provenance={**{k: identity[k] for k in ("training_config_sha256", "source_fingerprint")},
                            **cohort_identity(identity, task.index)})
            ranks.append({"rank": rank, "path": f"rank_{rank:02d}", "trace_hash": trace.trace_hash})
        bundle = write_operation_trace_bundle_manifest(directory, ranks)
        entries.append({"index": task.index, "task": task.to_dict(), "trace_bundle": str(directory),
                        "trace_hash": bundle["bundle_hash"], **cohort_identity(identity, task.index)})
    require(file_sha256(Path(training_config)) == identity["training_config_sha256"]
            and source_fingerprint() == identity["source_fingerprint"]
            and all(file_sha256(Path(row["path"])) == row["sha256"] for rows in inventories for row in rows),
            "inputs/source changed during capture", path="helios.inputs")
    document = {"schema_version": NATIVE_SCHEMA if native_policy is not None else SCHEMA,
                "status": "READY_FRESH_COHORT_STREAM", "identity": identity,
                "cohorts": entries, "gpu_execution": "NOT_RUN", "claimable_speedup": False}
    document["stream_sha256"] = fingerprint(document)
    write_new_json(root / "stream.json", document)
    return document


def load_stream(path, *, cycles, verify_rank=None, expected_kind=KIND):
    """Source-bound metadata admission; local tensor admission is worker-owned."""
    path = Path(path).resolve(); doc = json.loads(path.read_text()); unsigned = dict(doc)
    sha = unsigned.pop("stream_sha256", None)
    require(expected_kind in (KIND, NATIVE_KIND), "unsupported W2 input kind", path="helios.inputs")
    require(doc.get("schema_version") == "distillschedule." + expected_kind and doc.get("status") == "READY_FRESH_COHORT_STREAM"
            and sha == fingerprint(unsigned), "stream identity changed/incomplete", path="helios.inputs")
    identity = doc["identity"]; request = OriginalHeliosRequest.from_dict(identity["request"])
    native_policy = identity.get("native_history_policy")
    require((expected_kind == NATIVE_KIND and native_policy == NATIVE_HISTORY_POLICY)
            or (expected_kind == KIND and "native_history_policy" not in identity),
            "stream kind/native policy mismatch", path="helios.inputs")
    from functools import partial
    task_type = partial(HeliosCohortTask, request=request, native_policy=native_policy)
    sources = [HeliosCohortSources.from_dict(s) for s in identity["sources"]]
    require(type(cycles) is int and 2 <= cycles <= MAX_COHORTS
            and len(doc["cohorts"]) == len(sources) == len(identity["inventory"]) == cycles
            and identity["source_fingerprint"] == source_fingerprint(), "stream cycles/source changed", path="helios.inputs")
    from distillschedule.data.helios_epoch_inputs import validate_identity, cohort_identity
    if "input_policy" in identity:
        validate_identity(identity, task_type=task_type)
    else:
        _require_fresh_cohorts(sources, identity["inventory"])
    for index, entry in enumerate(doc["cohorts"]):
        task = task_type(index)
        require(entry["index"] == index and entry["task"] == task.to_dict()
                and Path(entry["trace_bundle"]).resolve() == path.parent / f"cohort-{index:04d}",
                "cohort order/path/task changed", path="helios.inputs")
        expected_policy = cohort_identity(identity, index)
        require({k: entry[k] for k in ("input_policy",) if k in entry} == expected_policy,
                "cohort policy/draw identity changed", path="helios.inputs")
        trace = load_cohort(task, entry["trace_bundle"], expected_hash=entry["trace_hash"])
        rows = identity["inventory"][index]
        require(len(rows) == 80 and [(r["step"], r["rank"]) for r in rows] ==
                [(s, r) for s in task.source_steps for r in range(16)], "incomplete sample inventory", path="helios.inputs")
        for rank, control in trace._control.items():
            source = control.manifest["source"]
            require(source == {"source_fingerprint": identity["source_fingerprint"],
                "training_config_sha256": identity["training_config_sha256"], "rank": rank,
                "samples": [r for r in rows if r["rank"] == rank], "seed": sources[index].seed, "cohort": index,
                "rng_scope": "new_native_CPU_streams_not_prior_training_rng_continuation", **expected_policy},
                "rank sample/RNG provenance differs", path="helios.inputs")
        require([r["path"] for r in rows] == [name for row in sources[index].to_dict()["samples"] for name in row],
                "sample inventory differs from explicit sources", path="helios.inputs")
        if verify_rank is not None:
            validate_rank_payload(trace, verify_rank, sources[index])
    return doc


def validate_rank_payload(trace, rank, source):
    """Hash-verified CPU payloads and exact original normal-path stage grids."""
    import torch
    from distillschedule.strategies.wan_capture import native_primitives, stage_control
    value = trace._load(rank)
    require(all(bool(torch.isfinite(t).all()) for t in value.tensors.values() if t.is_floating_point()),
            "non-finite original input payload", path="helios.inputs")
    native = native_primitives()
    for step in trace.logical_steps:
        for phase in (("generator", "critic") if step == trace.logical_steps[0] else ("critic",)):
            prefix = f"step/{step}/{phase}"
            timestep = value.tensor(prefix + ("/dmd_timestep" if phase == "generator" else "/critic_timestep"))
            require(bool(((timestep >= 20) & (timestep <= 980)).all()),
                    "original score timestep outside native clamp", path="helios.inputs")
            zero = all(bool(value.tensor(prefix+"/"+name).sum() == 0)
                       for name in ("history_short", "history_mid", "history_long"))
            options = history_control_options(trace.task, zero=zero, rank=rank)
            expected = stage_control(native, trace.task, source.seed, trace.task.index, step, phase, **options)
            require(value.operation(prefix + "/rollout")["control"] == expected,
                    "original stage grid/RNG exits changed", path="helios.inputs")
        mask = value.tensor(f"step/{step}/sample/attention_mask")
        require(bool(((mask == 0) | (mask == 1)).all()), "invalid native conditioning mask", path="helios.inputs")


def load_cohort(task, root, *, expected_hash):
    from runtime.operation_trace import OperationTrace, load_operation_trace_bundle_manifest, resolve_rank_trace_root
    from runtime.static_world16_trace import StaticWorld16Trace
    root = Path(root).resolve()
    bundle = load_operation_trace_bundle_manifest(root, verify=True)
    require(bundle is not None and bundle["bundle_hash"] == expected_hash and len(bundle["ranks"]) == 16,
            "cohort rank/bundle identity changed", path="helios.inputs")
    controls = {}
    for rank in range(16):
        directory = resolve_rank_trace_root(root, rank).resolve()
        require(root in directory.parents, "rank path escaped cohort", path="helios.inputs")
        controls[rank] = OperationTrace(root=directory, manifest=json.loads((directory/"trace.json").read_text()), tensors={})
    rank_validation = validate_controls(task, controls)
    obj = HeliosTrainingTrace.__new__(HeliosTrainingTrace)
    obj.task, obj.root, obj.variant, obj.gan, obj.request = task, root, "no-gan", False, None
    obj.logical_steps, obj.bundle_hash = task.source_steps, expected_hash
    obj.rank_control_validation = rank_validation
    obj.binding_hash = fingerprint({"schema_version":"distillschedule.helios_training_trace_binding.v1",
        "task":task.to_dict(), "bundle_hash":expected_hash, "rank_slots":list(range(16))})
    obj._control, obj._loaded = controls, {}
    # Unchanged legacy geometry/branch/rank validator, on the explicitly NEW
    # cohort object. StaticWorld16Trace.__init__ and its fixed ABI stay intact.
    if task.native_policy is None:
        StaticWorld16Trace._validate(obj)
    return obj


def validate_controls(task, controls):
    from runtime.operation_trace import validate_official_rank_stage_schedule_semantics
    require(set(controls) == set(range(16)), "complete rank16 cohort required", path="helios.inputs")
    conditioning = set()
    for value in controls.values():
        contract = value.manifest["contract"]
        require(contract.get(TRACE_KEY) == task.to_dict()
                and "distillschedule_original_wan_task" not in contract
                and "distillschedule_original_helios_request" not in contract,
                "new Helios cohort required; old traces cannot be relabelled", path="helios.inputs")
        if task.native_policy is not None:
            from distillschedule.runtime.helios_native_history import validate_contract
            validate_contract(task, value)
        keys = [op["key"] for op in value.manifest["operations"]]
        expected = [f"step/{s}/branch" for s in task.source_steps]
        for step in task.source_steps:
            for phase in (("generator", "critic") if step == task.first_step else ("critic",)):
                expected.extend((f"step/{step}/{phase}/rollout", f"step/{step}/{phase}/" +
                                 ("dmd" if phase == "generator" else "train")))
                for role, key in _keys(step, phase).items():
                    descriptor = value.manifest["tensors"][key]
                    if role == "text_embedding":
                        shape = descriptor["shape"]
                        require(len(shape) == 2 and type(shape[0]) is int and shape[0] > 0 and shape[1] == 4096
                                and descriptor["dtype"] == "bfloat16", "original conditioning shape/dtype changed",
                                path="helios.inputs")
                        conditioning.add(tuple(shape))
                    elif role == "attention_mask":
                        text = value.manifest["tensors"][_keys(step, phase)["text_embedding"]]
                        require(descriptor["shape"] == text["shape"][:1],
                                "original conditioning mask shape changed", path="helios.inputs")
                    else:
                        shape = list(task.latent_shape)
                        dtype = "bfloat16"
                        if role.startswith("indices_"):
                            shape = [1, {"indices_hidden_states":9, "indices_history_short":2,
                                         "indices_history_mid":2, "indices_history_long":16}[role]]
                            dtype = "int64"
                        elif role.endswith("_timestep"):
                            shape, dtype = [1], "int64"
                        elif role == "block_noise_stage1":
                            shape[-2:] = [24,40]
                        elif role in ("history_short", "history_mid", "history_long", "continuous_history"):
                            shape[2] = {"history_short":2, "history_mid":2, "history_long":16, "continuous_history":19}[role]
                        require(descriptor["shape"] == shape and descriptor["dtype"] == dtype,
                                f"native {role} shape/precision changed", path="helios.inputs")
                schedule = value.operation(f"step/{step}/{phase}/rollout")["control"]["pyramid_stage_call_sequence"]
                control = value.operation(f"step/{step}/{phase}/rollout")["control"]
                budget = 4 if (task.native_policy is not None or task.stage_schedule_policy == ALIGNED_SCHEDULE_POLICY) \
                    and control.get("all_history_zero") is True else 2
                if task.stage_schedule_policy == ALIGNED_SCHEDULE_POLICY:
                    require(control.get("stage_schedule_policy") == ALIGNED_SCHEDULE_POLICY
                            and type(control.get("all_history_zero")) is bool,
                            "aligned history/schedule scope missing", path="helios.inputs")
                require(len(schedule) == 3 and [s["stage"] for s in schedule] == [0,1,2]
                        and all(type(s["call_count"]) is int and 1 <= s["call_count"] <= (2 if task.native_policy is None else budget)
                                and s["denoising_step_budget"] == budget for s in schedule),
                        "native222 stage scope changed", path="helios.inputs")
        require(len(keys) == len(set(keys)) == len(expected) and set(keys) == set(expected),
                "another cycle/branch/work inventory", path="helios.inputs")
    require(len(conditioning) == 1, "rank/step conditioning geometry differs", path="helios.inputs")
    if task.native_policy is not None:
        from distillschedule.runtime.helios_native_history import validate_schedules
        return validate_schedules(task, controls)
    return validate_official_rank_stage_schedule_semantics(controls)


class HeliosTrainingTrace:
    """Original entry types/loaders, with genuinely step-qualified conditioning."""
    def _load(self, rank):
        from runtime.static_world16_trace import StaticWorld16Trace
        return StaticWorld16Trace._load(self, rank)

    def _physical_signature(self, operation):
        from runtime.static_world16_trace import StaticWorld16Trace
        return StaticWorld16Trace._physical_signature(operation)

    def _entry(self, rank, step, phase, device):
        from runtime.static_world16_trace import StaticWorld16Trace, _descriptor_hashes
        trace = self._load(rank); keys = _keys(step, phase)
        tensors = StaticWorld16Trace._tensors(self, trace, keys, device)
        identity = fingerprint({"task": self.task.to_dict(), "phase": phase, "rank": rank, "step": step,
                                "trace_hash": trace.trace_hash, "tensors": _descriptor_hashes(trace, keys)})
        return trace, tensors, identity, keys

    def load_generator_entry(self, *, rank, device):
        from runtime.static_world16_trace import StaticGeneratorEntry
        step = self.logical_steps[0]
        trace, tensors, identity, _ = self._entry(rank, step, "generator", device)
        return StaticGeneratorEntry(rank, step, trace, tensors, identity)

    def load_critic_entry(self, *, rank, step, device):
        from runtime.critic_pool_helios import LoadedPoolEntry
        require(step in self.logical_steps, "critic step outside cohort", path="helios.inputs")
        trace, tensors, identity, keys = self._entry(rank, step, "critic", device)
        entry = SimpleNamespace(source_trace_rank=rank, logical_step=step, identity_sha256=identity,
            trace_key=f"helios-training/{self.task.index}/step{step}/rank{rank}",
            rollout_operation=f"step/{step}/critic/rollout", train_operation=f"step/{step}/critic/train",
            logical_sample_slot=rank, update=self.logical_steps.index(step), production_round=self.task.index,
            producer_pool_rank=rank, rng_stream=f"helios/{self.task.index}/{rank}/{step}",
            trace_path=str(trace.root), operation_trace_hash=trace.trace_hash, tensor_keys=keys)
        return LoadedPoolEntry(entry, trace, tensors)
