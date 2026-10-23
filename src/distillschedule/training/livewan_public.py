"""Opt-in CPU1 FP32 original Self-Forcing, using the native complete G1/C5 loop.

This is a separate 32px spatial reference contract, not native368x640/B16,
mixed precision, C51, or pretrained quality. Export projects only current G.
The original worker owns every rollout/loss/recompute/clip/Adam commit.
"""
from __future__ import annotations

from copy import copy, deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from .adapters import ModelComponents
from .checkpoint import cpu_tree, rng_state, restore_rng, validate_rng

ASSETS = "distillschedule.livewan_cpu1_fp32_assets.v1"
OWNER = "distillschedule.livewan_cpu1_fp32_cycle_owner.v1"
SCOPE = "CPU1_FP32_ORIGINAL_SELF_FORCING_32PX_NOT_NATIVE_WORLD16"
PRECISION = "CPU_FP32_NO_AUTOCAST"
BACKEND = "torch-sdpa-cpu"
FIXTURE = "FULL_ORIGINAL_TINY_CPU_FIXTURE_NOT_PRETRAINED_OR_QUALITY"
EVIDENCE_KINDS = (FIXTURE, "LOCAL_ORIGINAL_WAN21_ASSETS_UNVERIFIED_QUALITY")
MODEL = "distillschedule.training.livewan_public:build_model"
ALGORITHM = "distillschedule.training.livewan_public:build_algorithm"
GEOMETRY = dict(contract="livewan_cpu1_32px_f81_latent21_v1", resolution=[32, 32], frames=81,
                latent_shape=[1, 16, 21, 4, 4], text_shape=[1, 512, 4096], world_size=1,
                global_batch=1, world_choices=[3, 6, 9], block_frames=3, steps_per_block=4,
                grad_blocks=3, cache_frames=22, max_frames=278)


def require(ok, message):
    if not ok:
        raise ValueError("LiveWan CPU1 FP32 reference: " + message)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)
    fingerprint(value)
    return value


def local_file(value):
    require(isinstance(value, str) and "://" not in value and Path(value).is_absolute(), "absolute local file required")
    path = Path(value)
    require(path.resolve() == path and path.is_file(), "missing/noncanonical/symlink source: " + value)
    return path


def pin_file(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha256(path))


def checked_pin(value):
    require(isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}, "exact file pin required")
    path = local_file(value["path"])
    require(type(value["bytes"]) is int and 0 < value["bytes"] <= 512_000_000
            and path.stat().st_size == value["bytes"] and file_sha256(path) == value["sha256"],
            "source bytes changed: " + str(path))
    return path


def tensor_content(value):
    """Stable content including dtype/shape and sequence types, never pickle bytes."""
    import torch
    if isinstance(value, torch.Tensor):
        require(type(value) is torch.Tensor or isinstance(value, torch.nn.Parameter), "plain CPU tensor required")
        require(value.device.type == "cpu" and value.layout == torch.strided, "dense CPU tensor required")
        require(not value.is_floating_point() or bool(value.isfinite().all()), "nonfinite state")
        tensor = value.detach().contiguous()
        return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                    sha256=hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(k): tensor_content(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return dict(sequence=type(value).__name__, items=[tensor_content(v) for v in value])
    require(value is None or type(value) in (str, bool, int, float), "unsupported state identity leaf")
    return dict(type=type(value).__name__, value=value)


def content_digest(value):
    return fingerprint(tensor_content(value))


def topology():
    import torch
    require(all(int(os.environ.get(k, default)) == expected for k, default, expected in
                (("WORLD_SIZE", "1", 1), ("LOCAL_WORLD_SIZE", "1", 1), ("RANK", "0", 0), ("LOCAL_RANK", "0", 0))),
            "CPU1 world1/rank0 only; no native world16 substitution")
    require(not torch.distributed.is_initialized() and not torch.cuda.is_initialized(), "no distributed group or CUDA initialization")
    require(torch.get_num_threads() == 1 and torch.get_default_dtype() == torch.float32
            and not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
            "single-thread explicit FP32/no-autocast required")


def runtime_identity():
    result = dict(python=platform.python_version())
    for name in ("torch", "diffusers", "safetensors", "numpy"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        result[name] = dict(version=importlib.metadata.version(name), imported_version=str(module.__version__),
                            path=str(path), entry_sha256=file_sha256(path))
    import torch
    result["policy"] = dict(threads=torch.get_num_threads(), default_dtype=str(torch.get_default_dtype()),
                            precision=PRECISION, backend=BACKEND)
    return result


@dataclass(frozen=True)
class CPUReferenceRequest:
    """Not a native LiveWanRequest: same temporal math, explicit32px CPU source."""
    @property
    def latent_shape(self):
        return tuple(GEOMETRY["latent_shape"])

    def new_blocks(self, world_frames):
        require(type(world_frames) is int and world_frames in (3, 6, 9), "original world3/6/9 required")
        return (21 - world_frames) // 3


@dataclass(frozen=True)
class LiveWanCPUContext:
    """Per-owner explicit provider/precision/geometry dispatch; no global aliases."""
    api: object
    source: dict

    @property
    def latent_shape(self):
        return tuple(GEOMETRY["latent_shape"])

    def validate(self):
        from distillschedule.reference import livewan_wan21 as provider
        from distillschedule.runtime import livewan_crossscale_worker as worker
        topology()
        require(worker._operation_calibrator is None, "CPU fixture cannot enter native operation calibration")
        require(self.source == provider.source_identity()
                and self.api is provider.load_cpu(backend=BACKEND, precision="float32"), "original provider/source changed")

    def validate_tensor(self, tensor):
        import torch
        require(type(tensor) is torch.Tensor and tensor.device.type == "cpu" and tensor.layout == torch.strided
                and tensor.dtype == torch.float32 and bool(tensor.isfinite().all()), "finite dense FP32 CPU tensor required")

    def validate_latent(self, tensor):
        self.validate_tensor(tensor)
        require(tuple(tensor.shape) == self.latent_shape, "CPU32px/81frame latent21 geometry changed; no native crop")

    def validate_streamer(self, streamer, student):
        import torch
        require(type(streamer) is self.api.stream.FewStepStreamer and streamer.model is student
                and str(streamer.device) == "cpu" and streamer.dtype == torch.float32
                and streamer.graphs is None and streamer.block_frames == 3 and streamer.S == 4
                and streamer.max_frames == 278 and streamer.schedule == [1., .75, .5, .25]
                and streamer.sampler == "renoise" and not streamer.noisy_context and streamer.latent_norm == 0.
                and streamer.window_frames is None and streamer.time_scale == 1000.
                and streamer.cache.max_tokens == 22 * 4, "original CPU streamer policy/ownership changed")

    def build_streamers(self, *, student, count, device):
        import torch
        require(count == 1 and str(device) == "cpu", "one owned CPU baseline streamer only")
        _, _, hp, wp = self.api.core.latent_geometry(32, 32)
        rope = self.api.core.make_rope_table(student, hp, wp, 278, device)
        result = [self.api.stream.FewStepStreamer(student, width=32, height=32, max_frames=278,
            device=device, dtype=torch.float32, window_frames=None, time_scale=1000., cache_frames=22,
            block_frames=3, num_steps=4, rope=rope, cuda_graphs=False)]
        self.validate_streamer(result[0], student)
        return result

    def build_score_context(self, *, bundle, loader, device):
        import torch
        from distillschedule.runtime.livewan_crossscale_worker import ScoreContext
        require(str(device) == "cpu" and bundle.critic is not None and bundle.teacher is not None,
                "CPU score requires actual Critic/Teacher")
        _, _, hp, wp = self.api.core.latent_geometry(32, 32)
        rope = self.api.core.make_rope_table(bundle.critic, hp, wp, 21 + 8, device)
        with torch.no_grad():
            negative = bundle.teacher.text_embedding(loader.negative.to(device).to(torch.float32))
        return ScoreContext(rope=rope, negative_teacher_context=negative, reference_context=self)


def _json_config_value(value):
    return list(value) if isinstance(value, tuple) else value


def _architecture(value):
    fields = {"model_type", "patch_size", "text_len", "in_dim", "dim", "ffn_dim", "freq_dim", "text_dim",
              "out_dim", "num_heads", "num_layers", "window_size", "qk_norm", "cross_attn_norm", "eps"}
    require(isinstance(value, dict) and set(value) == fields, "complete explicit original WanModel architecture required")
    require(value["model_type"] == "t2v" and value["patch_size"] == [1, 2, 2] and value["text_len"] == 512
            and value["in_dim"] == value["out_dim"] == 16 and value["text_dim"] == 4096
            and value["window_size"] == [-1, -1] and value["qk_norm"] is True
            and value["cross_attn_norm"] is True and type(value["eps"]) is float and value["eps"] == 1e-6,
            "original t2v conditioning/channel/patch/normalization policy changed")
    for key, limit in (("dim", 1024), ("ffn_dim", 4096), ("freq_dim", 256), ("num_heads", 64), ("num_layers", 4)):
        require(type(value[key]) is int and 0 < value[key] <= limit, "CPU architecture bound: " + key)
    require(value["dim"] % value["num_heads"] == 0 and value["dim"] // value["num_heads"] == 16
            and value["freq_dim"] % 2 == 0, "matching original head16/even time embeddings required")
    return value


def _safe_shapes(pin, *, evidence_kind, keys=None):
    from safetensors import safe_open
    path = checked_pin(pin)
    require(path.suffix == ".safetensors", "safetensors assets required; no unrestricted pickle")
    with safe_open(path, framework="pt", device="cpu") as handle:
        require((handle.metadata() or {}).get("evidence_kind") == evidence_kind, "safe payload evidence label changed")
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
        require(shapes and (keys is None or set(shapes) == set(keys)), "safe asset fields changed")
        require(all(handle.get_slice(key).get_dtype() == "F32" for key in shapes), "all-FP32 source assets required; no cast")
    checked_pin(pin)
    return shapes


def admit_assets(parameters):
    """Headers, configs and construction/attention budgets before real allocation."""
    from distillschedule.reference import livewan_wan21 as provider
    import torch
    require(isinstance(parameters, dict) and set(parameters) <= {"asset_manifest", "seed", "device"}
            and parameters.get("device", "cpu") == "cpu" and type(parameters.get("seed", 0)) is int,
            "explicit local assets/CPU only; no model or numerical overrides")
    topology()
    path = local_file(parameters.get("asset_manifest"))
    assets = read_json(path)
    require(isinstance(assets, dict) and set(assets) == {"schema_version", "scope", "precision", "backend", "evidence_kind",
        "source", "runtime", "geometry", "student", "teacher", "critic", "negative", "lora_seed", "budget"}, "asset fields changed")
    require(assets["schema_version"] == ASSETS and assets["scope"] == SCOPE and assets["precision"] == PRECISION
            and assets["backend"] == BACKEND and assets["evidence_kind"] in EVIDENCE_KINDS
            and assets["geometry"] == GEOMETRY and assets["source"] == provider.source_identity(),
            "CPU source/scope/precision/geometry changed")
    require(assets["runtime"] == runtime_identity(), "actual dependency runtime changed")
    require(type(assets["lora_seed"]) is int and 0 <= assets["lora_seed"] < 2**63, "original critic LoRA seed required")
    limits = dict(max_parameters=10_000_000, max_loaded_bytes=512_000_000, max_cycles=64, max_attention_elements=8_000_000)
    budget = assets["budget"]
    require(isinstance(budget, dict) and set(budget) == set(limits)
            and all(type(budget[k]) is int and 0 < budget[k] <= v for k, v in limits.items()), "explicit bounded CPU budget required")
    api = provider.load_cpu(backend=BACKEND, precision="float32")
    specs, count, loaded, paths = {}, 0, 0, set()
    for role in ("student", "teacher", "critic"):
        spec = assets[role]
        require(isinstance(spec, dict) and set(spec) == {"config", "weights"}
                and isinstance(spec["weights"], list) and spec["weights"], "complete named role config/safe weights required")
        architecture = _architecture(read_json(checked_pin(spec["config"])))
        shapes = {}
        for pin in spec["weights"]:
            actual = str(checked_pin(pin))
            require(actual not in paths, "role weights must have explicit independent pins")
            paths.add(actual)
            current = _safe_shapes(pin, evidence_kind=assets["evidence_kind"])
            require(not shapes.keys() & current.keys(), "duplicate full role state keys")
            shapes.update(current)
            loaded += pin["bytes"]
        count += sum(math.prod(shape) for shape in shapes.values())
        require(count <= budget["max_parameters"] and loaded <= budget["max_loaded_bytes"], "full models exceed declared CPU budget")
        # Upper bound from the declared full architecture also closes malicious
        # tiny-header/huge-config construction before entering even a meta model.
        d, f, n = architecture["dim"], architecture["ffn_dim"], architecture["num_layers"]
        require(4096*d + n*(12*d*d + 2*d*f) <= budget["max_parameters"], "architecture construction bound exceeded")
        require(2 * architecture["num_heads"] * (84**2 + 84*512) <= budget["max_attention_elements"],
                "CPU full-clip self/cross attention budget exceeded")
        outer = rng_state()
        try:
            with torch.device("meta"):
                meta = api.model.WanModel(**architecture)
            expected = {key: tuple(t.shape) for key, t in meta.state_dict().items()}
        finally:
            restore_rng(outer)
        require(shapes == expected, "complete original role tensor layout differs from frozen config: " + role)
        specs[role] = architecture
    # Include ORIGINAL rank32 LoRA parameters in the construction budget.
    d, f, n = (specs["critic"][key] for key in ("dim", "ffn_dim", "num_layers"))
    require(count + n * 32 * (8*d + 2*(d+f)) <= budget["max_parameters"], "original LoRA32 capacity exceeded")
    negative = _safe_shapes(assets["negative"], evidence_kind=assets["evidence_kind"], keys={"negative"})
    require(negative["negative"] == (1, 512, 4096)
            and loaded + assets["negative"]["bytes"] <= budget["max_loaded_bytes"], "negative prompt shape/payload budget changed")
    require(specs["student"] == specs["critic"] and specs["teacher"]["dim"] >= specs["student"]["dim"],
            "Student/Critic architecture and compatible Teacher head dimension required")
    return path, assets, specs, api


def _load_role(api, spec, architecture):
    from safetensors.torch import load_file
    model = api.model.WanModel(**architecture)
    state = {}
    for pin in spec["weights"]:
        state.update(load_file(str(checked_pin(pin)), device="cpu"))
        checked_pin(pin)
    require(all(bool(t.isfinite().all()) for t in state.values()), "nonfinite full role state")
    model.load_state_dict(state, strict=True)
    require(bool(model.head.head.weight.count_nonzero()), "nonzero supplied output head required; cold-zero fixture is vacuous")
    return model


class LiveWanCPUOwner:
    """Actual full Student, Teacher, Critic-LoRA/Adam, clocks, source and RNG."""

    def __init__(self, *, student, teacher, critic, lora, assets, asset_pin, api, architecture):
        import torch
        from distillschedule.runtime import livewan_crossscale_worker as worker
        self.assets, self.asset_pin = deepcopy(assets), deepcopy(asset_pin)
        self.architecture = deepcopy(_architecture(architecture))
        self.context = LiveWanCPUContext(api=api, source=deepcopy(assets["source"]))
        self.models = dict(student=student, teacher=teacher, critic=critic)
        self.optimizers = dict(generator=torch.optim.AdamW(list(student.parameters()), lr=2e-6,
            betas=(0., .99), weight_decay=0., eps=1e-8, foreach=True),
            critic=torch.optim.AdamW(lora, lr=4e-5, betas=(0., .99), weight_decay=0., eps=1e-8, foreach=True))
        self.bundle = worker.ModelBundle(student=student, critic=critic, teacher=teacher, critic_shadow=None,
            generator_optimizer=self.optimizers["generator"], critic_optimizer=self.optimizers["critic"],
            generator_optimizer_parameter_names=worker._bind_trajectory_optimizer_parameter_names(
                named_parameters=list(student.named_parameters()), optimizer=self.optimizers["generator"]),
            critic_optimizer_parameter_names=worker._bind_trajectory_optimizer_parameter_names(
                named_parameters=list(critic.named_parameters()), optimizer=self.optimizers["critic"],
                expected_parameter_ids={id(p) for p in lora}), critic_lora_params=lora,
            critic_shadow_lora_params=[], critic_gradient_reducer=None, generator_group=None, score_group=None,
            critic_group=None, score_local_group=None, score_leader_group=None, runahead_host_group=None,
            critic_shadow_sync_group=None, score_local_ranks=(), score_leader_ranks=(), generator_group_size=1,
            score_group_size=1, critic_group_size=1, metadata=dict(scope=SCOPE), reference_context=self.context)
        self.cycle, self.versions = 0, dict(generator=0, critic=0, ema=0)
        self.consumed, self.last_metrics, self._input_rows = [], {}, None
        self.poisoned, self._busy = False, False
        self._rng = cpu_tree(rng_state())
        self._named = self._parameters()
        self._parameter_ids = {name: id(p) for name, p in self._named.items()}
        require(len(set(self._parameter_ids.values())) == len(self._parameter_ids), "parameter aliases unsupported")
        self._modules = {role + "." + name: module for role, model in self.models.items() for name, module in model.named_modules()}
        self._classes = {name: type(module) for name, module in self._modules.items()}
        self._modes = {name: module.training for name, module in self._modules.items()}
        self._flags = {name: p.requires_grad for name, p in self._named.items()}
        self._policy = self._module_policy()
        self._frozen = {name: tensor_content(p) for name, p in self._named.items() if not p.requires_grad}
        self._detached = {name: tensor_content(p) for name, p in self._named.items() if name.startswith("student.text_embedding.")}
        self._fixed_tensors = {role: tensor_content(model.freqs) for role, model in self.models.items()}
        self._groups = {role: self._optimizer_value(role)["groups"] for role in self.optimizers}
        self._defaults = {role: cpu_tree(opt.defaults) for role, opt in self.optimizers.items()}
        self._optimizer_ids = {role: id(opt) for role, opt in self.optimizers.items()}
        self._source_files = {str(Path(m.__file__).resolve()): file_sha256(Path(m.__file__).resolve()) for m in self._source_modules()}
        self._bindings = self._live_bindings()
        self._admit_source_bindings()
        self._closures = self._closure_values()
        self.identity = dict(schema=OWNER, scope=SCOPE, precision=PRECISION, geometry=deepcopy(GEOMETRY),
            asset_manifest=deepcopy(asset_pin), assets=deepcopy(assets), architecture=deepcopy(self.architecture), frozen=deepcopy(self._frozen),
            source_files=deepcopy(self._source_files), parameter_layout={n: dict(shape=list(p.shape), dtype=str(p.dtype),
                requires_grad=p.requires_grad) for n, p in self._named.items()}, optimizer_policy=content_digest((self._groups, self._defaults)),
            module_policy=content_digest(self._policy), modes=deepcopy(self._modes), detached_text=deepcopy(self._detached),
            fixed_tensors=deepcopy(self._fixed_tensors))
        self._identity_digest = fingerprint(self.identity)
        self._validate_live()

    @staticmethod
    def expected(cycle):
        return dict(generator=cycle, critic=5*cycle, ema=0)

    def _parameters(self):
        return {role + "." + name: p for role, model in self.models.items() for name, p in model.named_parameters(remove_duplicate=False)}

    @staticmethod
    def _source_modules():
        return [importlib.import_module(name) for name in (
            "distillschedule.training.livewan_public", "distillschedule.data.livewan_reference_inputs",
            "distillschedule.runtime.livewan_crossscale_worker", "distillschedule.training.checkpoint",
            "distillschedule.training.tensor_state")]

    @classmethod
    def _live_bindings(cls):
        import torch
        result = {}
        for module in cls._source_modules():
            for name, value in vars(module).items():
                if inspect.isfunction(value):
                    result[(module.__name__, name)] = (value, value.__code__)
                    body = value
                    while hasattr(body, "__wrapped__"):
                        body = body.__wrapped__
                        if inspect.isfunction(body):
                            result[(module.__name__, name, "body", len(result))] = (body, body.__code__)
                elif inspect.isclass(value) and value.__module__ == module.__name__:
                    result[(module.__name__, name)] = (value, None)
                    for key, member in vars(value).items():
                        if isinstance(member, (classmethod, staticmethod)):
                            member = member.__func__
                        if inspect.isfunction(member):
                            result[(module.__name__, name, key)] = (member, member.__code__)
                            body = member
                            while hasattr(body, "__wrapped__"):
                                body = body.__wrapped__
                                if inspect.isfunction(body):
                                    result[(module.__name__, name, key, "body", len(result))] = (body, body.__code__)
        for name, value in (("adam", torch.optim.AdamW.step), ("clip", torch.nn.utils.clip_grad_norm_),
                            ("module_call", torch.nn.Module._call_impl), ("backward", torch.Tensor.backward),
                            ("linear", torch.nn.functional.linear), ("sdpa", torch.nn.functional.scaled_dot_product_attention)):
            result[("torch", name)] = (value, getattr(value, "__code__", None))
        return result

    def _admit_source_bindings(self):
        """Refuse a pre-construction patched worker, not just later redirection."""
        import ast
        from types import CodeType
        compiled, definitions = {}, {}
        def walk(code):
            yield code
            for value in code.co_consts:
                if isinstance(value, CodeType):
                    yield from walk(value)
        for key, (function, code) in self._bindings.items():
            if code is None or key[0] == "torch":
                continue
            if len(key) >= 3:
                owner = getattr(importlib.import_module(key[0]), key[1], None)
                if hasattr(owner, "__dataclass_fields__") and key[2] in {
                    "__init__", "__repr__", "__eq__", "__hash__", "__setattr__", "__delattr__"}:
                    # Stdlib-generated structural methods have no source body
                    # in this module. Their captured identities still stay fixed.
                    continue
            module = importlib.import_module(function.__module__)
            path = Path(code.co_filename).resolve()
            require(path.is_file() and (hasattr(function, "__wrapped__") or path == Path(module.__file__).resolve()),
                    "foreign live source function: " + str(key))
            if path not in compiled:
                compiled[path] = list(walk(compile(path.read_bytes(), str(path), "exec", dont_inherit=True)))
            matches = [candidate for candidate in compiled[path] if candidate.co_name == code.co_name
                       and candidate.co_firstlineno == code.co_firstlineno]
            require(len(matches) == 1 and code == matches[0], "live function differs from admitted source: " + str(key))
            expected = (key[1] if len(key) == 2 or len(key) == 4 and key[2] == "body" else key[2])
            if key[0] not in definitions:
                owner_path = Path(importlib.import_module(key[0]).__file__).resolve()
                definitions[key[0]] = {n.name for n in ast.parse(owner_path.read_bytes()).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            owns_body = len(key) != 2 or key[1] in definitions[key[0]]
            require(not owns_body or function.__module__ == key[0] and function.__name__ == expected,
                    "redirected named numerical binding: " + str(key))

    def _closure_values(self):
        return {key: tuple((id(cell), id(cell.cell_contents), getattr(cell.cell_contents, "__code__", None))
                           for cell in getattr(fn, "__closure__", ()) or ())
                for key, (fn, _) in self._bindings.items()}

    def _module_policy(self):
        def scalar(value):
            return value is None or type(value) in (int, float, bool, str) or (
                isinstance(value, (list, tuple)) and all(scalar(v) for v in value))
        return {name: {k: deepcopy(v) for k, v in vars(module).items() if not k.startswith("_")
                      and k != "training" and scalar(v)} for name, module in self._modules.items()}

    def _check_identity(self):
        import torch
        self.context.validate()
        require(not self._busy and fingerprint(self.identity) == self._identity_digest, "busy or changed source/input owner identity")
        require(self._frozen == self.identity["frozen"] and self._detached == self.identity["detached_text"]
                and self._fixed_tensors == self.identity["fixed_tensors"]
                and content_digest((self._groups, self._defaults)) == self.identity["optimizer_policy"]
                and content_digest(self._policy) == self.identity["module_policy"] and self._modes == self.identity["modes"]
                and self._flags == {n: spec["requires_grad"] for n, spec in self.identity["parameter_layout"].items()},
                "stored owner/layout/options policy changed")
        require(self.assets == self.identity["assets"] and runtime_identity() == self.assets["runtime"], "asset/runtime identity changed")
        require(self.architecture == self.identity["architecture"]
                and all(_json_config_value(getattr(self.models["student"], k)) == v for k, v in self.architecture.items()),
                "admitted Student architecture contents changed")
        checked_pin(self.asset_pin)
        for role in ("student", "teacher", "critic"):
            for pin in (self.assets[role]["config"], *self.assets[role]["weights"]):
                checked_pin(pin)
            require(type(self.models[role]) is self.context.api.model.WanModel
                    and self._fixed_tensors[role] == tensor_content(self.models[role].freqs), "original model class/RoPE changed")
        checked_pin(self.assets["negative"])
        current = self._parameters()
        modules = {role + "." + name: m for role, model in self.models.items() for name, m in model.named_modules()}
        require({name: id(p) for name, p in current.items()} == self._parameter_ids
                and modules.keys() == self._modules.keys()
                and all(modules[n] is self._modules[n] and type(modules[n]) is self._classes[n] for n in modules),
                "model/parameter/class ownership changed")
        require(self._flags == {n: p.requires_grad for n, p in current.items()}
                and self._modes == {n: m.training for n, m in modules.items()}
                and content_digest(self._module_policy()) == content_digest(self._policy), "model flags/modes/LoRA options changed")
        require(all(p.device.type == "cpu" and p.dtype == torch.float32 and p.layout == torch.strided
                    and list(p.shape) == self.identity["parameter_layout"][n]["shape"] for n, p in current.items()),
                "explicit CPU FP32 parameter layout changed")
        require(self._bindings == self._live_bindings()
                and self._closures == self._closure_values()
                and all(code is None or getattr(fn, "__code__", None) is code for fn, code in self._bindings.values())
                and all(file_sha256(Path(path)) == digest for path, digest in self._source_files.items()), "source/live numerical binding changed")
        # Numerical hooks and per-instance methods are not part of the admitted
        # source. Profilers may observe calls externally without changing them.
        import torch.nn.modules.module as module_hooks
        from torch.optim import optimizer as optimizer_hooks
        for registry in (module_hooks, optimizer_hooks):
            require(not any(v for k, v in vars(registry).items() if k.startswith("_global_") and isinstance(v, dict)), "global numerical hooks")
        for obj in (*modules.values(), *self.optimizers.values()):
            require(not any(v for k, v in vars(obj).items() if "hook" in k and isinstance(v, dict))
                    and not any(callable(v) and callable(getattr(type(obj), k, None)) for k, v in vars(obj).items()),
                    "injected numerical hook/instance method")
        require(self.bundle.reference_context is self.context and self.bundle.student is self.models["student"]
                and self.bundle.teacher is self.models["teacher"] and self.bundle.critic is self.models["critic"]
                and self.bundle.generator_optimizer is self.optimizers["generator"]
                and self.bundle.critic_optimizer is self.optimizers["critic"]
                and self.bundle.critic_gradient_reducer is None and self.bundle.score_group is None
                and self.bundle.score_group_size == self.bundle.generator_group_size == self.bundle.critic_group_size == 1,
                "original baseline bundle ownership changed")
        for role, opt in self.optimizers.items():
            require(type(opt) is torch.optim.AdamW and id(opt) == self._optimizer_ids[role], "original AdamW ownership changed")
            require(content_digest(self._optimizer_value(role)["groups"]) == content_digest(self._groups[role])
                    and content_digest(opt.defaults) == content_digest(self._defaults[role]), "original Adam options/parameters changed")
        expected = [p for n, p in self._named.items() if n.startswith("critic.") and p.requires_grad]
        require(len(expected) == len(self.bundle.critic_lora_params)
                and all(a is b for a, b in zip(expected, self.bundle.critic_lora_params)), "original Critic LoRA ownership changed")

    def _validate_live(self):
        self._check_identity()
        require(not self.poisoned, "partial/poisoned G/Teacher/C/read owner requires a previous complete boundary")
        require(type(self.cycle) is int and 0 <= self.cycle <= self.assets["budget"]["max_cycles"]
                and self.versions == self.expected(self.cycle)
                and all(type(v) is int for v in self.versions.values()), "native G/C/EMA clocks changed")
        require(all(p.grad is None and bool(p.isfinite().all()) for p in self._named.values()), "finite zero-gradient complete boundary required")
        require(self._frozen == {n: tensor_content(self._named[n]) for n in self._frozen}, "frozen Teacher/Critic base changed")
        require(self._detached == {n: tensor_content(self._named[n]) for n in self._detached}, "detached Student text projector changed")
        self._check_consumed(self.consumed, self.cycle)
        self._check_rng(self._rng)
        if self.cycle:
            self._check_metrics(self.last_metrics, self.cycle, self._rng)
        else:
            require(self.last_metrics == {}, "cold-zero owner cannot have a completed loss receipt")
        for role in self.optimizers:
            self._check_optimizer(self._optimizer_value(role), role, self.cycle)

    def _optimizer_value(self, role):
        names = {id(p): n for n, p in self._parameters().items()}
        opt = self.optimizers[role]
        require(all(id(p) in names for g in opt.param_groups for p in g["params"])
                and all(id(p) in names for p in opt.state), "foreign Adam parameter ownership")
        return dict(groups=[{**{k: cpu_tree(v) for k, v in g.items() if k != "params"},
                             "params": [names[id(p)] for p in g["params"]]} for g in opt.param_groups],
                    defaults=cpu_tree(opt.defaults), state={names[id(p)]: cpu_tree(slot) for p, slot in opt.state.items()})

    @staticmethod
    def _check_tensor(value, current, name):
        import torch
        require(type(value) is torch.Tensor and value.device.type == "cpu" and value.layout == torch.strided
                and value.dtype == current.dtype and value.shape == current.shape and bool(value.isfinite().all())
                and not value.requires_grad and value.grad_fn is None, "checkpoint shape/dtype/layout/nonfinite: " + name)

    def _check_optimizer(self, value, role, cycle):
        import torch
        require(isinstance(value, dict) and set(value) == {"groups", "defaults", "state"}
                and content_digest(value["groups"]) == content_digest(self._groups[role])
                and content_digest(value["defaults"]) == content_digest(self._defaults[role]), "Adam group/options/layout changed")
        expected = {n for group in self._groups[role] for n in group["params"]
                    if not n.startswith("student.text_embedding.")}
        require(isinstance(value["state"], dict) and set(value["state"]) == (expected if cycle else set()),
                "Adam slot presence changed; detached Student text must remain uninitialized")
        for name, slot in value["state"].items():
            require(isinstance(slot, dict) and set(slot) == {"step", "exp_avg", "exp_avg_sq"}, "Adam slot fields changed")
            self._check_tensor(slot["step"], torch.tensor(0.), name + ".step")
            require(slot["step"].item() == (cycle if role == "generator" else 5*cycle), "Adam/native clocks changed")
            for key in ("exp_avg", "exp_avg_sq"):
                self._check_tensor(slot[key], self._named[name], name + "." + key)
            require(bool((slot["exp_avg_sq"] >= 0).all()), "Adam variance must be nonnegative")

    def _check_rng(self, value):
        import torch
        require(isinstance(value, dict) and set(value) == {"python", "torch", "numpy"}, "CPU Python/Torch/NumPy RNG owners required")
        require(type(value["torch"]) is torch.Tensor and value["torch"].dtype == torch.uint8
                and value["torch"].device.type == "cpu" and value["torch"].shape == torch.get_rng_state().shape, "Torch RNG layout changed")
        numpy = value["numpy"]
        require(type(numpy) is tuple and len(numpy) == 5 and numpy[0] == "MT19937"
                and type(numpy[1]) is list and len(numpy[1]) == 624
                and all(type(n) is int and 0 <= n < 2**32 for n in numpy[1])
                and type(numpy[2]) is int and 0 <= numpy[2] <= 624 and numpy[3] in (0, 1)
                and type(numpy[4]) is float and math.isfinite(numpy[4]), "NumPy RNG layout changed")
        validate_rng(value)  # local generators only; no process RNG mutation.

    def _consumption(self, index):
        require(self._input_rows is not None and 0 <= index < len(self._input_rows), "unadmitted/exhausted CPU cohort")
        return deepcopy(self._input_rows[index])

    def _check_consumed(self, rows, cycle):
        require(isinstance(rows, list) and len(rows) == cycle, "source cursor changed")
        previous = "0"*64
        for index, row in enumerate(rows):
            desc = self._consumption(index)
            chain = fingerprint(dict(previous=previous, **desc))
            require(row == dict(**desc, chain=chain), "consumed source/control chain changed")
            previous = chain

    def _check_metrics(self, metrics, cycle, rng):
        from dataclasses import asdict
        from distillschedule.runtime.livewan_crossscale_worker import RunEvidence
        fields = {"cycle", "versions", "inputs", "losses", "optimizer_norms", "logical_score_calls", "rng_sha256",
                  "claimable_speedup", "gradient_sync", "cpu_contract", "source_control_sha256", "native_run_evidence"}
        require(isinstance(metrics, dict) and set(metrics) == fields and metrics["cycle"] == cycle
                and type(metrics["cycle"]) is int and metrics["versions"] == self.expected(cycle)
                and metrics["cpu_contract"] == GEOMETRY and metrics["claimable_speedup"] is False,
                "checkpoint loss/clock/CPU census receipt changed")
        source = self._consumption(cycle-1)
        require(metrics["source_control_sha256"] == source["control_sha256"] and len(metrics["inputs"]) == 5,
                "checkpoint last source/control census changed")
        keys = []
        for step, (row, entry) in enumerate(zip(metrics["inputs"], source["entries"], strict=True)):
            key = f"cycle={cycle-1}/step={step}/global_step={(cycle-1)*5+step}/slot=0"
            keys.append(key)
            require(isinstance(row, dict) and set(row) == {"operation_key", "record_id", "source", "content_sha256", "world_frames", "rope_gap", "seeds"}
                    and row["operation_key"] == key and row["record_id"] == entry["id"] and row["source"] == entry["source"]
                    and row["content_sha256"] == entry["content_sha256"] and type(row["world_frames"]) is int
                    and row["world_frames"] in (3, 6, 9) and type(row["rope_gap"]) is int and 0 <= row["rope_gap"] <= 240
                    and row["world_frames"] == entry["operation"]["world_frames"] and row["rope_gap"] == entry["operation"]["rope_gap"]
                    and row["seeds"] == entry["operation"]["seeds"],
                    "checkpoint actual source rows changed")
        expected_calls = {"teacher": 1, "teacher/nograd": 1, "teacher/nograd/plain": 1,
                          "critic": 6, "critic/nograd": 1, "critic/nograd/plain": 1, "critic/grad": 5, "critic/grad/checkpoint": 5}
        require(metrics["logical_score_calls"] == expected_calls
                and all(type(v) is int for v in metrics["logical_score_calls"].values()), "checkpoint original call census changed")
        losses = metrics["losses"]
        require(isinstance(losses, list) and len(losses) == 8, "checkpoint three recompute/five Critic losses required")
        for i, row in enumerate(losses):
            require(isinstance(row, dict) and set(row) == {"operation_key", "kind", "shape", "dtype", "finite", "mean", "norm", "max_abs"}
                    and row["kind"] == ("generator" if i < 3 else "critic") and row["operation_key"] == keys[0 if i < 3 else i-3]
                    and row["shape"] == [] and row["dtype"] == "torch.float32" and row["finite"] is True
                    and all(type(row[k]) is float and math.isfinite(row[k]) and row[k] >= 0. for k in ("mean", "norm", "max_abs")),
                    "checkpoint original loss fingerprints changed")
        norms = metrics["optimizer_norms"]
        require(isinstance(norms, list) and len(norms) == 6, "checkpoint original clip/commit census changed")
        for i, row in enumerate(norms):
            require(isinstance(row, dict) and set(row) == {"role", "step", "clip_input_norm", "gradient_tensor_metrics"}
                    and row["role"] == ("generator" if i == 0 else "critic") and row["step"] == (cycle-1)*5+max(0, i-1)
                    and type(row["clip_input_norm"]) is float and math.isfinite(row["clip_input_norm"]) and row["clip_input_norm"] > 0
                    and row["gradient_tensor_metrics"] == "NOT_COLLECTED_TRAJECTORY_LOW_OVERHEAD", "checkpoint original clip evidence changed")
        require(metrics["gradient_sync"] == [dict(step=(cycle-1)*5+i, role="critic", gradient_tensors=len(self.bundle.critic_lora_params), buckets=0)
                                             for i in range(5)], "checkpoint CPU1 identity-mean evidence changed")
        expected_rng = dict(python=fingerprint(repr(rng["python"])), torch=hashlib.sha256(rng["torch"].numpy().tobytes()).hexdigest(),
                            numpy=fingerprint(rng["numpy"]))
        require(metrics["rng_sha256"] == expected_rng, "checkpoint loss receipt/private RNG differs")
        expected = RunEvidence(loss_fingerprints=deepcopy(losses), optimizer_diagnostics=deepcopy(norms),
                               logical_score_calls=deepcopy(expected_calls), gradient_sync=deepcopy(metrics["gradient_sync"]))
        require(metrics["native_run_evidence"] == asdict(expected), "checkpoint native RunEvidence/census changed")

    def bind_inputs(self, identity, rows):
        self._validate_live()
        require(self.cycle == 0 and self._input_rows is None and rows and len(identity) == 64, "fresh input admission once only")
        self._input_rows = deepcopy(rows)
        self.identity["prepared"] = dict(identity=identity, cohorts=deepcopy(rows))
        self._identity_digest = fingerprint(self.identity)

    @contextmanager
    def _random_scope(self):
        outer = rng_state()
        try:
            restore_rng(self._rng)
            yield
        finally:
            self._rng = cpu_tree(rng_state())
            restore_rng(outer)

    def run(self, row, *, cycle):
        import torch
        from dataclasses import asdict
        from distillschedule.runtime import livewan_crossscale_worker as worker
        from distillschedule.data.livewan_reference_inputs import CPUCycleInputs, decode_cycle, descriptor
        self._validate_live()
        require(type(cycle) is int and cycle == self.cycle and cycle < self.assets["budget"]["max_cycles"], "CPU cycle cursor/capacity changed")
        require(descriptor(row) == self._consumption(cycle), "cohort replay/control/source change")
        evidence = worker.RunEvidence()
        # All subsequent source reads, Teacher projection, G/C work and receipt
        # admission belong to this one poisonable transaction. A partial native
        # update is never rolled back or retried as if it had not happened.
        self.poisoned = True
        self._busy = True
        try:
            with self._random_scope():
                loader = CPUCycleInputs(row, reference_context=self.context)
                streamers = worker._build_streamers(torch, student=self.bundle.student, count=1,
                    device=torch.device("cpu"), reference_context=self.context)
                score = worker.build_score_context(torch, bundle=self.bundle, loader=loader, device=torch.device("cpu"))
                worker._run_baseline(torch=torch, rank=0, trace=loader, loader=loader, bundle=self.bundle,
                    streamers=streamers, score_context=score, run_kind="trajectory", evidence=evidence,
                    versions=self.versions, device=torch.device("cpu"), step_offset=5*cycle)
                require(loader.read_count == 5 and self.versions == self.expected(cycle+1), "native five-entry commit/read census changed")
                from .livewan_training import _cycle_metrics
                native = _cycle_metrics(evidence, loader.consumed, self.versions, cycle+1)
                require(all(row["clip_input_norm"] > 0 for row in native["optimizer_norms"]), "nontrivial original G/C gradients required")
                require(not evidence.output_tensors and not evidence.output_fingerprints, "no training rollout-as-current-Student output")
                native["gradient_sync"] = deepcopy(evidence.gradient_sync)
                native["cpu_contract"] = deepcopy(GEOMETRY)
                native["source_control_sha256"] = row["control_sha256"]
                native["native_run_evidence"] = asdict(evidence)
                # Caches are scratch. Rebuild the next original prepare-rollout
                # cache, never serialize it or reuse recorded recomputes.
                del score, streamers, loader
            for optimizer in self.optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            self.cycle += 1
            desc = self._consumption(cycle)
            previous = self.consumed[-1]["chain"] if self.consumed else "0"*64
            self.consumed.append(dict(**desc, chain=fingerprint(dict(previous=previous, **desc))))
            self.last_metrics = deepcopy(native)
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self._busy = False
        self.poisoned = False
        try:
            self._validate_live()
        except BaseException:
            self.poisoned = True
            raise
        return deepcopy(native)

    def state_dict(self):
        if self.poisoned:
            # The common safe-state reader enumerates live tensors to discover
            # meshes BEFORE calling validate_state_dict. A poisoned CPU owner
            # must remain inspectable for that read-only step, but must never
            # masquerade as a resumable cycle. No clocks/Adam/RNG/caches are
            # represented by this explicitly unresumable diagnostic view.
            self._check_identity()
            return dict(schema=OWNER, checkpoint_status="POISONED_NOT_RESUMABLE",
                        diagnostic_parameters={name: cpu_tree(p) for name, p in self._named.items()})
        self._validate_live()
        return dict(schema=OWNER, checkpoint_status="COMPLETE_CYCLE", identity=deepcopy(self.identity), cycle=self.cycle, versions=deepcopy(self.versions),
            modules={role: cpu_tree(model.state_dict()) for role, model in self.models.items()},
            optimizers={role: self._optimizer_value(role) for role in self.optimizers},
            rng=cpu_tree(self._rng), consumed=deepcopy(self.consumed), last_metrics=deepcopy(self.last_metrics))

    def validate_checkpoint_boundary(self):
        self._validate_live()
        require(self.cycle > 0, "complete original G1/C5 boundary required")

    def _stage(self, value):
        # No live value mutation here, including when recovering a poisoned G,
        # Teacher, Critic, or failed copy. Validate ALL decoded owners first.
        import torch
        self._check_identity()
        fields = {"schema", "checkpoint_status", "identity", "cycle", "versions", "modules", "optimizers", "rng", "consumed", "last_metrics"}
        require(isinstance(value, dict) and set(value) == fields and value["schema"] == OWNER
                and value["checkpoint_status"] == "COMPLETE_CYCLE"
                and value["identity"] == self.identity, "checkpoint source/precision/config/input identity changed or poisoned diagnostic")
        cycle = value["cycle"]
        require(type(cycle) is int and 0 < cycle <= self.assets["budget"]["max_cycles"]
                and value["versions"] == self.expected(cycle) and all(type(v) is int for v in value["versions"].values()),
                "checkpoint incomplete/wrong G/C/EMA clock")
        self._check_consumed(value["consumed"], cycle)
        self._check_rng(value["rng"])
        self._check_metrics(value["last_metrics"], cycle, value["rng"])
        require(isinstance(value["modules"], dict) and set(value["modules"]) == set(self.models), "checkpoint role set changed")
        for role, model in self.models.items():
            current, saved = model.state_dict(), value["modules"][role]
            require(isinstance(saved, dict) and saved.keys() == current.keys(), "checkpoint full role names changed")
            for name, tensor in current.items():
                restored = saved[name]
                self._check_tensor(restored, tensor, role + "." + name)
                if role + "." + name in self._frozen:
                    require(tensor_content(restored) == self._frozen[role + "." + name], "checkpoint frozen Teacher/base changed")
                if role + "." + name in self._detached:
                    require(tensor_content(restored) == self._detached[role + "." + name], "checkpoint detached Student text changed")
        require(isinstance(value["optimizers"], dict) and set(value["optimizers"]) == set(self.optimizers), "checkpoint Adam owners changed")
        staged = {}
        for role, optimizer in self.optimizers.items():
            saved = value["optimizers"][role]
            self._check_optimizer(saved, role, cycle)
            native = optimizer.state_dict()
            mapping = {name: pid for group, group_native in zip(saved["groups"], native["param_groups"], strict=True)
                       for name, pid in zip(group["params"], group_native["params"], strict=True)}
            raw = dict(state={mapping[n]: cpu_tree(slot) for n, slot in saved["state"].items()},
                       param_groups=[{**{k: cpu_tree(v) for k, v in g.items() if k != "params"},
                                      "params": [mapping[n] for n in g["params"]]} for g in saved["groups"]])
            candidate = copy(optimizer)
            candidate.defaults = cpu_tree(saved["defaults"])
            candidate.param_groups = [dict(g, params=list(g["params"])) for g in optimizer.param_groups]
            candidate.state = {}
            candidate.load_state_dict(raw)
            staged[role] = candidate
        # Clone before mutation: Torch optimizers otherwise retain the caller's
        # saved tensor objects and corrupt that preceding recovery boundary.
        return cpu_tree(value), staged

    def validate_state_dict(self, value):
        try:
            self._stage(value)
        except BaseException:
            self.poisoned = True
            raise

    def load_state_dict(self, value):
        self.poisoned = True
        value, staged = self._stage(value)
        for role, model in self.models.items():
            model.load_state_dict(value["modules"][role], strict=True)
        for role, optimizer in self.optimizers.items():
            optimizer.state, optimizer.param_groups = staged[role].state, staged[role].param_groups
            optimizer.zero_grad(set_to_none=True)
        self.cycle, self.versions = value["cycle"], deepcopy(value["versions"])
        self._rng, self.consumed = cpu_tree(value["rng"]), deepcopy(value["consumed"])
        self.last_metrics = deepcopy(value["last_metrics"])
        self.poisoned = False
        try:
            self._validate_live()
        except BaseException:
            self.poisoned = True
            raise

    def validate_checkpoint_progress(self, value, progress):
        self.validate_state_dict(value)
        try:
            require(isinstance(progress, dict) and type(progress.get("cycle")) is type(progress.get("cursor")) is int
                    and progress["cycle"] == progress["cursor"] == value["cycle"]
                    and progress.get("versions") == {"native_cycle": value["cycle"]}
                    and type(progress["versions"]["native_cycle"]) is int, "inner/outer complete CPU1 cursor changed")
        except BaseException:
            self.poisoned = True
            raise


def build_model(parameters):
    import torch
    path, assets, specs, api = admit_assets(parameters)
    student = _load_role(api, assets["student"], specs["student"]).train()
    teacher = _load_role(api, assets["teacher"], specs["teacher"]).eval().requires_grad_(False)
    critic = _load_role(api, assets["critic"], specs["critic"]).eval().requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(assets["lora_seed"])
        lora, count, numel = api.lora.inject_lora(critic, rank=32)
    require(count == 6*specs["critic"]["num_layers"] and numel > 0, "complete original six-target LoRA32 required")
    owner = LiveWanCPUOwner(student=student, teacher=teacher, critic=critic, lora=lora, assets=assets,
                           asset_pin=pin_file(path), api=api, architecture=specs["student"])
    from .livewan_inference import project_current_generator
    return ModelComponents(state={"native_cycle": owner}, admit_training=admit_training, collect_metrics=collect_metrics,
        inference_projection=project_current_generator,
        metadata=dict(schema_version=OWNER, scope=SCOPE, precision=PRECISION, geometry=deepcopy(GEOMETRY),
            assets=deepcopy(assets), asset_manifest=pin_file(path), owner_identity=deepcopy(owner.identity),
            generation="CURRENT_G_EXPORT_NATIVE_FEWSTEPSTREAMER_CPU_FP32", decoded_media="OPTIONAL_EXTERNAL_PINNED_WAN21_CODEC", quality="NOT_RUN",
            native_world16="NOT_RUN", c51_restart="NOT_IMPLEMENTED", claimable_speedup=False)).validate()


def admit_training(*, state, request, prepared_identity, prepared_manifest):
    from distillschedule.data import PreparedDataset
    from distillschedule.data.livewan_reference_inputs import SCHEMA, producer_identity, decode_cycle, descriptor, validate_source_cycle
    from distillschedule.data.manifest import validate_source
    owner = state["native_cycle"]
    owner._validate_live()
    require(dict(request.execution) == dict(adapter="program", mode="reference")
            and request.trainer.get("device", "cpu") == "cpu" and not request.trainer.get("distributed")
            and not request.trainer.get("validation") and request.data.get("sharding", "single") == "single"
            and request.data.get("sampling", dict(kind="finite", shuffle=False)) == dict(kind="finite", shuffle=False),
            "CPU1 finite Program reference only; no auto/plan/native/distributed/validation")
    require(request.model["factory"] == MODEL and request.algorithm["factory"] == ALGORITHM
            and not request.algorithm.get("parameters"), "one original atomic baseline cycle required")
    meta = prepared_manifest["metadata"]
    require(meta == dict(schema_version=SCHEMA, producer=producer_identity(), scope=SCOPE, precision=PRECISION,
            geometry=GEOMETRY, evidence_kind=owner.assets["evidence_kind"], negative=owner.assets["negative"]),
            "CPU1 source/precision/geometry/negative prompt contract changed")
    for pin in prepared_manifest["provenance"]["documents"]:
        validate_source(pin)
    dataset = PreparedDataset(request.data["prepared"])
    require(dataset.identity == prepared_identity and dataset.manifest == prepared_manifest
            and 0 < request.trainer["cycles"] <= len(dataset) <= owner.assets["budget"]["max_cycles"], "finite prepared inventory/horizon changed")
    rows, ids, paths, hashes, contents = [], set(), set(), set(), set()
    for cycle, record in enumerate(prepared_manifest["records"]):
        row = decode_cycle(dataset.sample(cycle), cycle=cycle)
        require(descriptor(row) == record["metadata"], "prepared cohort metadata differs from safe tensors")
        validate_source_cycle(row)
        for entry in row["control"]["entries"]:
            for value, seen in ((entry["id"], ids), (entry["source"]["path"], paths),
                                (entry["source"]["sha256"], hashes), (entry["content_sha256"], contents)):
                require(value not in seen, "replayed/copied source record is not fresh")
                seen.add(value)
        rows.append(descriptor(row))
    owner.bind_inputs(prepared_identity, rows)


def native_cycle(ctx):
    from distillschedule.data.livewan_reference_inputs import decode_cycle
    cycle = ctx.attributes["cycle"]
    return ctx.state["native_cycle"].run(decode_cycle(ctx.inputs, cycle=cycle), cycle=cycle)


def build_algorithm(components, feeds, parameters, cycle, seed):
    from distillschedule.open_workload.torch_backend import Workload
    from distillschedule.workloads.program import Program, Call
    from distillschedule.data.livewan_reference_inputs import decode_cycle, descriptor
    require(not parameters and type(cycle) is int and cycle >= 0 and set(components.state) == {"native_cycle"}, "one original G1/C5 CPU owner required")
    owner = components.state["native_cycle"]
    owner._validate_live()
    require(cycle == owner.cycle and descriptor(decode_cycle(feeds, cycle=cycle)) == owner._consumption(cycle), "CPU cohort replay/order changed")
    call = Call(id="livewan.original_cycle", binding="original_g1_c5_ema0", inputs=tuple(feeds),
        reads=("native_cycle",), writes=("native_cycle",), grad="external_grad", replay_safe=False, attributes=dict(cycle=cycle))
    return Workload(program=Program((call,), initial_versions={"native_cycle": cycle}, feeds=tuple(feeds), outputs=(call.id,)),
        bindings={"original_g1_c5_ema0": native_cycle}, state=components.state, feeds=feeds, seed=seed, identity_extra=dict(scope=SCOPE))


def collect_metrics(outputs):
    require(set(outputs) == {"livewan.original_cycle"}, "one original complete cycle receipt required")
    losses = outputs["livewan.original_cycle"]["losses"]
    generator = [r["mean"] for r in losses if r["kind"] == "generator"]
    critic = [r["mean"] for r in losses if r["kind"] == "critic"]
    require(len(generator) == 3 and len(critic) == 5, "original recompute/loss census changed")
    return {"generator/loss": sum(generator), **{f"critic/loss_{i+1}": loss for i, loss in enumerate(critic)}}
