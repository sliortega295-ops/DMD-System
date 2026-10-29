"""Versioned ORIGINAL W4 reference assets, not a benchmark bootstrap.

Metadata discovery never imports Torch or opens weight payloads. Execution
verifies every inventoried byte and restricted-loads Stage1 before the unchanged
original reader. No assets entry is a command, callback or optimization knob.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import subprocess

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, SHA256_RE
from distillschedule.workloads.livewan_crossscale import LIVEWAN_COMMIT, WAN21_COMMIT
from .environment import credential_key, nonsecret_environment_snapshot

SCHEMA = "distillschedule.livewan_reference_assets.v1"
DEPENDENCIES = {"torch", "numpy", "safetensors", "diffusers", "transformers", "easydict",
                "accelerate", "torchvision", "flash-attn"}
BACKEND = {"attention": "original_flash_attention2", "generator_optimizer": "foreach_adamw",
           "critic_optimizer": "foreach_adamw", "generator_dtype": "torch.float32",
           "critic_lora_dtype": "torch.float32", "frozen_dtype": "torch.bfloat16",
           "default_dtype": "torch.float32", "matmul_tf32": False, "cudnn_tf32": True,
           "matmul_precision": "highest", "deterministic": False, "cudnn_benchmark": False}


def require(condition, message):
    if not condition:
        raise ValueError("LiveWan reference: " + message)


def document(path):
    value = json.loads(Path(path).read_text())
    require(type(value) is dict, "JSON object required")
    return value


def _absolute(path):
    require(type(path) is str and Path(path).is_absolute() and ".." not in Path(path).parts
            and "\0" not in path, "explicit absolute asset path required")
    return Path(path).resolve()


def source_identity(root, *, project):
    """Exact clean gitlink plus every potentially imported original Python file."""
    root = _absolute(str(root))
    expected = LIVEWAN_COMMIT if project == "livewan" else WAN21_COMMIT
    package = "wanstreamer/" if project == "livewan" else "wan/"
    license_file = "LICENSE" if project == "livewan" else "LICENSE.txt"
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)
    require(root.is_dir() and Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() == root
            and git("rev-parse", "HEAD").decode().strip() == expected,
            f"{project} source HEAD/root changed")
    require(not git("status", "--porcelain", "--untracked-files=all").strip(),
            f"{project} source must be clean (including untracked imports)")
    paths = [p for p in git("ls-files", "-z").decode().split("\0") if p and (
        p.startswith(package) and p.endswith(".py") or p == license_file
        or project == "livewan" and p in {"scripts/gen_teacher.py", "scripts/encode_prompts.py"})]
    require(paths and license_file in paths, f"{project} licensed original source missing")
    files = {}
    for name in paths:
        path = root / name
        expected_sha = hashlib.sha256(git("show", expected + ":" + name)).hexdigest()
        require(path.is_file() and path.resolve().is_relative_to(root)
                and file_sha256(path) == expected_sha, f"{project} original source hash changed: {name}")
        files[name] = expected_sha
    require(not git("status", "--porcelain", "--untracked-files=all").strip()
            and git("rev-parse", "HEAD").decode().strip() == expected, "source changed during admission")
    return {"root": str(root), "head": expected, "files": files}


def environment_snapshot(environment):
    prefixes = ("LIVEWAN_", "DISTILLSCHEDULE_", "DMD_", "NCCL_", "TORCH_", "PYTORCH_",
                "CUDA_", "CUBLAS_", "CUDNN_", "NSYS_", "NCU_")
    names = {"LOCAL_KERNELS", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "OMP_NUM_THREADS",
             "MKL_NUM_THREADS", "MALLOC_ARENA_MAX", "TRITON_CACHE_DIR"}
    # Test intake is never a production launch override.
    return nonsecret_environment_snapshot(environment, prefixes=prefixes, names=names,
        exclude={"DMD_LIVEWAN_TEST_UPSTREAM", "DMD_WAN21_TEST_UPSTREAM"})


def _environment(value):
    require(type(value) is dict and all(type(k) is str and type(v) is str for k, v in value.items()),
            "explicit effective runtime environment required")
    forbidden = ("LIVEWAN_", "DISTILLSCHEDULE_", "DMD_", "NSYS_", "NCU_", "TORCH_COMPILE", "TORCH_LOGS")
    require(not any(k.startswith(forbidden) for k in value)
            and not any(value.get(k) for k in ("LOCAL_KERNELS", "LD_PRELOAD", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")),
            "reference cannot inherit benchmark/optimization/profiling/allocator overrides")
    require(not any(credential_key(k) for k in value), "credentials must not be embedded in reference environment")


def _payload_files(root):
    one = root / "diffusion_pytorch_model.safetensors"
    shards = sorted(root.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    require(not (one.exists() and shards), "ambiguous single/sharded native checkpoint")
    weights = [one] if one.is_file() else shards
    require(weights and sorted(root.glob("*.safetensors")) == sorted(weights),
            "exact original used safetensors inventory required")
    config = root / "config.json"
    require(config.is_file(), "original model config missing")
    indices = sorted(root.glob("*.index.json"))
    require(len(indices) <= 1, "ambiguous native checkpoint index")
    if indices:
        index = document(indices[0])
        mapping = index.get("weight_map")
        require(type(mapping) is dict and mapping and all(type(k) is str and type(v) is str for k, v in mapping.items())
                and set(mapping.values()) == {p.name for p in weights}, "checkpoint index/shards incomplete or changed")
    return [config, *weights, *indices]


def frozen_inventory(root):
    """Operator helper: full used config/index/weight inventory, no copies."""
    root = _absolute(str(root))
    files = {p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in _payload_files(root)}
    value = {"root": str(root), "files": files}
    return {**value, "sha256": fingerprint(value)}


def _model(row, *, size, verify_payloads):
    require(type(row) is dict and set(row) == {"root", "files", "sha256"}, "explicit frozen model inventory required")
    root = _absolute(row["root"])
    require(row["root"] == str(root) and root.is_dir() and row["sha256"] == fingerprint({k: row[k] for k in ("root", "files")}),
            "frozen model inventory identity changed")
    paths = _payload_files(root)
    require(type(row["files"]) is dict and set(row["files"]) == {p.name for p in paths},
            "frozen config/index/safetensors inventory changed")
    for path in paths:
        pin = row["files"][path.name]
        require(type(pin) is dict and set(pin) == {"bytes", "sha256"} and type(pin["bytes"]) is int
                and pin["bytes"] > 0 and type(pin["sha256"]) is str and SHA256_RE.fullmatch(pin["sha256"])
                and path.stat().st_size == pin["bytes"] and path.resolve().is_relative_to(root), "frozen file pin/size changed")
        # Lazy metadata validation reads only configuration and index bytes.
        if verify_payloads or path.suffix == ".json":
            require(file_sha256(path) == pin["sha256"], "frozen file hash changed: " + str(path))
    # Original WanModel.ignore_for_config omits these constructor defaults
    # from save_pretrained configs. The source is pinned by source_identity;
    # only these ignored fields inherit defaults, never the core geometry.
    # Explicit values still take precedence and must pass the same checks.
    cfg = {"patch_size": [1, 2, 2], "window_size": [-1, -1],
           "qk_norm": True, "cross_attn_norm": True, **document(root / "config.json")}
    dims = ((1536, 8960, 12, 30) if size == "1.3B" else (5120, 13824, 40, 40))
    require(all(cfg.get(k) == v for k, v in zip(("dim", "ffn_dim", "num_heads", "num_layers"), dims))
            and cfg.get("freq_dim") == 256 and cfg.get("in_dim") == cfg.get("out_dim") == 16
            and cfg.get("patch_size") == [1, 2, 2] and cfg.get("text_len") == 512
            and cfg.get("qk_norm") is True and cfg.get("cross_attn_norm") is True
            and cfg.get("eps") == 1e-6 and cfg.get("window_size") == [-1, -1],
            "original Wan " + size + " geometry/config changed")


def asset_config(value, *, verify_payloads=False):
    """Public v1 is always the complete original pip-FA2 contract."""
    return _asset_config(value, verify_payloads=verify_payloads, dependencies=DEPENDENCIES)


def _asset_config(value, *, verify_payloads, dependencies):
    """Shared checks; only the separately versioned C51 provider owns substitution.

    No public reference flag can omit a dependency or admit a provider.
    """
    require(type(value) is dict and set(value) == {"schema_version", "sources", "models", "initial_student",
            "critic_lora_seed", "initial_versions", "optimizer_steps", "runtime"} and value["schema_version"] == SCHEMA,
            "explicit versioned original-reference assets required; no benchmark bootstrap")
    require(type(value["sources"]) is dict and set(value["sources"]) == {"livewan", "wan21"}, "both original sources required")
    for name, row in value["sources"].items():
        require(type(row) is dict and set(row) == {"root", "head", "files"}
                and row == source_identity(row["root"], project=name), "original source HEAD/hash/origin changed")
    require(type(value["models"]) is dict and set(value["models"]) == {"student", "critic", "teacher"},
            "explicit Student/Critic/Teacher frozen inventories required")
    for name, row in value["models"].items():
        _model(row, size="14B" if name == "teacher" else "1.3B", verify_payloads=verify_payloads)
    initial = value["initial_student"]
    require(type(initial) is dict and (initial == {"kind": "base_weights"} or (
        set(initial) == {"kind", "path", "sha256", "bytes"} and initial["kind"] == "stage1_state"
        and type(initial["bytes"]) is int and initial["bytes"] > 0
        and type(initial["sha256"]) is str and SHA256_RE.fullmatch(initial["sha256"]))),
        "explicit initial Student base or complete Stage1 state required")
    if initial["kind"] == "stage1_state":
        path = _absolute(initial["path"])
        require(path.is_file() and path.stat().st_size == initial["bytes"], "Stage1 state missing/size changed")
        if verify_payloads:
            validate_stage1(initial, value["models"]["student"])
    require(type(value["critic_lora_seed"]) is int and 0 <= value["critic_lora_seed"] < 2**63,
            "explicit native LoRA seed required")
    require(value["initial_versions"] == {"generator": 0, "critic": 0, "ema": 0}
            and all(type(v) is int for v in value["initial_versions"].values())
            and value["optimizer_steps"] == {"generator": 0, "critic": 0}
            and all(type(v) is int for v in value["optimizer_steps"].values()), "only fresh zero steps/versions admitted")
    runtime = value["runtime"]
    require(type(runtime) is dict and set(runtime) == {"dependency_versions", "environment", "backend"}
            and runtime["backend"] == BACKEND, "fixed original reference backend/precision policy required")
    versions = runtime["dependency_versions"]
    require(type(versions) is dict and set(versions) == dependencies
            and all(type(v) is str and v for v in versions.values()), "explicit complete original dependency versions required")
    _environment(runtime["environment"])
    return deepcopy(value)


def validate_stage1(initial, model):
    """Restricted admission before the existing weights_only=False mmap reader."""
    import torch
    from safetensors import safe_open
    path = Path(initial["path"])
    require(file_sha256(path) == initial["sha256"], "Stage1 state hash changed before restricted load")
    # Torch itself may register version-dependent auxiliary globals. None are
    # needed for this plain-tensor contract. Scope an EMPTY added allowlist and
    # restore it exactly, rather than trusting a caller's unsafe registration.
    added_globals = torch.serialization.get_safe_globals()
    torch.serialization.clear_safe_globals()
    try:
        require(not torch.serialization.get_unsafe_globals_in_checkpoint(path), "Stage1 unsafe globals forbidden")
        blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (OSError, ValueError, RuntimeError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError("LiveWan reference: Stage1 restricted weights_only load failed") from exc
    finally:
        torch.serialization.add_safe_globals(added_globals)
    state = blob.get("model", blob) if type(blob) is dict else None
    require(type(blob) is dict and (state is blob or set(blob) == {"model"}) and type(state) is dict and state,
            "complete Stage1 plain tensor state required")
    shapes = {}
    for weight in _payload_files(Path(model["root"])):
        if weight.suffix != ".safetensors":
            continue
        with safe_open(str(weight), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                require(name not in shapes, "duplicate frozen Student tensor names")
                shapes[name] = tuple(handle.get_slice(name).get_shape())
    require(set(state) == set(shapes), "Stage1 incomplete or unknown tensor inventory")
    for name, tensor in state.items():
        require(type(tensor) is torch.Tensor and tensor.device.type == "cpu" and tensor.layout == torch.strided
                and tensor.dtype in {torch.float32, torch.float16, torch.bfloat16}
                and tuple(tensor.shape) == shapes[name] and bool(torch.isfinite(tensor).all()),
                "Stage1 tensor geometry/dtype/nonfinite changed")
    require(file_sha256(path) == initial["sha256"], "Stage1 state hash changed during restricted load")


def admit_native_layouts(value, layouts):
    """Compare real safetensors headers with actual original meta-model shapes.

    This neither materializes weights nor substitutes another architecture.
    The unchanged original loader still owns real dtype conversion and strict
    load_state_dict, including the already restricted-admitted Stage1 state.
    """
    from safetensors import safe_open
    for role, row in value["models"].items():
        actual, locations = {}, {}
        indices = []
        for path in _payload_files(Path(row["root"])):
            if path.suffix != ".safetensors":
                if path.name.endswith(".index.json"):
                    indices.append(path)
                continue
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    require(name not in actual, "duplicate original checkpoint tensor name")
                    tensor = handle.get_slice(name)
                    require(tensor.get_dtype() in {"F32", "F16", "BF16"}, "original checkpoint floating tensor dtype required")
                    actual[name] = list(tensor.get_shape())
                    locations[name] = path.name
        require(actual == layouts["14B" if role == "teacher" else "1.3B"],
                f"{role} complete native parameter keys/shapes changed before allocation")
        if indices:
            require(document(indices[0])["weight_map"] == locations, "original checkpoint index tensor-to-shard binding changed")
    return {size: fingerprint(layout) for size, layout in layouts.items()}


@dataclass(frozen=True)
class ReferenceAssets:
    """Typed input to unchanged build_models, derived ONLY from verified files."""
    _document: dict

    @classmethod
    def load(cls, value):
        return cls(asset_config(value, verify_payloads=True))

    @property
    def sha256(self):
        return fingerprint(self._document)

    @property
    def versions(self):
        return deepcopy(self._document["initial_versions"])

    @property
    def value(self):
        rows = self._document["models"]
        initial = self._document["initial_student"]
        return {"student_base": deepcopy(rows["student"]), "critic_base": deepcopy(rows["critic"]),
                "teacher_base": deepcopy(rows["teacher"]),
                "student_state": ({k: initial[k] for k in ("path", "sha256", "bytes")}
                                  if initial["kind"] == "stage1_state" else None),
                "critic_lora": {"seed": self._document["critic_lora_seed"]},
                "optimizer": {"generator_step": 0, "critic_step": 0}, "initial_versions": self.versions}

    def native_plan(self):
        return {"workload": {role: {"checkpoint": row["root"]} for role, row in (
            ("generator", self._document["models"]["student"]), ("score", self._document["models"]["critic"]),
            ("teacher", self._document["models"]["teacher"]))}, "metadata": {},
            "training_backends": {"generator": {"options": {"optimizer_impl": "foreach_adamw"}}}}

    def unchanged(self):
        require(asset_config(self._document, verify_payloads=True) == self._document, "reference asset drift")


def admit_runtime(value):
    """Observe controls; never install, apply an environment, or patch a backend."""
    import torch
    runtime = value["runtime"]
    _environment(environment_snapshot(os.environ))
    require(environment_snapshot(os.environ) == runtime["environment"], "effective reference environment changed")
    for name, expected in runtime["dependency_versions"].items():
        require(importlib.metadata.version(name) == expected, "dependency version changed: " + name)
    actual = {**BACKEND, "default_dtype": str(torch.get_default_dtype()),
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_precision": torch.get_float32_matmul_precision(),
              "deterministic": torch.are_deterministic_algorithms_enabled(), "cudnn_benchmark": torch.backends.cudnn.benchmark}
    require(actual == runtime["backend"], "effective native backend/precision controls changed")
    return {"dependencies": deepcopy(runtime["dependency_versions"]), "backend": actual}
