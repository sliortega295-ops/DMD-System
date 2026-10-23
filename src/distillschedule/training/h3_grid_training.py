"""Opt-in original Grid1000 training, not the two-cycle optimized benchmark.

The original entrypoint builds the model/trainer and executes its existing
setup, loss and G1/F5 loop. This module owns only fail-closed admission and the
finite-data/checkpoint lifecycle. No numerical qualification is manufactured.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import timedelta
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, load_profile
from .h3_grid_inputs import SCHEMA as INPUT, document, require, data_contract, load_inputs, validate_payloads, NativeLatentCursor
from .environment import credential_key, nonsecret_environment_snapshot

SCHEMA = "distillschedule.original_h3_grid_training.v1"
ASSETS = "distillschedule.original_h3_grid_assets.v1"
POLICY = "distillschedule.original_h3_grid_checkpoint.v1"
LIGHTX2V = "d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be"
DIFFUSERS = "9284607295a09f759aadd65ed08f48b35feea6d9"
DEPENDENCIES = {"torch", "numpy", "loguru", "peft", "safetensors", "omegaconf", "PyYAML",
                "diffusers", "transformers", "accelerate", "imageio", "Pillow", "kernels", "triton"}
_ACTIVE = ContextVar("original_h3_grid_training", default=None)


def current_training():
    return _ACTIVE.get()


@contextmanager
def training_scope(lifecycle):
    require(current_training() is None, "nested original Grid training lifecycle")
    token = _ACTIVE.set(lifecycle)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _source(value, expected, label):
    require(isinstance(value, dict) and set(value) == {"path", "head"} and value["head"] == expected,
            f"explicit pinned {label} source identity required")
    path = Path(value["path"]).expanduser()
    require(path.is_absolute() and path.is_dir(), f"missing absolute {label} source")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    require(git("rev-parse", "--show-toplevel") == str(path.resolve())
            and git("rev-parse", "HEAD") == expected
            and not git("status", "--porcelain", "--untracked-files=all"), f"{label} source HEAD/clean state changed")
    return path.resolve()


def reference_config(value, public):
    """Validate a supplied original config; never fill it or mutate its policy.

    The canonical profile supplies the locked objective/geometry/dtype fields,
    NOT an incumbent implementation. Optional optimized transformations must
    all be explicitly OFF in this separate reference contract.
    """
    from distillschedule.search.h3_implementation import normalized_model_config
    from distillschedule.search.h3_request import H3Request
    request = H3Request.from_dict(document(public.execution["task"]))
    data, _ = data_contract(value)
    original = value.get("training", {})
    require(all(type(original.get(k)) is int for k in ("max_train_iters", "save_every_iters", "seed"))
            and original.get("max_train_iters") == public.trainer["cycles"]
            and original.get("save_every_iters") == public.trainer.get("checkpoint_every", 0)
            and original.get("seed") == public.trainer.get("seed")
            and original.get("output_dir") == public.trainer["output_dir"],
            "public cycles/checkpoint_every/seed/output_dir must equal the effective original config")
    require(value.get("resume") == {"auto_resume": False}, "implicit historical/cursorless auto-resume is forbidden")
    a100 = original.get("a100", {})
    policy = a100.get("activation_policy", {})
    require(set(policy) == {"name", "pin_memory", "detailed_events"}
            and policy["name"] in ("none", "checkpoint_boundary_cpu")
            and type(policy["pin_memory"]) is bool and policy["detailed_events"] is False,
            "explicit original activation policy required; no capacity policy is inferred")
    expected = deepcopy(load_profile("minimax_h3_grid1000_best")["runtime"]["model_config"])
    expected["data"] = {"train": data}
    expected["training"].update(max_train_iters=public.trainer["cycles"],
        save_every_iters=public.trainer.get("checkpoint_every", 0), seed=public.trainer["seed"],
        output_dir=public.trainer["output_dir"])
    expected["training"]["dmd"].update(height=request.height, width=request.width, num_frames=request.frames)
    # These are acceptance requirements, not launch-time overrides. The input
    # config must already contain each choice, including the physical policy.
    baseline = expected["training"]["a100"]
    baseline["activation_policy"] = policy
    reorder = a100.get("critic_rollout_reorder", {})
    require(set(reorder) == {"enabled", "group_size"} and type(reorder["enabled"]) is bool
            and type(reorder["group_size"]) is int and reorder["group_size"] == 5,
            "explicit original critic rollout schedule/group_size required")
    baseline["critic_rollout_reorder"] = reorder
    baseline["block_pointwise_fusion"] = {"enabled": False, "grad_enabled": False, "rmsnorm_modulate": False}
    baseline["rotary_fusion"] = {"enabled": False, "grad_enabled": False, "qk_rmsnorm": False, "qk_rmsnorm_grad": False}
    baseline["swiglu_fusion"] = {"enabled": False, "grad_enabled": False}
    baseline["fa3_nograd_splits"] = {"num_splits": 1, "grad_num_splits": 1}
    baseline["fa3_replay_cache"] = {"enabled": False, "blocks": [], "storage": "cuda",
                                   "max_d2h_inflight": 2, "trim_before_backward": False}
    baseline["lora_scale1_elision"] = {"enabled": False, "fused_grad_epilogue": False, "fused_nograd_epilogue": False}
    require(normalized_model_config(value) == normalized_model_config(expected),
            "supplied config is not the versioned original Grid1000 reference: shape/math/dtype/optimizer or optional optimization changed")
    return request, policy


def environment_snapshot(environment):
    prefixes = ("H3_", "DISTILLSCHEDULE_", "NCCL_", "TORCH_", "PYTORCH_", "CUBLAS_", "CUDA_")
    names = {"LOCAL_KERNELS", "LIGHTX2V_ROOT", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "MALLOC_ARENA_MAX",
             "LD_LIBRARY_PATH", "PYTHONPATH", "TRITON_CACHE_DIR"}
    return nonsecret_environment_snapshot(environment, prefixes=prefixes, names=names)


def _environment(value, config):
    require(isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()),
            "explicit effective reference environment required")
    allowed_h3 = {"H3_ADALN_GRID_MANIFEST", "H3_FLASH_ATTN3_LOCAL_PATH"}
    require({k for k in value if k.startswith("H3_")} <= allowed_h3
            and not any(k.startswith("DISTILLSCHEDULE_") for k in value)
            and not value.get("LOCAL_KERNELS"),
            "reference cannot inherit benchmark/planner/optimized environment overrides")
    require(value.get("H3_ADALN_GRID_MANIFEST") == config["training"]["a100"]["adaln_grid"]["manifest"]
            and bool(value.get("H3_FLASH_ATTN3_LOCAL_PATH")),
            "explicit matching Grid table and original local FA3 kernel paths required")
    require(not any(credential_key(k) for k in value), "credentials must not be embedded in reference environment")


def immutable_files(config, environment):
    """Hash every used frozen model/table/kernel file, not a size-only receipt."""
    model = Path(config["model"]["pretrained_model_name_or_path"])
    require(model.is_absolute() and model.is_dir(), "existing absolute original converted H3 model required")
    transformer = model / "transformer" if (model / "transformer/config.json").is_file() else model
    model_config = document(transformer / "config.json")
    require(model_config.get("_class_name") == "MiniMaxH3Transformer3DModel" and "num_refiner_layers" in model_config,
            "original trainable converted H3 transformer config required")
    weights = sorted(transformer.glob("*.safetensors"))
    require(weights, "original H3 safetensors shards missing")
    files = [transformer / "config.json", *weights, *sorted(transformer.glob("*.index.json"))]
    for index in transformer.glob("*.index.json"):
        mapped = document(index).get("weight_map", {})
        require(isinstance(mapped, dict) and mapped and all((transformer / name).is_file() for name in mapped.values()),
                "original H3 indexed weight shard missing")
    grid_path = Path(config["training"]["a100"]["adaln_grid"]["manifest"])
    require(grid_path.is_absolute(), "absolute Grid1000 manifest required")
    grid = document(grid_path)
    require(grid.get("schema_version") == "h3_a100.adaln_grid.v1" and grid.get("grid_size") == 1000
            and grid.get("num_blocks") == 50 and grid.get("rows_per_entry") == grid.get("modulation_chunks") == 6
            and grid.get("num_entries") == 1000 + grid.get("num_rollout_entries", -1000)
            and grid.get("renoise_sigma_min") == .02 and grid.get("renoise_sigma_max") == .98
            and grid.get("video_shift") == 6.0 and grid.get("audio_shift") == 3.0,
            "original Grid1000 manifest objective/dimensions changed")
    binary = (grid_path.parent / grid["binary_file"]).resolve()
    require(binary.is_file() and binary.stat().st_size == grid["num_entries"] * 50 * 6 * 6 * grid["hidden_size"] * 2,
            "Grid1000 BF16 binary incomplete")
    kernel = Path(environment["H3_FLASH_ATTN3_LOCAL_PATH"])
    require(kernel.is_absolute() and kernel.is_dir(), "explicit original local FA3 kernel directory required")
    kernel_files = sorted(p for p in kernel.rglob("*") if p.is_file() and ".git" not in p.parts
                          and "__pycache__" not in p.parts and p.suffix != ".pyc")
    require(kernel_files and any(p.suffix == ".so" for p in kernel_files), "original local FA3 compiled kernel missing")
    return {str(p.resolve()): file_sha256(p) for p in [*files, grid_path, binary, *kernel_files]}


def admit_reference(config, task_value, assets_value, paths):
    require(isinstance(assets_value, dict) and set(assets_value) == {
        "schema_version", "lightx2v", "diffusers", "native_config", "immutable_files", "dependency_versions", "environment"}
        and assets_value["schema_version"] == ASSETS, "versioned original H3 training assets required")
    _source(assets_value["lightx2v"], LIGHTX2V, "LightX2V")
    _source(assets_value["diffusers"], DIFFUSERS, "Diffusers")
    native_file = assets_value["native_config"]
    require(isinstance(native_file, dict) and set(native_file) == {"path", "sha256"}
            and Path(native_file["path"]).is_absolute() and Path(native_file["path"]).is_file()
            and file_sha256(Path(native_file["path"])) == native_file["sha256"], "original native config source hash changed")
    native = document(native_file["path"])
    request, policy = reference_config(native, config)
    require(request.to_dict() == task_value, "original task/config geometry differs")
    _environment(assets_value["environment"], native)
    versions = assets_value["dependency_versions"]
    require(isinstance(versions, dict) and set(versions) == DEPENDENCIES
            and all(isinstance(v, str) and bool(v) for v in versions.values())
            and versions["diffusers"] == "0.40.0.dev0", "complete explicit original dependency versions required")
    require(assets_value["immutable_files"] == immutable_files(native, assets_value["environment"]),
            "frozen original model/Grid/FA3 payload identity changed")
    inputs = load_inputs(config.data["prepared"], config_path=native_file["path"], cycles=config.trainer["cycles"])
    if config.trainer.get("resume_from"):
        from distillschedule.workloads.best_known.registry import git_head, repository_root
        identity = training_identity(git_head(repository_root()), assets_value, inputs, config.trainer["cycles"])
        checkpoint_policy = {"schema_version": POLICY, "identity_sha256": fingerprint(identity)}
        resume_metadata(config.trainer["resume_from"], identity=identity, policy=checkpoint_policy,
                        target=config.trainer["cycles"], torch_version=versions["torch"])
    return {"task": request.to_dict(), "plan": None, "assets": assets_value, "native_config": native, "inputs": inputs,
        "selected": {"schema_version": SCHEMA, "mode": "upstream-training/reference", "semantics": "approximate_grid1000",
                     "activation_policy": policy, "updates": {"student": 1, "fake": 5, "ema": 0},
                     "optional_optimized_paths": "FORBIDDEN", "numerical_cross_reference": "NOT_RUN"},
        "argv": ["--config", native_file["path"]],
        "lifecycle": {"schema_version": SCHEMA, "input_schema": INPUT, "same_topology_exact_resume": True,
                      "fixed_scheduler_horizon": config.trainer["cycles"], "known_optimized_backward_gate": "FAIL_UNCHANGED"},
        "validation_scope": "CPU_SOURCE_CONFIG_INPUT_FROZEN_PAYLOAD_IDENTITIES_ONLY; RUNTIME_DEPENDENCY_CAPACITY_AND_NUMERICAL_QUALIFICATION_NOT_RUN"}


def versions(cycle):
    return {"student": cycle, "fake": 5 * cycle, "ema": 0}


def training_identity(head, assets, inputs, target):
    return {"schema_version": SCHEMA, "source_head": head, "assets_sha256": fingerprint(assets),
            "input_sha256": inputs["inventory_sha256"], "native_config_sha256": assets["native_config"]["sha256"],
            "horizon": target}


def validate_progress(header, *, identity, policy, target):
    cycle = header.get("cycle")
    require(type(cycle) is int and 0 < cycle < target and header.get("cursor") == 6 * cycle
            and type(header.get("cursor")) is int and header.get("versions") == versions(cycle)
            and all(type(value) is int for value in header["versions"].values())
            and header.get("policy") == policy and header.get("identity", identity) == identity,
            "restart requires a versioned incomplete-target complete G1/F5/EMA0 boundary and exact next-sample cursor")


def resume_metadata(path, *, identity, policy, target, torch_version=None):
    """Lazy CPU rejection of legacy/partial/wrong-identity rank16 checkpoints."""
    from .checkpoint import SCHEMA as CHECKPOINT
    path = Path(path).resolve()
    require((path / "manifest.json").is_file() and (path / "_SUCCESS").is_file()
            and (path / "_SUCCESS").read_text().strip() == file_sha256(path / "manifest.json"),
            "complete versioned original checkpoint required; historical cursorless formats are not exact")
    header = document(path / "manifest.json")
    require(set(header) == {"schema_version", "kind", "identity", "cycle", "cursor", "versions", "policy", "topology", "torch", "files"}
            and header.get("schema_version") == CHECKPOINT and header.get("kind") == "RESUMABLE_COMPLETE_CYCLE"
            and header.get("identity") == identity and header.get("topology") == {
                "world_size": 16, "backend": "nccl", "node_groups": [list(range(8)), list(range(8, 16))]}
            and isinstance(header.get("torch"), str) and bool(header["torch"])
            and (torch_version is None or header["torch"] == torch_version),
            "original checkpoint schema/source/config/input/topology identity changed")
    validate_progress(header, identity=identity, policy=policy, target=target)
    names = {f"rank-{rank:05d}.pt" for rank in range(16)}
    require(set(header.get("files", {})) == names and all((path / name).is_file()
            and file_sha256(path / name) == header["files"][name] for name in names),
            "original checkpoint all-rank shards incomplete or changed")
    return header


def collective_admission(config, admit):
    """Called under the real all-rank Gloo plane before any model import/use."""
    from . import distributed
    result = distributed.checked(admit, "H3 original source/config/input admission")
    workload, source, paths, admitted = result
    distributed.checked(lambda: require(not source["identity"]["dirty"], "execution requires a clean committed source"),
                        "H3 original source cleanliness")
    identity = training_identity(source["identity"]["head"], admitted["assets"], admitted["inputs"], config.trainer["cycles"])
    distributed.admit_run_controls(config)
    require(all(row == identity for row in distributed.gather(identity)), "rank original source/config/input identity differs")
    return result, identity


class GridTrainingLifecycle:
    """State/data seam attached to one original trainer; never a second loop."""
    def __init__(self, config, admitted, identity):
        self.config, self.admitted, self.identity = config, admitted, identity
        self.policy = {"schema_version": POLICY, "identity_sha256": fingerprint(identity)}
        self.completed = 0
        self.cursor = None
        self.state = None

    def admit_loaded_config(self, loaded):
        from . import distributed
        from distillschedule.search.h3_implementation import normalized_model_config
        def local():
            path = Path(self.admitted["assets"]["native_config"]["path"]).resolve()
            require(Path(loaded.get("config_path", "")).resolve() == path
                    and file_sha256(path) == self.admitted["assets"]["native_config"]["sha256"]
                    and normalized_model_config(loaded) == normalized_model_config(self.admitted["native_config"]),
                    "actual original loader config differs from admitted config")
            reference_config(loaded, self.config)
            load_inputs(self.config.data["prepared"], config_path=path, cycles=self.config.trainer["cycles"])
            return validate_payloads(self.admitted["inputs"])
        semantic_inputs = distributed.checked(local, "H3 original effective loaded config/cached payloads")
        require(all(row == semantic_inputs for row in distributed.gather(semantic_inputs)),
                "rank original cached condition tensors differ")

    def checkpoint_header(self):
        from . import distributed
        from .checkpoint import read_manifest
        path = self.config.trainer.get("resume_from")
        if not path:
            return None
        def local():
            header = read_manifest(path, identity=self.identity)
            require(header.get("kind") == "RESUMABLE_COMPLETE_CYCLE", "original restart must be a complete-cycle checkpoint")
            validate_progress(header, identity=self.identity, policy=self.policy, target=self.config.trainer["cycles"])
            require(all(file_sha256(Path(path) / name) == sha for name, sha in header["files"].items()),
                    "restart shard missing or hash changed")
            return header
        return distributed.checked(local, "H3 original checkpoint metadata/shard admission")

    def start(self, trainer):
        from . import distributed
        def local():
            require(not trainer.auto_resume and getattr(trainer, "planner_cycle_observer", None) is None,
                    "original training cannot use historical auto-resume or the frozen two-cycle observer")
            require(trainer.max_train_iters == self.config.trainer["cycles"]
                    and trainer.save_every_iters == self.config.trainer.get("checkpoint_every", 0)
                    and trainer.output_train_dir == self.config.trainer["output_dir"],
                    "effective original target/save/output changed")
        distributed.checked(local, "H3 original trainer controls")
        header = self.checkpoint_header()
        self.completed = header["cycle"] if header else 0
        return self.completed

    def after_setup(self, trainer):
        from . import distributed
        from .checkpoint import restore_checkpoint
        from .h3_grid_state import trainer_state
        self.state = distributed.checked(lambda: trainer_state(trainer, self), "H3 original state ownership")
        resume = self.config.trainer.get("resume_from")
        if resume:
            restored = restore_checkpoint(resume, self.state, identity=self.identity)
            validate_progress(restored, identity=self.identity, policy=self.policy, target=trainer.max_train_iters)
        self.cursor = distributed.checked(lambda: NativeLatentCursor(trainer.dataloader_train,
            self.admitted["inputs"], cycle=self.completed, resume=bool(resume)), "H3 original native sampler/cursor")
        distributed.checked(lambda: self.state["clocks"].validate_checkpoint_boundary(), "H3 original restored boundary")
        return self.cursor

    def after_cycle(self, trainer, completed):
        from . import distributed
        from .checkpoint import save_checkpoint
        require(completed == self.completed + 1, "original loop skipped or repeated a cycle")
        self.completed = completed
        distributed.checked(lambda: self.state["clocks"].validate_checkpoint_boundary(), "H3 original complete-cycle boundary")
        cadence = self.config.trainer.get("checkpoint_every", 0)
        if cadence and completed % cadence == 0:
            save_checkpoint(Path(trainer.output_train_dir) / "checkpoints" / f"cycle-{completed:06d}", self.state,
                identity=self.identity, cycle=completed, cursor=self.cursor.cursor, versions=versions(completed), policy=self.policy)
        def receipt():
            rank = distributed.rank_world()[0]
            path = Path(trainer.output_train_dir) / "ranks" / f"rank-{rank:05d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {"schema_version": SCHEMA, "status": "COMPLETED" if completed == trainer.max_train_iters else "PARTIAL",
                   "rank": rank, "completed_cycles": completed, "next_sample_cursor": self.cursor.cursor,
                   "sampler_epoch": self.cursor.epoch, "versions": versions(completed), "identity": self.identity,
                   "numerical_cross_reference": "NOT_RUN", "known_optimized_backward_gate": "FAIL_UNCHANGED",
                   "claimable_speedup": False, "generation": "UNSUPPORTED", "export": "UNSUPPORTED"}
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, path)
        distributed.checked(receipt, "H3 original rank progress receipt")


def _backend(admitted):
    assets = admitted["assets"]
    for name, subdir in (("diffusers", "src"), ("lightx2v", "lightx2v_train")):
        directory = str(Path(assets[name]["path"]) / subdir)
        if directory not in sys.path:
            sys.path.insert(0, directory)
    for package, expected in (("diffusers", Path(assets["diffusers"]["path"]) / "src/diffusers"),
                              ("lightx2v_train", Path(assets["lightx2v"]["path"]) / "lightx2v_train/lightx2v_train")):
        loaded = importlib.import_module(package)
        require(Path(loaded.__file__).resolve().parent == expected.resolve(), f"{package} import origin differs from admitted source")
    for package, expected in assets["dependency_versions"].items():
        require(importlib.metadata.version(package) == expected, f"{package} dependency version changed")
    return importlib.import_module("distillschedule.backends.training.minimax_h3.grid_entrypoint")


def run_reference(config, *, admit):
    """Own one original torchrun execution, with collective pre-model failures.

    NCCL's default group is initialized early solely to allow a separate Gloo
    metadata plane before per-rank source/import checks. The original topology
    initializer still owns the exact CUDA mesh/data globals and all tensor math.
    """
    import torch
    import torch.distributed as dist
    from . import distributed
    from distillschedule.open_workload.replicated import control_plane
    names = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK")
    require(all(os.environ.get(k, "").isdigit() for k in names), "explicit original torchrun ranks required")
    world, rank, local, local_world, node = (int(os.environ[k]) for k in names)
    require(world == 16 and local_world == 8 and 0 <= rank < 16 and local == rank % 8 and node == rank // 8,
            "original Grid training requires two nodes of eight ranks")
    require(not dist.is_initialized(),
            "original Grid training owns its process group; enter on fresh torchrun ranks, not an embedding group")
    require(torch.cuda.is_available() and dist.is_gloo_available(), "original CUDA/NCCL plus Gloo metadata required")
    torch.cuda.set_device(local)
    metadata = None
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    try:
        require(dist.get_world_size() == world and dist.get_rank() == rank and str(dist.get_backend()) == "nccl",
                "initialized original rank group differs")
        metadata = dist.new_group(backend="gloo", timeout=timedelta(minutes=15))
        with control_plane(metadata):
            from distillschedule.open_workload.topology import discover
            allocated = discover()
            require(allocated["node_groups"] == [list(range(8)), list(range(8, 16))]
                    and not allocated["logical_nodes_on_one_host"], "actual original two-node topology differs")
            (_, source, _, admitted), identity = collective_admission(config, admit)
            distributed.checked(lambda: require(environment_snapshot(os.environ) == admitted["assets"]["environment"],
                "effective reference environment changed; no launch overrides are applied"), "H3 original environment")
            lifecycle = GridTrainingLifecycle(config, admitted, identity)
            lifecycle.checkpoint_header()  # all ranks/shards before model construction
            backend = distributed.checked(lambda: _backend(admitted), "H3 original import/dependency admission")
            old = sys.argv
            try:
                sys.argv = ["original-h3-grid-training-v1", *admitted["argv"]]
                with training_scope(lifecycle):
                    result = backend.main()
                    return 0 if result is None else result
            finally:
                sys.argv = old
    finally:
        # The original entrypoint normally cleans its group. Do not double
        # destroy it after a successful native cleanup or a native exception.
        if dist.is_initialized():
            if metadata is not None:
                dist.destroy_process_group(metadata)
            dist.destroy_process_group()
