"""Offline admission for original LiveWan teacher records, not a trainer.

No data download, prefix crop, tensor repair or training RNG draw is performed.
Supplied teacher-format archives do not attest pretrained quality.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import math
import json
import os
import tempfile
from pathlib import Path
import pickle
import random
import subprocess

from distillschedule.workloads.best_known.registry import (
    file_sha256, fingerprint,
)
from distillschedule.workloads.livewan_crossscale import LIVEWAN_COMMIT

KIND = "original_livewan_teacher_cohorts.v1"
SCHEMA = "distillschedule." + KIND
C51_KIND = "original_livewan_c51_teacher_stream.v1"
C51_SCHEMA = "distillschedule." + C51_KIND
STREAMS = ("rollout", "generator_dmd_t", "generator_dmd_noise",
           "generator_grad_blocks", "critic_t", "critic_noise")
SOURCE_FILES = ("wanstreamer/data.py", "scripts/gen_teacher.py", "scripts/encode_prompts.py", "LICENSE")
CONTRACT = {"resolution": [368, 640], "frames": 81, "latent_shape": [16, 21, 46, 80],
            "text_shape": [512, 4096], "archive_dtype": "torch.float16", "world_size": 16,
            "global_batch": 16, "steps_per_cycle": 5, "generator_updates": 1,
            "critic_updates": 5, "ema_updates": 0,
            "world_choices": [3, 6, 9], "rope_gap_max": 240, "rope_gap_probability": .7,
            "sigma": "continuous_shifted_uniform_5_clamp_0.02_0.98"}


def require(condition, message):
    if not condition:
        raise ValueError("original LiveWan inputs: " + message)


def _source_identity(source_root):
    root = Path(source_root).expanduser().resolve()
    require(root.is_dir(), "missing original LiveWan source root")
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("original LiveWan inputs: cannot qualify original source") from exc
    require(Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() == root
            and git("rev-parse", "HEAD").decode().strip() == LIVEWAN_COMMIT,
            "original source must be the exact LiveWan gitlink " + LIVEWAN_COMMIT)
    require(not git("status", "--porcelain", "--untracked-files=no").strip(),
            "original LiveWan tracked source changed; clean pinned source required")
    files = {}
    for name in SOURCE_FILES:
        path = root / name
        expected = hashlib.sha256(git("show", f"{LIVEWAN_COMMIT}:{name}")).hexdigest()
        require(path.is_file() and file_sha256(path) == expected, f"original source hash changed: {name}")
        files[name] = expected
    return {"schema_version": "distillschedule.livewan_teacher_source.v1", "root": str(root),
            "head": LIVEWAN_COMMIT, "license": "Apache-2.0", "files": files}


def _safe_load(path):
    import torch
    try:
        # Mandatory admission before ANY original TeacherLatents unsafe read.
        # Never retry unrestricted deserialization on an unsupported global.
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError, ValueError, OSError, EOFError) as exc:
        raise ValueError(f"original LiveWan inputs: cannot safely restricted-load archive: {path}") from exc


def _tensor(value, shape, label):
    import torch
    require(type(value) is torch.Tensor and value.layout == torch.strided
            and value.device.type == "cpu" and value.dtype == torch.float16
            and tuple(value.shape) == tuple(shape) and bool(torch.isfinite(value).all()),
            f"{label} geometry/dtype/nonfinite gate failed; exact original float16 tensors required")


def _tensor_hash(value):
    import torch
    digest = hashlib.sha256()
    digest.update(fingerprint({"shape": list(value.shape), "dtype": str(value.dtype)}).encode())
    digest.update(memoryview(value.detach().contiguous().view(torch.uint8).numpy()))
    return digest.hexdigest()


def _prompts(path):
    value = _safe_load(path)
    require(isinstance(value, dict) and set(value) == {"pos", "neg", "prompts", "neg_prompt"},
            "original prompt bundle fields required")
    names = value["prompts"]
    require(isinstance(names, list) and bool(names)
            and all(type(s) is str and bool(s.strip()) for s in names)
            and type(value["neg_prompt"]) is str, "prompt text identity is invalid")
    _tensor(value["pos"], (len(names), 512, 4096), "positive prompt")
    _tensor(value["neg"], (1, 512, 4096), "negative prompt")
    return value


def _sample(path, prompts):
    value = _safe_load(path)
    require(isinstance(value, dict) and set(value) == {
        "latents", "prompt_idx", "seed", "size", "frames", "steps", "cfg"},
        "record requires the original gen_teacher archive fields")
    _tensor(value["latents"], CONTRACT["latent_shape"], "record latent")
    require(type(value["prompt_idx"]) is int and 0 <= value["prompt_idx"] < len(prompts["prompts"]),
            "record prompt index is outside the original bundle")
    require(type(value["frames"]) is int and value["frames"] == 81
            and type(value["steps"]) is int and value["steps"] == 40 and value["size"] == "640*368"
            and type(value["seed"]) is int and value["seed"] >= 0
            and type(value["cfg"]) in (int, float) and math.isfinite(value["cfg"]) and value["cfg"] > 0,
            "record gen_teacher metadata changed; ORIGINAL 81 frames/640*368/40 steps required")
    return value


def _derived_seed(seed, operation_key, stream):
    # Same operation-keyed derivation as build_livewan_crossscale_trace.py.
    # Local control RNGs never consume restored Python/Torch training RNG.
    return int.from_bytes(hashlib.sha256(f"{seed}/{operation_key}/{stream}".encode()).digest()[:8], "big") & ((1 << 63) - 1)


def _rows(records, cycles, seed):
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    rows = []
    for ordinal, index in enumerate(order):
        step, slot = divmod(ordinal, 16)
        cycle, cycle_step = divmod(step, 5)
        key = (f"step={step}/slot={slot}" if cycles == 1 else
               f"cycle={cycle}/step={cycle_step}/global_step={step}/slot={slot}")
        control = random.Random(_derived_seed(seed, key, "control"))
        world_frames = (3, 6, 9)[control.randrange(3)]
        rope_gap = control.randint(1, 240) if control.random() < .7 else 0
        record = records[index]
        rows.append({"logical_step": step, "logical_slot": slot, "cycle_index": cycle,
                     "cycle_step": cycle_step, "dataset_index": index, "record_id": record["record_id"],
                     "sample_path": record["path"], "sample_sha256": record["sha256"],
                     "prompt_idx": record["prompt_idx"], "operation_key": key,
                     "world_frames": world_frames, "rope_gap": rope_gap,
                     "seeds": {name: _derived_seed(seed, key, name) for name in STREAMS},
                     "generator_update": cycle_step == 0, "critic_update": True})
    return rows


def _inventory(data_root, prompts_path, *, source_root, cycles, seed, evidence_kind, input_kind=KIND):
    require(input_kind in (KIND, C51_KIND), "unsupported native input contract")
    maximum = 100 if input_kind == C51_KIND else 64
    require(type(cycles) is int and 1 <= cycles <= maximum,
            f"finite complete cohort count must be 1..{maximum}")
    require(type(seed) is int and 0 <= seed < 2**63, "explicit 63-bit operation seed required")
    require(evidence_kind in {"NATIVE_TEACHER_ARCHIVES_UNVERIFIED_QUALITY", "CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED"},
            "explicit native archive or CPU fixture evidence kind required")
    source = _source_identity(source_root)
    root, prompts_path = Path(data_root).expanduser().resolve(), Path(prompts_path).expanduser().resolve()
    require(root.is_dir() and prompts_path.is_file(), "missing teacher record directory or prompt bundle")
    files = sorted(root.glob("*.pt"))
    require(len(files) == 80 * cycles and all(p.is_file() for p in files),
            "exactly 80 distinct teacher records per complete G1/C5/globalB16 cohort required; no padding/replay")
    require(len({p.resolve() for p in files}) == len(files), "duplicate teacher record sources")
    prompt_sha = file_sha256(prompts_path)
    prompts = _prompts(prompts_path)
    prompt_hashes = [_tensor_hash(p) for p in prompts["pos"]]
    records = []
    for index, path in enumerate(files):
        path = path.resolve()
        before = file_sha256(path)
        value = _sample(path, prompts)
        require(file_sha256(path) == before, "record hash changed during restricted admission")
        pi = value["prompt_idx"]
        record = {"dataset_index": index, "path": str(path), "sha256": before,
                  "latent_sha256": _tensor_hash(value["latents"]), "prompt_idx": pi,
                  "prompt_text": prompts["prompts"][pi], "prompt_tensor_sha256": prompt_hashes[pi],
                  "metadata": {k: v for k, v in value.items() if k != "latents"}}
        record["record_id"] = "teacher:" + fingerprint(record)
        records.append(record)
    require(all(len({r[k] for r in records}) == len(records)
                for k in ("record_id", "path", "sha256", "latent_sha256")),
            "duplicate/copied/replayed teacher record payloads are not fresh inputs")
    require(file_sha256(prompts_path) == prompt_sha and _source_identity(source_root) == source,
            "original source or prompt hash changed during admission")
    value = {"schema_version": "distillschedule." + input_kind, "status": "ADMITTED_FINITE_NATIVE_INPUTS", "source": source,
             "data_root": str(root), "prompts": {"path": str(prompts_path), "sha256": prompt_sha},
             "cycles": cycles, "seed": seed, "evidence_kind": evidence_kind, "contract": deepcopy(CONTRACT),
             "records": records, "rows": _rows(records, cycles, seed),
             "claim_boundary": {"training_lifecycle": "NOT_IMPLEMENTED", "gpu_execution": "NOT_RUN",
                                "rng": "independent_operation_streams_NOT_upstream_sequential_RNG_continuation",
                                "quality": "supplied teacher-format archives; generation/quality not attested"}}
    if input_kind == C51_KIND:
        value["claim_boundary"]["training_lifecycle"] = "EXPLICIT_C51_STREAM_SOURCE_CPU_ONLY_NATIVE_GPU_NOT_RUN"
    return {**value, "input_sha256": fingerprint(value)}


def _original_dataset(source, root, prompts):
    """Import the actual admitted source file, without replacing wanstreamer."""
    path = Path(source["root"]) / "wanstreamer/data.py"
    require(file_sha256(path) == source["files"]["wanstreamer/data.py"], "original source hash changed before import")
    spec = importlib.util.spec_from_file_location("_distillschedule_livewan_data_" + source["files"]["wanstreamer/data.py"][:16], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # frames=None is intentional: no prefix crop, transform, or synthetic data.
    return module, module.TeacherLatents(str(root), str(prompts), frames=None)


def _publish_manifest(path, value):
    """Publish complete fsynced bytes exclusively, using the existing link pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # atomic publication; an existing output is never replaced
    finally:
        temporary.unlink(missing_ok=True)


def prepare_inputs(data_root, prompts_path, output, *, source_root, cycles, seed,
                   evidence_kind="NATIVE_TEACHER_ARCHIVES_UNVERIFIED_QUALITY", input_kind=KIND):
    """Admit supplied original archives; publish only a complete immutable file.

    Preparation imports the actual dataset only after every restricted archive
    gate. The usable consumer is InputLoader(input_kind=KIND, prepared=output),
    not the public C51 training command (whose lifecycle still rejects this).
    """
    value = _inventory(data_root, prompts_path, source_root=source_root, cycles=cycles, seed=seed,
                       evidence_kind=evidence_kind, input_kind=input_kind)
    _, dataset = _original_dataset(value["source"], value["data_root"], value["prompts"]["path"])
    require([str(Path(p).resolve()) for p in dataset.files] == [r["path"] for r in value["records"]],
            "actual original dataset record inventory changed")
    _publish_manifest(Path(output).expanduser().resolve(), value)
    return value


def prepare_c51_inputs(*args, **kwargs):
    """Explicit finite C51 stream, up to 100 fresh complete native cohorts.

    All existing archive/source/geometry/content gates remain mandatory. This
    version extends the horizon, not Teacher quality or sequential-upstream RNG
    claims. Legacy preparation remains capped at 64 and byte-identical.
    """
    return prepare_inputs(*args, **kwargs, input_kind=C51_KIND)
