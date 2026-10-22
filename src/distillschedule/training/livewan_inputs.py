"""Finite fresh inputs owned by the pinned original LiveWan TeacherLatents.

This is a data-consumer seam, not a new C51 trainer or a restart checkpoint.
No trace replay, prefix cropping, model import, global RNG draw or data download
is performed. Supplied teacher-format archives do not attest pretrained quality.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import pickle
import random
import subprocess
from types import MappingProxyType

from distillschedule.runtime.livewan_crossscale_worker import OperationTraceRow
from distillschedule.search.livewan_request import LiveWanRequest
from distillschedule.workloads.best_known.registry import (
    _object, file_sha256, fingerprint, write_new_json,
)
from distillschedule.workloads.livewan_crossscale import LIVEWAN_COMMIT

KIND = "original_livewan_teacher_cohorts.v1"
SCHEMA = "distillschedule." + KIND
CURSOR_SCHEMA = "distillschedule.livewan_teacher_cursor.v1"
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


def _inventory(data_root, prompts_path, *, source_root, cycles, seed, evidence_kind):
    require(type(cycles) is int and 1 <= cycles <= 64, "finite complete cohort count must be 1..64")
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
    value = {"schema_version": SCHEMA, "status": "ADMITTED_FINITE_NATIVE_INPUTS", "source": source,
             "data_root": str(root), "prompts": {"path": str(prompts_path), "sha256": prompt_sha},
             "cycles": cycles, "seed": seed, "evidence_kind": evidence_kind, "contract": deepcopy(CONTRACT),
             "records": records, "rows": _rows(records, cycles, seed),
             "claim_boundary": {"training_lifecycle": "NOT_IMPLEMENTED", "gpu_execution": "NOT_RUN",
                                "rng": "independent_operation_streams_NOT_upstream_sequential_RNG_continuation",
                                "quality": "supplied teacher-format archives; generation/quality not attested"}}
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


def prepare_inputs(data_root, prompts_path, output, *, source_root, cycles, seed,
                   evidence_kind="NATIVE_TEACHER_ARCHIVES_UNVERIFIED_QUALITY"):
    """Admit supplied original archives; publish only a complete immutable file.

    Preparation imports the actual dataset only after every restricted archive
    gate. The usable consumer is InputLoader(input_kind=KIND, prepared=output),
    not the public C51 training command (whose lifecycle still rejects this).
    """
    value = _inventory(data_root, prompts_path, source_root=source_root, cycles=cycles, seed=seed,
                       evidence_kind=evidence_kind)
    _, dataset = _original_dataset(value["source"], value["data_root"], value["prompts"]["path"])
    require([str(Path(p).resolve()) for p in dataset.files] == [r["path"] for r in value["records"]],
            "actual original dataset record inventory changed")
    write_new_json(Path(output).expanduser().resolve(), value)
    return value


@dataclass(frozen=True)
class NativeTeacherRow(OperationTraceRow):
    record_id: str = ""
    prompt_idx: int = -1


class NativeTeacherInputs:
    """Source-qualified original dataset, consumed by the real worker loader."""
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        require(self.path.is_file(), "missing prepared native input manifest")
        manifest_sha = file_sha256(self.path)
        value = _object(self.path)
        unsigned = {k: v for k, v in value.items() if k != "input_sha256"}
        require(value.get("schema_version") == SCHEMA and value.get("input_sha256") == fingerprint(unsigned),
                "manifest identity/hash changed or unsupported input schema")
        expected = _inventory(value["data_root"], value["prompts"]["path"], source_root=value["source"]["root"],
                              cycles=value["cycles"], seed=value["seed"], evidence_kind=value["evidence_kind"])
        require(value == expected, "source/prompt/record/operation identity changed from prepared manifest")
        require(file_sha256(self.path) == manifest_sha, "manifest hash changed during native admission")
        self._document, self._manifest_sha256 = value, manifest_sha
        self.input_sha256 = value["input_sha256"]
        self.request = LiveWanRequest()
        self.cycles, self.steps, self.global_batch = value["cycles"], 5 * value["cycles"], 16
        self.prompts_path, self.prompts_sha256 = Path(value["prompts"]["path"]), value["prompts"]["sha256"]
        self.prompts = _prompts(self.prompts_path)
        self._unchanged()
        self.dataset_source, self.dataset = _original_dataset(value["source"], value["data_root"], self.prompts_path)
        require([str(Path(p).resolve()) for p in self.dataset.files] == [r["path"] for r in value["records"]]
                and self.dataset.frames is None, "actual original dataset inventory/geometry changed")
        rows = []
        for row in value["rows"]:
            original = OperationTraceRow.from_mapping(row, total_steps=self.steps)
            fields = {**original.__dict__, "seeds": MappingProxyType(dict(original.seeds))}
            rows.append(NativeTeacherRow(**fields, record_id=row["record_id"], prompt_idx=row["prompt_idx"]))
        self.rows = tuple(rows)
        self._unchanged()

    @property
    def records(self):
        return deepcopy(self._document["records"])

    def _unchanged(self):
        require(self.path.is_file() and file_sha256(self.path) == self._manifest_sha256,
                "prepared manifest hash changed")
        require(self.prompts_path.is_file() and file_sha256(self.prompts_path) == self.prompts_sha256,
                "prompt bundle hash changed")
        require(_source_identity(self._document["source"]["root"]) == self._document["source"],
                "original source identity changed")

    def _record(self, row):
        require(type(row) is NativeTeacherRow and type(row.logical_step) is int and type(row.logical_slot) is int
                and 0 <= row.logical_step < self.steps and 0 <= row.logical_slot < 16
                and row == self.rows[row.logical_step * 16 + row.logical_slot], "native input row identity changed")
        self._unchanged()
        record = self._document["records"][row.dataset_index]
        path = Path(record["path"])
        require(path.is_file() and file_sha256(path) == record["sha256"], "native record hash changed before consumption")
        return record

    @property
    def negative(self):
        self._unchanged()
        return self.prompts["neg"][0].float().unsqueeze(0)

    def load(self, row):
        import torch
        record = self._record(row)
        admitted = _sample(Path(record["path"]), self.prompts)
        require(admitted["prompt_idx"] == row.prompt_idx, "native record prompt binding changed")
        # This is the ORIGINAL TeacherLatents.__getitem__, not a replacement.
        latent, context, pi = self.dataset[row.dataset_index]
        require(pi == row.prompt_idx and tuple(latent.shape) == (16, 21, 46, 80)
                and tuple(context.shape) == (512, 4096) and latent.dtype == context.dtype == torch.float32
                and torch.equal(latent, admitted["latents"].float())
                and torch.equal(context, self.prompts["pos"][pi].float()),
                "actual original dataset sample/prompt/geometry differs from admission")
        self._record(row)
        return latent, context, pi

    def context(self, row, *, device):
        self._record(row)
        return self.prompts["pos"][row.prompt_idx].float().to(device, non_blocking=True).unsqueeze(0)


class NativeTeacherCursor:
    """Finite logical world16 rank cursor; serializable via checkpoint.pack_state.

    Cursor persistence is ONLY data state. It does not make C51's in-flight
    pipeline, models, named Adam or training RNG restartable. Each advancement
    succeeds only after actual original dataset + InputLoader consumption.
    """
    def __init__(self, loader, *, rank, world_size=16):
        require(getattr(loader, "input_kind", None) == KIND and type(loader.inputs) is NativeTeacherInputs,
                "actual fresh InputLoader owner required for native cursor")
        require(type(world_size) is int and world_size == 16 and type(rank) is int and 0 <= rank < 16,
                "native rank cursor requires explicit logical world16 topology")
        self.loader, self.inputs, self.rank, self.world_size = loader, loader.inputs, rank, world_size
        self.cursor = 0

    def __iter__(self):
        return self

    def peek(self):
        self.inputs._unchanged()
        if self.cursor == self.inputs.steps:
            raise StopIteration
        return self.inputs.rows[self.cursor * 16 + self.rank]

    def __next__(self):
        row = self.peek()
        sample = self.loader.load(row)
        self.cursor += 1
        return row, sample

    def validate_checkpoint_boundary(self):
        require(self.cursor % 5 == 0, "data cursor must be at a complete G1/C5 cycle boundary")
        self.inputs._unchanged()

    def _state(self, cursor):
        return {"schema_version": CURSOR_SCHEMA, "input_sha256": self.inputs.input_sha256,
                "rank": self.rank, "world_size": self.world_size, "cursor": cursor, "cycle": cursor // 5,
                "next_operation_key": (self.inputs.rows[cursor * 16 + self.rank].operation_key
                                       if cursor < self.inputs.steps else None)}

    def state_dict(self):
        self.validate_checkpoint_boundary()
        return self._state(self.cursor)

    def validate_state_dict(self, value):
        require(isinstance(value, dict) and type(value.get("cursor")) is int
                and 0 <= value["cursor"] <= self.inputs.steps and value["cursor"] % 5 == 0,
                "saved cursor must describe a complete G1/C5 cycle")
        require(fingerprint(value) == fingerprint(self._state(value["cursor"])),
                "saved data cursor rank/topology/input/seed identity changed")
        self.inputs._unchanged()

    def load_state_dict(self, value):
        # Existing checkpoint.unpack_state stages this same admission before
        # mutating any live model/optimizer/stateful owner on any rank.
        self.validate_state_dict(value)
        self.cursor = value["cursor"]

    def seek(self, cycle):
        require(type(cycle) is int and 0 <= cycle <= self.inputs.cycles, "invalid finite cursor cycle")
        self.load_state_dict(self._state(5 * cycle))
