"""Original native LiveWan data consumption and complete-cycle logical cursors.

Public preparation lives in data.livewan_inputs. This module does not provide
C51 model/pipeline restart or a new training loop.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from distillschedule.data.livewan_inputs import (
    KIND, SCHEMA, C51_KIND, C51_SCHEMA, EPOCH_POLICY,
    _inventory, _original_dataset, _prompts, _sample, _source_identity, require,
)
from distillschedule.runtime.livewan_crossscale_worker import OperationTraceRow
from distillschedule.search.livewan_request import LiveWanRequest
from distillschedule.workloads.best_known.registry import _object, file_sha256, fingerprint

CURSOR_SCHEMA = "distillschedule.livewan_teacher_cursor.v1"


@dataclass(frozen=True)
class NativeTeacherRow(OperationTraceRow):
    record_id: str = ""
    prompt_idx: int = -1


@dataclass(frozen=True)
class EpochTeacherRow(NativeTeacherRow):
    draw_index: int = -1
    epoch_index: int = -1
    epoch_offset: int = -1
    draw_id: str = ""
    cohort_id: str = ""


class NativeTeacherInputs:
    """Source-qualified original dataset, consumed by the real worker loader."""
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        require(self.path.is_file(), "missing prepared native input manifest")
        manifest_sha = file_sha256(self.path)
        value = _object(self.path)
        unsigned = {k: v for k, v in value.items() if k != "input_sha256"}
        require(value.get("schema_version") in (SCHEMA, C51_SCHEMA) and value.get("input_sha256") == fingerprint(unsigned),
                "manifest identity/hash changed or unsupported input schema")
        policy = value.get("input_policy")
        require("input_policy" not in value or type(policy) is dict and policy.get("schema_version") == EPOCH_POLICY,
                "unsupported native epoch input policy")
        expected = _inventory(value["data_root"], value["prompts"]["path"], source_root=value["source"]["root"],
                              cycles=value["cycles"], seed=value["seed"], evidence_kind=value["evidence_kind"],
                              input_kind=C51_KIND if value["schema_version"] == C51_SCHEMA else KIND,
                              input_policy=policy["schema_version"] if policy is not None else None)
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
        # The original owner must reopen the same canonical files we restricted-load.
        # Its constructor otherwise retains a mutable symlink spelling.
        self.dataset.files = tuple(record["path"] for record in value["records"])
        self._row_type = EpochTeacherRow if policy is not None else NativeTeacherRow
        rows = []
        for row in value["rows"]:
            original = OperationTraceRow.from_mapping(row, total_steps=self.steps)
            fields = {**original.__dict__, "seeds": MappingProxyType(dict(original.seeds))}
            if policy is not None:
                fields.update({key: row[key] for key in ("draw_index", "epoch_index", "epoch_offset", "draw_id", "cohort_id")})
            rows.append(self._row_type(**fields, record_id=row["record_id"], prompt_idx=row["prompt_idx"]))
        self.rows = tuple(rows)
        self._unchanged()

    @property
    def records(self):
        return deepcopy(self._document["records"])

    @property
    def input_policy(self):
        return deepcopy(self._document.get("input_policy"))

    @property
    def source_reuse_across_cycles(self):
        return self.input_policy is not None and len(self._document["records"]) < len(self.rows)

    def row(self, logical_step, logical_slot):
        """Exact native operation address, never cycling or consuming a record."""
        require(type(logical_step) is int and 0 <= logical_step < self.steps
                and type(logical_slot) is int and 0 <= logical_slot < self.global_batch,
                "finite native row address outside admitted horizon/topology")
        self._unchanged()
        return self.rows[logical_step * self.global_batch + logical_slot]

    def _unchanged(self):
        require(self.path.is_file() and file_sha256(self.path) == self._manifest_sha256,
                "prepared manifest hash changed")
        require(self.prompts_path.is_file() and file_sha256(self.prompts_path) == self.prompts_sha256,
                "prompt bundle hash changed")
        require(_source_identity(self._document["source"]["root"]) == self._document["source"],
                "original source identity changed")

    def _record(self, row):
        require(type(row) is self._row_type and type(row.logical_step) is int and type(row.logical_slot) is int
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
        if self.cursor == self.inputs.steps:
            raise StopIteration
        row = self.inputs.rows[self.cursor * 16 + self.rank]
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
