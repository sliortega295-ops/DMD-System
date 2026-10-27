"""Fresh-cohort lifetime policy for the SAME native frozen-down/FA3 runtime.

Only source qualification and cycle/version boundaries differ from the frozen
two-cycle policy. Named Linear/storage selection, SAC FIFO checks, autograd
issuing-thread scopes and original backward/commit hooks stay in NativeDownSAC.
"""
from dataclasses import dataclass
import math
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from .down_contract import GRAPHS, ScopeKey, require
from .down_replay import NativeDownSAC, CHECKPOINT_SHA, sha

# Already-reviewed integration overlay, not a rewrite of FSDP_SHA or its lock.
# The previous source's SAC functions are byte-identical; only the native
# optimizer-owned clipping and explicit dp_world_size metadata changed.
INTEGRATED_FSDP_SHA = "dd2fd0861f058b731145b9868524c13542d38c9ad19973de90b99088aa4a5d50"
INTEGRATED_FSDP_COMMIT = "0b93933107f4762b5d2cb29867eb749b4cb67a5d"


def validate_fsdp_source(fsdp):
    import json
    from distillschedule.workloads.best_known.registry import repository_root
    root = repository_root()
    target = "src/distillschedule/runtime/fsdp2_static.py"
    require(Path(fsdp.__file__).resolve() == (root / target).resolve()
            and sha(fsdp.__file__) == INTEGRATED_FSDP_SHA, "EXACT_REVIEWED_STREAM_FSDP_SOURCE")
    lock = json.loads((root / "docs/workloads/source_layout.lock.json").read_text())
    # The source-layout registry is versioned independently of this seam.
    row = lock["files"][target]
    require(row["target"] == target and row["source_commit"] == INTEGRATED_FSDP_COMMIT
            and row["reviewed_source_sha256"] == row["integrated_sha256"] == INTEGRATED_FSDP_SHA,
            "EXISTING_REVIEWED_STREAM_FSDP_OVERLAY_CHANGED")
    return row


@dataclass(frozen=True)
class StreamScopeKey(ScopeKey):
    def validate(self):
        require(all(type(getattr(self, name)) is int for name in
                    ("rank", "cycle", "index", "logical_step", "version", "block")), "EXACT_SCOPE_INTEGER_TYPES")
        require(0 <= self.rank < 16 and self.role == "critic" and self.stage == "CRITIC", "NATIVE_C_TRAINING_ONLY")
        require(self.cycle >= 1 and 0 <= self.index < 5
                and self.logical_step == 1000 + 5 * (self.cycle - 1) + self.index,
                "FRESH_CYCLE_INDEX_SOURCE_STEP")
        require(self.version == 10 + 5 * (self.cycle - 1) + self.index, "EXACT_PHYSICAL_C_VERSION")
        require(self.graph in GRAPHS and 0 <= self.block < 40, "EXACT_GRAPH_BLOCK")
        return self


def qualify_rows(trace, rank):
    """Bind each fresh row count to its real rank manifest and tensor hashes.

    This is the original transformer's floor current/short and replicate-pad
    ceil mid/long patch geometry, not permission for a new model or shape.
    Actual named-down operands are checked again by select_down at every call.
    """
    from distillschedule.training.helios_gan_inputs import GANCohortTask, validate_controls, keys
    require(type(trace.task) is GANCohortTask and type(rank) is int and 0 <= rank < 16,
            "EXPLICIT_FRESH_GAN_COHORT_REQUIRED")
    validate_controls(trace.task, trace._control)
    control = trace._control[rank]
    source = Path(control.root) / "trace.json"
    # Verify the actual manifest bytes against the already admitted object;
    # immutable stream/bundle hashes are authenticated by load_cohort.
    import json
    require(json.loads(source.read_text()) == control.manifest, "FRESH_RANK_SOURCE_CHANGED")
    tensors = control.manifest["tensors"]
    rows, descriptors = {}, []
    for step in trace.logical_steps:
        selected = keys(step, "critic")
        shape = tensors[selected["entry_noise"]]["shape"]
        require(shape == [1, 16, 9, 48, 80], "QUALIFIED_GAN_CURRENT_GEOMETRY")
        history = 0
        for role, frames, patch in (("history_short", 2, (1, 2, 2)),
                                    ("history_mid", 2, (2, 4, 4)),
                                    ("history_long", 16, (4, 8, 8))):
            value = tensors[selected[role]]
            require(value["shape"] == [1, 16, frames, 48, 80]
                    and value["dtype"] == "bfloat16", "QUALIFIED_UNCROPPED_HISTORY_GEOMETRY")
            sizes = value["shape"][2:]
            history += math.prod((size // stride if role == "history_short" else
                                  (size + stride - 1) // stride)
                                 for size, stride in zip(sizes, patch, strict=True))
        current = math.prod((shape[2], shape[3] // 2, shape[4] // 2))
        cropped = math.prod((shape[2], shape[3] // 4, shape[4] // 4))
        require(history == 2400 and current == 8640 and cropped == 2160,
                "SOURCE_PINNED_GAN_TOKEN_GEOMETRY")
        rows.update({(step, "B1_DENOISE"): current + history,
                     (step, "B3_GAN_R1"): 3 * (cropped + history)})
        descriptors.append({"step": step, "tensor_sha256": {
            role: tensors[key]["sha256"] for role, key in selected.items()}})
    identity = {"schema": "distillschedule.gan_fresh_down_rows.v1", "rank": rank,
                "task": trace.task.to_dict(), "bundle_hash": trace.bundle_hash,
                "source_entry_path": str(source), "source_entry_sha256": file_sha256(source),
                "graphs": [{"logical_step": step, "graph": graph, "rows": value}
                           for (step, graph), value in rows.items()], "descriptors": descriptors}
    identity["qualification_sha256"] = fingerprint(identity)
    return rows, identity


class NativeDownStreamSAC(NativeDownSAC):
    """One hook installation, fresh stores per graph, never serialized caches."""
    def __init__(self, *, torch, rank):
        super().__init__(torch=torch, rank=rank, enabled=True, rows={})
        self.completed = self.started_at = None
        self.cohort = None
        self.state.update(schema="gan.native_down_stream_sac.v1", qualifications=[])

    def scope_key(self, *args):
        return StreamScopeKey(*args).validate()

    def validate_sources(self, fsdp, checkpoint):
        record = validate_fsdp_source(fsdp)
        require(sha(checkpoint.__file__) == CHECKPOINT_SHA, "EXACT_NATIVE_PUBLIC_SAC_SOURCE")
        self.state["current_fsdp_source"] = record

    def validate_native_scope(self, generator, critic, ema):
        require((int(generator.global_steps), int(critic.global_steps), int(ema.optimization_step)) == (2, 10, 800),
                "EXACT_NATIVE_INITIAL_VERSIONS")
        require(not self.rows, "STREAM_ROWS_MUST_BE_QUALIFIED_AFTER_RESTORE")

    def begin_cycle(self, index, trace):
        from distillschedule.training.helios_gan_training import expected_versions
        require(type(index) is int and index >= 0 and self.cohort is None
                and self.current is None and not self.stores, "FRESH_DRAINED_STREAM_CYCLE_REQUIRED")
        if self.started_at is None:
            self.started_at = self.completed = index
        require(index == self.completed and self.calls == 5 * (index - self.started_at),
                "SEQUENTIAL_STREAM_COHORT_REQUIRED")
        wanted = expected_versions(index)
        require((int(self.generator.global_steps), int(self.critic.global_steps), int(self.ema.optimization_step)) ==
                tuple(wanted[k] for k in ("generator", "critic", "ema")), "STREAM_CYCLE_VERSION_HISTORY")
        require(trace.task.index == index, "FRESH_TRACE_COHORT_INDEX")
        self.rows, qualification = qualify_rows(trace, self.rank)
        self.state["qualifications"].append(qualification)
        self.cohort = index

    def next_critic_scope(self, kwargs):
        require(self.cohort is not None and self.started_at is not None, "QUALIFY_FRESH_COHORT_BEFORE_CRITIC")
        index = self.calls - 5 * (self.cohort - self.started_at)
        require(0 <= index < 5, "EXACT5_NATIVE_C_UPDATES_PER_COHORT")
        cycle = self.cohort + 1
        step, version = 1000 + 5 * self.cohort + index, 10 + 5 * self.cohort + index
        require(kwargs.get("step") == step and kwargs.get("rank") == self.rank
                and int(self.critic.global_steps) == version and int(self.generator.global_steps) == 2 + cycle
                and int(self.ema.optimization_step) == 800 + self.cohort + (index > 0),
                "SEQUENTIAL_FRESH_NATIVE_CYCLE_SOURCE_VERSION_HISTORY")
        return cycle, index, step, version

    def cycle_boundary(self, completed):
        from distillschedule.training.helios_gan_training import expected_versions
        require(self.cohort is not None and completed == self.cohort + 1, "ONE_COMPLETE_STREAM_COHORT")
        cycles = completed - self.started_at
        require(self.calls == 5 * cycles and self.current is None and not self.stores
                and len(self.state["graph_drains"]) == 10 * cycles, "COMPLETE_STREAM_UPDATES_AND_DRAINED_GRAPHS")
        require(self.state["combined_context_count"] == 400 * cycles
                and self.state["extra_down"] == dict(original=400 * cycles, replay=400 * cycles)
                and self.state["extra_FA3"] == dict(original=0, replay=0) and not self.state["audit_errors"],
                "EXACT_STREAM_DOWNONLY_TRAINING_COUNTERS")
        wanted = expected_versions(completed)
        require((int(self.generator.global_steps), int(self.critic.global_steps), int(self.ema.optimization_step)) ==
                tuple(wanted[k] for k in ("generator", "critic", "ema")), "COMPLETE_STREAM_NATIVE_CLOCKS")
        self.completed, self.cohort, self.rows = completed, None, {}

    def finish(self):
        require(self.started_at is not None and self.completed > self.started_at and self.cohort is None
                and self.current is None and not self.stores and not self.rows
                and not self.state["audit_errors"], "COMPLETE_DRAINED_GAN_STREAM")

    def abort(self):
        # No autograd/FSDP/Parameter mutation: drop only policy-owned references.
        self.stores.clear()
        self.current, self.cohort, self.rows = None, None, {}
        self.state["failed"] = True
