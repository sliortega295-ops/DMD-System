"""Use the pinned H3 trainer; emit completion evidence after its gates.

No new loss, update, checkpoint, or cache implementation lives here. This is a
one-cycle receipt wrapper by default, not an alternative training loop. Its
default elapsed includes setup and is NOT the historical timing denominator.
H3_PLANNER_TWO_CYCLE=1 observes two native cycles, never a replay/sampler loop.
"""
from __future__ import annotations
import functools
import json
import os
from pathlib import Path
import sys
import time


def _measure_planner_two_cycle(trainer, original, *, torch, rank: int) -> None:
    from .planner_cycle_observer import H3PlannerCycleObserver

    if getattr(trainer, "planner_cycle_observer", None) is not None:
        raise RuntimeError("H3 planner cycle observer is already installed")
    observer = H3PlannerCycleObserver(torch=torch, rank=rank,
        source_head=os.environ["EXPECTED_INTEGRATION_HEAD"],
        calibration=os.environ.get("H3_PLANNER_CALIBRATION", "0") == "1",
        host_fenced_clock=os.environ.get("H3_PLANNER_CLOCK") == "host_fenced")
    trainer.planner_cycle_observer = observer
    tuple_capture = None
    start = time.perf_counter()
    try:
        if bool(os.environ.get("H3_ATTENTION_TUPLE_CAPTURE_CONFIG")) != bool(os.environ.get("H3_ATTENTION_TUPLE_CAPTURE_CONFIG_SHA256")):
            raise ValueError("H3 capture opt-in requires both packet path and checksum")
        if os.environ.get("H3_ATTENTION_TUPLE_CAPTURE_CONFIG"):
            # Source/capability/owner admission precedes attachment. The original
            # train/observer gates below remain authoritative and unchanged.
            from .attention_tuple_capture import attach_native_capture
            tuple_capture = attach_native_capture(trainer, observer, torch=torch, rank=rank)
        original(trainer)
        # Host-fenced mode already completed both stop events at its fourth
        # boundary fence. Preserve the existing final fence only in CUDA-event
        # mode, rather than adding a fifth device-wide synchronization.
        if not observer.host_fenced_clock:
            torch.cuda.synchronize()
        wrapper_elapsed = time.perf_counter() - start
        row = observer.finish_after_sync(trainer)
        if tuple_capture is not None:
            from .attention_tuple_capture import finish_native_capture
            finish_native_capture(tuple_capture, row)
        row["elapsed_including_trainer_setup_and_cycle_probes_s"] = wrapper_elapsed
        path = Path(os.environ["DISTILLSCHEDULE_RUN_ROOT"]) / "execution/ranks" / f"rank_{rank:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as f:
            json.dump(row, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
    except BaseException as error:
        if observer.action_trace is not None:
            # An interrupted optimizer cannot be rolled back; preserve the
            # observed prefix separately, never a successful completion receipt.
            try:
                path = Path(os.environ["DISTILLSCHEDULE_RUN_ROOT"]) / "execution/action_failures" / f"rank_{rank:03d}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x") as f:
                    json.dump(dict(status="FAIL_NATIVE_ACTION_RUN", error_type=type(error).__name__,
                        error=str(error), rank=rank, integration_head=observer.source_head,
                        cycle=observer.action_trace.cycle, completed_cycles=observer.cycles,
                        actual_updates=observer.updates, rows=observer.action_trace.rows,
                        dispatch=observer.action_program.receipt() if observer.action_program is not None else None,
                        rollback="NOT_SUPPORTED", retry="FORBIDDEN_SAME_PROCESS"), f, indent=2, allow_nan=False)
            except Exception as persistence_error:
                print(f"H3 action failure receipt could not be persisted: {persistence_error!r}; "
                      f"preserving native {type(error).__name__}: {error}", file=sys.stderr)
        raise
    finally:
        try:
            if tuple_capture is not None:
                from .attention_tuple_capture import close_native_capture
                close_native_capture(trainer, tuple_capture)
        finally:
            observer.close(trainer)
            del trainer.planner_cycle_observer


def main() -> None:
    from .planner_cycle_observer import planner_two_cycle_enabled
    two_cycle = planner_two_cycle_enabled()
    if bool(os.environ.get("H3_ATTENTION_TUPLE_CAPTURE_CONFIG")) != bool(os.environ.get("H3_ATTENTION_TUPLE_CAPTURE_CONFIG_SHA256")):
        raise ValueError("H3 capture opt-in requires both packet path and checksum")
    if os.environ.get("H3_ATTENTION_TUPLE_CAPTURE_CONFIG") and not two_cycle:
        raise ValueError("H3 attention capture requires the original two-cycle correctness entry")
    if os.environ.get("H3_ATTENTION_TUPLE_CAPTURE_CONFIG") and os.environ.get("DISTILLSCHEDULE_RUN_KIND") != "correctness":
        raise ValueError("H3 attention capture is correctness-only, never profile/native timing")
    from . import grid_entrypoint
    from .trainer import MiniMaxH3A100DmdTrainer
    from lightx2v_train.runtime.distributed import get_rank
    import torch
    original = MiniMaxH3A100DmdTrainer.train

    @functools.wraps(original)
    def measured(self):
        if two_cycle:
            return _measure_planner_two_cycle(self, original, torch=torch, rank=int(get_rank()))
        start = time.perf_counter()
        original(self)
        # Original train() executes all matched/grid/fusion/replay gates first.
        torch.cuda.synchronize()
        row = {
            "status": "PASS_NATIVE_RUNTIME_CONTRACTS", "rank": int(get_rank()),
            "integration_head": os.environ["EXPECTED_INTEGRATION_HEAD"],
            "semantics": "approximate_grid1000", "census": self.matched_cycle_census.snapshot(),
            "elapsed_including_trainer_setup_s": time.perf_counter() - start,
            "formal_timing_eligible": False, "numerical_cross_reference": "NOT_RUN_BY_THIS_WRAPPER",
            "boundary_policy": os.environ["H3_ACTIVATION_POLICY"],
            "fa3_replay_storage": os.environ["H3_FA3_REPLAY_CACHE_STORAGE"],
            "grid_replay": dict(self.grid_replay_registration.stats),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        for name in ("fa3_replay_cache_registration", "fused_block_pointwise_registration", "lora_scale1_elision_registration"):
            registration = getattr(self, name, None)
            if registration is not None and callable(getattr(registration, "snapshot", None)):
                row[name] = registration.snapshot()
        path = Path(os.environ["DISTILLSCHEDULE_RUN_ROOT"]) / "execution/ranks" / f"rank_{get_rank():03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as f:
            json.dump(row, f, indent=2, sort_keys=True)
            f.write("\n")
    MiniMaxH3A100DmdTrainer.train = measured
    try:
        grid_entrypoint.main()
    finally:
        MiniMaxH3A100DmdTrainer.train = original


if __name__ == "__main__":
    main()
