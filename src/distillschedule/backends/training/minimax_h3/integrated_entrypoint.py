"""Use the pinned H3 trainer unchanged; emit completion evidence after its gates.

No new loss, update, checkpoint, or cache implementation lives here. This is a
one-cycle receipt wrapper, not an alternative training loop. Its elapsed time
includes trainer setup and is explicitly NOT the historical timing denominator.
"""
from __future__ import annotations
import functools
import json
import os
from pathlib import Path
import time


def main() -> None:
    from . import grid_entrypoint
    from .trainer import MiniMaxH3A100DmdTrainer
    from lightx2v_train.runtime.distributed import get_rank
    import torch
    original = MiniMaxH3A100DmdTrainer.train

    @functools.wraps(original)
    def measured(self):
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
