"""Actual CPU/Gloo rank failures and atomic H3 lifecycle shards.

Two CPU ranks qualify control/checkpoint mechanics, never native world16
pretrained parity, CUDA capacity or FSDP numerical equivalence.
"""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import pytest

from test_training_h3_grid import (
    original_source as original_source,
    cached_data as cached_data,
    tiny_config,
    tiny_original_trainer,
)


def _rank(rank, root, native_path, public_raw, inputs_path, source_path, scenario):
    import torch
    import torch.distributed as dist
    sys.path.insert(0, str(Path(source_path) / "lightx2v_train"))
    root = Path(root)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2", GROUP_RANK="0")
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        from distillschedule.training.config import TrainingConfig
        from distillschedule.training.h3_grid_training import collective_admission, _source, LIGHTX2V
        from distillschedule.training.h3_grid_inputs import load_inputs
        from distillschedule.training.checkpoint import rng_state, pack_state, restore_rng
        from distillschedule.search.resources import observe_source_identity
        from distillschedule.workloads.best_known.registry import file_sha256
        from test_training_end_to_end import equal
        public = TrainingConfig.from_dict(public_raw, source=root / "public.json")
        if scenario in ("source_failure", "input_failure", "controls_failure"):
            if scenario == "controls_failure" and rank == 1:
                raw = public.to_dict(); raw["trainer"]["checkpoint_every"] = 2
                public = TrainingConfig.from_dict(raw, source=root / "public.json")
            def admit():
                # Actual source/input owners, not a fake model or missing
                # dependency shim. This exercises the collective seam only.
                source = Path(source_path) if scenario != "source_failure" or rank == 0 else root / "missing-source"
                _source({"path": str(source), "head": LIGHTX2V}, LIGHTX2V, "LightX2V")
                path = inputs_path if scenario != "input_failure" or rank == 0 else root / "missing-input.json"
                inputs = load_inputs(path, config_path=native_path, cycles=3)
                return ("W5", observe_source_identity(source), {}, {
                    "assets": {"native_config": {"sha256": file_sha256(Path(native_path))}}, "inputs": inputs})
            with pytest.raises(ValueError, match="rank failures|ranks disagree"):
                collective_admission(public, admit)
            assert not (root / f"model-constructed-{rank}").exists()
        else:
            native = json.loads(Path(native_path).read_text())
            inputs = json.loads(Path(inputs_path).read_text())
            torch.manual_seed(19 + rank)
            initial_rng = rng_state()
            control, full = tiny_original_trainer(root, native, inputs, public)
            control.train()
            expected, expected_rng = pack_state(full.state), rng_state()
            dist.barrier()
            if rank == 0:
                import shutil
                shutil.rmtree(root / "run")
            dist.barrier()
            restore_rng(initial_rng)
            trainer, partial = tiny_original_trainer(root, native, inputs, public)
            original = partial.after_cycle
            def stop(trainer, cycle):
                original(trainer, cycle)
                raise InterruptedError("published all-rank complete boundary")
            partial.after_cycle = stop
            with pytest.raises(InterruptedError):
                trainer.train()
            checkpoint = root / "run/checkpoints/cycle-000001"
            manifest = json.loads((checkpoint / "manifest.json").read_text())
            assert set(manifest["files"]) == {"rank-00000.pt", "rank-00001.pt"}
            assert manifest["cursor"] == 6 and manifest["versions"] == {"student": 1, "fake": 5, "ema": 0}
            if rank == 0 and scenario != "exact_resume":
                shard = checkpoint / "rank-00001.pt"
                if scenario == "missing_shard":
                    shard.unlink()
                else:
                    payload = torch.load(shard, weights_only=True)
                    payload["state"]["clocks"]["value"]["score_serial"] += 1
                    torch.save(payload, shard)
                    manifest["files"][shard.name] = file_sha256(shard)
                    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
                    (checkpoint / "_SUCCESS").write_text(file_sha256(checkpoint / "manifest.json") + "\n")
            dist.barrier()
            resumed, continuation = tiny_original_trainer(root, native, inputs, public, resume=checkpoint)
            before = deepcopy(resumed.model.denoiser_module().state_dict())
            if scenario == "exact_resume":
                resumed.train()
                assert equal(expected, pack_state(continuation.state))
                assert equal(expected_rng, rng_state())
                assert continuation.cursor.cursor == 18
            else:
                with pytest.raises(ValueError, match="rank failures"):
                    resumed.train()
                assert equal(before, resumed.model.denoiser_module().state_dict())
        assert torch.cuda.is_initialized() is False
        (root / f"rank-{rank}.json").write_text(json.dumps({"status": "PASS_CPU_GLOO_ONLY", "scenario": scenario}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scenario", ["source_failure", "input_failure", "controls_failure", "exact_resume", "missing_shard", "malformed_rank_clock"])
def test_actual_gloo_rank_failure_and_complete_original_state(
        tmp_path, original_source, cached_data, scenario):
    import torch.multiprocessing as mp
    _, _, public = tiny_config(tmp_path, cached_data)
    mp.spawn(_rank, args=(str(tmp_path), str(cached_data[0]), public.to_dict(),
             str(tmp_path / "tiny-inputs.json"), str(original_source), scenario), nprocs=2, join=True)
    assert [json.loads((tmp_path / f"rank-{rank}.json").read_text())["status"] for rank in range(2)] == ["PASS_CPU_GLOO_ONLY"] * 2
