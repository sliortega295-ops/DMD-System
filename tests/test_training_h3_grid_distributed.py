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
    real_loader,
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


def _cursor_failure_rank(rank, root, native_path, inputs_path, source_path, failure):
    """Actual dataset/cursor only: never construct a trainer or model."""
    import torch
    import torch.distributed as dist
    from distillschedule.training import distributed
    from distillschedule.open_workload.replicated import control_plane
    from distillschedule.training.h3_grid_inputs import NativeLatentCursor, validate_payloads
    sys.path.insert(0, str(Path(source_path) / "lightx2v_train"))
    root = Path(root)
    dist.init_process_group("gloo", init_method=(root / "cursor-rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    metadata = None
    original_gather = dist.all_gather_object
    original_load = torch.load
    groups = []
    def observed_gather(*args, **kwargs):
        groups.append(kwargs.get("group"))
        return original_gather(*args, **kwargs)
    try:
        metadata = dist.new_group(backend="gloo", timeout=timedelta(seconds=30))
        dist.all_gather_object = observed_gather
        with control_plane(metadata):
            native = json.loads(Path(native_path).read_text())
            inputs = json.loads(Path(inputs_path).read_text())
            semantic = distributed.checked(lambda: validate_payloads(inputs), "CPU actual cached input startup")
            assert all(row == semantic for row in distributed.gather(semantic))
            cursor = distributed.checked(lambda: NativeLatentCursor(
                real_loader(native, None), inputs, cycle=0, resume=False), "CPU actual cursor startup")
            first = next(cursor)
            assert first["meta"]["id"] == [inputs["rows"][cursor.order[0]]["record"]["id"]]
            target = Path(inputs["rows"][cursor.order[1]]["condition_path"])
            dist.barrier(group=metadata)
            if rank == 1:
                if failure == "missing":
                    target.unlink()
                elif failure == "changed":
                    with target.open("ab") as handle:
                        handle.write(b"post-startup mutation")
                else:
                    def fail_read(path, *args, **kwargs):
                        if Path(path).resolve() == target and kwargs.get("weights_only") is False:
                            raise OSError("rank-local original condition read failure")
                        return original_load(path, *args, **kwargs)
                    torch.load = fail_read
            dist.barrier(group=metadata)
            checks_before = len(groups)
            error = None
            try:
                next(cursor)
            except Exception as caught:
                error = caught
            else:
                # A successful return here would enter the next native model
                # call while its peer has already failed. No model is invoked.
                (root / f"model-edge-entered-{rank}").touch()
            result = {"rank": rank, "failure": failure,
                "error_type": type(error).__name__ if error else None, "error": str(error) if error else None,
                "metadata_checks_for_next": len(groups) - checks_before,
                "metadata_group_only": all(group is metadata for group in groups),
                "model_edge_entered": (root / f"model-edge-entered-{rank}").exists(),
                "cuda_initialized": torch.cuda.is_initialized()}
            (root / f"cursor-rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
            assert isinstance(error, ValueError) and "rank failures" in str(error), result
            expected_error = {"missing": "FileNotFoundError", "changed": "cached condition changed",
                              "load_error": "rank-local original condition read failure"}[failure]
            assert expected_error in str(error)
            assert result["metadata_checks_for_next"] == 1 and result["metadata_group_only"]
            assert not result["model_edge_entered"] and not result["cuda_initialized"]
    finally:
        torch.load = original_load
        dist.all_gather_object = original_gather
        if metadata is not None:
            dist.destroy_process_group(metadata)
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", ["missing", "changed", "load_error"])
def test_actual_gloo_rejects_rank_local_post_startup_cursor_failure_before_model(
        tmp_path, original_source, cached_data, failure):
    import torch.multiprocessing as mp
    tiny_config(tmp_path, cached_data)
    mp.spawn(_cursor_failure_rank, args=(str(tmp_path), str(cached_data[0]),
        str(tmp_path / "tiny-inputs.json"), str(original_source), failure), nprocs=2, join=True)
    results = [json.loads((tmp_path / f"cursor-rank-{rank}.json").read_text()) for rank in range(2)]
    assert results[0]["error"] == results[1]["error"]
    assert all(row["metadata_checks_for_next"] == 1 and row["metadata_group_only"]
               and not row["model_edge_entered"] and not row["cuda_initialized"] for row in results)
