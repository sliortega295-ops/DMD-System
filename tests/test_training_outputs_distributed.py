"""Collective checkpoint output, with actual CPU DDP/FSDP2 state and forwards."""
import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_training_end_to_end import equal

from distillschedule.data import prepare
from distillschedule.examples.tiny_av import build_fixture
from distillschedule.training.config import TrainingConfig, load_training_config


def _output_worker(rank, rendezvous, config_path, root, reduction, world):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(rank % 2),
                      LOCAL_WORLD_SIZE="2", GROUP_RANK=str(rank // 2))
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=world, timeout=timedelta(seconds=60))
    try:
        from distillschedule.training.session import evaluate, export, train
        config = load_training_config(config_path)
        raw = config.to_dict()
        raw["data"]["sharding"] = "rank_strided"
        raw["trainer"].update(cycles=1, output_dir=str(Path(root) / "training"))
        raw["trainer"]["distributed"] = {"backend": "gloo", "timeout_seconds": 60}
        raw["execution"] = {"mode": "manual", "implementation": {
            "backend": "torch-callable-replicated-v1", "reduction": reduction,
            "bucket_cap_mb": 1 if reduction == "ddp_overlap" else 0,
            "local": {"backend": "torch-callable-resident-v1", "share_immutable": False,
                      "checkpoint_bindings": []}}}
        if reduction.startswith("hsdp2"):
            raw["execution"]["implementation"]["node_groups"] = [[0, 1], [2, 3]]
        config = TrainingConfig.from_dict(raw, source=config.source)
        trained = train(config)
        from distillschedule.training import session
        original_load = session.load_callable
        def custom_exporter(reference):
            factory, identity = original_load(reference)
            if reference != config.model["factory"]:
                return factory, identity
            def build(parameters):
                components = factory(parameters)
                components.export = lambda *args: pytest.fail("unsupported custom distributed callback ran")
                components.load_export = lambda *args: None
                return components
            return build, identity
        session.load_callable = custom_exporter
        try:
            with pytest.raises(ValueError, match="custom distributed export"):
                export(config, trained["checkpoint"], Path(root) / "unsupported-export")
        finally:
            session.load_callable = original_load
        assert not (Path(root) / "unsupported-export").exists()
        exported = export(config, trained["checkpoint"], Path(root) / "weights")
        assert exported["world_size"] == world
        original_generate = session._generate_values
        def unexpected_generation(*args, **kwargs):
            pytest.fail("generation callback ran before collective parameter admission")
        session._generate_values = unexpected_generation
        try:
            with pytest.raises(ValueError, match="shared evaluation request"):
                evaluate(config, trained["checkpoint"], Path(root) / "divergent-evaluation",
                         parameters={"steps": rank + 1})
            with pytest.raises(ValueError, match="evaluation request admission"):
                evaluate(config, trained["checkpoint"], Path(root) / "invalid-evaluation",
                         parameters={"steps": float("nan") if rank == 1 else 1})
        finally:
            session._generate_values = original_generate
        assert not (Path(root) / "divergent-evaluation").exists()
        assert not (Path(root) / "invalid-evaluation").exists()
        if reduction == "post_backward":
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor import DTensor, Shard
            mesh = init_device_mesh("cpu", (world,))
            def dtensor_factory(reference):
                factory, identity = original_load(reference)
                if reference != config.model["factory"]:
                    return factory, identity
                def build(parameters):
                    components = factory(parameters)
                    generate = components.generate
                    def distributed_output(state, feeds, parameters):
                        local = generate(state, feeds, parameters).detach().contiguous()
                        if parameters.get("nonfinite_rank") == rank:
                            local.fill_(float("nan"))
                        shape = torch.Size((local.shape[0] * world, *local.shape[1:]))
                        return DTensor.from_local(local, mesh, [Shard(0)], run_check=False,
                                                  shape=shape, stride=torch.empty(shape).stride())
                    components.generate, components.decode = distributed_output, None
                    return components
                return build, identity
            session.load_callable = dtensor_factory
            try:
                with pytest.raises(ValueError, match="nonfinite"):
                    evaluate(config, trained["checkpoint"], Path(root) / "nonfinite-dtensor",
                             parameters={"nonfinite_rank": 1})
                assert not (Path(root) / "nonfinite-dtensor").exists()
                dtensor = evaluate(config, trained["checkpoint"], Path(root) / "valid-dtensor")
                values = torch.load(dtensor["artifact"], weights_only=True)
                assert type(values["generated"]) is torch.Tensor
                assert values["generated"].shape[0] == world
                assert dtensor["video"]["status"] == "NOT_AVAILABLE"
            finally:
                session.load_callable = original_load
        evaluated = evaluate(config, trained["checkpoint"], Path(root) / "evaluated")
        assert evaluated["rank"] == rank
        assert evaluated["sample_id"] == f"sample-{rank:06d}"
        assert evaluated["generation_parameters"] == {}
        assert evaluated["inference_seed"] == config.trainer["seed"]
        assert f"rank-{rank:05d}" in evaluated["artifact"]
        assert Path(evaluated["artifact"]).is_file()
        original_save = torch.save
        if rank == 1:
            def fail_write(*args, **kwargs):
                raise OSError("injected evaluation disk failure")
            torch.save = fail_write
        try:
            with pytest.raises(ValueError, match="evaluation disk failure"):
                evaluate(config, trained["checkpoint"], Path(root) / "failed-evaluation")
        finally:
            torch.save = original_save
        assert not (Path(root) / "failed-evaluation/_SUCCESS").exists()
        assert not (Path(root) / "failed-evaluation/generation.json").exists()
        assert dist.is_initialized(), "output lifecycle destroyed the embedding group"
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("reduction,world", [("post_backward", 2), ("ddp_overlap", 2),
                                           ("fsdp2_reshard", 2), ("hsdp2_reshard", 4)])
def test_sharded_checkpoint_exports_dense_weights_and_evaluates_all_ranks(tmp_path, reduction, world):
    if "sdp2" in reduction and tuple(map(int, torch.__version__.split(".")[:2])) < (2, 10):
        pytest.skip("CPU FSDP2 requires Torch >= 2.10")
    media = tmp_path / "media"
    build_fixture(media, samples=world * 2)
    prepare(media / "raw/manifest.json", media / "prepared")
    mp.spawn(_output_worker, args=(str(tmp_path / "rendezvous"), str(media / "train.json"),
                                  str(tmp_path), reduction, world), nprocs=world, join=True)
    manifest = json.loads((tmp_path / "weights/export.json").read_text())
    assert manifest["kind"] == "INFERENCE_WEIGHTS_NOT_RESUMABLE_CHECKPOINT"
    assert (tmp_path / "weights/_SUCCESS").is_file()
    from torch.distributed.tensor import DTensor
    weights = torch.load(tmp_path / "weights/weights.pt", weights_only=True)
    assert all(not isinstance(value, DTensor) for model in weights.values() for value in model.values())
    from distillschedule.training.session import generate_from_export
    config = load_training_config(media / "train.json")
    for rank in range(world):
        generated = generate_from_export(config, tmp_path / "weights", tmp_path / f"portable-{rank}",
                                         prepared=media / "prepared/manifest.json", sample_index=rank)
        portable = torch.load(generated["artifact"], weights_only=True)
        evaluated = torch.load(tmp_path / f"evaluated/ranks/rank-{rank:05d}/generation.pt", weights_only=True)
        assert equal(portable, evaluated)
    summary = json.loads((tmp_path / "evaluated/generation.json").read_text())
    assert [row["sample_id"] for row in summary["ranks"]] == [f"sample-{rank:06d}" for rank in range(world)]
    assert (tmp_path / "evaluated/_SUCCESS").is_file()
