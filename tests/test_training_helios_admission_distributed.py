"""Synthetic CPU/Gloo pre-model metadata proof, NOT world16/GPU admission.

Only the original world16 topology guard and original main/model boundary are
represented by fixtures. Input capture/consumer, frozen-file verification,
checked/gather, metadata groups and checkpoint metadata are production code.
"""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from test_training_helios_original import fresh_stream as fresh_stream
from test_training_helios_stream import config


@pytest.fixture(scope="module")
def frozen_assets(tmp_path_factory):
    """Real tiny pre-cast/negative artifacts, not pretrained model evidence."""
    from safetensors.torch import save_file
    from runtime.precast_checkpoint import convert_pretrained_safetensors
    from runtime.gst_pool_contract import write_negative_prompt_asset
    root = tmp_path_factory.mktemp("helios-admission-assets")
    assets = {"assets": {}}
    for role, subfolder in (("generator", "transformer_ode"), ("score", "transformer")):
        model = root / role
        source = model / subfolder; source.mkdir(parents=True)
        (source / "config.json").write_text('{"scope":"CPU_ASSET_IDENTITY_ONLY"}')
        save_file({"weight": torch.ones(2, 2)}, str(source / "model.safetensors"))
        (source / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 16}, "weight_map": {"weight": "model.safetensors"}}))
        precast = root / (role + "-precast")
        convert_pretrained_safetensors(source, precast, target_dtype=torch.bfloat16)
        assets["assets"][role + "_model"] = {"path": str(model)}
        assets["assets"][role + "_precast"] = {"path": str(precast)}
    negative = root / "negative"
    write_negative_prompt_asset(negative, embedding=torch.zeros(1, 512, 4096, dtype=torch.bfloat16), model_root=root)
    assets["assets"]["negative_prompt"] = {"path": str(negative)}
    return assets


def _launcher_metadata(rank):
    # run_reference still enforces its unchanged production 2 x 8 topology.
    # This is only a CPU metadata fixture: the actual caller group has 2 ranks.
    os.environ.update(WORLD_SIZE="16", RANK=str(rank), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="8")
    os.environ.pop("GROUP_RANK", None)


def _metadata_worker(rank, rendezvous, stream_path, assets, case, output):
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    from distillschedule.training import helios_worker as worker
    from distillschedule.training import helios_lifecycle as lifecycle
    from distillschedule.training.checkpoint import save_checkpoint
    from distillschedule.open_workload.replicated import control_plane

    torch.set_num_threads(1)
    for name in ("LOCAL_KERNELS", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        os.environ.pop(name, None)
    _launcher_metadata(rank)
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    root = Path(output)
    raw = config(root).to_dict()
    raw["data"]["prepared"] = stream_path
    conf = type(config(root)).from_dict(raw, source=root / "training.json")
    admitted = {"assets": deepcopy(assets), "stream": worker.load_stream(stream_path, cycles=3),
                "selected": {"scope": "SYNTHETIC_CPU_REFERENCE_METADATA"}, "argv": []}
    source = {"identity": {"head": "SYNTHETIC_CPU_SOURCE_NOT_A_GPU_RUN"}}
    # Only the uniform world guard is a facade; new_group and every metadata
    # collective use actual torch.distributed CPU/Gloo with actual rank order.
    metadata_dist = SimpleNamespace(get_world_size=lambda: 16, new_group=lambda **kw:
        dist.new_group(backend=kw["backend"], timeout=timedelta(seconds=45)))
    args = SimpleNamespace(variant="no-gan", timing_mode=True, profile_mode=False, profile_diagnostics=False,
        attention_backend="sdpa", selective_flash_checkpoint_stride=0, ema_device="cuda",
        generator_residual_storage="cuda_out_of_place", critic_residual_storage="cuda_out_of_place",
        **{name.replace("-", "_"): False for name in worker._OFF})
    policy = StaticFSDP2P0Policy()
    bindings, reached = [], []
    held = None
    if case == "bad_frozen" and rank == 1:
        local = root / "rank1-precast"
        shutil.copytree(assets["assets"]["generator_precast"]["path"], local)
        admitted["assets"]["assets"]["generator_precast"]["path"] = str(local)

    def original_entry():
        nonlocal held
        binding = lifecycle.current_training()
        bindings.append(binding)
        # Original construction/launcher is not run by this CPU fixture. Use
        # real 2-rank launcher metadata for generic checkpoint topology checks.
        os.environ.update(WORLD_SIZE="2", LOCAL_WORLD_SIZE="2", GROUP_RANK="0")
        if case == "missing_payload" and rank == 1:
            path = Path(admitted["stream"]["cohorts"][0]["trace_bundle"]) / "rank_01/tensors.pt"
            held = path.with_suffix(".held")
            path.rename(held)
        if case == "bad_frozen" and rank == 1:
            shard = next(Path(admitted["assets"]["assets"]["generator_precast"]["path"]).glob("*.safetensors"))
            value = shard.read_bytes()
            shard.write_bytes(value[:-1] + bytes([value[-1] ^ 1]))
        if case == "identity" and rank == 1:
            binding["identity"]["source_head"] = "rank1-different-source"
        if case == "policy" and rank == 1:
            args.attention_backend = "foreign"
        if case == "off_flag" and rank == 1:
            args.elide_critic_scalar_logs = True
        dist.barrier()
        lifecycle.admit_original_rank(torch=torch, dist=metadata_dist, rank=rank, args=args, policy=policy)
        reached.append("SYNTHETIC_MODEL_BOUNDARY_NOT_A_MODEL")
        frozen = worker.immutable_assets(admitted["assets"])
        assert binding["identity"]["verified_immutable_assets"] == frozen
        assert binding["checkpoint_policy"]["identity_sha256"] == fingerprint(binding["identity"])
        assert binding["stream"] == admitted["stream"]
        return 23

    worker._backend = lambda: SimpleNamespace(main=original_entry)
    try:
        if case == "environment" and rank == 1:
            os.environ["LOCAL_KERNELS"] = "rank-local-unbound-override"
        if case == "old_receipt" and rank == 1:
            receipt = Path(conf.trainer["output_dir"]) / "ranks/rank_01.json"
            receipt.parent.mkdir(parents=True, exist_ok=True); receipt.write_text("preserved")
        error = None
        try:
            worker.run_reference(conf, admitted, source)
        except Exception as exc:
            error = str(exc)
        report = {"case": case, "rank": rank, "error": error, "model_boundaries": len(reached),
                  "metadata_group_created": bool(bindings and bindings[-1].get("metadata_group") is not None)}
        reports = [None, None]; dist.all_gather_object(reports, report)
        caller_value = torch.tensor(rank + 1)
        dist.all_reduce(caller_value)
        assert caller_value.item() == 3, "admission damaged the embedding caller group"
        assert lifecycle.current_training() is None and not torch.cuda.is_initialized()
        if held is not None:
            held.rename(held.with_suffix(".pt"))
            held = None
        dist.barrier()
        (root / f"{case}-rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        if case == "success":
            assert all(row["error"] is None and row["model_boundaries"] == 1 for row in reports), reports
            binding = bindings[-1]
            with control_plane(binding["metadata_group"]):
                saved = save_checkpoint(root / "checkpoint", {}, identity=binding["identity"], cycle=1, cursor=1,
                    versions=lifecycle.expected_versions(1), policy=binding["checkpoint_policy"])
            # A real published CPU metadata checkpoint is accepted at the same
            # hook. Neither call constructs models or restores tensor state.
            resumed = conf.to_dict(); resumed["trainer"]["resume_from"] = str(saved)
            resumed = type(conf).from_dict(resumed, source=conf.source)
            _launcher_metadata(rank)
            assert worker.run_reference(resumed, admitted, source) == 23
            assert bindings[-1]["identity"] == binding["identity"]
            assert bindings[-1]["checkpoint_policy"] == binding["checkpoint_policy"]
            # Rehashing a foreign immutable identity cannot bypass admission.
            if rank == 0:
                manifest = saved / "manifest.json"
                changed = json.loads(manifest.read_text())
                changed["identity"]["verified_immutable_assets"]["generator_precast"]["manifest_sha256"] = "foreign"
                manifest.write_text(json.dumps(changed))
                (saved / "_SUCCESS").write_text(file_sha256(manifest) + "\n")
            dist.barrier()
            _launcher_metadata(rank)
            with pytest.raises(ValueError, match="Helios restart metadata.*checkpoint identity changed"):
                worker.run_reference(resumed, admitted, source)
            assert len(reached) == 2, "foreign checkpoint crossed the synthetic model boundary"
        else:
            messages = {"missing_payload": "tensors.pt", "bad_frozen": "shard hash mismatch",
                        "identity": "rank source/stream/reference differs", "policy": "physical policy changed",
                        "off_flag": "unbound optimization", "environment": "attention/allocator environment differs",
                        "old_receipt": "old rank receipt is preserved"}
            assert all(row["error"] and messages[case] in row["error"] and row["model_boundaries"] == 0
                       and row["metadata_group_created"] for row in reports), reports
            assert reports[0]["error"] == reports[1]["error"], reports
            if case != "identity":
                assert "rank failures [None," in error, "healthy rank did not finish its local gate"
        caller_value = torch.tensor(rank + 1)
        dist.all_reduce(caller_value)
        assert caller_value.item() == 3 and not torch.cuda.is_initialized()
    finally:
        if held is not None:
            held.rename(held.with_suffix(".pt"))
        for binding in bindings:
            if binding.get("metadata_group") is not None:
                dist.destroy_process_group(binding["metadata_group"])
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["missing_payload", "bad_frozen", "identity", "policy", "off_flag",
                                  "environment", "old_receipt", "success"])
def test_rank_admission_is_collective_before_synthetic_model_boundary(fresh_stream, frozen_assets, tmp_path, case):
    root, _ = fresh_stream
    mp.spawn(_metadata_worker, args=(str(tmp_path / "rendezvous"), str(root / "prepared/stream.json"),
                                    frozen_assets, case, str(tmp_path)), nprocs=2, join=True)
    reports = [json.loads((tmp_path / f"{case}-rank-{rank}.json").read_text()) for rank in range(2)]
    assert {row["rank"] for row in reports} == {0, 1}
