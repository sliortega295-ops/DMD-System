"""Portable W2 admission through actual P0; CPU facades are not GPU evidence."""
from copy import deepcopy
from contextlib import contextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import shutil
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.training.config import TrainingConfig
from distillschedule.workloads.best_known.registry import fingerprint
from helios_reference_fixtures import (original_logical as original_logical,
    frozen_assets as frozen_assets, portable_asset_document)
from test_training_helios_original import fresh_stream as fresh_stream
from test_training_helios_stream import config


def test_direct_reference_templates_are_separate_explicit_inputs():
    from distillschedule.workloads.best_known.registry import repository_root
    from distillschedule.training.native import _options
    from distillschedule.training.helios_inputs import HeliosCohortSources
    from distillschedule.strategies.wan_capture import read_training_config
    root = repository_root() / "recipes/helios_nogan"
    training = TrainingConfig.from_dict(json.loads((root / "training.reference.json").read_text()),
                                        source=root / "training.reference.json")
    assert _options(training, "W2") == "reference" and "plan" not in training.execution
    assets = json.loads((root / "assets.reference.template.json").read_text())
    assert set(assets) == {"schema_version", "assets"} and "trace" not in assets["assets"]
    sources = [HeliosCohortSources.from_dict(row) for row in
               json.loads((root / "sources.reference.template.json").read_text())]
    assert len(sources) == training.trainer["cycles"] == 2
    assert len({p for source in sources for row in source.samples for p in row}) == 160
    assert read_training_config(root / "inputs.reference.yaml")["ema_decay"] == .99


def test_portable_assets_bind_new_step10_without_experiment_manifest(fresh_stream, frozen_assets, original_logical, tmp_path):
    from distillschedule.training import native
    from distillschedule.training import helios_worker as worker
    root, stream = fresh_stream
    value = portable_asset_document(frozen_assets, original_logical)
    (tmp_path / "assets.json").write_text(json.dumps(value))
    (tmp_path / "task.json").write_text(json.dumps(stream["identity"]["request"]))
    raw = config(tmp_path).to_dict(); raw["data"]["prepared"] = str(root / "prepared/stream.json")
    conf = TrainingConfig.from_dict(raw, source=tmp_path / "training.json")
    admitted = worker.admit_reference(conf, stream["identity"]["request"], value,
        {"assets": tmp_path / "assets.json", "output": tmp_path / "run"})
    assert native.validate_native(conf)["selected"] == admitted["selected"]
    args = worker._backend().impl._parser().parse_args(admitted["argv"])
    logical_hash = json.loads((original_logical / "manifest.json").read_text())["manifest_hash"]
    assert args.expected_logical_hash == logical_hash
    assert args.expected_trace_hash == stream["cohorts"][0]["trace_hash"]
    assert args.trace_bundle == Path(stream["cohorts"][0]["trace_bundle"])
    assert set(admitted["assets"]) == {"schema_version", "assets"}
    assert set(admitted["assets"]["assets"]) == {"generator_model", "score_model", "generator_precast",
        "score_precast", "logical_checkpoint", "negative_prompt"}
    from distillschedule.workloads.best_known.registry import load_profile
    old = load_profile("helios14b_nogan_best")["runtime"]["environment"]
    assert logical_hash != old["DISTILLSCHEDULE_LOGICAL_STEP10_HASH"]
    assert args.expected_trace_hash != old["DISTILLSCHEDULE_TRACE_BUNDLE_HASH"]
    assert not (tmp_path / "run").exists() and not torch.cuda.is_initialized()


@pytest.mark.parametrize("field", ["global_step", "optimizer_step", "parameter_version", "scheduler", "ema", "ema_payload_clock"])
def test_portable_bootstrap_rejects_rehashed_wrong_clocks(frozen_assets, original_logical, tmp_path, field):
    from distillschedule.training.helios_worker import load_reference_bootstrap
    local = tmp_path / "logical"; shutil.copytree(original_logical, local)
    manifest = json.loads((local / "manifest.json").read_text())
    if field == "global_step": manifest["global_step"] = 0
    elif field in ("optimizer_step", "parameter_version"): manifest["components"]["generator"][field] = 0
    elif field == "scheduler": manifest["components"]["generator"]["scheduler_state"]["last_epoch"] = 0
    elif field == "ema": manifest["ema"]["optimization_step"] = 0
    else:
        path = local / "ema/sparse_ema_manifest.json"
        metadata = json.loads(path.read_text()); metadata["optimization_step"] = 11
        path.write_text(json.dumps(metadata))
    manifest.pop("manifest_hash"); manifest["manifest_hash"] = fingerprint(manifest)
    (local / "manifest.json").write_text(json.dumps(manifest))
    value = portable_asset_document(frozen_assets, local)
    with pytest.raises(ValueError): load_reference_bootstrap(value)


def test_reference_asset_schema_does_not_accept_launcher_or_historical_trace_fields(frozen_assets, original_logical):
    from distillschedule.training.helios_reference_assets import asset_config
    value = portable_asset_document(frozen_assets, original_logical)
    for field, entry in (("nodes", []), ("master_addr", "host"), ("hf_home", "/unused")):
        changed = deepcopy(value); changed[field] = entry
        with pytest.raises(ValueError, match="direct-reference assets"): asset_config(changed)
    changed = deepcopy(value); changed["assets"]["trace"] = {"path":"/unused/old-trace"}
    with pytest.raises(ValueError, match="six native roles"): asset_config(changed)
    changed = deepcopy(value); changed["assets"]["logical_checkpoint"]["sha256"] = "unbound"
    with pytest.raises(ValueError, match="SHA256"): asset_config(changed)


def test_portable_ema_validation_releases_each_shard_before_loading_next(
        frozen_assets, original_logical, tmp_path, monkeypatch):
    from distillschedule.training.helios_reference_assets import load_bootstrap
    from distillschedule.workloads.best_known.registry import file_sha256
    local = tmp_path / "two-ema-shards"
    shutil.copytree(original_logical, local)
    manifest_path = local / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    component = manifest["components"]["generator"]
    component["logical_numel"] = 2
    component["parameter_names"].append("other_weight")
    component["records"]["other_weight"] = {"shape": [1], "numel": 1, "offset_elements": 1}
    for payload in component["payloads"].values():
        path = local / payload["path"]
        path.write_bytes(path.read_bytes() * 2)
        payload.update(bytes=path.stat().st_size, sha256=file_sha256(path))
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = fingerprint(manifest)
    manifest_path.write_text(json.dumps(manifest))
    ema_path = local / "ema/sparse_ema_manifest.json"
    ema = json.loads(ema_path.read_text())
    ema["tracked_names"].append("other_weight")
    path = ema_path.parent / "other-weight.pt"
    torch.save({"other_weight": torch.tensor([.2])}, path)
    ema["shards"].append({"file": path.name, "names": ["other_weight"], "sha256": file_sha256(path)})
    ema_path.write_text(json.dumps(ema))
    original_load = torch.load
    references = []

    def witnessed_load(path, *args, **kwargs):
        if Path(path).parent == ema_path.parent:
            assert all(ref() is None for ref in references), "previous EMA shard is still resident"
            values = original_load(path, *args, **kwargs)
            references.extend(weakref.ref(tensor) for tensor in values.values())
            return values
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", witnessed_load)
    load_bootstrap(portable_asset_document(frozen_assets, local))
    assert len(references) == 2 and all(ref() is None for ref in references)


def _portable_worker(rank, rendezvous, stream_path, frozen, logical, case, output):
    """Actual public/P0 chain; only launcher/GPU resource surfaces are facades."""
    from distillschedule.training import native, helios_worker as worker, helios_lifecycle as lifecycle
    from distillschedule.open_workload.replicated import control_plane
    torch.set_num_threads(1)
    for name in ("LOCAL_KERNELS", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        os.environ.pop(name, None)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2", GROUP_RANK="0")
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    os.environ.update(WORLD_SIZE="16", LOCAL_WORLD_SIZE="8")
    with pytest.raises(ValueError, match="do not enter on an embedding group"):
        with worker.reference_process_group():
            raise AssertionError("borrowed a caller group")
    os.environ.update(WORLD_SIZE="2", LOCAL_WORLD_SIZE="2")
    root = Path(output); held = None; reached = []; groups = []
    local = root / f"logical-{rank}"; shutil.copytree(logical, local)
    assets = portable_asset_document(frozen, local)
    # Local spelling is itself part of the W2 identity, so both ranks normally
    # use the same package. A failure case alone uses a corrupt local copy.
    if case not in {"missing_logical", "late_missing_logical", "bad_logical", "bad_ema"}:
        assets = portable_asset_document(frozen, Path(logical))
    else:
        assets = portable_asset_document(frozen, Path(logical)) if rank == 0 else assets
    asset_path = root / f"assets-{rank}.json"; asset_path.write_text(json.dumps(assets))
    task = root / f"task-{rank}.json"
    task.write_text(json.dumps(worker.load_stream(stream_path, cycles=3)["identity"]["request"]))
    raw = config(root).to_dict(); raw["data"]["prepared"] = stream_path
    raw["execution"].update(assets=str(asset_path), task=str(task))
    conf = TrainingConfig.from_dict(raw, source=root / f"training-{rank}.json")
    observed = native._source(conf)
    def cpu_source(_):
        source = deepcopy(observed); source["identity"]["dirty"] = False
        if case == "source" and rank == 1: source["identity"]["head"] = "foreign-source"
        return source
    native._source = cpu_source  # source clean is explicitly NOT proved here
    p0 = worker._backend().impl
    backend = worker._backend()
    def import_admitted_backend():
        if case == "late_missing_logical" and rank == 1:
            (local / "manifest.json").unlink()
        return backend
    worker._backend = import_admitted_backend
    # The facade replaces only production 16-rank NCCL/CUDA allocation. The
    # initialized group, every metadata operation, file read and P0.main are real.
    facade = SimpleNamespace(get_rank=dist.get_rank, get_world_size=lambda:16, get_backend=lambda:"nccl",
        is_initialized=dist.is_initialized, group=dist.group, barrier=dist.barrier,
        new_group=dist.new_group, destroy_process_group=lambda:None,
        init_process_group=lambda *a, **k: (_ for _ in ()).throw(AssertionError("P0 double initialized")))
    with lifecycle.original_training_scope({"owns_process_group":True, "process_group":object(), "metadata_group":object()}):
        with pytest.raises(ValueError, match="process-group ownership changed"):
            lifecycle.adopt_original_process_group(dist=facade)
    class CpuTorch:
        def __getattr__(self, name): return getattr(torch, name)
        def device(self, *args): return torch.device("cpu")
    p0.dist = facade; p0.torch = CpuTorch()
    p0.base._memory = lambda *a: {"scope":"SYNTHETIC_CPU_TELEMETRY_NOT_GPU_CAPACITY"}
    torch.cuda.set_device = lambda *a:None
    torch.cuda.manual_seed = lambda *a:None
    torch.cuda.get_device_capability = lambda *a:(8,0)
    class Boundary(RuntimeError): pass
    original_attention = p0._attention_backend_receipt
    def attention_receipt(expected):
        value = original_attention(expected)
        if case == "attention_rng":
            # A dependency import/SDK may consume RNG. The original runtime
            # stream must be initialized after setup, just before model build.
            random.random(); torch.rand(1)
        return value
    p0._attention_backend_receipt = attention_receipt
    def model_boundary(**kwargs):
        binding = lifecycle.current_training()
        assert binding.get("metadata_group") is not None and binding.get("startup") is not None
        assert kwargs["package"].manifest_hash == binding["identity"]["bootstrap_logical_hash"]
        if case == "attention_rng":
            seed = binding["rank_seed"]
            assert random.getstate() == random.Random(seed).getstate(), "original Python runtime seed moved before dependency setup"
            assert torch.equal(torch.random.get_rng_state(), torch.Generator().manual_seed(seed).get_state()), \
                "original Torch runtime seed moved before dependency setup"
        reached.append("SYNTHETIC_MODEL_BOUNDARY_NOT_A_MODEL")
        raise Boundary("CPU_SYNTHETIC_MODEL_BOUNDARY")
    p0._build_engines = model_boundary
    @contextmanager
    def cpu_group():
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=75)); groups.append(group)
        with control_plane(group):
            yield {"metadata_group":group, "process_group":dist.group.WORLD, "owns_process_group":True}
    worker.reference_process_group = cpu_group
    try:
        if rank == 1:
            if case == "missing_assets": asset_path.unlink()
            elif case == "bad_assets": asset_path.write_text('{"corrupt":true}')
            elif case == "missing_stream":
                raw["data"]["prepared"] = str(root / "absent-stream.json")
                conf = TrainingConfig.from_dict(raw, source=conf.source)
            elif case == "bad_stream":
                path = root / "changed-stream.json"; value = json.loads(Path(stream_path).read_text())
                value["identity"]["source_fingerprint"] = "obsolete"
                value.pop("stream_sha256"); value["stream_sha256"] = fingerprint(value)
                path.write_text(json.dumps(value)); raw["data"]["prepared"] = str(path)
                conf = TrainingConfig.from_dict(raw, source=conf.source)
            elif case == "missing_logical": (local / "manifest.json").unlink()
            elif case == "bad_logical":
                path = local / "generator-first_moment.f32"; value = path.read_bytes(); path.write_bytes(value[:-1]+b"x")
            elif case == "bad_ema":
                path = next((local / "ema").glob("*.pt")); value = path.read_bytes(); path.write_bytes(value[:-1]+b"x")
            elif case == "missing_startup":
                entry = worker.load_stream(stream_path, cycles=3)["cohorts"][0]
                path = Path(entry["trace_bundle"]) / "rank_01/tensors.pt"; held = path.with_suffix(".held")
                path.rename(held)
            elif case == "bad_frozen":
                local_precast = root / "rank1-precast"; shutil.copytree(assets["assets"]["generator_precast"]["path"], local_precast)
                assets["assets"]["generator_precast"]["path"] = str(local_precast)
                shard = next(local_precast.glob("*.safetensors")); value = shard.read_bytes()
                shard.write_bytes(value[:-1]+bytes([value[-1]^1])); asset_path.write_text(json.dumps(assets))
        dist.barrier()
        error = None
        try: native.run_native(conf)
        except Exception as exc: error = str(exc)
        report = {"rank":rank, "error":error, "model_boundaries":len(reached), "metadata_groups":len(groups)}
        reports = [None,None]; dist.all_gather_object(reports, report)
        if case in {"success", "attention_rng"}:
            assert all(r["error"] == "CPU_SYNTHETIC_MODEL_BOUNDARY" and r["model_boundaries"] == 1 for r in reports), reports
        else:
            messages = {"missing_assets":"assets-1.json", "bad_assets":"assets", "missing_stream":"absent-stream",
                "bad_stream":"stream cycles/source changed", "missing_logical":"manifest.json", "bad_logical":"logical payload",
                "late_missing_logical":"manifest.json",
                "bad_ema":"EMA shard", "missing_startup":"tensors.pt", "bad_frozen":"shard hash mismatch",
                "source":"rank source/stream/reference differs"}
            assert all(r["error"] and messages[case] in r["error"] and r["model_boundaries"] == 0
                       and r["metadata_groups"] == 1 for r in reports), reports
            assert reports[0]["error"] == reports[1]["error"], reports
            if case != "source":
                assert "rank failures [None," in error, "healthy rank did not finish its local gate"
        value = torch.tensor(rank+1); dist.all_reduce(value)
        assert value.item() == 3 and lifecycle.current_training() is None and not torch.cuda.is_initialized()
        (root / f"{case}-{rank}.json").write_text(json.dumps(report))
    finally:
        dist.barrier()
        if held is not None: held.rename(held.with_suffix(".pt"))
        for group in groups: dist.destroy_process_group(group)
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["missing_assets", "bad_assets", "missing_stream", "bad_stream", "missing_logical", "late_missing_logical",
                                  "bad_logical", "bad_ema", "missing_startup", "bad_frozen", "source", "success", "attention_rng"])
def test_public_to_original_p0_collectively_admits_before_model(fresh_stream, frozen_assets, original_logical, tmp_path, case):
    root, _ = fresh_stream
    mp.spawn(_portable_worker, args=(str(tmp_path / "rendezvous"), str(root / "prepared/stream.json"), frozen_assets,
        str(original_logical), case, str(tmp_path)), nprocs=2, join=True)
