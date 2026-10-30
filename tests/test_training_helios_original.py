"""Original Helios producer/helpers + real CPU gradients/Adam/DTensor/EMA.

The root model is a scalar CPU test fixture, not Helios14B quality evidence.
Only GPU telemetry is disabled. Dataset/history/noise/scheduler, rollout, DMD,
denoising loss, dispatch, optimizer, EMA recurrence and checkpoint code are real.
"""
from copy import deepcopy
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import pickle
import random
import shutil
import weakref
from types import SimpleNamespace

import pytest
import torch

from distillschedule.training.helios_inputs import (
    HeliosCohortTask, HeliosCohortSources, prepare_stream, load_stream, load_cohort,
)
from distillschedule.workloads.best_known.registry import repository_root, fingerprint
from helios_reference_fixtures import (original_logical as original_logical,
    frozen_assets as frozen_assets, portable_asset_document)


@pytest.fixture(scope="module")
def fresh_stream(tmp_path_factory, request):
    import yaml
    root = tmp_path_factory.mktemp("original-helios-stream")
    training = yaml.safe_load((repository_root() / "configs/distillschedule/templates/helios_stage3_384x640.yaml").read_text())
    # Explicit supported input settings. Never resample runtime failures to
    # hide the unsupported all-zero/rank-local amplified branch.
    training["training_config"]["is_random_drop"] = False
    training["training_config"]["noise_corrupt_clean_prob_history"] = 1.
    config = root / "training.yaml"; config.write_text(yaml.safe_dump(training))
    sources = []
    for index in range(getattr(request, "param", 3)):
        samples = []
        seed = 41 + index
        for offset in range(5):
            names = []
            for rank in range(16):
                # Dataset section selection remains native. This deterministic
                # fixture supplies enough encoded sections for nonzero history.
                local_seed = seed + index * 1000000 + rank * 5 + offset
                sections = next(n for n in range(2, 32) if all(torch.randint(0, n, (1,),
                    generator=torch.Generator().manual_seed(s)).item() > 0
                    for s in (local_seed, local_seed + 999999)))
                path = root / f"feature-{index}-{offset}-{rank}.pt"
                value = .1 * (index + 1) + .001 * (offset + rank)
                # Expanded views keep source-fixture storage small; the real
                # native producer materializes and hashes every sampled tensor.
                torch.save({"vae_latent": torch.tensor(value, dtype=torch.bfloat16).expand(sections,16,9,48,80),
                    "prompt_embed": torch.full((2,4096), value, dtype=torch.bfloat16),
                    "attention_mask": torch.ones(2, dtype=torch.bool)}, path)
                names.append(str(path))
            samples.append(names)
        sources.append(HeliosCohortSources.from_dict({"seed": seed, "samples": samples}))
    # This fixture also runs before the source-qualified GAN/helper tests.
    # Admit the same pinned closure before legacy capture reuses its imports.
    from distillschedule.strategies.wan_capture import native_primitives
    native_primitives(source_only=True)
    document = prepare_stream(HeliosCohortTask(0).request, sources, config, root / "prepared")
    return root, document


def test_original_stream_fixture_keeps_native_helpers_source_admitted(fresh_stream):
    from distillschedule.adapters.helios import reference_helpers
    from distillschedule.strategies import wan_capture
    # The same fixture runs first in the collective-admission module. Its
    # real native capture must leave a source-admitted closure for later CPU
    # proofs, without deleting or replacing any already loaded helper.
    before = reference_helpers._cached_modules()
    identity = reference_helpers.reference_helper_identity(backend="torch-sdpa")
    original = wan_capture.native_primitives()
    admitted = wan_capture.native_primitives(source_only=True)
    assert len(before) == len(identity["modules"]) == 11
    assert reference_helpers._cached_modules() == before
    assert original.post is admitted.post is before["helios.utils.utils_helios_post"]
    assert original.history is admitted.history is before["helios.utils.utils_helios_base"]
    assert original.scheduler is admitted.scheduler
    assert original.dataset is admitted.dataset
    assert not torch.cuda.is_initialized()


def test_original_producer_consumer_and_full_rank_step_rng_admission(fresh_stream):
    root, document = fresh_stream
    assert load_stream(root / "prepared/stream.json", cycles=3, verify_rank=0) == document
    traces = [load_cohort(HeliosCohortTask(i), entry["trace_bundle"], expected_hash=entry["trace_hash"])
              for i, entry in enumerate(document["cohorts"])]
    assert len({entry["trace_hash"] for entry in document["cohorts"]}) == 3
    seen = []
    for trace in traces:
        g = trace.load_generator_entry(rank=0, device="cpu")
        for step in trace.logical_steps:
            c = trace.load_critic_entry(rank=0, step=step, device="cpu")
            assert c.pool_entry.logical_step == step and c.pool_entry.source_trace_rank == 0
            assert c.tensors["entry_noise"].shape == (1,16,9,48,80)
            assert c.tensors["text_embedding"].shape == (1,2,4096)
            seen.append(c.pool_entry.identity_sha256)
        assert g.logical_step == trace.logical_steps[0]
    assert len(set(seen)) == 15
    from distillschedule.training.helios_inputs import validate_rank_payload
    traces[0]._loaded[0].tensors["step/10/generator/entry_noise"].flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_rank_payload(traces[0], 0, HeliosCohortSources.from_dict(document["identity"]["sources"][0]))
    assert torch.cuda.is_initialized() is False


def test_stream_metadata_tampering_and_wrong_legacy_bindings_reject(fresh_stream):
    from distillschedule.training.helios_inputs import validate_controls, TRACE_KEY
    from runtime.operation_trace import OperationTrace
    root, document = fresh_stream
    original = root / "prepared/stream.json"
    for change in (lambda d: d["cohorts"].reverse(),
                   lambda d: d["identity"]["sources"][0].update(seed=1001),
                   lambda d: d["identity"]["inventory"][0].pop()):
        bad = deepcopy(document); change(bad); bad.pop("stream_sha256")
        bad["stream_sha256"] = fingerprint(bad)
        path = original.with_name("changed.json"); path.write_text(json.dumps(bad))
        with pytest.raises((ValueError, KeyError)):
            load_stream(path, cycles=3)
    entry = document["cohorts"][0]
    trace = load_cohort(HeliosCohortTask(0), entry["trace_bundle"], expected_hash=entry["trace_hash"])
    controls = {rank: OperationTrace(root=value.root, manifest=deepcopy(value.manifest), tensors={})
                for rank, value in trace._control.items()}
    controls[0].manifest["tensors"]["step/10/sample/text_embedding"]["shape"] = [2,2048]
    with pytest.raises(ValueError, match="conditioning"):
        validate_controls(HeliosCohortTask(0), controls)
    controls[0].manifest["tensors"]["step/10/sample/text_embedding"]["shape"] = [2,4096]
    controls[0].manifest["contract"].pop(TRACE_KEY)
    with pytest.raises(ValueError, match="new Helios cohort"):
        validate_controls(HeliosCohortTask(0), controls)


def test_fresh_stream_rejects_unsafe_pickle_without_execution(fresh_stream, tmp_path):
    root, document = fresh_stream
    marker = tmp_path / "pickle-executed"

    class UnsupportedPayload:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",)

    payload = Path(document["cohorts"][0]["trace_bundle"]) / "rank_00/tensors.pt"
    original = payload.read_bytes()
    tensors = torch.load(payload, map_location="cpu", weights_only=True)
    tensors["step/10/generator/entry_noise"] = UnsupportedPayload()
    try:
        torch.save(tensors, payload)
        # Real prepared stream -> HeliosTrainingTrace -> StaticWorld16Trace ->
        # OperationTrace.load. No substitute loader or allowlisted globals.
        with pytest.raises(pickle.UnpicklingError, match="Unsupported global"):
            load_stream(root / "prepared/stream.json", cycles=3, verify_rank=0)
    finally:
        payload.write_bytes(original)
        assert not marker.exists(), "unsupported pickle global executed during stream admission"


def replayed_sources(document, tmp_path, replay):
    values = deepcopy(document["identity"]["sources"])
    first = values[0]["samples"][0][0]
    if replay == "seed_only":
        values[1]["samples"] = deepcopy(values[0]["samples"])
    else:
        replacement = Path(first)
        if replay in ("symlink_overlap", "copied_payload"):
            replacement = tmp_path / "repeated-feature.pt"
            if replay == "symlink_overlap":
                replacement.symlink_to(first)
            else:
                shutil.copyfile(first, replacement)
        values[1]["samples"][0][0] = str(replacement)
    assert values[1]["seed"] != values[0]["seed"]
    return [HeliosCohortSources.from_dict(value) for value in values]


@pytest.mark.parametrize("replay", ["seed_only", "overlap", "symlink_overlap", "copied_payload"])
def test_prepare_stream_rejects_cross_cohort_replay(fresh_stream, tmp_path, monkeypatch, replay):
    import distillschedule.training.helios_inputs as inputs
    root, document = fresh_stream
    sources = replayed_sources(document, tmp_path, replay)

    def unexpected_capture(*args, **kwargs):
        raise AssertionError("repeated sources reached native capture")

    monkeypatch.setattr(inputs, "capture_rank", unexpected_capture)
    output = tmp_path / "rejected-stream"
    with pytest.raises(ValueError, match="globally distinct feature"):
        prepare_stream(HeliosCohortTask(0).request, sources, root / "training.yaml", output)
    assert not output.exists()


@pytest.mark.parametrize("replay", ["seed_only", "overlap", "symlink_overlap", "copied_payload"])
def test_load_stream_rejects_cross_cohort_replay(fresh_stream, tmp_path, replay):
    from distillschedule.workloads.best_known.registry import file_sha256
    from runtime.operation_trace import _manifest_hash, write_operation_trace_bundle_manifest
    _, document = fresh_stream
    sources = replayed_sources(document, tmp_path, replay)
    changed = deepcopy(document)
    changed["identity"]["sources"] = [source.to_dict() for source in sources]
    output = tmp_path / "replayed-stream"
    for index, entry in enumerate(changed["cohorts"]):
        rows = changed["identity"]["inventory"][index]
        for row, name in zip(rows, (p for names in sources[index].samples for p in names), strict=True):
            row.update(path=str(Path(name).resolve()), sha256=file_sha256(Path(name)))
        original = Path(entry["trace_bundle"])
        directory = output / f"cohort-{index:04d}"
        ranks = []
        for rank in range(16):
            name = f"rank_{rank:02d}"
            destination = directory / name; destination.mkdir(parents=True)
            manifest = json.loads((original / name / "trace.json").read_text())
            manifest["source"]["samples"] = [row for row in rows if row["rank"] == rank]
            manifest["trace_hash"] = _manifest_hash(manifest)
            (destination / "trace.json").write_text(json.dumps(manifest))
            (destination / "tensors.pt").symlink_to(original / name / "tensors.pt")
            ranks.append({"rank": rank, "path": name, "trace_hash": manifest["trace_hash"]})
        bundle = write_operation_trace_bundle_manifest(directory, ranks)
        entry.update(trace_bundle=str(directory), trace_hash=bundle["bundle_hash"])
    changed.pop("stream_sha256")
    changed["stream_sha256"] = fingerprint(changed)
    path = output / "stream.json"; path.write_text(json.dumps(changed))
    # Re-sign all metadata so rejection must be freshness admission, not an
    # unrelated stale hash/provenance failure. Tensor descriptors stay pinned.
    with pytest.raises(ValueError, match="globally distinct feature"):
        load_stream(path, cycles=3, verify_rank=0)


@pytest.mark.parametrize("deferred_gate", ["payload", "immutable"])
def test_public_config_source_admission_to_original_entrypoint(fresh_stream, tmp_path, monkeypatch, deferred_gate):
    from distillschedule.training.config import TrainingConfig
    import distillschedule.training.native as native
    import distillschedule.training.helios_worker as worker
    from distillschedule.training.helios_lifecycle import current_training
    from test_training_native import search_fixture
    root, _ = fresh_stream
    source_dir = tmp_path / "asset-fixture"; source_dir.mkdir()
    bounded, _, _ = search_fixture(source_dir, "W2")
    raw = bounded.to_dict(); raw["execution"].pop("plan"); raw["execution"]["mode"] = "reference"
    raw["data"] = {"input_kind": "original_helios_cohort_stream.v1", "prepared": str(root / "prepared/stream.json")}
    raw["trainer"].update(cycles=3, checkpoint_every=1, stop_after_cycle=1, output_dir=str(tmp_path / "training-output"))
    conf = TrainingConfig.from_dict(raw, source=bounded.source)
    metadata_load = worker.load_stream
    if deferred_gate == "payload":
        def metadata_only(*args, **kwargs):
            assert kwargs.get("verify_rank") is None, "rank payload read before original entry"
            return metadata_load(*args, **kwargs)
        monkeypatch.setattr(worker, "load_stream", metadata_only)
    def deferred_immutable(_):
        raise AssertionError("immutable payload read before original entry")
    monkeypatch.setattr(worker, "immutable_assets", deferred_immutable)
    described = native.validate_native(conf)
    assert described["selected"]["mesh"] == "world16-fsdp"
    assert described["native_api"].endswith(":run_reference")
    assert native.run_native(conf, dry_run=True)["status"] == "DRY_RUN_NATIVE_BINDING_NOT_EXECUTED"
    assert not Path(conf.trainer["output_dir"]).exists()
    observed_source = native._source(conf)
    # This test crosses the real admission/entrypoint boundary, not physical
    # execution: explicitly isolate the dirty-check claim and pretrained gate.
    test_source = deepcopy(observed_source); test_source["identity"]["dirty"] = False
    monkeypatch.setattr(native, "_source", lambda _: deepcopy(test_source))
    for name, value in {"WORLD_SIZE":"16", "LOCAL_WORLD_SIZE":"8", "RANK":"0", "LOCAL_RANK":"0"}.items():
        monkeypatch.setenv(name, value)
    for name in ("LOCAL_KERNELS", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        monkeypatch.delenv(name, raising=False)
    parser = worker._backend().impl._parser()
    seen = []
    def entrypoint():
        args = parser.parse_args()
        binding = current_training()
        assert args.mesh_policy == "world16-fsdp" and args.rollout_batch_size == 1
        assert args.timing_mode and args.ema_device == "cuda" and not args.profile_diagnostics
        assert args.selective_flash_checkpoint_stride == 0 and args.attention_backend == "sdpa"
        assert all(getattr(args, name.replace("-", "_")) is False for name in worker._OFF)
        assert binding["identity"]["source_head"] == observed_source["identity"]["head"]
        assert binding["config"].trainer["stop_after_cycle"] == 1
        assert len(binding["stream"]["cohorts"]) == 3
        assert "verified_immutable_assets" not in binding["identity"]
        assert "checkpoint_policy" not in binding
        seen.append(binding["identity"])
        return 23
    monkeypatch.setattr(worker, "_backend", lambda: SimpleNamespace(main=entrypoint))
    # This source-to-entrypoint test does not execute the original worker or
    # resource launcher. Real two-rank public/P0 admission is covered separately.
    monkeypatch.setattr(worker, "reference_process_group", lambda: nullcontext({}))
    from distillschedule.cli import main
    conf.source.write_text(json.dumps(conf.to_dict()))
    assert main(["train",str(conf.source)]) == 23
    assert len(seen) == 1 and current_training() is None
    assert torch.cuda.is_initialized() is False


@pytest.fixture
def original_cpu_engines(tmp_path, monkeypatch, original_logical, frozen_assets):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Replicate
    from distillschedule.runtime.fsdp2_static import build_fsdp2_engine, FSDP2ShardEMA
    from distillschedule.training.helios_worker import _backend, load_reference_bootstrap
    from runtime.critic_pool_checkpoint import load_logical_master_into_trainables
    require_clean = not dist.is_initialized()
    assert require_clean
    dist.init_process_group("gloo", init_method="file://" + str(tmp_path / "rendezvous"), rank=0, world_size=1)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("cpu-proof",))
    # P0 helpers contain GPU telemetry even for real CPU tensors. These are
    # deliberately NOT evidence of real allocator/fences/capacity/admission.
    from distillschedule.data import wan_reference_inputs
    # This legacy ScalarRoot fixture keeps its existing mocked CUDA-telemetry
    # route. It is NOT the admitted full-model W1 CPU reference and must not
    # widen that production trace gate merely to run scalar state tests.
    monkeypatch.setattr(wan_reference_inputs, "cpu_telemetry", lambda *a, **kw: False)
    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: None)
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **kw: (2**40,2**40))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    package, _ = load_reference_bootstrap(portable_asset_document(frozen_assets, original_logical))

    class ScalarRoot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.register_buffer("base", torch.tensor(.15))
            self.adapters_enabled = True

        def enable_adapters(self):
            self.adapters_enabled = True

        def disable_adapters(self):
            self.adapters_enabled = False

        def forward(self, hidden_states, encoder_hidden_states, **kwargs):
            weight = self.weight.to_local().reshape(()) if self.adapters_enabled else self.base
            return ((hidden_states.float() * .01 + weight + encoder_hidden_states.float().mean() * .01
                     ).to(hidden_states.dtype),)

    def build():
        def engine(role):
            model = ScalarRoot()
            with torch.no_grad():
                load_logical_master_into_trainables(model, package, component=role, verify_payload_sha256=True)
            model.weight = torch.nn.Parameter(distribute_tensor(model.weight.detach(), mesh, [Replicate()]))
            return build_fsdp2_engine(module=model, package=package, component=role,
                lr=2e-6 if role == "generator" else 4e-7, betas=(0., .999), eps=1e-8, weight_decay=1e-3)[0]
        g, c = engine("generator"), engine("critic")
        for owned, step in ((g,2), (c,10)):
            assert owned.global_steps == owned.lr_scheduler.last_epoch == step
            adam = owned.optimizer.state[owned.module.weight]
            assert adam["step"].item() == step
            assert torch.equal(adam["exp_avg"].to_local(), torch.tensor([.03]))
            assert torch.equal(adam["exp_avg_sq"].to_local(), torch.tensor([.004]))
        # Actual original master/Adam/scheduler/EMA restoration, on a CPU
        # DTensor tiny owner. No model-construction or CUDA/FSDP capacity claim.
        ema = FSDP2ShardEMA.load_from_logical(g.module, package.root, device="cpu")
        assert ema.optimization_step == 10 and ema.decay == .99 and ema.update_after_step == 750
        return g, c, ema
    try:
        yield _backend().impl, build
    finally:
        dist.destroy_process_group()


def test_actual_original_helpers_three_cycles_checkpoint_resume_equal(fresh_stream, original_cpu_engines,
        original_logical, frozen_assets, tmp_path, monkeypatch):
    from distillschedule.training.helios_lifecycle import execute_cycles, expected_versions
    from distillschedule.training.native_state import engine_state
    from distillschedule.training.checkpoint import pack_state, rng_state
    from distillschedule.profiling.static_profile_diagnostics import StaticProfileDiagnostics
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    from test_training_end_to_end import equal
    from test_training_helios_stream import config
    _, stream = fresh_stream
    p0, build = original_cpu_engines
    args = SimpleNamespace(timing_mode=True, profile_mode=False, profile_diagnostics=False, ema_device="cpu")
    from distillschedule.training.helios_worker import immutable_assets
    identity = {"proof": "CPU_ORIGINAL_HELPERS_NOT_PRETRAINED_OR_GPU", "stream": stream["stream_sha256"],
        "assets":portable_asset_document(frozen_assets, original_logical),
        "verified_immutable_assets":immutable_assets(portable_asset_document(frozen_assets, original_logical)),
        "bootstrap_logical_hash": json.loads((original_logical / "manifest.json").read_text())["manifest_hash"]}
    def run(engines, destination, *, every=0, resume=None, progress=None, stop=None):
        conf = config(destination)
        raw = conf.to_dict(); raw["trainer"].update(output_dir=str(destination), checkpoint_every=every)
        if resume:
            raw["trainer"]["resume_from"] = str(resume)
        if stop is not None:
            raw["trainer"]["stop_after_cycle"] = stop
        conf = type(conf).from_dict(raw, source=conf.source)
        return execute_cycles(p0=p0, args=args, policy=StaticFSDP2P0Policy(rollout_batch_size=1,
            pack_real_score_pair=False), generator=engines[0], critic=engines[1], ema=engines[2],
            diagnostics=StaticProfileDiagnostics(enabled=False, rank=0), negative=torch.zeros(1,2,4096,dtype=torch.bfloat16),
            rank=0, device=torch.device("cpu"), binding={"config":conf,"stream":stream,"identity":identity,
                "checkpoint_policy":{"cpu_original_proof": True}}, result={}, progress=progress)

    torch.manual_seed(73); random.seed(73)
    control = build()
    before = control[0].module.weight.to_local().detach().clone()
    progress_path = tmp_path / "control-progress.json"
    result = run(control, tmp_path / "control", progress=lambda value: p0.base._atomic_json(progress_path, value))
    assert json.loads(progress_path.read_text())["completed_cycles"] == 3
    expected_rng = rng_state()
    expected_state = pack_state(engine_state({"generator":control[0], "critic":control[1]}, ema=control[2]))
    assert result["final_versions"] == expected_versions(3)
    assert not torch.equal(before, control[0].module.weight.to_local())
    assert [r["source_steps"] for r in result["cycles"]] == [list(range(s,s+5)) for s in (10,15,20)]

    torch.manual_seed(73); random.seed(73)
    interrupted = build()
    stopped = run(interrupted, tmp_path / "first", every=1, stop=1)
    assert stopped["status"] == "STOPPED_ORIGINAL_HELIOS_AT_CHECKPOINT"
    assert stopped["training_segment"] == dict(horizon_cycles=3,start_cycle=0,
        requested_stop_cycle=1,completed_cycle=1,executed_cycles=1,complete_horizon=False)
    assert [row["source_steps"] for row in stopped["cycles"]] == [list(range(10,15))]
    checkpoint = tmp_path / "first/checkpoints/cycle-000001"
    assert (checkpoint / "_SUCCESS").is_file()
    assert sorted(p.name for p in checkpoint.parent.iterdir()) == ["cycle-000001"]
    from test_training_helios_stream import fresh_process_segment
    child=fresh_process_segment(tmp_path,conf=config(tmp_path),
        binding={"stream":stream,"identity":identity,"checkpoint_policy":{"cpu_original_proof":True}},
        args=args,policy=StaticFSDP2P0Policy(rollout_batch_size=1,pack_real_score_pair=False),
        logical=original_logical,frozen=frozen_assets,checkpoint=checkpoint,optimized=False)
    assert equal(expected_state,child["state"])
    assert expected_rng.keys() == child["rng"].keys()
    assert {key: equal(value, child["rng"][key]) for key, value in expected_rng.items()} == dict.fromkeys(expected_rng, True)
    assert child["calls"] == [list(range(15,20)),list(range(20,25))]
    assert child["receipt"]["training_segment"] == dict(horizon_cycles=3,start_cycle=1,
        requested_stop_cycle=3,completed_cycle=3,executed_cycles=2,complete_horizon=True)
    for actual, uninterrupted in zip(child["receipt"]["cycles"],result["cycles"][1:],strict=True):
        for key in ("source_steps","trace_hash","selected_order","actions","final_versions"):
            assert actual[key] == uninterrupted[key], key
    for stale in (1,):
        untouched=build(); before_state=pack_state(engine_state({"generator":untouched[0],"critic":untouched[1]},ema=untouched[2]))
        before_rng=rng_state()
        with pytest.raises(ValueError,match="stop_after_cycle.*resume"):
            run(untouched,tmp_path/"stale",every=1,resume=checkpoint,stop=stale)
        assert equal(before_state,pack_state(engine_state({"generator":untouched[0],"critic":untouched[1]},ema=untouched[2])))
        assert equal(before_rng,rng_state())
    torch.randn(37); random.random()
    restored = build(); parameter_ids = [id(engine.module.weight) for engine in restored[:2]]
    from distillschedule.training import checkpoint as checkpoint_io
    from distillschedule.runtime import helios_planner_cycle as original_cycle
    original_restore, original_prepare = checkpoint_io.restore_checkpoint, original_cycle._prepare_cycle
    payload_refs, copied_tensor_refs, resumed_preparations = [], [], []

    class ObservedPayload(dict):
        """Weak-referenceable view of the real, unmodified restore return."""

    def observed_restore(*args, **kwargs):
        payload = ObservedPayload(original_restore(*args, **kwargs))
        assert set(payload["state"]) == set(engine_state(
            {"generator": restored[0], "critic": restored[1]}, ema=restored[2]))
        payload_refs.append(weakref.ref(payload))
        copied = [payload["state"][role + ".weights"]["value"]["tensors"]["weight"]["local"]
                  for role in ("generator", "critic")]
        copied.extend(payload["state"]["ema"]["value"]["shadows"])
        copied_tensor_refs.extend(weakref.ref(tensor) for tensor in copied)
        return payload

    def observed_prepare(*args, **kwargs):
        gc.collect()
        assert payload_refs and all(ref() is None for ref in payload_refs), \
            "returned checkpoint CPU payload is still resident before resumed original cycle"
        assert copied_tensor_refs and all(ref() is None for ref in copied_tensor_refs), \
            "copied checkpoint weights/EMA payload is still resident before resumed original cycle"
        resumed_preparations.append(tuple(kwargs["steps"]))
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(checkpoint_io, "restore_checkpoint", observed_restore)
    monkeypatch.setattr(original_cycle, "_prepare_cycle", observed_prepare)
    resumed_progress = tmp_path / "resumed-progress.json"
    final = run(restored, tmp_path / "restored", resume=checkpoint,
                progress=lambda value: p0.base._atomic_json(resumed_progress, value))
    assert json.loads(resumed_progress.read_text())["completed_cycles"] == 3
    assert resumed_preparations == [tuple(range(15,20)), tuple(range(20,25))]
    monkeypatch.setattr(checkpoint_io, "restore_checkpoint", original_restore)
    monkeypatch.setattr(original_cycle, "_prepare_cycle", original_prepare)
    assert [id(engine.module.weight) for engine in restored[:2]] == parameter_ids
    assert final["resume_cycle"] == 1 and [r["cohort_index"] for r in final["cycles"]] == [1,2]
    assert equal(expected_state, pack_state(engine_state({"generator":restored[0], "critic":restored[1]}, ema=restored[2])))
    assert equal(expected_rng, rng_state())
    assert torch.cuda.is_initialized() is False
    # A rehashed manifest must not turn a partial/wrong update boundary into a
    # restart. Validate before any actual owned tensor or clock is changed.
    from distillschedule.workloads.best_known.registry import file_sha256
    manifest = checkpoint / "manifest.json"
    original_header = json.loads(manifest.read_text())
    changed = deepcopy(original_header); changed["identity"]["bootstrap_logical_hash"] = "foreign-bootstrap"
    manifest.write_text(json.dumps(changed)); (checkpoint / "_SUCCESS").write_text(file_sha256(manifest)+"\n")
    fresh = build()
    before_state = pack_state(engine_state({"generator":fresh[0], "critic":fresh[1]}, ema=fresh[2]))
    with pytest.raises(ValueError, match="checkpoint identity changed"):
        run(fresh, tmp_path / "foreign-resume", resume=checkpoint)
    assert equal(before_state, pack_state(engine_state({"generator":fresh[0], "critic":fresh[1]}, ema=fresh[2])))
    changed = deepcopy(original_header); changed["versions"]["ema"] -= 1
    manifest.write_text(json.dumps(changed))
    (checkpoint / "_SUCCESS").write_text(file_sha256(manifest) + "\n")
    fresh = build()
    before_state = pack_state(engine_state({"generator":fresh[0], "critic":fresh[1]}, ema=fresh[2]))
    with pytest.raises(ValueError, match="complete G1/C5/EMA5 boundary"):
        run(fresh, tmp_path / "invalid-resume", resume=checkpoint)
    assert equal(before_state, pack_state(engine_state({"generator":fresh[0], "critic":fresh[1]}, ema=fresh[2])))
    (tmp_path / "restart-lifetime-proof.json").write_text(json.dumps({
        "scope":"REAL_ORIGINAL_HELPERS_CPU_SCALAR_REPLICATED_DTENSOR_NOT_PRETRAINED_OR_GPU",
        "restore_return_container_probes":len(payload_refs), "copied_weight_ema_tensor_probes":len(copied_tensor_refs),
        "payload_collected_before_original_preparations":resumed_preparations,
        "actual_state_matches_uninterrupted":True, "rng_matches_uninterrupted":True,
        "live_parameter_identity_preserved":True, "final_versions":final["final_versions"],
        "control_and_resumed_progress_published":True, "foreign_identity_and_wrong_boundary_rejected":True,
    }, indent=2) + "\n")
