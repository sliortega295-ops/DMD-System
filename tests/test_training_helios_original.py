"""Original Helios producer/helpers + real CPU gradients/Adam/DTensor/EMA.

The root model is a scalar CPU test fixture, not Helios14B quality evidence.
Only GPU telemetry is disabled. Dataset/history/noise/scheduler, rollout, DMD,
denoising loss, dispatch, optimizer, EMA recurrence and checkpoint code are real.
"""
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from distillschedule.training.helios_inputs import (
    HeliosCohortTask, HeliosCohortSources, prepare_stream, load_stream, load_cohort,
)
from distillschedule.workloads.best_known.registry import repository_root, fingerprint


@pytest.fixture(scope="module")
def fresh_stream(tmp_path_factory):
    import yaml
    root = tmp_path_factory.mktemp("original-helios-stream")
    training = yaml.safe_load((repository_root() / "configs/distillschedule/templates/helios_stage3_384x640.yaml").read_text())
    # Explicit supported input settings. Never resample runtime failures to
    # hide the unsupported all-zero/rank-local amplified branch.
    training["training_config"]["is_random_drop"] = False
    training["training_config"]["noise_corrupt_clean_prob_history"] = 1.
    config = root / "training.yaml"; config.write_text(yaml.safe_dump(training))
    sources = []
    for index in range(3):
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
    document = prepare_stream(HeliosCohortTask(0).request, sources, config, root / "prepared")
    return root, document


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
    raw["trainer"].update(cycles=3, checkpoint_every=1, output_dir=str(tmp_path / "training-output"))
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
        assert len(binding["stream"]["cohorts"]) == 3
        assert "verified_immutable_assets" not in binding["identity"]
        assert "checkpoint_policy" not in binding
        seen.append(binding["identity"])
        return 23
    monkeypatch.setattr(worker, "_backend", lambda: SimpleNamespace(main=entrypoint))
    assert native.run_native(conf) == 23
    assert len(seen) == 1 and current_training() is None
    assert torch.cuda.is_initialized() is False


@pytest.fixture
def original_cpu_engines(tmp_path, monkeypatch):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Replicate
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter, FSDP2ShardEMA
    from distillschedule.training.helios_worker import _backend
    require_clean = not dist.is_initialized()
    assert require_clean
    dist.init_process_group("gloo", init_method="file://" + str(tmp_path / "rendezvous"), rank=0, world_size=1)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("cpu-proof",))
    # P0 helpers contain GPU telemetry even for real CPU tensors. These are
    # deliberately NOT evidence of real allocator/fences/capacity/admission.
    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: None)
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **kw: (2**40,2**40))

    class ScalarRoot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(distribute_tensor(torch.tensor([.2]), mesh, [Replicate()]))
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
        def engine(step):
            model = ScalarRoot()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(0., .999))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.)
            scheduler.last_epoch = step
            return FSDP2EngineAdapter(model, optimizer, scheduler, physical_step=step)
        g, c = engine(2), engine(10)
        # CPU construction harness only. The actual FSDP2ShardEMA.step/get_decay
        # methods own every recurrence, over an actual CPU DTensor local shard.
        ema = FSDP2ShardEMA.__new__(FSDP2ShardEMA)
        ema._names, ema._parameters = ["weight"], [g.module.weight]
        ema._shadows = [g.module.weight.to_local().detach().clone()]
        ema.decay, ema.min_decay, ema.update_after_step = .99, 0., 0
        ema.optimization_step, ema.cur_decay_value = 10, None
        ema.device, ema.nvtx_enabled, ema.last_transfer_bytes = "cpu", False, 0
        return g, c, ema
    try:
        yield _backend().impl, build
    finally:
        dist.destroy_process_group()


def test_actual_original_helpers_three_cycles_checkpoint_resume_equal(fresh_stream, original_cpu_engines, tmp_path):
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
    identity = {"proof": "CPU_ORIGINAL_HELPERS_NOT_PRETRAINED_OR_GPU", "stream": stream["stream_sha256"]}
    def run(engines, destination, *, every=0, resume=None, progress=None):
        conf = config(destination)
        raw = conf.to_dict(); raw["trainer"].update(output_dir=str(destination), checkpoint_every=every)
        if resume:
            raw["trainer"]["resume_from"] = str(resume)
        conf = type(conf).from_dict(raw, source=conf.source)
        return execute_cycles(p0=p0, args=args, policy=StaticFSDP2P0Policy(rollout_batch_size=1,
            pack_real_score_pair=False), generator=engines[0], critic=engines[1], ema=engines[2],
            diagnostics=StaticProfileDiagnostics(enabled=False, rank=0), negative=torch.zeros(1,2,4096,dtype=torch.bfloat16),
            rank=0, device=torch.device("cpu"), binding={"config":conf,"stream":stream,"identity":identity,
                "checkpoint_policy":{"cpu_original_proof": True}}, result={}, progress=progress)

    torch.manual_seed(73); random.seed(73)
    control = build()
    before = control[0].module.weight.to_local().detach().clone()
    result = run(control, tmp_path / "control")
    expected_rng = rng_state()
    expected_state = pack_state(engine_state({"generator":control[0], "critic":control[1]}, ema=control[2]))
    assert result["final_versions"] == expected_versions(3)
    assert not torch.equal(before, control[0].module.weight.to_local())
    assert [r["source_steps"] for r in result["cycles"]] == [list(range(s,s+5)) for s in (10,15,20)]

    class StopAfterCheckpoint(RuntimeError):
        pass
    def stop(result):
        assert result["completed_cycles"] == 1 and result["latest_checkpoint"]
        raise StopAfterCheckpoint()
    torch.manual_seed(73); random.seed(73)
    interrupted = build()
    with pytest.raises(StopAfterCheckpoint):
        run(interrupted, tmp_path / "first", every=1, progress=stop)
    checkpoint = tmp_path / "first/checkpoints/cycle-000001"
    assert (checkpoint / "_SUCCESS").is_file()
    torch.randn(37); random.random()
    restored = build(); parameter_ids = [id(engine.module.weight) for engine in restored[:2]]
    final = run(restored, tmp_path / "restored", resume=checkpoint)
    assert [id(engine.module.weight) for engine in restored[:2]] == parameter_ids
    assert final["resume_cycle"] == 1 and [r["cohort_index"] for r in final["cycles"]] == [1,2]
    assert equal(expected_state, pack_state(engine_state({"generator":restored[0], "critic":restored[1]}, ema=restored[2])))
    assert equal(expected_rng, rng_state())
    assert torch.cuda.is_initialized() is False
    # A rehashed manifest must not turn a partial/wrong update boundary into a
    # restart. Validate before any actual owned tensor or clock is changed.
    from distillschedule.workloads.best_known.registry import file_sha256
    manifest = checkpoint / "manifest.json"
    changed = json.loads(manifest.read_text()); changed["versions"]["ema"] -= 1
    manifest.write_text(json.dumps(changed))
    (checkpoint / "_SUCCESS").write_text(file_sha256(manifest) + "\n")
    fresh = build()
    before_state = pack_state(engine_state({"generator":fresh[0], "critic":fresh[1]}, ema=fresh[2]))
    with pytest.raises(ValueError, match="complete G1/C5/EMA5 boundary"):
        run(fresh, tmp_path / "invalid-resume", resume=checkpoint)
    assert equal(before_state, pack_state(engine_state({"generator":fresh[0], "critic":fresh[1]}, ema=fresh[2])))
