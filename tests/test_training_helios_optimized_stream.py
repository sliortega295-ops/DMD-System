"""W2 SafeBundle fresh-stream source/CPU proofs, NOT native GPU validation.

Tiny original engines exercise real rollout/loss/Adam/LR/EMA/state owners.
Synthetic kernel counters never establish FA2/Triton availability or parity.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillschedule.training.config import TrainingConfig
from test_training_helios_stream import config
from test_training_helios_original import fresh_stream, original_cpu_engines
from helios_reference_fixtures import original_logical, frozen_assets, portable_asset_document

PRESET = "helios_nogan.safe_stream_ed8.v1"


def optimized_config(tmp_path, *, cycles=4):
    value = config(tmp_path, cycles=cycles, mode="manual").to_dict()
    value["execution"]["preset"] = PRESET
    return TrainingConfig.from_dict(value, source=tmp_path / "training.json")


def test_new_opt_in_is_not_the_frozen_two_cycle_or_reference_policy(tmp_path):
    from distillschedule.training.native import _options, describe_native
    from distillschedule.training.native_presets import describe_preset, preset_implementation
    from distillschedule.search.helios_safe_bundle import HeliosSafeBundle
    from distillschedule.search.helios_implementation import HeliosImplementation
    conf = optimized_config(tmp_path)
    assert _options(conf, "W2") == "manual"
    assert preset_implementation("W2", PRESET) == HeliosImplementation(
        "no-gan", "node-hsdp", 512, 5, False, HeliosSafeBundle.safe())
    result = describe_native(conf)
    assert result["native_api"].endswith("helios_worker:run_safe_stream")
    assert result["capabilities"]["modes"] == ["manual"]
    assert result["capabilities"]["cycles"]["max"] == 100
    assert result["capabilities"]["resume"]["supported"] is True
    assert result["preset"]["numerical_status"] == "REVIEW"
    assert result["gpu_execution"] == "NOT_RUN"
    old = describe_preset("W2", "helios_nogan.safe_ed8.v1")
    assert old["cycles"] == 2 and old["continuous_training"] is False
    reference = describe_native(config(tmp_path))
    assert reference["capabilities"]["modes"] == ["reference"]
    assert reference["native_api"].endswith(":run_reference")


def test_source_bound_flags_do_not_load_historical_model_or_trace_environment(monkeypatch):
    from distillschedule.training import helios_safe_stream as safe
    from distillschedule.search.helios_implementation import HeliosImplementation
    monkeypatch.setattr(HeliosImplementation, "environment", lambda *a, **kw:
        pytest.fail("fresh stream inherited frozen benchmark assets"))
    flags = safe.effective_flags()
    assert flags["all_role_lora_scale1_add"] is True
    assert flags["optimizer_dispatch_warmup"] is True
    assert flags["generator_nograd_residual_fusion"] is True
    assert flags["exact_grad_residual_fusion"] is True
    assert flags["lora_checkpoint_replay_scope"] is True
    assert flags["lora_b_gemm_epilogue"] is True
    assert flags["generator_nograd_cross_kv_cache"] is False
    assert flags["generator_nograd_persistent_unshard"] is False
    assert safe.implementation().request is None
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    fields = StaticFSDP2P0Policy.__dataclass_fields__
    policy = StaticFSDP2P0Policy(mesh_policy="node-hsdp", **{k:v for k,v in flags.items() if k in fields})
    args = SimpleNamespace(variant="no-gan", timing_mode=True, profile_mode=False, profile_diagnostics=False,
        attention_backend="local-fa2", ema_device="cuda", selective_flash_checkpoint_stride=0,
        generator_residual_storage="cuda_out_of_place", critic_residual_storage="cuda_out_of_place",
        mesh_policy="node-hsdp", modulation_chunk_tokens=512, rollout_batch_size=1,
        rollout_backend="native-fsdp2-batch", critic_only_step=None, operation_source=False,
        generator_nograd_persistent_unshard_flush_allocator_cache=True,
        **flags)
    bound = safe.bind_effective(args=args, policy=policy, environment={})
    assert bound["implementation"] == safe.implementation().assignment
    assert bound["candidate_bundle"]["id"] == "safe-nogan-ed8-v1"
    for change in ({"exact_grad_residual_fusion":False}, {"selective_flash_checkpoint_stride":2},
                   {"mesh_policy":"world16-fsdp"}, {"generator_nograd_cross_kv_cache":True}):
        with pytest.raises(ValueError):
            safe.bind_effective(args=args, policy=replace(policy, **change), environment={})
    for environment in ({"DISTILLSCHEDULE_HELIOS_SELECTIVE_FLASH_CHECKPOINT_STRIDE":"2"},
                        {"DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_MERGED_EXECUTION":"on"},
                        {"DISTILLSCHEDULE_TRACE_BUNDLE_HASH":"stale"}):
        with pytest.raises(ValueError):
            safe.bind_effective(args=args, policy=policy, environment=environment)


@pytest.mark.parametrize("change", ["old_preset", "wrong_mode", "plan", "implementation", "transport",
                                   "missing_stream", "replay", "cycles1", "cycles101"])
def test_safe_stream_never_widens_other_policies_or_ignores_controls(tmp_path, change):
    from distillschedule.training.native import _options
    raw = optimized_config(tmp_path).to_dict()
    if change == "old_preset":
        raw["execution"]["preset"] = "helios_nogan.safe_ed8.v1"
    elif change == "wrong_mode":
        raw["execution"]["mode"] = "reference"
    elif change in ("plan", "implementation", "transport"):
        raw["execution"][change] = "unbound.json"
    elif change == "missing_stream":
        raw["data"] = {}
    elif change == "replay":
        raw["data"]["replay_fixed_trace"] = True
    else:
        raw["trainer"]["cycles"] = 1 if change == "cycles1" else 101
    with pytest.raises(ValueError):
        _options(TrainingConfig.from_dict(raw, source=tmp_path / "training.json"), "W2")


@pytest.mark.parametrize("fresh_stream", [4], indirect=True)
def test_actual_producer_worker_and_original_helpers_four_cycles_resume_at_two(
        fresh_stream, original_cpu_engines, original_logical, frozen_assets, tmp_path, monkeypatch):
    """Real original math/state; synthetic CUDA telemetry and kernel counters.

    The scalar CPU DTensor is NOT an admitted native 14B/node-HSDP owner. The
    real production layout refusal is checked below, not disabled globally.
    """
    from contextlib import nullcontext
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    from distillschedule.training import native, helios_worker as worker, helios_lifecycle as lifecycle
    from distillschedule.training import helios_safe_stream as safe
    from distillschedule.training import checkpoint as checkpoint_io
    from distillschedule.training.native_state import engine_state
    from distillschedule.profiling.static_profile_diagnostics import StaticProfileDiagnostics
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    from distillschedule.runtime import helios_planner_cycle as original_cycle
    from distillschedule.workloads.best_known.registry import file_sha256
    from test_training_end_to_end import equal
    from test_refactor_helios_safe_incumbent import installed_build, increment_build
    root, stream = fresh_stream
    p0, build = original_cpu_engines
    assets = portable_asset_document(frozen_assets, original_logical)
    task_path, assets_path = tmp_path / "task.json", tmp_path / "assets.json"
    task_path.write_text(json.dumps(stream["identity"]["request"]))
    assets_path.write_text(json.dumps(assets))
    raw = optimized_config(tmp_path).to_dict()
    raw["trainer"].update(stop_after_cycle=2,checkpoint_every=2)
    raw["data"]["prepared"] = str(root / "prepared/stream.json")
    raw["execution"].update(task=str(task_path), assets=str(assets_path))
    conf = TrainingConfig.from_dict(raw, source=tmp_path / "training.json")
    original_source = native._source(conf)
    source = deepcopy(original_source); source["identity"]["dirty"] = False  # Wiring fixture only.
    monkeypatch.setattr(native, "_source", lambda _: deepcopy(source))
    # Metadata does not read a local extension or claim its availability.
    report = native.validate_native(conf)
    assert report["selected"] == safe.descriptor()
    assert native.run_native(conf, dry_run=True)["status"] == "DRY_RUN_NATIVE_BINDING_NOT_EXECUTED"
    assert not Path(conf.trainer["output_dir"]).exists()
    # Explicit fake file only reaches the parser/control seam. Original P0's
    # unchanged SHA/import gate rejects this same fake below.
    kernels = tmp_path / "SYNTHETIC_FA2_NOT_A_BINARY"; kernels.mkdir()
    extension = kernels / "_flash_attn2_cuda_a511a4c.abi3.so"
    extension.write_bytes(b"SYNTHETIC_CPU_TEST_NOT_GPU_ACCEPTANCE")
    monkeypatch.setenv("LOCAL_KERNELS", "kernels-community/flash-attn2=" + str(kernels))
    # Isolate only the Python dispatch import so this source-only file/hash
    # refusal can run on Torch CPU. This is NOT native extension import proof.
    import sys
    from types import ModuleType
    kernel_import = ModuleType("helios.modules.helios_kernels")
    kernel_import.attention_dispatch = SimpleNamespace(flash_attn_func=None)
    with monkeypatch.context() as isolated:
        isolated.setitem(sys.modules, "helios.modules.helios_kernels", kernel_import)
        with pytest.raises(AssertionError, match="extension identity changed"):
            p0._attention_backend_receipt("local-fa2")  # Actual unchanged gate rejects the fake .so.
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {"WORLD_SIZE":"16", "LOCAL_WORLD_SIZE":"8", "RANK":"0", "LOCAL_RANK":"0"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(worker, "reference_process_group", lambda: nullcontext({}))
    captured, metadata_groups = [], []
    parser = p0._parser()

    def entrypoint():
        args = parser.parse_args()
        binding = lifecycle.current_training()
        fields = StaticFSDP2P0Policy.__dataclass_fields__
        policy = StaticFSDP2P0Policy(**{key:getattr(args, key) for key in fields})
        assert binding["config"].trainer.get("stop_after_cycle", 2) == 2
        assert args.mesh_policy == "node-hsdp" and args.attention_backend == "local-fa2"
        assert binding["implementation"] == safe.implementation()
        assert binding["identity"]["execution_policy"] == safe.descriptor()
        assert binding["identity"]["bootstrap_trace_hash"] == stream["cohorts"][0]["trace_hash"]
        assert "reference_policy" not in binding["identity"]
        # Same actual preconstruction hook, original asset and fresh payload
        # validation; only world16 launcher/CUDA allocation is a CPU facade.
        metadata_dist = SimpleNamespace(get_world_size=lambda:16, new_group=dist.new_group)
        try:
            lifecycle.admit_original_rank(torch=torch, dist=metadata_dist, rank=0,
                                          args=args, policy=policy, seed_runtime=False)
        finally:
            if binding.get("metadata_group") is not None:
                metadata_groups.append(binding["metadata_group"])
        assert binding["effective_configuration"] == safe.bind_effective(args=args, policy=policy)
        captured.append((dict(binding), args, policy))
        return 23

    monkeypatch.setattr(worker, "_backend", lambda: SimpleNamespace(main=entrypoint))
    from distillschedule.cli import main
    conf.source.write_text(json.dumps(conf.to_dict()))
    assert main(["train",str(conf.source)]) == 23
    admitted, args, policy = captured[0]
    assert lifecycle.current_training() is None
    # Generic checkpoint topology uses the actual group, never this facade.
    monkeypatch.setenv("WORLD_SIZE", "1")
    calls, counters = [], []
    real_prepare = original_cycle._prepare_cycle

    def prepare(**kwargs):
        assert kwargs["observer"].impl == safe.implementation()
        prepared, graph, order = real_prepare(**kwargs)
        assert order.index("r4") < order.index("c0"), "window5 was silently replaced by reference window1"
        calls.append(tuple(kwargs["steps"]))
        real_run = prepared.run
        def run():
            value = real_run()
            increment_build(counters[-1])  # SYNTHETIC: no Triton/FA2 is executed on the scalar CPU owner.
            return value
        prepared.run = run
        return prepared, graph, order

    monkeypatch.setattr(original_cycle, "_prepare_cycle", prepare)
    def run(engines, output, *, every=0, resume=None, progress=None, effective=True, stop=None):
        value = conf.to_dict(); value["trainer"].update(output_dir=str(output), checkpoint_every=every)
        if resume:
            value["trainer"]["resume_from"] = str(resume)
        value["trainer"].pop("stop_after_cycle", None)
        if stop is not None:
            value["trainer"]["stop_after_cycle"] = stop
        selected = TrainingConfig.from_dict(value, source=conf.source)
        binding = dict(admitted, config=selected)
        if not effective:
            binding.pop("effective_configuration")
        counters.append(installed_build())
        return lifecycle.execute_cycles(p0=p0, args=args, policy=policy,
            generator=engines[0], critic=engines[1], ema=engines[2],
            diagnostics=StaticProfileDiagnostics(enabled=False, rank=0),
            negative=torch.zeros(1,2,4096,dtype=torch.bfloat16), rank=0, device=torch.device("cpu"),
            binding=binding, result={"build":counters[-1]}, progress=progress)

    def seed():
        torch.manual_seed(73); random.seed(73); np.random.seed(73)
    def state(engines):
        return checkpoint_io.pack_state(engine_state({"generator":engines[0], "critic":engines[1]}, ema=engines[2]))
    seed(); control = build()
    ids = [(id(e), id(e.module.weight), id(e.optimizer), id(e.lr_scheduler)) for e in control[:2]]
    result = run(control, tmp_path / "control")
    assert ids == [(id(e), id(e.module.weight), id(e.optimizer), id(e.lr_scheduler)) for e in control[:2]]
    assert result["final_versions"] == lifecycle.expected_versions(4) == {"generator":6, "critic":30, "ema":30}
    assert calls == [tuple(range(s,s+5)) for s in (10,15,20,25)]
    assert [row["trace_hash"] for row in result["cycles"]] == [row["trace_hash"] for row in stream["cohorts"]]
    assert all(row["peak_live_rollouts"] == 5 and row["retained_rollouts"] == 0
               and all(item["released_after_action"] is not None for item in row["rollout_lifetimes"])
               for row in result["cycles"])
    expected_state, expected_rng = state(control), checkpoint_io.rng_state()
    seed(); interrupted = build(); calls.clear()
    stopped = run(interrupted, tmp_path / "first", every=2, stop=2)
    assert stopped["status"] == "STOPPED_ORIGINAL_HELIOS_AT_CHECKPOINT"
    assert stopped["training_segment"] == dict(horizon_cycles=4,start_cycle=0,
        requested_stop_cycle=2,completed_cycle=2,executed_cycles=2,complete_horizon=False)
    saved = tmp_path / "first/checkpoints/cycle-000002"
    assert (saved / "_SUCCESS").is_file()
    assert sorted(p.name for p in saved.parent.iterdir()) == ["cycle-000002"]
    from test_training_helios_stream import fresh_process_segment
    child=fresh_process_segment(tmp_path,conf=conf,binding=admitted,args=args,policy=policy,
        logical=original_logical,frozen=frozen_assets,checkpoint=saved,optimized=True)
    assert equal(expected_state,child["state"])
    assert expected_rng.keys() == child["rng"].keys()
    assert {key: equal(value, child["rng"][key]) for key, value in expected_rng.items()} == dict.fromkeys(expected_rng, True)
    assert child["calls"] == [list(range(20,25)),list(range(25,30))]
    assert child["receipt"]["training_segment"] == dict(horizon_cycles=4,start_cycle=2,
        requested_stop_cycle=4,completed_cycle=4,executed_cycles=2,complete_horizon=True)
    for actual, uninterrupted in zip(child["receipt"]["cycles"],result["cycles"][2:],strict=True):
        for key in ("source_steps","trace_hash","selected_order","actions","final_versions"):
            assert actual[key] == uninterrupted[key], key
    for stale in (1,2):
        untouched=build(); before_state,before_rng=state(untouched),checkpoint_io.rng_state()
        with pytest.raises(ValueError,match="stop_after_cycle.*resume"):
            run(untouched,tmp_path/f"stale-{stale}",every=1,resume=saved,stop=stale)
        assert equal(before_state,state(untouched)) and equal(before_rng,checkpoint_io.rng_state())
    assert calls == [tuple(range(s,s+5)) for s in (10,15)]
    torch.rand(37); random.random(); np.random.random()
    restored = build(); parameter_ids = [id(e.module.weight) for e in restored[:2]]
    calls.clear()
    final = run(restored, tmp_path / "restored", resume=saved)
    assert calls == [tuple(range(s,s+5)) for s in (20,25)]
    assert final["resume_cycle"] == 2 and final["initializations"] == 1
    assert [id(e.module.weight) for e in restored[:2]] == parameter_ids
    assert equal(expected_state, state(restored)) and equal(expected_rng, checkpoint_io.rng_state())
    with pytest.raises(ValueError, match="topology/layout"):
        safe.owner_layout(restored[0], restored[1])  # A CPU tiny owner is NOT node-HSDP GPU admission.
    # Missing physical admission must refuse before loading saved tensor/RNG
    # state; direct lifecycle use cannot mutate an owner and then reject.
    fresh = build(); before = state(fresh)
    with pytest.raises(ValueError, match="preconstruction physical binding"):
        run(fresh, tmp_path / "no-effective", resume=saved, effective=False)
    assert equal(before, state(fresh))
    # Actual all-state reader rejects corrupt/missing shard and incomplete
    # boundary before any named trainable/Adam/LR/EMA mutation.
    shard = saved / "state.pt"; original_bytes = shard.read_bytes()
    manifest_path = saved / "manifest.json"; original_header = json.loads(manifest_path.read_text())
    for case in ("missing_shard", "corrupt_shard", "boundary", "source", "topology"):
        changed = deepcopy(original_header)
        if case == "missing_shard": shard.rename(saved / "held.pt")
        elif case == "corrupt_shard": shard.write_bytes(original_bytes[:-1] + bytes([original_bytes[-1] ^ 1]))
        elif case == "boundary": changed["versions"]["ema"] -= 1
        elif case == "source": changed["identity"]["source_head"] = "foreign"
        else: changed["topology"] = {"world_size":16}
        manifest_path.write_text(json.dumps(changed)); (saved / "_SUCCESS").write_text(file_sha256(manifest_path)+"\n")
        fresh = build(); before, before_rng = state(fresh), checkpoint_io.rng_state()
        with pytest.raises(ValueError):
            run(fresh, tmp_path / ("bad-" + case), resume=saved)
        assert equal(before, state(fresh)) and equal(before_rng, checkpoint_io.rng_state())
        if case == "missing_shard": (saved / "held.pt").rename(shard)
        shard.write_bytes(original_bytes)
    manifest_path.write_text(json.dumps(original_header)); (saved / "_SUCCESS").write_text(file_sha256(manifest_path)+"\n")
    # Public restart metadata binds the saved native layout digest. This
    # explicitly synthetic metadata checkpoint has no model state; actual
    # construction/restore must still reject the CPU owner at the native hook.
    native_layout = {"topology":deepcopy(safe.TOPOLOGY), "trainables_sha256":"a"*64}
    native_identity = dict(admitted["identity"], native_owner_layout=native_layout)
    metadata_saved = checkpoint_io.save_checkpoint(tmp_path / "SYNTHETIC-layout-metadata", {},
        identity=native_identity, cycle=2, cursor=2, versions=lifecycle.expected_versions(2),
        policy=lifecycle._checkpoint_policy(native_identity, optimized=True))
    resumed_raw = conf.to_dict(); resumed_raw["trainer"].pop("stop_after_cycle")
    resumed_raw["trainer"].update(
        resume_from=str(metadata_saved), output_dir=str(tmp_path / "metadata-resume"))
    resumed_conf = TrainingConfig.from_dict(resumed_raw, source=conf.source)
    monkeypatch.setenv("WORLD_SIZE", "16")
    assert native.run_native(resumed_conf) == 23  # Synthetic model boundary only, not a GPU/restore success.
    rebound = captured[-1][0]
    assert rebound["identity"] == native_identity
    assert rebound["checkpoint_policy"] == lifecycle._checkpoint_policy(native_identity, optimized=True)
    logical, _ = worker.load_reference_bootstrap(assets)
    with lifecycle.original_training_scope(rebound):
        with pytest.raises(ValueError, match="topology/layout"):
            lifecycle.run_training_cycles(p0=p0, args=args, policy=policy, result={},
                output=tmp_path / "NOT_WRITTEN.json", rank=0, device=torch.device("cpu"), logical=logical,
                generator_engine=restored[0], critic_engine=restored[1], ema=restored[2], diagnostics=None, negative=None)
    metadata_manifest = metadata_saved / "manifest.json"
    malformed = json.loads(metadata_manifest.read_text())
    malformed["identity"]["native_owner_layout"]["topology"]["mesh"] = [[0,1]]
    metadata_manifest.write_text(json.dumps(malformed))
    (metadata_saved / "_SUCCESS").write_text(file_sha256(metadata_manifest)+"\n")
    with pytest.raises(ValueError, match="checkpoint native node-HSDP layout"):
        native.run_native(resumed_conf)
    # Both successful and refused admissions own their fresh CPU metadata
    # groups; the embedding group's lifecycle remains with the fixture.
    for group in metadata_groups:
        dist.destroy_process_group(group)
    assert not torch.cuda.is_initialized()


def test_missing_effective_admission_refuses_before_reading_or_restoring_state(tmp_path, monkeypatch):
    """Fast ordering proof: invalid binding cannot reach checkpoint IO."""
    import torch
    from test_training_native_state import _engine
    from distillschedule.training import helios_lifecycle as lifecycle, helios_safe_stream as safe
    from distillschedule.training import checkpoint as checkpoint_io
    generator, critic = _engine(), _engine()
    generator.global_steps, critic.global_steps = 2, 10
    selected = {name:value for name,value in generator.module.named_parameters() if value.requires_grad}
    ema = SimpleNamespace(optimization_step=10, _parameters=list(selected.values()), _names=list(selected))
    conf = optimized_config(tmp_path)
    raw = conf.to_dict(); raw["trainer"]["resume_from"] = str(tmp_path / "checkpoint")
    conf = TrainingConfig.from_dict(raw, source=conf.source)
    touched = []
    def read(*a, **kw):
        touched.append("checkpoint_read")
        raise AssertionError("checkpoint IO preceded physical admission")
    monkeypatch.setattr(checkpoint_io, "read_manifest", read)
    with pytest.raises(ValueError, match="preconstruction physical binding"):
        lifecycle.execute_cycles(p0=SimpleNamespace(torch=torch), args=None, policy=None,
            generator=generator, critic=critic, ema=ema, diagnostics=None, negative=None, rank=0, device="cpu",
            binding={"config":conf, "stream":{"cohorts":[None]*4}, "implementation":safe.implementation(),
                     "identity":{}, "checkpoint_policy":{}}, result={})
    assert touched == []


def test_explicit_local_fa2_dependency_is_not_installed_downloaded_or_faked(tmp_path):
    from distillschedule.training import helios_safe_stream as safe
    for environment in ({}, {"LOCAL_KERNELS":"auto"},
                        {"LOCAL_KERNELS":"kernels-community/flash-attn2=" + str(tmp_path)}):
        with pytest.raises(ValueError, match="LOCAL_KERNELS|extension is missing"):
            safe.validate_environment(environment, require_kernel=True)
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        with pytest.raises(ValueError, match="allocator"):
            safe.validate_environment({key:"expandable_segments:True"})
    with pytest.raises(ValueError, match="unbound"):
        safe.validate_environment({"DISTILLSCHEDULE_UNREGISTERED_OPTIMIZATION":"on"})
    assert not (tmp_path / "_flash_attn2_cuda_a511a4c.abi3.so").exists()


def _gloo_state_worker(rank, rendezvous, output):
    """Real 2-rank local shards/Adam/LR/RNG; synthetic scalar update, NOT W2 math."""
    from datetime import timedelta
    import os
    import random
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Replicate, Shard
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    from distillschedule.training.native_state import engine_state
    from distillschedule.training import checkpoint as checkpoint_io, distributed
    from distillschedule.training import helios_safe_stream as safe, helios_lifecycle as lifecycle
    from test_training_end_to_end import equal
    from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2", GROUP_RANK="0")
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    root = Path(output)
    try:
        mesh = init_device_mesh("cpu", (1,2), mesh_dim_names=("replicate", "shard"))
        def build():
            engines = []
            for step in (2,10):
                model = torch.nn.Module()
                model.weight = torch.nn.Parameter(distribute_tensor(torch.full((2,2), .1), mesh, [Replicate(), Shard(0)]))
                optimizer = torch.optim.AdamW([model.weight], lr=.002, betas=(0.,.999))
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=.9)
                engines.append(FSDP2EngineAdapter(model,optimizer,scheduler,physical_step=step,global_batch=16,dp_world_size=2))
            return engines
        def update(engines):
            for engine, count in zip(engines, (1,5), strict=True):
                for _ in range(count):
                    # Each rank's actual RNG drives the local DTensor gradient.
                    # No claim this independent scalar loss is Helios/DMD.
                    noise = torch.rand(()) + random.random()
                    engine.backward((engine.module.weight * noise).sum())
                    engine.step()
        def state(engines):
            return engine_state({"generator":engines[0], "critic":engines[1]})
        def seed():
            torch.manual_seed(701+rank); random.seed(701+rank)
        identity = {"scope":"CPU_GLOO_LOCAL_SHARD_TEST_NOT_NATIVE_W2_GPU",
            "strategy":safe.descriptor(), "source":"SYNTHETIC_SAME_TEST_SOURCE",
            "stream":"SYNTHETIC_FOUR_DISTINCT_COHORT_IDS", "native_layout":"CPU_1x2_NOT_GPU_2x8"}
        seed(); control=build()
        for _ in range(4): update(control)
        expected, expected_rng = checkpoint_io.pack_state(state(control)), checkpoint_io.rng_state()
        seed(); first=build()
        for _ in range(2): update(first)
        saved = checkpoint_io.save_checkpoint(root / "checkpoint", state(first), identity=identity,
            cycle=2, cursor=2, versions=lifecycle.expected_versions(2),
            policy=lifecycle._checkpoint_policy(identity, optimized=True))
        header = checkpoint_io.read_manifest(saved, identity=identity)
        assert set(header["files"]) == {"rank-00000.pt", "rank-00001.pt"}
        torch.rand(7); random.random()
        restored=build(); ids=[id(e.module.weight) for e in restored]
        checkpoint_io.restore_checkpoint(saved, state(restored), identity=identity)
        for _ in range(2): update(restored)
        assert ids == [id(e.module.weight) for e in restored]
        assert equal(expected, checkpoint_io.pack_state(state(restored)))
        assert equal(expected_rng, checkpoint_io.rng_state())
        with pytest.raises(ValueError, match="topology/layout"):
            safe.owner_layout(*restored)  # Never admits this CPU topology as the native node-HSDP one.
        shard = saved / "rank-00001.pt"
        original_bytes = shard.read_bytes()
        for case in ("missing_rank", "corrupt_rank", "pending_gradient", "wrong_topology", "wrong_source"):
            fresh = build(); before = checkpoint_io.pack_state(state(fresh)); before_rng=checkpoint_io.rng_state()
            if rank == 0:
                if case == "missing_rank": shard.rename(saved / "held.pt")
                elif case == "corrupt_rank": shard.write_bytes(original_bytes[:-1] + bytes([original_bytes[-1]^1]))
            dist.barrier()
            if case in ("missing_rank", "corrupt_rank"):
                with pytest.raises(ValueError, match="shard missing or hash changed"):
                    checkpoint_io.restore_checkpoint(saved, state(fresh), identity=identity)
            elif case == "pending_gradient":
                if rank == 1:
                    fresh[0].module.weight.grad = torch.ones_like(fresh[0].module.weight)
                with pytest.raises(ValueError, match="unconsumed gradients"):
                    checkpoint_io.save_checkpoint(root / "invalid-pending", state(fresh), identity=identity,
                        cycle=2,cursor=2,versions=lifecycle.expected_versions(2))
            else:
                other=dict(identity, **({"source":"foreign"} if case == "wrong_source" else {"native_layout":"foreign"}))
                with pytest.raises(ValueError, match="identity changed"):
                    checkpoint_io.restore_checkpoint(saved, state(fresh), identity=other)
            assert equal(before, checkpoint_io.pack_state(state(fresh))) and equal(before_rng, checkpoint_io.rng_state())
            dist.barrier()
            if rank == 0 and case in ("missing_rank", "corrupt_rank"):
                if case == "missing_rank": (saved / "held.pt").rename(shard)
                shard.write_bytes(original_bytes)
            dist.barrier()
        assert not (root / "invalid-pending/_SUCCESS").exists()
        (root / f"gloo-rank-{rank}.json").write_text(json.dumps({"rank":rank,"status":"PASS",
            "scope":identity["scope"], "versions":lifecycle.expected_versions(4),
            "state_and_rank_rng_equal":True, "missing_corrupt_rank_and_pending_boundary_rejected":True}))
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_local_native_state_resume_and_failure_boundaries(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_gloo_state_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    assert all(json.loads((tmp_path / f"gloo-rank-{r}.json").read_text())["state_and_rank_rng_equal"] for r in range(2))
