"""W2 native continuation proof; never pretrained-model or GPU evidence."""
import json
from pathlib import Path

import pytest

from distillschedule.training.config import TrainingConfig

ROOT = Path(__file__).resolve().parents[1]
KIND = "original_helios_cohort_stream.v1"


def config(tmp_path, *, cycles=3, mode="reference"):
    return TrainingConfig.from_dict({
        "schema_version": "distillschedule.training.v1", "model": {"workload": "W2"},
        "algorithm": {"name": "original_helios_nogan"},
        "data": {"input_kind": KIND, "prepared": "stream.json"},
        "execution": {"adapter": "native", "mode": mode, "task": "task.json", "assets": "assets.json"},
        "trainer": {"cycles": cycles, "checkpoint_every": 1, "output_dir": "run"},
    }, source=tmp_path / "training.json")


def test_versioned_w2_reference_options_do_not_widen_fixed_measurements(tmp_path):
    from distillschedule.training.native import _options, describe_native
    conf = config(tmp_path)
    assert _options(conf, "W2") == "reference"
    described = describe_native(conf)
    assert described["capabilities"]["modes"] == ["reference"]
    assert described["source"]["selected_plan_is_authoritative"] is False
    assert "no node reservation/launch" in described["launch"]
    assert described["capabilities"]["bootstrap"] == {
        "global_step":10, "versions":{"generator":2,"critic":10,"ema":10},
        "cold_start_supported":False, "asset_contract":"distillschedule.helios_reference_assets.v1 or unchanged legacy native assets"}
    assert described["capabilities"]["resume"]["supported"] is True
    assert described["capabilities"]["generate"]["supported"] is False
    assert described["capabilities"]["export"]["supported"] is True
    assert "CURRENT" in described["capabilities"]["export"]["scope"]
    raw = conf.to_dict()
    raw["data"] = {}
    raw["execution"]["mode"] = "plan"
    raw["execution"]["plan"] = "bounded-plan.json"
    raw["trainer"]["checkpoint_every"] = 0
    with pytest.raises(ValueError, match="cycles"):
        _options(TrainingConfig.from_dict(raw, source=conf.source), "W2")
    for section, key, value in (("execution", "mode", "manual"),
                                ("data", "replay_fixed_trace", True),
                                ("trainer", "seed", 5)):
        changed = conf.to_dict(); changed[section][key] = value
        with pytest.raises(ValueError):
            _options(TrainingConfig.from_dict(changed, source=conf.source), "W2")
    changed = conf.to_dict(); changed["execution"].update(mode="plan", plan="selected.json")
    with pytest.raises(ValueError, match="reference only"):
        _options(TrainingConfig.from_dict(changed, source=conf.source), "W2")


def test_original_helios_training_task_never_relabels_canonical_request():
    from distillschedule.search.helios_request import OriginalHeliosRequest
    from distillschedule.training.helios_inputs import HeliosCohortTask
    request = OriginalHeliosRequest()
    assert request.source_steps == (10, 11, 12, 13, 14)
    assert HeliosCohortTask(2).source_steps == (20, 21, 22, 23, 24)
    assert request.source_steps == (10, 11, 12, 13, 14)
    for value in (-1, True, 100):
        with pytest.raises(ValueError):
            HeliosCohortTask(value)
    with pytest.raises(ValueError, match="reference"):
        HeliosCohortTask(0, request=OriginalHeliosRequest(video_frames=17))


def test_explicit_aligned_request_survives_public_cohort_roundtrip():
    from distillschedule.search.helios_request import OriginalHeliosRequest
    from distillschedule.training.helios_inputs import HeliosCohortTask
    request = OriginalHeliosRequest(stage_schedule_policy="benchmark_v4_strict_global")
    task = HeliosCohortTask(1, request)
    assert HeliosCohortTask.from_dict(task.to_dict()) == task
    assert task.source_steps == (15,16,17,18,19)
    assert HeliosCohortTask(1).stage_schedule_policy == "source_normal_only"
    from distillschedule.strategies.wan_task import NATIVE_HISTORY_POLICY
    with pytest.raises(ValueError, match="cannot combine"):
        HeliosCohortTask(1, request, NATIVE_HISTORY_POLICY)


@pytest.mark.parametrize("variant", ["no-gan", "gan"])
def test_data_cli_passes_explicit_aligned_scope_to_real_stream_producer(tmp_path, monkeypatch, capsys, variant):
    from distillschedule.data.cli import main
    from distillschedule.training import helios_inputs, helios_gan_inputs
    sources = tmp_path / "sources.json"
    sources.write_text("[]")
    seen = []
    def capture(*args, **kwargs):
        seen.append((args, kwargs))
        return {"status":"SYNTHETIC_TEST_ONLY", "gpu_execution":"NOT_RUN"}
    module = helios_inputs if variant == "no-gan" else helios_gan_inputs
    monkeypatch.setattr(module, "prepare_stream", capture)
    command = "prepare-helios-stream" if variant == "no-gan" else "prepare-helios-gan-stream"
    assert main([command,"--sources",str(sources),"--training-config",str(tmp_path/"train.yaml"),
        "--output",str(tmp_path/"prepared"),"--stage-schedule-policy","benchmark_v4_strict_global"]) == 0
    request = seen[0][0][0] if variant == "no-gan" else seen[0][1]["request"]
    assert request.variant == variant and request.stage_schedule_policy == "benchmark_v4_strict_global"
    assert not (tmp_path/"prepared").exists()


def test_rollout_math_reads_explicit_helios_geometry_not_wan_default():
    from types import SimpleNamespace
    from distillschedule.search.helios_request import OriginalHeliosRequest
    from distillschedule.runtime.rollout_geometry import rollout_geometry
    from distillschedule.training.helios_inputs import HeliosCohortTask, TRACE_KEY
    trace = SimpleNamespace(manifest={"contract": {TRACE_KEY: HeliosCohortTask(1).to_dict()}})
    assert type(rollout_geometry(trace)) is OriginalHeliosRequest
    trace.manifest["contract"]["distillschedule_original_wan_task"] = {}
    with pytest.raises(ValueError, match="ambiguous"):
        rollout_geometry(trace)


def test_original_p0_opt_in_owns_setup_before_lifecycle_without_editing_old_two_cycles():
    from distillschedule.training.helios_worker import reference_argv
    source = (ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py").read_text()
    assert source.index("run_training_cycles(", source.index("def main")) > source.index("ema = FSDP2ShardEMA")
    assert source.index("run_training_cycles(") < source.index("run_two_cycles(")
    args = reference_argv({"assets": {name: {"path": "/assets/" + name} for name in (
        "generator_model", "generator_precast", "score_model", "score_precast", "logical_checkpoint",
        "trace", "negative_prompt")}}, Path("/run/cpu-source-test"))
    assert "--mesh-policy" in args and args[args.index("--mesh-policy") + 1] == "world16-fsdp"
    assert "--no-pack-real-score-pair" in args
    assert "--timing-mode" in args
    assert "--no-exact-modulated-norm-epilogue" in args


def test_five_actual_source_steps_not_one_repeated_feature_set():
    from distillschedule.training.helios_inputs import HeliosCohortSources
    rows = [[f"/features/step{s}/rank{rank}.pt" for rank in range(16)] for s in range(5)]
    source = HeliosCohortSources.from_dict({"seed": 42, "samples": rows})
    assert len(source.samples) == 5 and all(len(row) == 16 for row in source.samples)
    for bad in (rows[:1], [rows[0]] * 5, [row[:15] for row in rows]):
        with pytest.raises(ValueError):
            HeliosCohortSources.from_dict({"seed": 42, "samples": bad})
    with pytest.raises(ValueError):
        HeliosCohortSources.from_dict({"seed": True, "samples": rows})


def test_public_prepare_errors_are_reported_without_traceback(tmp_path, capsys):
    from distillschedule.cli import main
    result = main(["data", "prepare-helios-stream", "--sources", str(tmp_path / "missing.json"),
                   "--training-config", str(tmp_path / "training.yaml"), "--output", str(tmp_path / "prepared")])
    assert result == 2
    assert "ERROR:" in capsys.readouterr().err


def test_new_restart_identity_checks_actual_precast_shards_not_just_sizes(tmp_path):
    """Tiny files test asset-integrity admission, NOT native-model numerics."""
    import torch
    from safetensors.torch import save_file
    from runtime.precast_checkpoint import convert_pretrained_safetensors
    from runtime.gst_pool_contract import write_negative_prompt_asset
    import distillschedule.training.helios_worker as worker
    assets = {"assets": {}}
    for role, directory in (("generator", "transformer_ode"), ("score", "transformer")):
        model = tmp_path / role
        source = model / directory; source.mkdir(parents=True)
        (source / "config.json").write_text('{"test":"asset_identity_only_not_a_model"}')
        save_file({"weight": torch.ones(2,2)}, str(source / "model.safetensors"))
        (source / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({
            "metadata":{"total_size":16}, "weight_map":{"weight":"model.safetensors"}}))
        precast = tmp_path / (role + "-precast")
        convert_pretrained_safetensors(source, precast, target_dtype=torch.bfloat16)
        assets["assets"][role + "_model"] = {"path":str(model)}
        assets["assets"][role + "_precast"] = {"path":str(precast)}
    negative = tmp_path / "negative"
    write_negative_prompt_asset(negative, embedding=torch.zeros(1,512,4096,dtype=torch.bfloat16), model_root=tmp_path)
    assets["assets"]["negative_prompt"] = {"path":str(negative)}
    identity = worker.immutable_assets(assets)
    assert set(identity) == {"generator_precast", "score_precast", "negative_prompt"}
    assert identity["generator_precast"]["payload_hashes_verified"] is True
    shard = next(Path(assets["assets"]["generator_precast"]["path"]).glob("*.safetensors"))
    original = shard.read_bytes()
    shard.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    assert shard.stat().st_size == len(original)
    with pytest.raises(RuntimeError, match="shard hash mismatch"):
        worker.immutable_assets(assets)


def _segment_config(tmp_path, variant, *, stop=50, every=50, cycles=100):
    raw = config(tmp_path, cycles=cycles).to_dict()
    if variant == "safe":
        raw["execution"].update(mode="manual", preset="helios_nogan.safe_stream_ed8.v1")
    elif variant == "gan":
        raw["model"]["workload"] = "W3"
        raw["algorithm"]["name"] = "original_helios_gan"
        raw["data"]["input_kind"] = "original_helios_gan_cohort_stream.v1"
        raw["execution"].update(mode="manual", preset="helios_gan.fa3_down40_nomerge_stream.v1")
    raw["trainer"].update(checkpoint_every=every, stop_after_cycle=stop)
    return TrainingConfig.from_dict(raw, source=tmp_path / "training.json")


@pytest.mark.parametrize("variant", ["reference", "safe", "gan"])
def test_stop_keeps_full_horizon_and_exposes_checkpoint_only_capability(tmp_path, variant):
    from distillschedule.training import native
    conf = _segment_config(tmp_path, variant)
    assert native._options(conf, conf.model["workload"]) == ("reference" if variant == "reference" else "manual")
    assert conf.trainer["cycles"] == 100 and conf.trainer["stop_after_cycle"] == 50
    assert native.describe_native(conf)["capabilities"]["stop"]["checkpoint_required"] is True


@pytest.mark.parametrize("variant", ["reference", "safe", "gan"])
@pytest.mark.parametrize("stop,every", [(0,50),(101,50),(True,50),(50.0,50),(50,0),(49,50),(100,0),(None,50)])
def test_invalid_stop_fails_before_worker_or_model(tmp_path, monkeypatch, variant, stop, every):
    from distillschedule.training import native
    conf = _segment_config(tmp_path, variant, stop=stop, every=every)
    monkeypatch.setattr(native, "_source", lambda _: pytest.fail("invalid stop reached source/model admission"))
    with pytest.raises(ValueError, match="stop_after_cycle.*checkpoint boundary"):
        native.run_native(conf, dry_run=True)


@pytest.mark.parametrize("variant", ["reference", "safe", "gan"])
def test_stop_does_not_widen_fixed_trace_contracts(tmp_path, variant):
    from distillschedule.training import native
    raw = _segment_config(tmp_path, variant, stop=1, every=1, cycles=2).to_dict()
    raw["data"] = {}
    if variant != "reference":
        raw["execution"]["preset"] = ("helios_nogan.safe_ed8.v1" if variant == "safe"
            else "helios_gan.fa3_down40_nomerge.v1")
    with pytest.raises(ValueError):
        native._options(TrainingConfig.from_dict(raw, source=tmp_path / "training.json"), raw["model"]["workload"])


def _resume_cpu_segment_child(spec_path, output):
    """Fresh interpreter reuses real original scalar owner and native state codec.

    CPU/Gloo plus synthetic memory/kernel counters only, never native GPU proof.
    """
    from types import SimpleNamespace
    import torch
    from distillschedule.training import helios_lifecycle as lifecycle, helios_safe_stream as safe
    from distillschedule.training.checkpoint import pack_state, rng_state
    from distillschedule.training.native_state import engine_state
    from distillschedule.profiling.static_profile_diagnostics import StaticProfileDiagnostics
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    from distillschedule.runtime import helios_planner_cycle as cycle
    from test_training_helios_original import original_cpu_engines
    from test_refactor_helios_safe_incumbent import installed_build, increment_build
    spec = json.loads(Path(spec_path).read_text())
    output = Path(output); output.mkdir()
    patch = pytest.MonkeyPatch()
    owners = original_cpu_engines.__wrapped__(output, patch, Path(spec["logical"]), spec["frozen"])
    try:
        p0, build = next(owners)
        engines = build()
        config = TrainingConfig.from_dict(spec["config"], source=spec["config_source"])
        binding = dict(spec["binding"], config=config)
        optimized = spec["optimized"]
        if optimized:
            binding["implementation"] = safe.implementation()
        build_record = installed_build() if optimized else {}
        calls = []
        prepare = cycle._prepare_cycle
        def observed_prepare(**kwargs):
            calls.append(list(kwargs["steps"]))
            prepared, graph, order = prepare(**kwargs)
            if optimized:
                original_run = prepared.run
                def run():
                    value = original_run()
                    increment_build(build_record)  # SYNTHETIC, not FA2/Triton execution.
                    return value
                prepared.run = run
            return prepared, graph, order
        patch.setattr(cycle, "_prepare_cycle", observed_prepare)
        result = lifecycle.execute_cycles(p0=p0, args=SimpleNamespace(**spec["args"]),
            policy=StaticFSDP2P0Policy(**spec["policy"]), generator=engines[0],critic=engines[1],ema=engines[2],
            diagnostics=StaticProfileDiagnostics(enabled=False,rank=0),
            negative=torch.zeros(1,2,4096,dtype=torch.bfloat16),rank=0,device=torch.device("cpu"),
            binding=binding,result={"build":build_record})
        torch.save(dict(state=pack_state(engine_state({"generator":engines[0],"critic":engines[1]},ema=engines[2])),
                        rng=rng_state(),calls=calls,receipt=result),output/"child.pt")
        assert not torch.cuda.is_initialized()
    finally:
        owners.close(); patch.undo()


def fresh_process_segment(tmp_path, *, conf, binding, args, policy, logical, frozen, checkpoint, optimized):
    """Test-only JSON packet: no opaque owner state is sent to the child."""
    from dataclasses import asdict
    import subprocess
    import sys
    import torch
    from test_training_helios_gan_stream import _cpu_lifecycle_child_env
    raw = conf.to_dict()
    raw["trainer"].pop("stop_after_cycle", None)
    raw["trainer"].update(resume_from=str(checkpoint), output_dir=str(tmp_path/"child-run"),checkpoint_every=0)
    fields = ("stream", "identity", "checkpoint_policy", "effective_configuration")
    spec = dict(config=raw,config_source=str(conf.source), binding={k:binding[k] for k in fields if k in binding},
                args=vars(args),policy=asdict(policy),logical=str(logical),frozen=frozen,optimized=optimized)
    path=tmp_path/"child-spec.json"; path.write_text(json.dumps(spec,default=str))
    code="from test_training_helios_stream import _resume_cpu_segment_child; import sys; _resume_cpu_segment_child(*sys.argv[1:])"
    child=subprocess.run([sys.executable,"-c",code,str(path),str(tmp_path/"child-run")],
        env=_cpu_lifecycle_child_env(),capture_output=True,text=True,timeout=120)
    assert child.returncode == 0, child.stdout+child.stderr
    return torch.load(tmp_path/"child-run/child.pt",weights_only=False)


def _stop_gloo_worker(rank, rendezvous, output, case):
    """Real two-rank controls; physical world16 and model are deliberately absent."""
    from datetime import timedelta
    from types import SimpleNamespace
    import torch
    import torch.distributed as dist
    from distillschedule.training.helios_lifecycle import admit_stop_control
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        trainer=dict(cycles=100,checkpoint_every=50,stop_after_cycle=50)
        if rank == 1:
            if case == "different": trainer["stop_after_cycle"]=100
            else: trainer["checkpoint_every"]=0
        with pytest.raises(ValueError,match="stop_after_cycle") as caught:
            admit_stop_control(SimpleNamespace(trainer=trainer))
        errors=[None,None]; dist.all_gather_object(errors,str(caught.value))
        assert errors[0] == errors[1]
        assert ("ranks disagree" if case == "different" else "checkpoint boundary") in errors[0]
        # No model boundary was entered; the healthy rank also exits the gate.
        assert not torch.cuda.is_initialized()
        (Path(output)/f"rank-{rank}.json").write_text(json.dumps(dict(scope="CPU_GLOO_CONTROLS_ONLY",error=errors[0])))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["different","invalid_cadence"])
def test_stop_collective_rejects_rank_divergence_before_owner(tmp_path,case):
    import torch.multiprocessing as mp
    mp.spawn(_stop_gloo_worker,args=(str(tmp_path/"rendezvous"),str(tmp_path),case),nprocs=2,join=True)
    assert all((tmp_path/f"rank-{rank}.json").is_file() for rank in range(2))
