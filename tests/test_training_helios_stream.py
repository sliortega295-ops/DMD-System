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
    assert described["capabilities"]["export"]["supported"] is False
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
    for value in (-1, True, 64):
        with pytest.raises(ValueError):
            HeliosCohortTask(value)
    with pytest.raises(ValueError, match="reference"):
        HeliosCohortTask(0, request=OriginalHeliosRequest(video_frames=17))


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
