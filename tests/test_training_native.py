"""Public native bindings: CPU admission, not six-model GPU validation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.training.config import TrainingConfig
from distillschedule.training.native import describe_native, validate_native, run_native


ROOT = Path(__file__).resolve().parents[1]


def config(tmp_path, workload="W6", **trainer):
    return TrainingConfig.from_dict({
        "schema_version": "distillschedule.training.v1", "model": {"workload": workload},
        "data": {}, "algorithm": {}, "execution": {"adapter": "native", "mode": "plan",
            "task": str(ROOT / "configs/distillschedule/original_h3_ar/w6_reference.json"),
            "plan": "selected-plan.json", "assets": "native-assets.json"},
        "trainer": {"cycles": 2, "checkpoint_every": 0, "output_dir": "run", **trainer},
    }, source=tmp_path / "training.json")


@pytest.mark.parametrize("workload", [f"W{i}" for i in range(1, 7)])
def test_describe_is_truthful_and_does_not_open_model_assets(tmp_path, workload):
    result = describe_native(config(tmp_path, workload))
    assert result["workload"] == workload
    assert result["gpu_execution"] == "NOT_RUN"
    assert result["capabilities"]["resume"]["supported"] is (workload == "W1")
    if workload == "W1":
        assert "stream" in result["capabilities"]["resume"]["scope"]
    assert result["capabilities"]["generate"]["supported"] is False


@pytest.mark.parametrize("field,value", [("cycles", 3), ("checkpoint_every", 1),
    ("resume_from", "checkpoint"), ("seed", 7), ("generate", True), ("export", "weights")])
def test_unsupported_native_lifecycle_is_rejected_not_ignored(tmp_path, field, value):
    with pytest.raises(ValueError, match="cycles|checkpoint|resume|seed|generate|export"):
        validate_native(config(tmp_path, **{field: value}))


def write(path, value):
    path.write_text(json.dumps(value, indent=2))
    return path


def replace_config(config, **sections):
    raw = config.to_dict()
    for key, value in sections.items():
        raw[key].update(value)
    return TrainingConfig.from_dict(raw, source=config.source)


def search_fixture(tmp_path, workload):
    """Real original compiler/validator with small serialized CPU assets.

    Identity files here do NOT represent pretrained checkpoints or native
    correctness. No GPU or physical asset admission is asserted by this test.
    """
    from distillschedule.search.helios_implementation import HeliosImplementation
    from distillschedule.search.helios_request import OriginalHeliosRequest
    from distillschedule.search.livewan_implementation import LiveWanImplementation
    from distillschedule.search.livewan_request import LiveWanRequest
    from distillschedule.search.h3_implementation import H3Implementation
    from distillschedule.search.h3_request import H3Request
    from distillschedule.search.execution import resolve_candidate
    from distillschedule.workloads.best_known.registry import file_sha256
    if workload in ("W2", "W3"):
        variant = "no-gan" if workload == "W2" else "gan"
        impl = HeliosImplementation(variant, "world16-fsdp", 512, 2, False)
        task = OriginalHeliosRequest(variant=variant)
        profile_id = impl.workload
    elif workload == "W4":
        impl = LiveWanImplementation("pipeline", True, True, True, bundle="c51_full_v1")
        task = LiveWanRequest()
        profile_id = "livewan_crossscale_pipeline_best"
    else:
        impl = H3Implementation(cache_policy="cpu_staged_trim")
        task = H3Request()
        profile_id = "minimax_h3_grid1000_best"
    profile = impl.lower_profile()
    assets = dict(schema_version=1, nodes=[dict(host=f"cpu-fixture-node{i}",
        repo_root="/srv/fixture/DMD-System", python_env="/srv/fixture/python") for i in (0, 1)],
        master_addr="192.0.2.1", master_port=29500, run_root="/srv/fixture/runs",
        node_local_cache_root="/tmp/fixture-cache", assets={})
    for name, spec in profile["required_assets"].items():
        path = tmp_path / name
        if name == "source_receipt":
            write(path, dict(status="PASS", source_root="/srv/fixture/wan-source"))
        elif spec["kind"] == "directory":
            path.mkdir()
            (path / "identity.json").write_text("CPU fixture, not pretrained weights")
        else:
            path.write_text("CPU fixture, not native input tensors")
        identity = path / "identity.json" if path.is_dir() else path
        assets["assets"][name] = dict(path=str(path), sha256=file_sha256(identity))
        if path.is_dir():
            assets["assets"][name]["identity_file"] = "identity.json"
    assets_path = write(tmp_path / "native-assets.json", assets)
    receipt = resolve_candidate(profile_id, impl.id, assets_path=assets_path, run_id="cpu-selected-replay")
    write(tmp_path / "selected-plan.json", receipt)
    task_path = write(tmp_path / "task.json", task.to_dict())
    conf = replace_config(config(tmp_path, workload), execution={"task": str(task_path)})
    return conf, receipt, impl


@pytest.mark.parametrize("workload", ["W2", "W3", "W4", "W5"])
def test_selected_receipt_uses_actual_native_compiler_and_validator(tmp_path, workload):
    conf, plan, impl = search_fixture(tmp_path, workload)
    result = validate_native(conf)
    assert result["selected"]["id"] == impl.id
    assert result["native_run_id"] == plan["run_id"]
    assert result["hardware"]["world_size"] == 16
    assert result["gpu_execution"] == "NOT_RUN"
    assert "PHYSICAL_GATES" in result["validation_scope"]
    assert run_native(conf, dry_run=True)["status"] == "DRY_RUN_NATIVE_BINDING_NOT_EXECUTED"
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("workload", ["W2", "W3", "W4", "W5"])
def test_dispatch_passes_the_selected_plan_without_reselecting(tmp_path, monkeypatch, workload):
    conf, plan, _ = search_fixture(tmp_path, workload)
    import distillschedule.training.native as native
    import distillschedule.search.execution as execution
    original = native._source
    def clean_source(config):
        observed = original(config)
        observed["identity"]["dirty"] = False  # unit fixture, not a live-source clean claim
        return observed
    monkeypatch.setattr(native, "_source", clean_source)
    calls = []
    def dispatch(selected, **kwargs):
        calls.append((selected, kwargs))
        return 19
    monkeypatch.setattr(execution, "run_plan", dispatch)
    assert run_native(conf) == 19
    assert calls == [(plan, dict(coordinator_output=tmp_path / "run", transport_config=None))]


@pytest.mark.parametrize("kind", ["source", "shape", "cycles", "command", "assets", "candidate", "profile"])
def test_selected_plan_rejects_changed_binding_before_dispatch(tmp_path, kind):
    conf, plan, _ = search_fixture(tmp_path, "W2")
    if kind == "source":
        plan["integration_head"] = "f" * 40
    elif kind == "shape":
        task_path = Path(conf.execution["task"])
        task = json.loads(task_path.read_text()); task["video_frames"] = 17
        write(task_path, task)
    elif kind == "cycles":
        conf = replace_config(conf, trainer={"cycles": 7})
    elif kind == "command":
        plan["nodes"][0]["command"][-1] = "undeclared-extra-argument"
    elif kind == "assets":
        value = json.loads(Path(conf.execution["assets"]).read_text())
        value["assets"]["trace"]["sha256"] = "f" * 64
        write(Path(conf.execution["assets"]), value)
    elif kind == "candidate":
        plan["search_candidate"]["id"] = "canonical-reference"
    else:
        plan["fully_resolved_profile"]["contract"]["updates"]["critic"] = 1
    write(Path(conf.execution["plan"]), plan)
    with pytest.raises(ValueError):
        run_native(conf, dry_run=True)


def test_w4_c34_is_not_accepted_as_self_forcing_c51(tmp_path):
    conf, _, _ = search_fixture(tmp_path, "W4")
    from distillschedule.search.livewan_implementation import LiveWanImplementation
    from distillschedule.search.execution import resolve_candidate
    assets_path = Path(conf.execution["assets"])
    assets = json.loads(assets_path.read_text())
    assets["assets"].pop("score_flash_attn3_kernel")
    write(assets_path, assets)
    other = resolve_candidate("livewan_crossscale_pipeline_best",
        LiveWanImplementation("pipeline", True, True, True).id,
        assets_path=assets_path, run_id="cpu-selected-replay")
    write(Path(conf.execution["plan"]), other)
    with pytest.raises(ValueError, match="C51 full bundle"):
        validate_native(conf)


def test_w5_does_not_accept_w6_continuous_task(tmp_path):
    conf, _, _ = search_fixture(tmp_path, "W5")
    conf = replace_config(conf, execution={"task": str(ROOT / "configs/distillschedule/original_h3_ar/w6_reference.json")})
    with pytest.raises(ValueError, match="H3 request"):
        validate_native(conf)


def test_w6_real_source_admission_rejects_unadmitted_source(tmp_path):
    from distillschedule.strategies.h3_ar_assets import H3ARAssets
    fields = {name: {} for name in H3ARAssets.__dataclass_fields__}
    fields.update(schema_version="distillschedule.original_h3_ar_assets.v1",
                  source_root=str(tmp_path / "source"), checkpoint_root=str(tmp_path / "checkpoint"))
    write(tmp_path / "native-assets.json", fields)
    write(tmp_path / "policy.json", dict(gpu_limit_bytes=1000, gpu_reserve_bytes=10,
        host_limit_bytes=3000, host_reserve_bytes=100))
    conf = replace_config(config(tmp_path), execution={"capacity_policy": "policy.json"})
    with pytest.raises(ValueError, match="wrong exact native source receipt"):
        validate_native(conf)


@pytest.mark.parametrize("workload", ["W1", "W5"])
def test_doctor_real_native_admission_never_imports_torch(tmp_path, workload):
    conf = wan_reference_fixture(tmp_path) if workload == "W1" else search_fixture(tmp_path, workload)[0]
    path = write(tmp_path / "public-config.json", conf.to_dict())
    script = ("import sys; from distillschedule.training.config import load_training_config; "
        "from distillschedule.training.native import validate_native; "
        "validate_native(load_training_config(sys.argv[1])); "
        "assert 'torch' not in sys.modules; assert 'diffusers' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", script, str(path)], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("slug,workload", [("wan_ar", "W1"), ("helios_nogan", "W2"),
    ("helios_gan", "W3"), ("self_forcing", "W4"), ("h3_fullseq", "W5"), ("h3_ar", "W6")])
def test_portable_templates_declare_exact_native_workload_and_fail_on_unfilled_input(slug, workload):
    from distillschedule.training.config import load_training_config
    path = ROOT / "recipes" / slug / "training.json"
    text = path.read_text()
    assert all(prefix not in text for prefix in ("/home/", "/share/", "/root/"))
    conf = load_training_config(path)
    assert describe_native(conf)["workload"] == workload
    assert conf.trainer["checkpoint_every"] == 0
    with pytest.raises(ValueError, match="placeholder"):
        validate_native(conf)


def test_explicit_placeholders_and_missing_paths_have_actionable_errors(tmp_path):
    conf = replace_config(config(tmp_path), execution={"assets": "REPLACE_WITH_NATIVE_ASSETS.json"})
    with pytest.raises(ValueError, match="placeholder.*execution.assets"):
        validate_native(conf)
    with pytest.raises(ValueError, match="missing execution.assets"):
        validate_native(config(tmp_path, "W2"))


def wan_reference_fixture(tmp_path):
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.strategies.wan_strategy import WanStrategy
    from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
    model, bootstrap, trace, negative = [tmp_path / n for n in ("model", "bootstrap", "trace", "negative")]
    for path in (model, bootstrap, trace, negative):
        path.mkdir()
    payload = bootstrap / "cpu-fixture.safetensors"
    payload.write_text("CPU bootstrap identity fixture, not trained weights")
    manifest = dict(schema_version="distillschedule.wan13b_bootstrap.v1", model_root=str(model),
        block_count=30, lora_rank=256, components={name: dict(trainable_payload=payload.name,
        sha256=file_sha256(payload)) for name in ("generator", "critic")})
    manifest["manifest_hash"] = fingerprint(manifest)
    write(bootstrap / "manifest.json", manifest)
    write(tmp_path / "native-assets.json", dict(schema_version="distillschedule.original_wan_assets.v1",
        model_root=str(model), bootstrap=str(bootstrap), expected_bootstrap_hash=manifest["manifest_hash"],
        trace_bundle=str(trace), expected_trace_hash="1" * 64, negative_prompt=str(negative)))
    write(tmp_path / "task.json", OriginalWanTask().to_dict())
    write(tmp_path / "reference.json", WanStrategy("fsdp", 512, False).to_dict())
    raw = config(tmp_path, "W1").to_dict()
    raw["execution"] = dict(adapter="native", mode="reference", task="task.json", assets="native-assets.json",
                            implementation="reference.json")
    raw["data"] = {"replay_fixed_trace": True}
    return TrainingConfig.from_dict(raw, source=tmp_path / "training.json")


def test_w1_reference_real_bootstrap_and_strategy_admission(tmp_path):
    conf = wan_reference_fixture(tmp_path)
    result = validate_native(conf)
    assert result["selected"]["placement"] == "fsdp"
    assert result["native_argv"] == ["benchmark-manual", "--task", str(tmp_path / "task.json"),
        "--assets", str(tmp_path / "native-assets.json"), "--output", str(tmp_path / "run"),
        "--execute", "--implementation", str(tmp_path / "reference.json"), "--cycles", "2"]
    assert result["native_physical_admission"].startswith("REQUIRED")
    assert not (tmp_path / "run").exists()


def test_w1_bootstrap_source_and_task_not_relabelled(tmp_path):
    conf = wan_reference_fixture(tmp_path)
    payload = tmp_path / "bootstrap/cpu-fixture.safetensors"
    payload.write_text("changed")
    with pytest.raises(ValueError, match="payload hash changed"):
        validate_native(conf)


def test_w1_reference_does_not_masquerade_as_arbitrary_long_training(tmp_path):
    conf = wan_reference_fixture(tmp_path)
    with pytest.raises(ValueError, match="reference/manual supports exactly 2 cycles"):
        validate_native(replace_config(conf, trainer={"cycles": 3}))
    raw = conf.to_dict(); raw["data"] = {}
    with pytest.raises(ValueError, match="fixed trace is replay"):
        validate_native(TrainingConfig.from_dict(raw, source=conf.source))


def test_w1_enters_existing_native_cli_exactly(tmp_path, monkeypatch):
    conf = wan_reference_fixture(tmp_path)
    import distillschedule.training.native as native
    from distillschedule.strategies import wan_cli
    source = native._source(conf); source["identity"]["dirty"] = False
    monkeypatch.setattr(native, "_source", lambda _: copy.deepcopy(source))
    calls = []
    monkeypatch.setattr(wan_cli, "main", lambda argv: calls.append(argv) or 23)
    assert run_native(conf) == 23
    assert calls[0] == validate_native(conf)["native_argv"]


def test_w6_reference_enters_real_task_and_capacity_api(tmp_path, monkeypatch):
    """Command-wiring unit only: real W6 source rejection is tested above."""
    import distillschedule.training.native as native
    from distillschedule.strategies.h3_ar_assets import H3ARAssets
    from distillschedule.strategies import h3_ar_cli
    fields = {name: {} for name in H3ARAssets.__dataclass_fields__}
    fields.update(schema_version="distillschedule.original_h3_ar_assets.v1",
                  source_root="/cpu-unit/source", checkpoint_root="/cpu-unit/checkpoint")
    write(tmp_path / "native-assets.json", fields)
    write(tmp_path / "policy.json", dict(gpu_limit_bytes=1000, gpu_reserve_bytes=10,
        host_limit_bytes=3000, host_reserve_bytes=100))
    raw = config(tmp_path).to_dict()
    raw["execution"].pop("plan")
    raw["execution"].update(mode="reference", implementation="zero3-native", capacity_policy="policy.json")
    conf = TrainingConfig.from_dict(raw, source=tmp_path / "training.json")
    monkeypatch.setattr(H3ARAssets, "validate", lambda _: {"unit_fixture_only": True})
    source = native._source(conf); source["identity"]["dirty"] = False
    monkeypatch.setattr(native, "_source", lambda _: copy.deepcopy(source))
    calls = []
    monkeypatch.setattr(h3_ar_cli, "main", lambda argv: calls.append(argv) or 29)
    assert run_native(conf) == 29
    assert calls == [["benchmark-manual", "--task", conf.execution["task"], "--assets",
        str(tmp_path / "native-assets.json"), "--capacity-policy", str(tmp_path / "policy.json"),
        "--output", str(tmp_path / "run"), "--execute", "--implementation", "zero3-native"]]
    value = json.loads((tmp_path / "policy.json").read_text()); value["gpu_limit_bytes"] = -1
    write(tmp_path / "policy.json", value)
    with pytest.raises(ValueError, match="invalid capacity policy"):
        validate_native(conf)


def test_w1_fresh_reference_stream_reaches_original_checkpoint_worker(tmp_path, monkeypatch):
    """Metadata wiring fixture; original tensor/capacity checks still run in worker."""
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.strategies.wan_stream import SCHEMA, WanCohortSources, task_at
    from distillschedule.strategies.wan_inputs import WanInputSources
    from distillschedule.strategies.wan_planning import source_fingerprint
    from distillschedule.workloads.best_known.registry import fingerprint
    from dataclasses import replace
    import distillschedule.strategies.wan_stream as streams
    import distillschedule.strategies.wan_worker as worker
    conf = wan_reference_fixture(tmp_path)
    task = OriginalWanTask.from_dict(json.loads((tmp_path / 'task.json').read_text()))
    source = WanInputSources(tuple(str(tmp_path / f'rank-{i}.pt') for i in range(16)))
    cohorts = WanCohortSources(tuple(replace(source, cohort=i) for i in range(3)))
    document = {'schema_version': SCHEMA, 'status': 'READY_FRESH_COHORT_STREAM', 'task': task.to_dict(),
        'identity': {'task': task.to_dict(), 'sources': cohorts.to_dict(),
                     'source_fingerprint': source_fingerprint(), 'inventory': [{}] * 3},
        'cohorts': [{'index': i, 'task': task_at(task, i).to_dict()} for i in range(3)]}
    document['stream_sha256'] = fingerprint(document)
    write(tmp_path / 'stream.json', document)
    raw = conf.to_dict()
    raw['data'] = {'prepared': 'stream.json', 'input_kind': 'original_wan_cohort_stream.v1'}
    raw['trainer'].update(cycles=3, checkpoint_every=1)
    conf = TrainingConfig.from_dict(raw, source=conf.source)
    argv = validate_native(conf)['native_argv']
    assert argv[0] == 'train-stream' and argv[-2:] == ['--checkpoint-every', '1']
    assert '--implementation' in argv and '--plan' not in argv
    calls = []
    monkeypatch.setattr(streams, 'load_stream', lambda *a: document)
    monkeypatch.setattr(worker, 'run_worker', lambda *a, **kw: calls.append(kw) or {'status': 'CPU_WIRING_ONLY'})
    from distillschedule.strategies.wan_cli import main
    assert main(argv) == 0
    assert calls[0]['mode'] == 'train' and calls[0]['cycles'] == 3
    assert calls[0]['checkpoint_every'] == 1 and calls[0]['strategy'].placement == 'fsdp'
    assert calls[0]['stream_path'] == tmp_path / 'stream.json'
    assert not (tmp_path / 'run').exists()
