"""Direct public preset -> original compiler -> runtime binding, CPU only.

Identity fixtures are not pretrained assets. Only the final physical dispatch
is intercepted; source, typed implementations and launch lowering are real.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from distillschedule.training.config import TrainingConfig, load_training_config
from distillschedule.training.native import describe_native, run_native, validate_native
from test_training_native import (
    ROOT, config, replace_config, search_fixture, wan_reference_fixture, write,
)


PRESETS = {
    "W1": "wan_ar.incumbent21a.v1",
    "W2": "helios_nogan.safe_ed8.v1",
    "W3": "helios_gan.fa3_down40_nomerge.v1",
    "W4": "self_forcing.c51.v1",
    "W5": "h3_fullseq.cpu_staged_trim.v1",
    "W6": "h3_ar.zero3_ap.v1",
}


def implementation(workload):
    from distillschedule.search.helios_implementation import HeliosImplementation
    from distillschedule.search.helios_safe_bundle import HeliosSafeBundle
    from distillschedule.search.helios_gan_bundle import HeliosGANBundle
    from distillschedule.search.livewan_implementation import LiveWanImplementation
    from distillschedule.search.h3_implementation import H3Implementation
    if workload == "W2":
        return HeliosImplementation("no-gan", "node-hsdp", 512, 5, False, HeliosSafeBundle.safe())
    if workload == "W3":
        return HeliosImplementation("gan", "world16-fsdp", 2048, 5, True, HeliosGANBundle.manual())
    if workload == "W4":
        return LiveWanImplementation("pipeline", True, True, True, bundle="c51_full_v1")
    return H3Implementation(cache_policy="cpu_staged_trim")


def direct(conf, workload):
    raw = conf.to_dict()
    raw["execution"].pop("plan", None)
    raw["execution"].pop("implementation", None)
    raw["execution"].update(mode="manual", preset=PRESETS[workload])
    return TrainingConfig.from_dict(raw, source=conf.source)


def clean_source(monkeypatch):
    import distillschedule.training.native as native
    original = native._source
    def observe(conf):
        value = original(conf)
        value["identity"]["dirty"] = False  # CPU wiring fixture, not a clean-tree claim.
        return value
    monkeypatch.setattr(native, "_source", observe)


@pytest.mark.parametrize("workload", ["W2", "W3", "W4", "W5"])
def test_direct_preset_without_plan_uses_same_real_lowering_and_dispatch(tmp_path, monkeypatch, workload):
    impl = implementation(workload)
    conf, saved, _ = search_fixture(tmp_path, workload, implementation=impl)
    conf = direct(conf, workload)
    Path(tmp_path / "selected-plan.json").unlink()  # No historical/user-created plan exists.
    clean_source(monkeypatch)
    import distillschedule.workloads.best_known.dispatch as dispatch
    calls = []
    monkeypatch.setattr(dispatch, "run_resolved", lambda receipt, **kw: calls.append((receipt, kw)) or 17)
    result = validate_native(conf)
    assert result["selected"]["id"] == impl.id
    assert result["preset"]["id"] == PRESETS[workload]
    assert result["preset"]["continuous_training"] is False
    assert "selected_plan" not in result
    assert run_native(conf) == 17
    receipt, kwargs = calls[0]
    assert receipt["mode"] == "correctness"
    assert receipt["fully_resolved_profile"] == saved["fully_resolved_profile"]
    assert receipt["asset_config"] == saved["asset_config"]
    assert kwargs == {"dry_run": False, "coordinator_output": tmp_path / "run"}
    assert not (tmp_path / "run").exists()
    runtime = receipt["fully_resolved_profile"]["runtime"]
    if workload in ("W2", "W3"):
        from distillschedule.search.helios_implementation import current_implementation
        environment = receipt["nodes"][0]["environment"]
        assert current_implementation(environment) == impl
        assert environment["DISTILLSCHEDULE_HELIOS_LORA_B_GEMM_EPILOGUE"] == "on"
        assert environment["DISTILLSCHEDULE_HELIOS_LORA_CHECKPOINT_REPLAY_SCOPE"] == "on"
        # Enter the actual setup-time owner, not just an environment assertion.
        args = SimpleNamespace(variant=impl.variant, timing_mode=True, profile_mode=False,
            profile_diagnostics=False, ema_device="cuda", attention_backend="local-fa2",
            generator_residual_storage="cuda_out_of_place", critic_residual_storage="cuda_out_of_place",
            critic_only_step=None, operation_source=False, lora_checkpoint_replay_scope=True)
        flags = {key.removeprefix("DISTILLSCHEDULE_").removeprefix("HELIOS_").lower(): value == "on"
                 for key, value in environment.items() if key.startswith("DISTILLSCHEDULE_") and value in ("on", "off")}
        flags.update(mesh_policy=impl.mesh, modulation_chunk_tokens=impl.chunk_tokens,
            generator_nograd_persistent_unshard=impl.persistent_unshard, rollout_batch_size=1,
            pack_real_score_pair=False, rollout_backend="native-fsdp2-batch",
            selective_flash_checkpoint_stride=2 if workload == "W3" else 0)
        binding = impl.receipt_binding(args=args, policy=SimpleNamespace(**flags), environment=environment)
        assert binding["candidate_bundle"]["id"] == impl.candidate_bundle.id
        with pytest.raises(ValueError):
            impl.candidate_bundle.effect_snapshot({})  # Installation/metadata is not executed counters.
        if workload == "W3":
            assert environment["DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_MERGED_EXECUTION"] == "off"
            assert impl.candidate_bundle.cost_identity()["merged_lora"] is False
    elif workload == "W4":
        assert runtime["planner_implementation"]["bundle"] == "c51_full_v1"
        assert runtime["planner_c51_bundle"]
        assert "score_flash_attn3_kernel" in receipt["asset_config"]["assets"]
    else:
        binding = impl.receipt_binding(receipt["nodes"][0]["resolved_model_config"],
                                       receipt["nodes"][0]["environment"])
        assert binding["candidate_id"] == impl.id
        assert result["preset"]["numerical_status"] == "FAIL_UNRESOLVED"
        assert runtime["strict_pair_gate"] == "NOT_RUN"


@pytest.mark.parametrize("workload", ["W2", "W3", "W4", "W5"])
def test_equivalent_saved_plan_and_preset_have_identical_effective_selection(tmp_path, workload):
    conf, _, _ = search_fixture(tmp_path, workload, implementation=implementation(workload))
    saved = validate_native(conf)
    preset = validate_native(direct(conf, workload))
    assert preset["selected"] == saved["selected"]
    assert preset["task"] == saved["task"]
    assert preset["hardware"] == saved["hardware"]


def test_w1_preset_passes_typed_incumbent_to_existing_worker_without_strategy_file(tmp_path, monkeypatch):
    conf = direct(wan_reference_fixture(tmp_path), "W1")
    (tmp_path / "reference.json").unlink()
    clean_source(monkeypatch)
    import distillschedule.strategies.wan_worker as worker
    calls = []
    monkeypatch.setattr(worker, "run_worker", lambda task, assets, **kw: calls.append((task, assets, kw)) or {"status": "CPU_WIRING_ONLY"})
    report = validate_native(conf)
    selected = report["selected"]
    assert selected["candidate_bundle"]["generator_rollout_cuda_graph"] is True
    assert selected["candidate_bundle"]["critic_grad_cuda_graph"] is True
    assert selected["placement"] == "resident" and selected["chunk_tokens"] == 2048
    assert not selected.get("retained_critic_blocks")
    assert run_native(conf) == {"status": "CPU_WIRING_ONLY"}
    _, _, kwargs = calls[0]
    assert kwargs["mode"] == "manual" and kwargs["cycles"] == 2
    assert kwargs["strategy"].candidate_bundle.enabled
    assert kwargs["strategy"].to_dict() == selected
    assert kwargs["output"] == tmp_path / "run"
    assert not (tmp_path / "run").exists()


def test_w6_preset_enters_original_zero3_ap_worker_and_retains_capacity_gate(tmp_path, monkeypatch):
    from distillschedule.strategies.h3_ar_assets import H3ARAssets
    import distillschedule.strategies.h3_ar_worker as worker
    fields = {name: {} for name in H3ARAssets.__dataclass_fields__}
    fields.update(schema_version="distillschedule.original_h3_ar_assets.v1",
                  source_root="/cpu-unit/source", checkpoint_root="/cpu-unit/checkpoint")
    write(tmp_path / "native-assets.json", fields)
    write(tmp_path / "policy.json", dict(gpu_limit_bytes=1000, gpu_reserve_bytes=10,
        host_limit_bytes=3000, host_reserve_bytes=100))
    conf = direct(replace_config(config(tmp_path), execution={"capacity_policy": "policy.json"}), "W6")
    monkeypatch.setattr(H3ARAssets, "validate", lambda _: {"unit_fixture_only": True})
    clean_source(monkeypatch)
    calls = []
    monkeypatch.setattr(worker, "run_worker", lambda task, assets, output, **kw: calls.append(kw) or {"unit": True})
    assert run_native(conf) == 0
    assert calls[0]["implementation"].id == "zero3-ap"
    assert calls[0]["capacity_policy"]["gpu_limit_bytes"] == 1000
    policy = json.loads((tmp_path / "policy.json").read_text()); policy["gpu_limit_bytes"] = -1
    write(tmp_path / "policy.json", policy)
    with pytest.raises(ValueError, match="invalid capacity policy"):
        validate_native(conf)


@pytest.mark.parametrize("workload", ["W2", "W3", "W4", "W5"])
@pytest.mark.parametrize("change", ["shape", "source", "cycles", "assets", "override"])
def test_direct_preset_binding_drift_rejects_before_dispatch(tmp_path, workload, change):
    conf, _, _ = search_fixture(tmp_path, workload, implementation=implementation(workload))
    conf = direct(conf, workload)
    if change == "shape":
        task_path = Path(conf.execution["task"])
        task = json.loads(task_path.read_text())
        field = "video_frames" if workload in ("W2", "W3") else "frames"
        task[field] = 17
        write(task_path, task)
    elif change == "source":
        conf = replace_config(conf, execution={"expected_head": "f" * 40})
    elif change == "cycles":
        conf = replace_config(conf, trainer={"cycles": 100})
    elif change == "assets":
        path = Path(conf.execution["assets"])
        assets = json.loads(path.read_text()); assets["assets"].pop(next(iter(assets["assets"])))
        write(path, assets)
    else:
        conf = replace_config(conf, execution={"implementation": "merged-lora"})
    with pytest.raises(ValueError):
        run_native(conf, dry_run=True)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("workload", list(PRESETS))
def test_preset_requires_explicit_manual_opt_in_and_matching_identity(tmp_path, workload):
    raw = direct(config(tmp_path, workload), workload).to_dict()
    raw["execution"]["preset"] = PRESETS["W2"] if workload != "W2" else "unknown.v1"
    with pytest.raises(ValueError, match="preset"):
        validate_native(TrainingConfig.from_dict(raw, source=tmp_path / "training.json"))
    raw["execution"].update(preset=PRESETS[workload], mode="reference")
    with pytest.raises(ValueError, match="manual"):
        validate_native(TrainingConfig.from_dict(raw, source=tmp_path / "training.json"))


@pytest.mark.parametrize("slug,workload", [("wan_ar", "W1"), ("helios_nogan", "W2"),
    ("helios_gan", "W3"), ("self_forcing", "W4"), ("h3_fullseq", "W5"), ("h3_ar", "W6")])
def test_six_public_recipes_select_versioned_presets_not_private_plans(slug, workload):
    conf = load_training_config(ROOT / "recipes" / slug / "training.json")
    assert conf.execution["mode"] == "manual"
    assert conf.execution["preset"] == PRESETS[workload]
    assert "plan" not in conf.execution and "implementation" not in conf.execution
    description = describe_native(conf)
    assert description["preset"]["gpu_execution"] == "NOT_RUN"
    assert description["preset"]["continuous_training"] is False
    assert description["capabilities"]["cycles"]["exact"] == 2
    assert description["capabilities"]["checkpoint"]["supported"] is False
    assert description["capabilities"]["resume"]["supported"] is False


@pytest.mark.parametrize("workload", ["W2", "W5"])
def test_public_direct_admission_imports_no_torch_or_model_and_writes_nothing(tmp_path, workload):
    conf, _, _ = search_fixture(tmp_path, workload, implementation=implementation(workload))
    conf = direct(conf, workload)
    path = write(tmp_path / "public-config.json", conf.to_dict())
    (tmp_path / "selected-plan.json").unlink()
    script = ("import sys; from distillschedule.training.config import load_training_config; "
        "from distillschedule.training.native import validate_native; "
        "validate_native(load_training_config(sys.argv[1])); "
        "assert 'torch' not in sys.modules; assert 'diffusers' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", script, str(path)], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("workload,input_kind", [
    ("W2", "original_helios_cohort_stream.v1"),
    ("W4", "original_livewan_teacher_cohorts.v1"),
    ("W5", "original_h3_grid_latent_dataset.v1"),
])
def test_direct_presets_cannot_claim_fresh_reference_lifecycle(tmp_path, workload, input_kind):
    conf = direct(config(tmp_path, workload), workload)
    conf = replace_config(conf, data={"input_kind": input_kind, "prepared": "fresh.json"})
    with pytest.raises(ValueError, match="reference only"):
        validate_native(conf)


def test_w1_literal_preset_rejects_fresh_stream_and_long_horizon(tmp_path):
    conf = direct(wan_reference_fixture(tmp_path), "W1")
    with pytest.raises(ValueError, match="exactly 2 cycles"):
        validate_native(replace_config(conf, trainer={"cycles": 3}))
    conf = replace_config(conf, data={"input_kind": "original_wan_cohort_stream.v1", "prepared": "stream.json"})
    with pytest.raises(ValueError, match="fixed.*trace|fresh.*stream"):
        validate_native(conf)


def test_w5_failed_original_gate_is_not_relabelled_or_retried(tmp_path, monkeypatch):
    conf, _, _ = search_fixture(tmp_path, "W5", implementation=implementation("W5"))
    conf = direct(conf, "W5")
    clean_source(monkeypatch)
    import distillschedule.workloads.best_known.dispatch as dispatch
    calls = []
    def failed_gate(receipt, **kwargs):
        calls.append(receipt)
        raise ValueError("original sampled numerical gate FAIL")
    monkeypatch.setattr(dispatch, "run_resolved", failed_gate)
    with pytest.raises(ValueError, match="numerical gate FAIL"):
        run_native(conf)
    assert len(calls) == 1
    assert calls[0]["fully_resolved_profile"]["runtime"]["strict_pair_gate"] == "NOT_RUN"
