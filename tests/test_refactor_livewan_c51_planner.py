"""CPU/source fixtures only: C51 costs and GPU admission are never fabricated."""
import copy
import inspect
import json
from pathlib import Path

import pytest

from distillschedule.search.livewan_implementation import LIVEWAN, LiveWanImplementation, implementations
from distillschedule.workloads.best_known.registry import fingerprint, load_profile


def test_versioned_bundle_opt_in_keeps_legacy_defaults_and_ids():
    from distillschedule.search.workflow_cli import candidate_inventory
    legacy = implementations()
    assert len(legacy) == 5
    assert all(i.bundle == "legacy" for i in legacy)
    assert not any("c51" in i.id for i in legacy)
    values = implementations(include_c51=True)
    assert len(values) == 6
    assert {i.bundle for i in values} == {"control_v1", "c51_full_v1"}
    for impl in values:
        assert LiveWanImplementation.parse(impl.id) == impl
        assert LiveWanImplementation.from_assignment(impl.assignment) == impl
        assert "bundle" in impl.assignment
    c51 = values[-1]
    assert c51.source_id == "c51" and c51.bundle == "c51_full_v1"
    inventory = candidate_inventory(LIVEWAN, include_livewan_c51=True)
    assert len(inventory["candidates"]) == 6
    assert len({r["semantic_fingerprint"] for r in inventory["candidates"]}) == 1
    assert inventory["gpu_execution"] == "NOT_RUN" and not inventory["historical_seconds_are_costs"]
    assert load_profile(LIVEWAN)["contract"]["variant"] == "c34-continuous-b16-residual-layout"
    assert values[0].lower_profile()["required_assets"] == values[-1].lower_profile()["required_assets"]


def test_c51_lowering_reaches_real_native_mechanisms():
    from distillschedule.search.execution import resolve_candidate, validated_profile
    from distillschedule.runtime.livewan_generator_fused_rms import CONTRACT as RMS
    from distillschedule.runtime.livewan_generator_splitkv import CONTRACT as KV
    from distillschedule.runtime.livewan_score_fa3 import BACKEND_CONTRACT, SAC_CONTRACT
    from distillschedule.workloads.best_known.registry import repository_root
    impl = implementations(include_c51=True)[-1]
    receipt = resolve_candidate(LIVEWAN, impl.id, mode="calibration", run_id="cpu-c51-only")
    validated_profile(receipt, root=repository_root())
    raw = receipt["native_plan"]["template"]
    metadata = raw["metadata"]
    assert metadata["generator_rms_epilogue"] == RMS
    assert metadata["generator_readonly_splitkv"] == KV
    assert metadata["score_attention_backend"] == BACKEND_CONTRACT
    assert metadata["critic_attention_checkpoint"] == SAC_CONTRACT
    assert all(k not in metadata for k in ("critic_step1_overlap", "critic_ffn_down_checkpoint", "step0_wave1_early_delivery"))
    assert receipt["fully_resolved_profile"]["runtime"]["planner_implementation"] == impl.assignment
    assert receipt["fully_resolved_profile"]["runtime"]["environment"]["LIVEWAN_PIPELINE_VARIANT"] == "c51-continuous-b16-generator-fused-rms"
    assert raw["workload"]["shape"]["latent"]["frames"] == 21


def test_c51_cannot_relax_geometry_prerequisites_or_decision_schema():
    from distillschedule.search.livewan_request import LiveWanRequest
    with pytest.raises(ValueError):
        implementations(include_c51=True, request=LiveWanRequest(frames=69))
    with pytest.raises(ValueError):
        LiveWanImplementation("pipeline", False, False, False, bundle="c51_full_v1")
    with pytest.raises(ValueError):
        LiveWanImplementation("fsdp", bundle="c51_full_v1")
    impl = implementations(include_c51=True)[-1]
    with pytest.raises(ValueError):
        LiveWanImplementation.from_assignment({**impl.assignment, "merged_lora": "on"})


def test_current_observation_geometry_and_calibration_hooks_survive_port():
    from distillschedule.runtime import livewan_crossscale_worker as worker
    for function in (worker.prepare_rollout, worker._dmd_fake_velocity,
                     worker._dmd_teacher_velocity, worker.critic_backward_for_batch,
                     worker.generator_backward_for_sample):
        assert "@_observe_native_action" in inspect.getsource(function)
    source = inspect.getsource(worker.run_worker)
    for token in ("install_geometry(", "AsyncRangeCollector(", "TransportObserver(",
                  '"operation_calibration"', "installed_plan_sha256", "actual_latent_shapes"):
        assert token in source
    assert "_operation_calibrator.boundary(" in inspect.getsource(worker._run_pipeline_8x8)


def test_unified_cli_c51_is_source_only_unless_explicit_native_execution(tmp_path, monkeypatch, capsys):
    from distillschedule.search.workflow_cli import main
    from distillschedule.search.execution import NativeTrialRunner
    from distillschedule.workloads.best_known.registry import git_head, repository_root, load_assets
    monkeypatch.setattr(NativeTrialRunner, "_rpc", lambda *a, **kw: pytest.fail("CPU source-only intent must not SSH/launch"))
    assert main(["candidates", "--workload", LIVEWAN, "--include-livewan-c51"]) == 0
    assert len(json.loads(capsys.readouterr().out)["candidates"]) == 6
    values = implementations(include_c51=True)
    out = tmp_path / "source-only"
    assert main(["calibrate", "--workload", LIVEWAN, "--expected-head", git_head(repository_root()),
        "--candidate", values[0].id, values[-1].id, "--run-id", "cpu-only-c51",
        "--output", str(out)]) == 0
    intent = json.loads((out / "intent.json").read_text())
    assert intent["gpu_execution"] == "NOT_RUN" and intent["formal_timing_attempts"] == 0
    assert intent["source_only_plans"][-1]["search_candidate"]["id"] == values[-1].id
    # New kernel assets must be explicit and common to the entire v1 cohort.
    with pytest.raises(ValueError, match="exactly match"):
        load_assets(repository_root() / "configs/distillschedule/best_known/assets_examples" / (LIVEWAN + ".json"),
                    values[-1].lower_profile())


def test_pinned_score_fa3_artifact_drift_fails_before_import_or_model(tmp_path, monkeypatch):
    from distillschedule.runtime.livewan_score_fa3 import load_flash_attn3
    monkeypatch.setenv("DISTILLSCHEDULE_SCORE_FA3_KERNEL", str(tmp_path))
    (tmp_path / "metadata.json").write_text("{}")
    with pytest.raises(RuntimeError, match="pinned FA3 artifact missing/changed"):
        load_flash_attn3()


@pytest.mark.parametrize("path,value", [
    (("workload", "shape", "batch_size"), 32),
    (("workload", "shape", "frames"), 18),
    (("workload", "metadata", "sigma_distribution"), "grid1000"),
    (("workload", "metadata", "grad_blocks"), 2),
    (("workload", "metadata", "steps_per_block"), 3),
    (("workload", "metadata", "guidance"), 3.0),
    (("workload", "generator", "precision"), "bf16"),
    (("metadata", "critic_b2_preserves_per_sample_rng"), False),
    (("metadata", "generator_rollout_cast_cache", "cache_cross_optimizer"), True),
    (("metadata", "score_attention_backend", "artifact_interface_sha256"), "f" * 64),
    (("metadata", "generator_rms_epilogue", "name"), "generator_nograd_rms_cast_square_v1"),
])
def test_exact_bundle_rejects_shape_source_rng_precision_and_training_drift(path, value):
    from distillschedule.core.plan import ExecutionPlan
    from distillschedule.runtime.livewan_c51_bundle import source_template, validate_plan
    plan = ExecutionPlan.from_mapping(source_template()).to_dict()
    assert validate_plan(plan, mode="pipeline")
    parent = plan
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises((ValueError, RuntimeError)):
        validate_plan(plan, mode="pipeline")


def _runtime_fixture(rank):
    """Synthetic counts for real validators; not collected GPU observations."""
    from distillschedule.runtime import (
        livewan_generator_fused_rms as rms, livewan_generator_splitkv as kv,
        livewan_score_gate as gate, livewan_score_layernorm as ln, livewan_score_fa3 as fa,
        livewan_generator_modulation as mod, livewan_generator_gate as ggate,
        livewan_rollout_cast_cache as casts, livewan_score_rmsnorm as srms,
    )
    from distillschedule.runtime.livewan_c51_bundle import receipt_contract
    na = dict(execution_status="NOT_APPLICABLE")
    row = dict(fixture_scope="CPU_SYNTHETIC_OPERATOR_COUNTS_NOT_GPU_VALIDATION",
        workload_request={"frames": 81, "schema_version": "distillschedule.livewan_request.v1"},
        actual_latent_shapes=[[1, 16, 21, 46, 80]], installed_plan_sha256="f" * 64,
        planner_c51_bundle=dict(receipt_contract(), execution_status="PASS"),
        call_inventory={"teacher/nograd/plain": 4, "critic/grad/checkpoint": 10},
        score_residual_gate=copy.deepcopy(na), score_layernorm=copy.deepcopy(na),
        score_rmsnorm_kernel=dict(na, forward_calls=0, backward_calls=0),
        critic_attention_checkpoint=dict(na, checkpoint_calls=0),
        score_attention_backend=dict(fa.BACKEND_CONTRACT, **na,
            function_patched=False, invocation_count=0, contract_failure_count=0),
        generator_attention_backend=dict(invocation_count=2, readonly_splitkv=dict(
            enabled=True, name=kv.CONTRACT["name"], contract=kv.CONTRACT,
            production_installed=False, **na, eligible_nograd_calls=0)),
        generator_rms_epilogue=copy.deepcopy(na), generator_rollout_cast_cache=copy.deepcopy(na),
        generator_modulation=copy.deepcopy(na), generator_gate=copy.deepcopy(na))
    if rank < 8:
        scope = dict(execution_status="PASS", scope_entries=20, scope_exits=20,
                     active=False, scope_failures=0, contract_failures=0)
        row["generator_rms_epilogue"] = dict(rms.CONTRACT, **scope,
            scope_forward_calls=[1920] * 20, calls=38400, shape_histogram={"1x2760x1536": 38400})
        row["generator_rollout_cast_cache"] = dict(scope, name=casts.NAME,
            effective_dtype_changed=False, cache_cross_optimizer=False)
        row["generator_modulation"] = dict(mod.PLAN_CONTRACT, **scope,
            scope_forward_calls=[1281] * 20, calls=25620)
        row["generator_gate"] = dict(ggate.PLAN_CONTRACT, **scope, output_layout="row_major",
            scope_forward_calls=[1260] * 20, calls=25200)
        row["generator_attention_backend"]["readonly_splitkv"].update(
            execution_status="PASS", production_installed=True,
            api_source_sha256=kv.EXPECTED_KVCACHE_SOURCE_SHA256, eligible_nograd_calls=1,
            reference_calls=1, contract_failure_count=0, shape_histogram={"synthetic_qkv": 1})
    else:
        row["score_residual_gate"] = dict(gate.PLAN_CONTRACT, execution_status="PASS",
            install_status="PASS", contract_failures=0, forward_calls=1520, backward_calls=600)
        row["score_layernorm"] = dict(ln.PLAN_CONTRACT, execution_status="PASS", install_status="PASS",
            contract_failures=0, total_norm_calls=2294, selected_entries=2294, forward_calls=2294,
            native_fallback_calls=0, native_fallback_histogram={}, backward_calls=100)
        row["score_attention_backend"].update(execution_status="PASS", install_status="PASS",
            function_patched=True, invocation_count=10, backward_invocation_count=10,
            artifact_root="/cpu-fixture-only")
        row["critic_attention_checkpoint"] = dict(name=fa.SAC_CONTRACT["name"],
            source=dict(interface_sha256=fa.INTERFACE_SHA256,
                        operator=fa.OP_NAME.replace("::", ".") + ".default"),
            execution_status="PASS", production_installed=True, self_attention_only=True,
            contract_failures=0, checkpoint_calls=300, original_fa_calls=300,
            replay_fa_cache_reads=300, expected_checkpoint_calls=300,
            allow_cache_entry_mutation=False, cpu_offload=False)
        row["score_rmsnorm_kernel"] = dict(srms.PLAN_CONTRACT, install_status="PASS",
            contract_failures=0, forward_calls=10, backward_calls=10,
            shape_histogram={"1x19320x1536": 1, "1x19320x5120": 1})
    return row


@pytest.mark.parametrize("rank", range(16))
def test_actual_runtime_counters_required_not_config_only(rank):
    from distillschedule.runtime.livewan_c51_bundle import runtime_receipt_errors
    raw = _runtime_fixture(rank)
    assert not runtime_receipt_errors(raw, rank=rank, cycles=2)
    keys = (["generator_rms_epilogue", "generator_gate", "generator_modulation"] if rank < 8 else
            ["score_layernorm", "score_residual_gate", "critic_attention_checkpoint", "score_attention_backend"])
    for key in keys:
        broken = copy.deepcopy(raw)
        broken.pop(key)
        assert runtime_receipt_errors(broken, rank=rank, cycles=2)
    raw["planner_c51_bundle"]["source_template_sha256"] = "f" * 64
    assert runtime_receipt_errors(raw, rank=rank, cycles=2)
    raw = _runtime_fixture(rank)
    raw["actual_latent_shapes"] = [[1, 16, 18, 46, 80]]
    assert runtime_receipt_errors(raw, rank=rank, cycles=2)


def test_real_common_calibration_search_lowering_rejects_fake_costs_and_capacity(tmp_path, monkeypatch):
    """Common chain with labeled CPU native envelopes, never empirical costs."""
    from dataclasses import replace
    import test_refactor_native_macro_calibration as fixtures
    from distillschedule.search.native_macro_calibration import assemble_calibrations
    from distillschedule.search.native_macro_adapter import lower_prediction, validate_lowered_plan, build_problem
    from distillschedule.search.model_search import search_plans
    from distillschedule.search.cost_model import CostModel, MissingCalibration
    from distillschedule.search.simulator import simulate
    from distillschedule.workloads.best_known.registry import git_head, repository_root
    head = git_head(repository_root())
    monkeypatch.setattr(fixtures, "HEAD", head)
    values = implementations(include_c51=True)
    results = []
    for index, impl in enumerate((values[0], values[-1])):
        folder = tmp_path / f"fixture{index}"
        folder.mkdir()
        fixture = fixtures._fixture(folder, monkeypatch, LIVEWAN, impl)
        if impl.bundle == "c51_full_v1":
            # Prove the legacy rank envelope cannot authorize new operators.
            with pytest.raises(ValueError, match="actual full-bundle"):
                fixtures._assemble(fixture)
            for path in fixture.raw_paths:
                raw = json.loads(path.read_text())
                counts = _runtime_fixture(raw["rank"])
                counts.pop("installed_plan_sha256")
                raw.update(counts)
                path.write_text(json.dumps(raw))
        result = fixtures._assemble(fixture)
        path = folder / "result.json"
        path.write_text(json.dumps(result))
        results.append(path)
    problem, costs = assemble_calibrations(results, output=tmp_path / "assembled", expected_head=head)
    result = search_plans(problem, costs, exhaustive=True, max_proposals=32)
    assert result["predicted_best"]["assignment"] == values[-1].assignment
    assert result["gpu_execution"] == "NOT_RUN" and not result["claimable_speedup"]
    selected = lower_prediction(result, problem, costs, run_id="cpu-synthetic-c51-selected")
    validate_lowered_plan(json.loads(json.dumps(selected)))
    assert selected["native_plan"]["template"]["metadata"]["generator_rms_epilogue"]["name"] == "generator_nograd_fused_rms_row_v1"
    assert selected["fully_resolved_profile"]["runtime"]["planner_implementation"]["bundle"] == "c51_full_v1"
    assert simulate(replace(problem, memory_limits={r: 1 for r in range(16)}),
                    values[-1].assignment, costs)["status"] == "PREDICTED_CAPACITY_FAIL"
    with pytest.raises(ValueError, match="missing lowering rule"):
        simulate(problem, values[-2].assignment, costs)
    with pytest.raises(MissingCalibration):
        simulate(problem, values[-1].assignment, CostModel(costs.binding, costs.records[:-1]))
    forged = CostModel(costs.binding, (replace(costs.records[0],
        samples_ns=(costs.records[0].samples_ns[0] + 1,)), *costs.records[1:]))
    with pytest.raises(ValueError, match="samples changed"):
        forged.verify_receipts()
    with pytest.raises(ValueError, match="mismatch"):
        CostModel({**costs.binding, "source_head": "f" * 40}, costs.records).bind(
            problem.binding, problem_scope="MEASURED")
    layout_path = Path(problem.lowering["layout_receipt"])
    layout = json.loads(layout_path.read_text())
    layout.pop("calibration_results")
    layout_path.write_text(json.dumps(layout))
    with pytest.raises(ValueError, match="source-qualified calibration"):
        build_problem(layout_path)
