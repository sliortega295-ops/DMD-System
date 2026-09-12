from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from distillschedule.evidence.trajectory_state_sketch import (
    canonical_state_name,
    deterministic_flat_indices,
)
from distillschedule.runtime import fsdp2_static, wan13b_fair
from distillschedule.runtime.fsdp2_p0 import (
    expected_trace_rollout_call_inventory,
    validate_exact_rollout_call_inventory,
)


ROOT = Path(__file__).resolve().parents[1]


def test_trajectory_is_an_explicit_nonlegacy_mode() -> None:
    controller = (ROOT / "scripts/distillschedule/run_wan13b_fair_world16.py").read_text()
    launcher = (ROOT / "scripts/distillschedule/launch_wan13b_fair_world16_node.sh").read_text()
    assert '"trajectory"' in controller
    assert '"${MODE}" == trajectory' in launcher
    assert "probe_wan13b_trajectory_world16.py" in launcher
    assert "DISTILLSCHEDULE_TRAJECTORY_CYCLES" in launcher
    assert '"--trajectory-cycles", type=int, choices=(1, 2, 5, 50)' in controller
    assert "probe_wan13b_fair_world16.py" in launcher


def test_controller_supports_managed_platform_ssh_transport() -> None:
    controller = (
        ROOT / "scripts/distillschedule/run_wan13b_fair_world16.py"
    ).read_text()
    assert 'value.add_argument("--ssh-port", type=int, default=22)' in controller
    assert '"--ssh-tty"' in controller
    assert 'result.append("-tt")' in controller
    assert '"-p",' in controller
    assert "port=args.ssh_port" in controller
    assert 'decoded.index("{")' in controller
    assert 'decoded.rindex("}")' in controller


def test_trajectory_worker_preserves_g1_c5_ema5_and_append_only_receipts() -> None:
    path = ROOT / "scripts/distillschedule/probe_wan13b_trajectory_world16.py"
    source = path.read_text()
    ast.parse(source)
    assert "AppendOnlyTrajectoryWriter" in source
    assert '"generator": 1, "critic": 5, "ema": 5' in source
    assert "for source_step in SOURCE_STEPS" in source
    assert '"FIXED_FIVE_STEP_COHORT_REPLAY"' in source
    assert 'RESIDENT_WINNER = "resident-async128-tree-chunk2048"' in source
    assert 'FSDP_BASELINE = "world16-fsdp"' in source
    from distillschedule.search.wan_contract import arm
    assert arm("world16-fsdp")["name"] == "WORLD16_FSDP_BASELINE"
    assert "expected_trace_rollout_call_inventory(" in source
    assert '"trace_rollout_contract_sha256"' in source
    assert "rollout_call_count_each" not in source
    assert "if not finite:" in source
    assert "trajectory cycles must be one of 1, 2, 5, or 50" in source


def test_realistic_b1_rollout_inventory_schema_is_executed_fail_closed() -> None:
    generator_update = {
        "call_count": 4,
        "stage_call_count": {0: 2, 1: 1, 2: 1},
        "logical_call_count": 4,
        "logical_stage_call_count": {0: 2, 1: 1, 2: 1},
        "dummy_call_count": 0,
    }
    critic_rollout_b1 = {
        "batch_size": 1,
        "call_count": 6,
        "stage_call_count": {0: 2, 1: 2, 2: 2},
        "logical_call_count": 6,
        "logical_stage_call_count": {0: 2, 1: 2, 2: 2},
        "dummy_call_count": 0,
    }
    rows = [generator_update, *(critic_rollout_b1.copy() for _ in range(5))]
    normalized = [
        validate_exact_rollout_call_inventory(row, label=f"rollout{index}")
        for index, row in enumerate(rows)
    ]
    assert len(normalized) == 6
    assert [row["call_count"] for row in normalized] == [4, 6, 6, 6, 6, 6]

    missing_logical = dict(critic_rollout_b1)
    missing_logical.pop("logical_call_count")
    with pytest.raises(ValueError, match="missing.*logical_call_count"):
        validate_exact_rollout_call_inventory(missing_logical, label="critic B1")

    drifted = dict(critic_rollout_b1, logical_call_count=7)
    with pytest.raises(AssertionError, match="logical stage-call census"):
        validate_exact_rollout_call_inventory(drifted, label="critic B1")

    producer = (ROOT / "runtime/static_fsdp2_p0_helios.py").read_text()
    assert '"logical_call_count": expected_call_count' in producer
    assert '"logical_stage_call_count": expected_stage_calls' in producer
    assert "validate_exact_rollout_call_inventory(" in producer


def _rank_control_validation() -> dict[str, object]:
    rows = (
        ("step/10/generator/rollout", (1, 0, 0)),
        ("step/10/critic/rollout", (1, 1, 1)),
        ("step/11/critic/rollout", (1, 1, 1)),
        ("step/12/critic/rollout", (1, 1, 0)),
        ("step/13/critic/rollout", (1, 0, 0)),
        ("step/14/critic/rollout", (1, 1, 1)),
    )
    operations = [
        {
            "operation": operation,
            "synchronized_exit": list(exits),
            "per_stage_call_counts": [value + 1 for value in exits],
            "logical_call_count": sum(value + 1 for value in exits),
            "physical_call_count": sum(value + 1 for value in exits),
            "rank_count": 16,
        }
        for operation, exits in rows
    ]
    return {
        "status": "PASS",
        "semantics": "official_helios_normal_rank0_broadcast_strict",
        "rank_count": 16,
        "operation_count": len(operations),
        "global_entry_count": 16 * len(operations),
        "local_exception_entry_count": 0,
        "cross_rank_call_count_parity": True,
        "operations": operations,
    }


def test_trace_derived_rollout_inventories_preserve_variable_exits() -> None:
    validation = _rank_control_validation()
    operations = [row["operation"] for row in validation["operations"]]  # type: ignore[index]
    inventories = [
        expected_trace_rollout_call_inventory(
            validation, operation=str(operation), expected_rank_count=16
        )
        for operation in operations
    ]
    assert [row["call_count"] for row in inventories] == [4, 6, 6, 5, 4, 6]
    assert inventories[0]["stage_call_count"] == {0: 2, 1: 1, 2: 1}
    assert inventories[3]["stage_call_count"] == {0: 2, 1: 2, 2: 1}
    assert inventories[4]["stage_call_count"] == {0: 2, 1: 1, 2: 1}
    assert inventories[5]["stage_call_count"] == {0: 2, 1: 2, 2: 2}
    assert inventories[0]["call_count"] + sum(
        row["call_count"] for row in inventories[1:]
    ) == 31
    assert sum(row["call_count"] for row in inventories[1:]) == 27


def test_trace_derived_rollout_inventory_rejects_malformed_trace_receipts() -> None:
    bad_total = _rank_control_validation()
    bad_total["operations"][3]["physical_call_count"] = 6  # type: ignore[index]
    with pytest.raises(ValueError, match="inconsistent exact call census"):
        expected_trace_rollout_call_inventory(
            bad_total,
            operation="step/12/critic/rollout",
            expected_rank_count=16,
        )

    bad_stage = _rank_control_validation()
    bad_stage["operations"][3]["per_stage_call_counts"][2] = 2  # type: ignore[index]
    with pytest.raises(ValueError, match=r"does not equal exit \+ 1"):
        expected_trace_rollout_call_inventory(
            bad_stage,
            operation="step/12/critic/rollout",
            expected_rank_count=16,
        )

    duplicate = _rank_control_validation()
    duplicate["operations"].append(dict(duplicate["operations"][0]))  # type: ignore[index]
    duplicate["operation_count"] = 7
    duplicate["global_entry_count"] = 16 * 7
    with pytest.raises(ValueError, match="matched 2 rows"):
        expected_trace_rollout_call_inventory(
            duplicate,
            operation="step/10/generator/rollout",
            expected_rank_count=16,
        )


def test_scalar_and_gradient_instrumentation_is_opt_in() -> None:
    p0 = (
        ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
    ).read_text()
    assert "collect_scalars: bool = False" in p0
    assert p0.count("if collect_scalars") >= 2
    fsdp_source = inspect.getsource(fsdp2_static.FSDP2EngineAdapter.step)
    async_source = inspect.getsource(wan13b_fair.GradReadyAsyncResidentEngine.step)
    for source in (fsdp_source, async_source):
        assert '_trajectory_capture_grad_norm' in source
        assert "last_gradient_norm" in source
    tree = ast.parse(p0)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for name in ("_run_nogan_generator_update_p0", "_run_nogan_critic_precomputed"):
        function = functions[name]
        defaults = dict(
            zip(
                [argument.arg for argument in function.args.kwonlyargs],
                function.args.kw_defaults,
                strict=True,
            )
        )
        assert isinstance(defaults["collect_scalars"], ast.Constant)
        assert defaults["collect_scalars"].value is False


def test_state_anchors_expose_canonical_numeric_coordinates() -> None:
    source = (
        ROOT / "scripts/distillschedule/probe_wan13b_trajectory_world16.py"
    ).read_text()
    assert "trajectory_state_sketch" in source
    assert "collect_numeric_state_sketch" in source
    assert 'backend="gloo"' in source
    assert "object_collective_group=evidence_group" in source
    assert "workload_nccl_algo=os.environ.get" in source
    launcher = (
        ROOT / "scripts/distillschedule/launch_wan13b_fair_world16_node.sh"
    ).read_text()
    assert "export NCCL_ALGO=Tree" in launcher
    assert "cycle_{cycle:02d}_global.json" in source
    assert "(tensor, flat_index)" in source
    assert canonical_state_name("module._orig_mod.blocks.0.lora_A") == (
        "blocks.0.lora_A"
    )
    first = deterministic_flat_indices("blocks.0.lora_A", 10_007, count=16)
    second = deterministic_flat_indices("blocks.0.lora_A", 10_007, count=16)
    assert first == second
    assert len(first) == len(set(first)) == 16
    assert first[0] == 0
    assert first[-1] == 10_006


def test_trajectory_summarizer_requires_all_16_rank_cycles() -> None:
    source = (
        ROOT / "scripts/distillschedule/summarize_wan13b_trajectory_world16.py"
    ).read_text()
    assert "for rank in range(WORLD_SIZE)" in source
    assert "WORLD_SIZE * args.cycles" in source
    assert '"generator": 2 + args.cycles' in source
    assert '"critic": 10 + 5 * args.cycles' in source
    assert '"ema": 10 + 5 * args.cycles' in source
    assert '"numeric_state_anchors": anchors' in source
    assert '"generator_ema"' in source
    assert 'sketch.get("coordinates")' in source
    assert '/ "execution"' in source
    assert 'final_path = args.run_root / "execution" / "ranks"' in source
    assert "payload_mapping_by_cycle[cycle] != payload_mapping_by_cycle[1]" in source


def test_trajectory_comparator_is_tied_to_numeric_sketch_schema() -> None:
    source = (
        ROOT / "scripts/distillschedule/compare_wan13b_trajectory_world16.py"
    ).read_text()
    ast.parse(source)
    assert "STATE_SKETCH_SCHEMA_VERSION" in source
    assert "compare_numeric_state_sketches" in source
    assert 'choices=(1, 2, 5, 50)' in source
    assert 'return (1, 10, 25, 50)' in source
    assert 'required_state_cycles=anchors' in source
