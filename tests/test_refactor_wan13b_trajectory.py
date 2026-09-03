from __future__ import annotations

import ast
import inspect
from pathlib import Path

from distillschedule.evidence.trajectory_state_sketch import (
    canonical_state_name,
    deterministic_flat_indices,
)
from distillschedule.runtime import fsdp2_static, wan13b_fair


ROOT = Path(__file__).resolve().parents[1]


def test_trajectory_is_an_explicit_nonlegacy_mode() -> None:
    controller = (ROOT / "scripts/distillschedule/run_wan13b_fair_world16.py").read_text()
    launcher = (ROOT / "scripts/distillschedule/launch_wan13b_fair_world16_node.sh").read_text()
    assert '"trajectory"' in controller
    assert '"${MODE}" == trajectory' in launcher
    assert "probe_wan13b_trajectory_world16.py" in launcher
    assert "DISTILLSCHEDULE_TRAJECTORY_CYCLES" in launcher
    assert '"--trajectory-cycles", type=int, choices=(1, 5, 50)' in controller
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
    assert "WORLD16_FSDP_BASELINE" in source
    assert "if not finite:" in source
    assert "trajectory cycles must be one of 1, 5, or 50" in source


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
    assert "for rank in range(16)" in source
    assert "16 * args.cycles" in source
    assert '"generator": 2 + args.cycles' in source
    assert '"critic": 10 + 5 * args.cycles' in source
    assert '"ema": 10 + 5 * args.cycles' in source
    assert '"numeric_state_anchors": anchors' in source
    assert '"generator_ema"' in source
    assert 'sketch.get("coordinates")' in source
    assert '/ "execution"' in source
    assert 'final_path = args.run_root / "execution" / "ranks"' in source


def test_trajectory_comparator_is_tied_to_numeric_sketch_schema() -> None:
    source = (
        ROOT / "scripts/distillschedule/compare_wan13b_trajectory_world16.py"
    ).read_text()
    ast.parse(source)
    assert "STATE_SKETCH_SCHEMA_VERSION" in source
    assert "compare_numeric_state_sketches" in source
    assert 'choices=(1, 5, 50)' in source
    assert 'return (1, 10, 25, 50)' in source
    assert 'required_state_cycles=anchors' in source
