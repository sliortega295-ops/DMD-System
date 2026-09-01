from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_wrapper():
    path = ROOT / "scripts/distillschedule/livewan_ncu_rank_wrapper.py"
    spec = importlib.util.spec_from_file_location("livewan_ncu_rank_wrapper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(module):
    return module.parser().parse_args(
        [
            "--target-script", "/repo/entry.py",
            "--ncu-bin", "/usr/local/bin/ncu",
            "--output-base", "/runs/ncu/rank8_flash",
            "--manifest", "/runs/ncu/rank8_flash.manifest.json",
            "--target-node-rank", "1",
            "--target-local-rank", "0",
            "--kernel-regex", "flash_fwd_kernel",
            "--launch-skip-before-match", "640",
            "--launch-count", "1",
            "--metric-set", "full",
            "--", "--plan", "/runs/plan.json",
        ]
    )


def test_rank_wrapper_profiles_only_preregistered_worker() -> None:
    module = _load_wrapper()
    args = _args(module)
    executable, command, target = module.build_exec_command(
        args,
        {"DISTILLSCHEDULE_NODE_RANK": "1", "LOCAL_RANK": "0", "RANK": "8"},
    )
    assert target is True
    assert executable == "/usr/local/bin/ncu"
    assert command[:3] == ["/usr/local/bin/ncu", "--set", "full"]
    assert command[command.index("--kernel-name") + 1] == "regex:flash_fwd_kernel"
    assert command[command.index("--launch-skip-before-match") + 1] == "640"
    assert command[command.index("--launch-count") + 1] == "1"
    assert command[-3:] == ["/repo/entry.py", "--plan", "/runs/plan.json"]

    executable, command, target = module.build_exec_command(
        args,
        {"DISTILLSCHEDULE_NODE_RANK": "1", "LOCAL_RANK": "1", "RANK": "9"},
    )
    assert target is False
    assert executable == sys.executable
    assert command == [sys.executable, "/repo/entry.py", "--plan", "/runs/plan.json"]


def test_controller_and_launcher_keep_ncu_diagnostic_bounded() -> None:
    controller = (ROOT / "scripts/distillschedule/run_livewan_crossscale_world16.py").read_text()
    launcher = (
        ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh"
    ).read_text()
    assert 'value.add_argument("--ncu-bin", type=Path)' in controller
    assert 'args.run_kind != "correctness"' in controller
    assert 'args.ncu_launch_count != 1' in controller
    assert "livewan_ncu_rank_wrapper.py" in launcher
    assert '[[ "${RUN_KIND}" == correctness ]]' in launcher
    assert '[[ "${NCU_LAUNCH_COUNT}" == 1 ]]' in launcher
    assert "PASS_BOUNDED_CAPTURE" in launcher
