from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "distillschedule"
    / "run_gan_pipeline_offload_search.py"
)
SPEC = importlib.util.spec_from_file_location("gan_pipeline_offload_search", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tail_blocks_are_nested() -> None:
    assert MODULE.tail_blocks(0) == []
    assert MODULE.tail_blocks(1) == [39]
    assert MODULE.tail_blocks(4) == [36, 37, 38, 39]
    assert MODULE.tail_blocks(40) == list(range(40))


def test_candidate_command_is_capacity_only_and_preserves_pipeline_family(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        repo_root=Path("/dmd"),
        workload_contract=Path("/dmd/contract.json"),
        timeout_seconds=1234,
    )
    command = MODULE._candidate_command(
        args,
        run_id="pipeline-search-k20",
        streamed_count=20,
        result_path=tmp_path / "result.json",
        port=32201,
    )
    assert command[command.index("--family") + 1] == "pipeline"
    assert "--capacity-search" in command
    assert command[command.index("--streamed-block-count") + 1] == "20"
    assert "--profile-nsys" not in command


def test_cgroup_oom_delta_is_an_actual_capacity_signal() -> None:
    before = MODULE._memory_events("oom 2\noom_kill 1\n")
    after = MODULE._memory_events("oom 3\noom_kill 1\n")
    assert after["oom"] > before["oom"]
