from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from distillschedule.core.config import load_plan

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/distillschedule/plans/helios_routea_gan_full5_world16_baseline.json"
SCRIPT = ROOT / "scripts/distillschedule/run_gan_offload_search.py"


def load_script():
    spec = importlib.util.spec_from_file_location("gan_offload_search", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SEARCH = load_script()


def set_plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    values = {
        "HELIOS_GAN_DATA_ROOT": tmp_path / "gan_data",
        "DISTILLSCHEDULE_GAN_TRACE_BUNDLE": tmp_path / "trace",
        "DISTILLSCHEDULE_GAN_CHECKPOINT": tmp_path / "checkpoint-1000",
        "DISTILLSCHEDULE_OUTPUT_DIR": tmp_path / "run",
        "DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE": tmp_path / "frozen",
        "DISTILLSCHEDULE_IMMUTABLE_MODEL_MMAP_CACHE": tmp_path / "immutable",
        "HELIOS_RUNTIME_ROOT": tmp_path / "helios",
        "DISTILLSCHEDULE_PYTHON": tmp_path / "bin/python",
        "DISTILLSCHEDULE_ACCELERATE": tmp_path / "bin/accelerate",
        "DISTILLSCHEDULE_ACCELERATE_CONFIG": tmp_path / "accelerate.yaml",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))


def test_tail_blocks_are_nested_and_k4_means_last_four() -> None:
    assert SEARCH.tail_blocks(40, 40) == list(range(40))
    assert SEARCH.tail_blocks(40, 4) == [36, 37, 38, 39]
    assert SEARCH.tail_blocks(40, 1) == [39]
    assert SEARCH.tail_blocks(40, 0) == []
    assert set(SEARCH.tail_blocks(40, 4)).issubset(SEARCH.tail_blocks(40, 8))


def test_candidate_environment_disables_streamer_only_at_k0() -> None:
    k4 = SEARCH.candidate_environment(40, 4)
    assert k4["DISTILLSCHEDULE_GAN_STREAMING_WINDOW"] == "1"
    assert k4["DISTILLSCHEDULE_GAN_STREAMING_COMPONENTS"] == "generator,score"
    assert k4["DISTILLSCHEDULE_GAN_STREAMED_BLOCK_COUNT"] == "4"
    assert k4["DISTILLSCHEDULE_GAN_STREAMED_BLOCK_INDICES"] == "36,37,38,39"
    zero = SEARCH.candidate_environment(40, 0)
    assert zero["DISTILLSCHEDULE_GAN_STREAMING_WINDOW"] == "0"
    assert zero["DISTILLSCHEDULE_GAN_STREAMED_BLOCK_COUNT"] == "0"
    assert zero["DISTILLSCHEDULE_GAN_STREAMED_BLOCK_INDICES"] == ""


def test_gan_plan_defaults_to_full40_but_accepts_controlled_k4_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_plan_env(monkeypatch, tmp_path)
    default = load_plan(PLAN).plan
    env = default.legacy.extra_env
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_WINDOW"] == "1"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS"] == "generator,score"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_BLOCK_COUNT"] == "40"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_INDICES"] == ",".join(
        str(index) for index in range(40)
    )
    assert env["DISTILLSCHEDULE_GAN_CAPACITY_SEARCH"] == "0"

    monkeypatch.setenv("DISTILLSCHEDULE_GAN_CAPACITY_SEARCH", "1")
    monkeypatch.setenv("DISTILLSCHEDULE_GAN_STREAMING_WINDOW", "1")
    monkeypatch.setenv("DISTILLSCHEDULE_GAN_STREAMING_COMPONENTS", "generator,score")
    monkeypatch.setenv("DISTILLSCHEDULE_GAN_STREAMED_BLOCK_COUNT", "4")
    monkeypatch.setenv("DISTILLSCHEDULE_GAN_STREAMED_BLOCK_INDICES", "36,37,38,39")
    candidate = load_plan(PLAN).plan
    candidate_env = candidate.legacy.extra_env
    assert candidate_env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_BLOCK_COUNT"] == "4"
    assert candidate_env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_INDICES"] == "36,37,38,39"
    assert candidate_env["DISTILLSCHEDULE_GAN_CAPACITY_SEARCH"] == "1"
    assert candidate.lifecycle.frozen_block_streaming.block_count == 40
    assert candidate.workload.branches.gan is True
    assert candidate.evidence.start_step == 1000
    assert candidate.evidence.steps == 5


def test_non_capacity_failure_does_not_form_a_search_bound() -> None:
    observations = {
        40: {"classification": "PASS"},
        20: {"classification": "ABORT"},
    }
    assert SEARCH.monotonic_violation(observations) is False


def test_monotonicity_violation_detects_larger_oom_after_smaller_pass() -> None:
    observations = {
        8: {"classification": "PASS"},
        16: {"classification": "CAPACITY_FAIL"},
        40: {"classification": "PASS"},
    }
    assert SEARCH.monotonic_violation(observations) is True


def test_actual_oom_detection_uses_cuda_or_cgroup_evidence(tmp_path: Path) -> None:
    run = tmp_path / "artifacts/runs/candidate/launcher/node_0"
    run.mkdir(parents=True)
    (run / "console.log").write_text("torch.OutOfMemoryError: CUDA out of memory\n")
    assert SEARCH.actual_oom_observed(tmp_path, "candidate") is True

    (run / "console.log").write_text("ordinary failure\n")
    (run / "launch_manifest.txt").write_text(
        "memory_events_before=oom 0;oom_kill 0;\n"
        "memory_events_after=oom 1;oom_kill 0;\n"
    )
    assert SEARCH.actual_oom_observed(tmp_path, "candidate") is True
