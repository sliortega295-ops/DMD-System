from __future__ import annotations

from pathlib import Path

from distillschedule.core.config import load_plan
from distillschedule.runtime import DistillExecutor


ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_writes_resolved_and_compiled_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DISTILLSCHEDULE_OUTPUT_DIR", str(tmp_path / "run"))
    loaded = load_plan(ROOT / "configs/distillschedule/plans/helios_14b_384x640_full5.json")
    result = DistillExecutor().run(loaded, repo_root=ROOT, dry_run=True)
    assert result.dry_run
    assert result.return_code is None
    assert (result.output_dir / "resolved_plan.json").is_file()
    assert (result.output_dir / "compiled_run.json").is_file()
    assert (result.output_dir / "resolved_helios_config.yaml").is_file()
    assert result.command[0].endswith("accelerate")
    assert "launch" in result.command
    assert any(token.endswith("train_helios.py") for token in result.command)



def test_world16_dry_run_writes_node_qualified_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ID", "typed-world16-test")
    monkeypatch.setenv("DISTILLSCHEDULE_OUTPUT_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE", str(tmp_path / "frozen"))
    monkeypatch.setenv("DISTILLSCHEDULE_IMMUTABLE_MODEL_MMAP_CACHE", str(tmp_path / "immutable"))
    monkeypatch.setenv("DISTILLSCHEDULE_NODE_RANK", "1")
    monkeypatch.setenv("DISTILLSCHEDULE_MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("DISTILLSCHEDULE_MASTER_PORT", "29677")
    monkeypatch.setenv("HELIOS_RUNTIME_ROOT", str(ROOT))
    monkeypatch.setenv("DISTILLSCHEDULE_PYTHON", "/runtime/python")
    monkeypatch.setenv("DISTILLSCHEDULE_ACCELERATE", "/runtime/accelerate")
    plan = ROOT / "configs/distillschedule/plans/helios_routea_full5_world16_typed.json"
    loaded = load_plan(plan)
    result = DistillExecutor().run(loaded, repo_root=ROOT, dry_run=True)
    assert (result.output_dir / "resolved_plan.node1.json").is_file()
    assert (result.output_dir / "compiled_run.node1.json").is_file()
    assert (result.output_dir / "resolved_helios_config.node1.yaml").is_file()
    assert not (result.output_dir / "resolved_plan.json").exists()
