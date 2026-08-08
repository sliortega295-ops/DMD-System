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
