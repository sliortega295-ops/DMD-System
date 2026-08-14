from __future__ import annotations

import importlib.util
from pathlib import Path


def test_gan_paired_formal_script_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/distillschedule/run_gan_paired_formal.py"
    spec = importlib.util.spec_from_file_location("run_gan_paired_formal", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SCHEMA == "distillschedule.gan_paired_formal_timing.v2"
