from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillschedule.core.config import load_plan
from distillschedule.core.contracts import ContractViolation


ROOT = Path(__file__).resolve().parents[1]


def test_composed_plan_loads_and_derives_profile_key() -> None:
    loaded = load_plan(ROOT / "tests/fixtures/dmd_full_cycle.json")
    plan = loaded.plan
    assert plan.workload.shape.profile_key == "b1-f81-lc16-lf9-384x640-l48x80-t512-p12x20_24x40_48x80"
    assert plan.workload.profile_key.startswith("g-fixture-14b-bf16-r256__s-fixture-14b-bf16-r256__")
    assert plan.evidence.start_step == 10
    assert plan.evidence.steps == 5
    assert len(loaded.source_files) == 3
    assert len(loaded.digest) == 64


def test_include_order_allows_final_plan_override(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    plan = tmp_path / "plan.json"
    base.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "id": "base",
                "workload": {
                    "name": "w",
                    "adapter": "generic",
                    "generator": {
                        "name": "g", "family": "f", "size": "1b", "role": "generator", "checkpoint": "/g"
                    },
                    "score": {
                        "name": "s", "family": "f", "size": "1b", "role": "score", "checkpoint": "/s"
                    },
                    "shape": {
                        "batch_size": 1,
                        "frames": 1,
                        "resolution": [8, 8],
                        "latent": {"channels": 4, "frames": 1, "height": 1, "width": 1},
                        "pyramid": [[1, 1]]
                    }
                },
                "hardware": {"nodes": 1, "local_world_size": 1, "world_size": 1, "gpu_memory_gib": 40},
                "rank_groups": [{"name": "all", "ranks": [0], "components": ["generator", "score"]}],
                "training_backends": {
                    "generator": {"kind": "native"},
                    "score": {"kind": "native"}
                },
                "evidence": {"run_id": "r", "output_dir": "out"}
            }
        )
    )
    plan.write_text(json.dumps({"include": ["base.json"], "id": "override"}))
    loaded = load_plan(plan)
    assert loaded.plan.plan_id == "override"


def test_unknown_fields_fail_closed(tmp_path: Path) -> None:
    source = ROOT / "configs/distillschedule/plans/generic_autoregressive_dmd_vllm_critic_only.json"
    raw = json.loads(source.read_text())
    raw["surprise"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ContractViolation, match="unknown fields: surprise"):
        load_plan(path)
