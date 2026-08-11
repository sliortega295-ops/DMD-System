from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillschedule.core.contracts import ContractViolation
from distillschedule.runtime.gan_baseline import validate_systems_gan_bootstrap
from scripts.distillschedule.build_systems_gan_bootstrap import gan_head_specs
from scripts.distillschedule.build_systems_gan_trace import _crop, _seed


def test_official_gan_head_inventory_is_frozen() -> None:
    specs = gan_head_specs()
    assert len(specs) == 130
    assert sum(item.numel for item in specs) == 988_359_941
    assert specs[0].name == "gan_heads.5.head3d.0.weight"
    assert specs[-1].name == "gan_final_head.head3d.20.bias"
    assert specs[-3].shape == (768,)


def test_gan_trace_rng_is_operation_keyed() -> None:
    a = _crop(seed=_seed(7, 3, 1000, "gan_g_crop"))
    b = _crop(seed=_seed(7, 3, 1000, "gan_g_crop"))
    c = _crop(seed=_seed(7, 4, 1000, "gan_g_crop"))
    assert a.equal(b)
    assert not a.equal(c) or _seed(7, 3, 1000, "gan_g_crop") != _seed(7, 4, 1000, "gan_g_crop")


def _write_bootstrap(root: Path, *, artifact_hash: str = "a" * 64) -> None:
    root.mkdir()
    paths = ["gan_bootstrap/heads.f32", "critic/model.pt", "model_ema/manifest.json"]
    paths += [f"critic/optim_{rank}.pt" for rank in range(16)]
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    marker = {
        "schema_version": "distillschedule.systems_only_gan_bootstrap.v1",
        "status": "PASS",
        "classification": "systems_only_gan_bootstrap",
        "logical_control_window": [1000, 1001, 1002, 1003, 1004],
        "logical_schedule_versions": {"generator": 200, "critic": 1000, "ema": 800},
        "official_checkpoint_1000": False,
        "source_faithful_continuation": False,
        "generator_optimizer_materialized_from_source": True,
        "critic_lora_optimizer_materialized_from_source": True,
        "gan_head_optimizer_materialized": True,
        "artifact_hash": artifact_hash,
        "source_checkpoint": "/source/checkpoint-10",
        "source_parameter_versions": {"generator": 2, "critic": 10, "ema": 10},
        "gan_head_manifest": {"path": paths[0], "logical_numel": 1},
        "critic_model": {"path": paths[1]},
        "ema": {"path": paths[2]},
        "critic_optimizer": {
            "zero_stage": 1,
            "world_size": 16,
            "files": [{"rank": rank, "path": f"critic/optim_{rank}.pt"} for rank in range(16)],
        },
    }
    (root / "systems_only_gan_bootstrap.json").write_text(json.dumps(marker))


def test_systems_bootstrap_is_explicitly_not_official_checkpoint1000(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoint-1000"
    _write_bootstrap(root)
    monkeypatch.setenv("DISTILLSCHEDULE_GAN_BOOTSTRAP_ARTIFACT_HASH", "a" * 64)
    result = validate_systems_gan_bootstrap(root)
    assert result["classification"] == "systems_only_gan_bootstrap"
    assert result["official_checkpoint_1000"] is False
    assert result["optimizer_materialized"] is True


def test_systems_bootstrap_rejects_pinned_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoint-1000"
    _write_bootstrap(root)
    monkeypatch.setenv("DISTILLSCHEDULE_GAN_BOOTSTRAP_ARTIFACT_HASH", "b" * 64)
    with pytest.raises(ContractViolation, match="differs from pinned"):
        validate_systems_gan_bootstrap(root)
