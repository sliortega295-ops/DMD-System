from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

# See test_refactor_systems_gan_bootstrap.py: these are tensor/safetensors
# integration checks, not dependency-free typed-runtime tests.
torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file

from tools.assets.build_systems_gan_precast import (
    INDEX,
    MARKER,
    PRECAST_MANIFEST,
    build_systems_gan_precast,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_systems_gan_precast_adds_exact_rounded_bootstrap_heads(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}\n")
    shard = base / "base.safetensors"
    save_file({"base.weight": torch.ones(2, dtype=torch.bfloat16)}, str(shard))
    index = {"metadata": {"total_size": 4}, "weight_map": {"base.weight": shard.name}}
    (base / INDEX).write_text(json.dumps(index))
    manifest = {
        "schema_version": "distillschedule.precast_safetensors.v1",
        "source": {"path": "/source"},
        "artifact": {
            "target_dtype": "bfloat16",
            "keep_in_fp32_modules": ["norm1"],
            "config_file": "config.json",
            "config_sha256": _sha(base / "config.json"),
            "index_file": INDEX,
            "index_sha256": _sha(base / INDEX),
            "tensor_count": 1,
            "tensor_bytes": 4,
            "dtype_tensor_counts": {"bfloat16": 1},
            "shards": [{"file": shard.name, "size_bytes": shard.stat().st_size, "sha256": _sha(shard), "tensor_count": 1}],
        },
    }
    (base / PRECAST_MANIFEST).write_text(json.dumps(manifest))

    checkpoint = tmp_path / "checkpoint-1000"
    (checkpoint / "gan_bootstrap").mkdir(parents=True)
    values = torch.tensor([1.125, -2.25, 3.5], dtype=torch.float32)
    raw = checkpoint / "gan_bootstrap" / "heads.f32"
    raw.write_bytes(values.numpy().tobytes())
    records = [
        {"name": "gan_heads.5.head3d.0.weight", "offset_elements": 0, "numel": 2, "shape": [2]},
        {"name": "gan_final_head.head3d.20.bias", "offset_elements": 2, "numel": 1, "shape": [1]},
    ]
    bootstrap = {
        "classification": "systems_only_gan_bootstrap",
        "artifact_hash": "a" * 64,
        "gan_head_manifest": {
            "path": "gan_bootstrap/heads.f32",
            "sha256": _sha(raw),
            "logical_numel": 3,
            "parameter_count": 2,
            "records": records,
        },
    }
    (checkpoint / "systems_only_gan_bootstrap.json").write_text(json.dumps(bootstrap))

    output = tmp_path / "gan-precast"
    result = build_systems_gan_precast(
        base_precast=base, bootstrap_checkpoint=checkpoint, output=output
    )
    assert result["status"] == "PASS"
    assert (output / MARKER).is_file()
    updated_index = json.loads((output / INDEX).read_text())
    assert updated_index["metadata"]["total_size"] == 10
    assert len(updated_index["weight_map"]) == 3
    merged = {}
    for filename in set(updated_index["weight_map"].values()):
        merged.update(load_file(str(output / filename)))
    assert torch.equal(merged["gan_heads.5.head3d.0.weight"], values[:2].to(torch.bfloat16))
    assert torch.equal(merged["gan_final_head.head3d.20.bias"], values[2:].to(torch.bfloat16))
