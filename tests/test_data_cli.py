from __future__ import annotations

import json
import subprocess
import sys

from distillschedule.data.cli import main
from test_training_h3_grid import cached_data as cached_data


def test_inspect_and_imports_do_not_load_torch(tmp_path):
    import hashlib
    import torch
    from distillschedule.data import import_preencoded

    feature = tmp_path / "tensor.pt"
    torch.save({"latent": torch.ones(1, 2, 3)}, feature)
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "schema_version": "distillschedule.data.preencoded.v1",
        "records": [{"id": "one", "features": {"path": feature.name,
                    "sha256": hashlib.sha256(feature.read_bytes()).hexdigest()}}],
    }))
    manifest = import_preencoded(source, tmp_path / "prepared")
    script = f"from distillschedule.data.cli import main; import sys; assert 'torch' not in sys.modules; assert main(['inspect', {str(manifest)!r}]) == 0; assert 'torch' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["validation"] == "HASHES_ONLY_TENSORS_NOT_LOADED"


def test_invalid_data_cli_returns_error_not_traceback(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "missing.json")]) == 2
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err
    assert "Traceback" not in captured.err


def test_native_preparation_commands_are_discoverable_without_model_imports():
    script = """
import sys
from distillschedule.cli import main
try:
    main(['data', '--help'])
except SystemExit as exc:
    assert exc.code == 0
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'lightx2v_train', 'peft'))
"""
    completed = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert "prepare-helios-stream" in completed.stdout
    assert "prepare-h3-grid" in completed.stdout
    assert "prepare-livewan-inputs" in completed.stdout


def test_original_grid_inventory_cli_does_not_decode_or_change_native_data(tmp_path, cached_data, capsys):
    import torch
    # The ordinary helper fixture uses tiny Tensor files. This command hashes
    # supplied data; semantic native tensor admission is still worker-owned.
    config, native, _ = cached_data
    output = tmp_path / "inventory.json"
    assert main(["prepare-h3-grid", "--native-config", str(config), "--output", str(output)]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row == json.loads(output.read_text())
    assert row["schema_version"] == "original_h3_grid_latent_dataset.v1"
    assert len(row["rows"]) == 288 and row["cycles"] == native["training"]["max_train_iters"]
    assert torch.cuda.is_initialized() is False


def test_native_preparation_missing_input_is_a_normal_cli_error(tmp_path, capsys):
    assert main(["prepare-h3-grid", "--native-config", str(tmp_path / "missing.json"),
                 "--output", str(tmp_path / "output.json")]) == 2
    assert "ERROR:" in capsys.readouterr().err
    assert not (tmp_path / "output.json").exists()
