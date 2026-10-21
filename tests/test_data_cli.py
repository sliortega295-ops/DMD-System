from __future__ import annotations

import json
import subprocess
import sys

from distillschedule.data.cli import main


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
