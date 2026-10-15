"""CPU-only coverage for metadata RPC on stdin-hostile SSH gateways."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from distillschedule.search.execution import native_rpc
from distillschedule.workloads.best_known.registry import load_profile, validate_asset_config


def _assets(tmp_path, transport=None):
    node = {"host": "allocated-node", "repo_root": str(tmp_path),
            "python_env": str(Path(sys.executable).parent.parent)}
    if transport is not None:
        node["rpc_transport"] = transport
    return {"nodes": [node, {**node, "host": "peer-node"}]}


def test_default_rpc_keeps_stdin_transport(tmp_path):
    with patch("subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout='{"status":"CPU"}', stderr="")) as called:
        result = native_rpc(transport_assets=_assets(tmp_path), source_head="head",
                            output=tmp_path, action="resolve")
    assert result == {"status": "CPU"}
    assert json.loads(called.call_args.kwargs["input"])["action"] == "resolve"
    assert "-tt" not in called.call_args.args[0]


def test_tty_rpc_passes_payload_in_argv_and_frames_output(tmp_path):
    request = {"unicode": "独立样本", "quoted": "'\";exit", "values": [0, 1]}
    worker = tmp_path / "scripts/distillschedule/run_best_known_workload.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("import json,sys\nassert sys.argv[1:]==['_search']\n"
                      "print(json.dumps(json.load(sys.stdin)))\nraise SystemExit(0)\n")
    real_run = subprocess.run
    def fake_ssh(command, **kwargs):
        assert "-n" in command and "-tt" in command
        assert kwargs.get("input") is None
        local = real_run(["bash", "-c", command[-1]], capture_output=True, text=True)
        assert local.returncode == 0, local.stderr
        return SimpleNamespace(returncode=0, stdout="gateway banner\n" + local.stdout,
                               stderr="gateway closed\n")
    with patch("subprocess.run", side_effect=fake_ssh):
        result = native_rpc(transport_assets=_assets(tmp_path, "argv-json-tty-v1"),
                            source_head="head", output=tmp_path, action="resolve", data=request)
    assert result == {"action": "resolve", "source_head": "head", "data": request}


@pytest.mark.parametrize("output", ["no frame\n", "DMD_NATIVE_RPC_JSON=%%%\n",
    "DMD_NATIVE_RPC_JSON=e30=\nDMD_NATIVE_RPC_JSON=e30=\n"])
def test_tty_rpc_missing_invalid_or_duplicate_frames_fail_closed(tmp_path, output):
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=output, stderr="")):
        with pytest.raises(ValueError):
            native_rpc(transport_assets=_assets(tmp_path, "argv-json-tty-v1"),
                       source_head="head", output=tmp_path, action="resolve")


def test_unknown_rpc_transport_rejected_before_subprocess(tmp_path):
    with patch("subprocess.run") as called:
        with pytest.raises(ValueError, match="RPC transport"):
            native_rpc(transport_assets=_assets(tmp_path, "arbitrary-shell"),
                       source_head="head", output=tmp_path, action="resolve")
    called.assert_not_called()


@pytest.mark.parametrize("mode", ["stdin", "argv-json-tty-v1", "arbitrary-shell"])
def test_asset_schema_admits_only_explicit_registered_rpc_modes(mode):
    profile = load_profile("minimax_h3_grid1000_best", root=ROOT)
    value = {"schema_version": 1, "nodes": [
        {"host": f"node{rank}", "repo_root": "/source", "python_env": "/env",
         "rpc_transport": mode} for rank in range(2)],
        "master_addr": "node0", "master_port": 29000, "run_root": "/project/runs",
        "node_local_cache_root": "/tmp/planner-cache", "assets": {
            name: {"path": f"/assets/{name}", "sha256": "0" * 64,
                   **({"identity_file": "identity.json"} if spec["kind"] == "directory" else {})}
            for name, spec in profile["required_assets"].items()}}
    if mode == "arbitrary-shell":
        with pytest.raises(ValueError, match="RPC transport"):
            validate_asset_config(value, profile)
    else:
        assert validate_asset_config(value, profile) == value


@pytest.mark.parametrize('transport', ['stdin', 'argv-json-tty-v1'])
def test_metadata_ssh_does_not_inherit_python_loader_but_remote_python_keeps_it(tmp_path, monkeypatch, transport):
    import base64
    monkeypatch.setenv('LD_LIBRARY_PATH', '/isolated-python/lib')
    monkeypatch.setenv('LD_PRELOAD', '/isolated-python/lib/override.so')
    output = '{"status":"CPU"}'
    if transport == 'argv-json-tty-v1':
        output = 'DMD_NATIVE_RPC_JSON=' + base64.b64encode(output.encode()).decode() + '\n'
    with patch('subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=output, stderr='')) as called:
        result = native_rpc(transport_assets=_assets(tmp_path, transport), source_head='head',
                            output=tmp_path, action='resolve')
    assert result == {'status':'CPU'}
    assert 'LD_LIBRARY_PATH' not in called.call_args.kwargs['env']
    assert 'LD_PRELOAD' not in called.call_args.kwargs['env']
    assert 'LD_LIBRARY_PATH=' in called.call_args.args[0][-1]
