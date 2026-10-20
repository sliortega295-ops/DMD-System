"""Exact private W6 source/dependency and CPU import gate; no GPU context/model."""
import datetime
import hashlib
import importlib
import io
import json
import os
import pathlib
import socket
import sys
import time
import traceback
import unittest

os.environ.update(CUDA_VISIBLE_DEVICES="", USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false", DIFFSYNTH_ATTENTION_IMPLEMENTATION="torch", PYTHONDONTWRITEBYTECODE="1")
ROOT = pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100")
ASSET = ROOT / "assets/five_workload_h100_20260930/h3_ar_frozen_r1"
SOURCE = ASSET / "assets/exact_reference_source_r2"
RUN = ROOT / "research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1"
OUT = RUN / "artifacts/h3_w6_preparation"
RECEIPT = OUT / "private_environment_cpu_gate_r1.json"
started = time.monotonic()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(path):
    h = hashlib.sha256()
    rows = []
    for f in sorted(path.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
            name = str(f.relative_to(path))
            data = f.read_bytes()
            h.update(name.encode() + b"\0")
            h.update(data)
            rows.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return h.hexdigest(), rows


def main():
    assert not RECEIPT.exists(), "preserve prior evidence"
    OUT.mkdir(exist_ok=True)
    result = {"run_id": "h100_w6_exact_environment_prepare_20261014_r1", "hostname": socket.gethostname(), "python": sys.version, "executable": sys.executable, "source_root": str(SOURCE), "source_identity": "KS3 exact object snapshot; NOT a verified Git revision", "GPU": "HIDDEN_NOT_RUN", "model": "NOT_LOADED", "training": "NOT_RUN", "timing": "NOT_RUN", "shared_environment_changes": "NONE", "modules": {}, "status": "CHECKING"}
    try:
        receipt_hash = sha(SOURCE / "source_receipt.json")
        assert receipt_hash == "9ca4ad8ca13b0a9e7397f6313e4eca225f10f52802e90423b06f7e786a58b989"
        rows = json.loads((SOURCE / "source_receipt.json").read_text())["objects"]
        for item in rows:
            assert sha(SOURCE / item["relative_path"]) == item["sha256"], item["relative_path"]
        original = pathlib.Path("/share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/h3_diffsynth_dmd2_world16_20261010/assets/exact_reference_source_r2")
        supplement = json.loads((ASSET / "reports/source_config_supplement_r5.json").read_text())
        for item in supplement["rows"]:
            relative = pathlib.Path(item["path"]).relative_to(original)
            assert sha(SOURCE / relative) == item["sha256"], str(relative)
        ds_root = ASSET / "zero3_200s_r2/assets/zero3_dependency_overlay_r1/deepspeed"
        ds_hash, ds_rows = tree(ds_root)
        assert ds_hash == "b0cea87060d389e6e67c48c13d538dde77967be989ea40363d8f84b188e9d28a"
        assert len(ds_rows) == 928
        with (OUT / "deepspeed_tree_rows_r1.json").open("x") as f:
            json.dump(ds_rows, f, indent=2)
            f.write("\n")
        runtime_hash, runtime_rows = tree(ASSET / "assets/runtime_overlay_r1")
        with (OUT / "runtime_dependency_rows_r1.json").open("x") as f:
            json.dump(runtime_rows, f, indent=2)
            f.write("\n")
        result.update(source_receipt_sha256=receipt_hash, source_verified_files=len(rows), supplement_verified_files=len(supplement["rows"]), deepspeed_tree_sha256=ds_hash, deepspeed_verified_files=len(ds_rows), runtime_dependency_tree_sha256=runtime_hash, runtime_dependency_files=len(runtime_rows))
        paths = [ASSET / "assets/runtime_overlay_r1", ASSET / "zero3_200s_r2/assets/zero3_dependency_overlay_r1", ASSET / "assets/python_overlay_r3", SOURCE / "examples/minimax_h3/model_training", SOURCE]
        sys.path[:0] = list(map(str, paths))
        result["PYTHONPATH"] = ":".join(map(str, paths))
        for name in ("torch", "accelerate", "deepspeed", "modelscope", "transformers", "tokenizers", "huggingface_hub", "safetensors", "peft", "diffsynth", "diffsynth.models.minimax_h3_dit", "diffsynth.pipelines.minimax_h3_audio_video", "h3_dmd2", "train_dmd2", "dmd2_zero3", "test_train_dmd2"):
            module = importlib.import_module(name)
            file = getattr(module, "__file__", None)
            result["modules"][name] = {"version": getattr(module, "__version__", None), "file": file, "source_sha256": sha(pathlib.Path(file)) if file else None}
        import torch
        from torch.distributed.fsdp import fully_shard
        assert torch.__version__ == "2.10.0+cu128"
        assert torch.version.cuda == "12.8"
        assert not torch.cuda.is_initialized(), "CPU gate must not create GPU contexts"
        expected = {"accelerate": "1.14.0", "deepspeed": "0.18.4", "modelscope": "1.31.0", "transformers": "4.57.6", "tokenizers": "0.22.2", "huggingface_hub": "0.34.4", "safetensors": "0.6.2", "peft": "0.18.1"}
        for name, version in expected.items():
            assert result["modules"][name]["version"] == version, (name, version)
        stream = io.StringIO()
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules["test_train_dmd2"]))
        result["tests"] = {"run": tests.testsRun, "failures": len(tests.failures), "errors": len(tests.errors), "skipped": len(tests.skipped), "log": stream.getvalue()}
        assert tests.testsRun == 13 and tests.wasSuccessful(), "exact native CPU gate"
        assert not torch.cuda.is_initialized()
        result.update(status="PASS_EXACT_SOURCE_PRIVATE_OVERLAY_CPU_TESTS", torch_cuda=torch.version.cuda, CUDA_initialized=False, fully_shard_import="PASS", attention_backend_CPU_test_only="torch", GPU_attention="NOT_RUN", model_weight_identity_vs_A100="NOT_VERIFIED_IN_THIS_GATE")
    except BaseException as exc:
        result.update(status="FAIL_CPU_ENVIRONMENT_GATE", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    result.update(elapsed_seconds=time.monotonic() - started, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with RECEIPT.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"status": result["status"], "receipt": str(RECEIPT), "source_files": result.get("source_verified_files"), "deepspeed_files": result.get("deepspeed_verified_files"), "tests": {k: v for k, v in result.get("tests", {}).items() if k != "log"}, "error": result.get("error"), "GPU": "NOT_RUN"}), flush=True)
    return 0 if result["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
