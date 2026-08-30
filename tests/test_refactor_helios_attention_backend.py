from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINNED_EXTENSION_SHA256 = (
    "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
)


def _policy(backend: str) -> dict[str, object]:
    return {
        "mesh_policy": "node-hsdp",
        "rollout_batch_size": 1,
        "pack_real_score_pair": False,
        "rollout_backend": "native-fsdp2-batch",
        "attention_backend": backend,
        "modulation_chunk_tokens": 512,
        "selective_flash_checkpoint_stride": 0,
        "generator_nograd_cross_kv_cache": False,
        "generator_nograd_static_context_cache": False,
        "generator_residual_storage": "cuda_out_of_place",
        "critic_residual_storage": "cuda_out_of_place",
        "generator_nograd_lora_fused_epilogue": False,
        "all_role_lora_scale1_add": True,
        "optimizer_dispatch_warmup": True,
        "generator_nograd_residual_fusion": True,
        "exact_grad_residual_fusion": True,
        "exact_modulated_norm_epilogue": True,
        "exact_modulated_norm_materialization": True,
        "exact_modulated_norm_backward": False,
        "elide_explicit_rollout_sync": False,
    }


def _attention_receipt(backend: str) -> dict[str, object]:
    if backend == "sdpa":
        return {
            "status": "PASS",
            "requested": backend,
            "runtime": backend,
            "local_kernels": "",
            "flash_attn_func_available": False,
            "flash_attn_func_module": None,
            "extension_sha256": None,
        }
    return {
        "status": "PASS",
        "requested": backend,
        "runtime": backend,
        "local_kernels": "kernels-community/flash-attn2=/pinned/kernel",
        "flash_attn_func_available": True,
        "flash_attn_func_module": "flash_attn2.flash_attn_interface",
        "extension_sha256": PINNED_EXTENSION_SHA256,
    }


def _run_validate_summary(tmp_path: Path, backend: str) -> dict[str, object]:
    run_root = tmp_path / backend
    rank_root = run_root / "execution" / "ranks"
    rank_root.mkdir(parents=True)
    for rank in range(16):
        receipt = {
            "rank": rank,
            "status": "PASS",
            "variant": "no-gan",
            "p0_policy": _policy(backend),
            "attention_backend": _attention_receipt(backend),
        }
        (rank_root / f"rank_{rank:02d}.json").write_text(
            json.dumps(receipt) + "\n"
        )
    output = run_root / "summary.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py"),
        "--run-root",
        str(run_root),
        "--variant",
        "no-gan",
        "--mode",
        "validate",
        "--mesh-policy",
        "node-hsdp",
        "--rollout-batch-size",
        "1",
        "--pack-real-score-pair",
        "off",
        "--attention-backend",
        backend,
        "--modulation-chunk-tokens",
        "512",
        "--generator-residual-storage",
        "cuda_out_of_place",
        "--critic-residual-storage",
        "cuda_out_of_place",
        "--generator-nograd-lora-fused-epilogue",
        "off",
        "--all-role-lora-scale1-add",
        "on",
        "--optimizer-dispatch-warmup",
        "on",
        "--generator-nograd-residual-fusion",
        "on",
        "--exact-grad-residual-fusion",
        "on",
        "--exact-modulated-norm-epilogue",
        "on",
        "--exact-modulated-norm-materialization",
        "on",
        "--exact-modulated-norm-backward",
        "off",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(output.read_text())


def test_launcher_and_controller_pin_attention_backend() -> None:
    launcher = (
        ROOT / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh"
    ).read_text()
    controller = (
        ROOT / "scripts/distillschedule/run_static_fsdp2_p0_world16.py"
    ).read_text()
    worker = (
        ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
    ).read_text()
    for source in (launcher, controller, worker):
        assert "attention-backend" in source or "ATTENTION_BACKEND" in source
    assert "unset LOCAL_KERNELS" in launcher
    assert "kernels-community/flash-attn2=" in launcher
    assert PINNED_EXTENSION_SHA256 in launcher
    assert PINNED_EXTENSION_SHA256 in worker
    assert "flash_attn_extension_hash_mismatch" in launcher


def test_validate_summary_accepts_sdpa_receipts(tmp_path: Path) -> None:
    summary = _run_validate_summary(tmp_path, "sdpa")
    assert summary["status"] == "PASS"
    assert summary["policy"]["attention_backend"] == "sdpa"
    assert len(summary["rank_attention_backends"]) == 16


def test_validate_summary_accepts_pinned_local_fa2_receipts(tmp_path: Path) -> None:
    summary = _run_validate_summary(tmp_path, "local-fa2")
    assert summary["status"] == "PASS"
    assert summary["policy"]["attention_backend"] == "local-fa2"
    assert {
        row["extension_sha256"] for row in summary["rank_attention_backends"]
    } == {PINNED_EXTENSION_SHA256}


def test_validate_summary_rejects_backend_identity_drift(tmp_path: Path) -> None:
    run_root = tmp_path / "drift"
    rank_root = run_root / "execution" / "ranks"
    rank_root.mkdir(parents=True)
    for rank in range(16):
        receipt = {
            "rank": rank,
            "status": "PASS",
            "variant": "no-gan",
            "p0_policy": _policy("local-fa2"),
            "attention_backend": _attention_receipt("local-fa2"),
        }
        if rank == 7:
            receipt["attention_backend"]["extension_sha256"] = "0" * 64
        (rank_root / f"rank_{rank:02d}.json").write_text(
            json.dumps(receipt) + "\n"
        )
    output = run_root / "summary.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py"),
        "--run-root",
        str(run_root),
        "--variant",
        "no-gan",
        "--mode",
        "validate",
        "--mesh-policy",
        "node-hsdp",
        "--rollout-batch-size",
        "1",
        "--pack-real-score-pair",
        "off",
        "--attention-backend",
        "local-fa2",
        "--modulation-chunk-tokens",
        "512",
        "--generator-residual-storage",
        "cuda_out_of_place",
        "--critic-residual-storage",
        "cuda_out_of_place",
        "--generator-nograd-lora-fused-epilogue",
        "off",
        "--all-role-lora-scale1-add",
        "on",
        "--optimizer-dispatch-warmup",
        "on",
        "--generator-nograd-residual-fusion",
        "on",
        "--exact-grad-residual-fusion",
        "on",
        "--exact-modulated-norm-epilogue",
        "on",
        "--exact-modulated-norm-materialization",
        "on",
        "--exact-modulated-norm-backward",
        "off",
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 1
    summary = json.loads(output.read_text())
    assert summary["status"] == "FAIL"
    assert any("rank7 attention backend" in error for error in summary["errors"])
