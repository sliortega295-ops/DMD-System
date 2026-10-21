"""Read-only A100 source -> private H100 overlay, streamed without local payload.

Only source/dependency code is transferred. No weights, checkpoint, cache,
GPU work, package installation, credentials, or shared-environment changes.
"""
import datetime
import hashlib
import json
import pathlib
import shlex
import subprocess
import time

RUN = pathlib.Path(__file__).resolve().parents[1]
SOURCE = "job-c6cfcde5-db4b-4527-8eb8-7518319a0277-master-0.shuchenweng.baai-vision_vision.dx-calc1.job@ssh.platform-multi.baai.ac.cn"
TARGET = "inference.jiangzhikun.baai-worldsound_prod.cn-neimongol-helingeer.ws@ssh.platform-cuihu.jingneng-inner.ac.cn"
SSH = ["ssh", "-T", "-4", "-S", "none", "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", "-p", "2222"]
A100 = "/share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/h3_diffsynth_dmd2_world16_20261010"
SITE = "/share/project/shuchenweng/yongyan_liu/DMD-System_assets/lightx2v_minimax_dmd_20260817/env/lib/python3.10/site-packages"
H100 = "/home/jiangzhikun/yongyan_liu/Helios-H100/assets/five_workload_h100_20260930/h3_ar_frozen_r1"


def transfer(label, source_root, relative_paths, destination):
    started = time.monotonic()
    # Never follows an arbitrary model directory; the admitted list is code only.
    assert all(".safetensors" not in p and "checkpoint" not in p for p in relative_paths)
    source_cmd = "set -eu; " + shlex.join(["tar", "--dereference", "--exclude=__pycache__", "--exclude=*.pyc", "-czf", "-", "-C", source_root, *relative_paths])
    dest_cmd = "set -eu; umask 022; test ! -e {p}; mkdir -p {p}; tar --extract --gzip --keep-old-files --no-same-owner -C {p}".format(p=shlex.quote(destination))
    log_dir = RUN / "logs"
    with (log_dir / f"w6_{label}_source_stderr_r1.log").open("xb") as slog, (log_dir / f"w6_{label}_target_r1.log").open("xb") as dlog:
        src = subprocess.Popen([*SSH, SOURCE, source_cmd], stdout=subprocess.PIPE, stderr=slog)
        dst = subprocess.Popen([*SSH, TARGET, dest_cmd], stdin=subprocess.PIPE, stdout=dlog, stderr=subprocess.STDOUT)
        digest = hashlib.sha256()
        size = 0
        try:
            while chunk := src.stdout.read(1 << 20):
                if time.monotonic() - started > 360:
                    raise TimeoutError("bounded source transfer deadline")
                digest.update(chunk)
                size += len(chunk)
                dst.stdin.write(chunk)
            dst.stdin.close()
            src.stdout.close()
            src_rc = src.wait(timeout=30)
            dst_rc = dst.wait(timeout=30)
            if src_rc or dst_rc:
                raise RuntimeError(f"streamed transfer failed {label}: source={src_rc}, target={dst_rc}")
        finally:
            # Only these local SSH children, never remote foreign GPU processes.
            for proc in (src, dst):
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=30)
    return {"label": label, "source_root": source_root, "relative_paths": relative_paths, "destination": destination, "archive_stream_bytes": size, "archive_stream_sha256": digest.hexdigest(), "elapsed_seconds": time.monotonic() - started, "source_exit": src_rc, "target_exit": dst_rc, "local_payload_staging": "NONE"}


def main():
    manifest = RUN / "manifests/w6_source_environment_stream_transfer_r1.json"
    assert not manifest.exists(), "preserve existing transfer evidence"
    rows = [transfer("exact_source", A100, [
        "assets/exact_reference_source_r2", "assets/python_overlay_r3",
        "zero3_200s_r2/assets/zero3_dependency_overlay_r1",
        "zero3_200s_r2/reports/zero3_import_node0_r1.json",
        "reports/source_config_supplement_r5.json"], H100)]
    rows.append(transfer("runtime_dependencies", SITE, [
        "transformers", "transformers-4.57.6.dist-info",
        "accelerate", "accelerate-1.14.0.dist-info",
        "tokenizers", "tokenizers-0.22.2.dist-info",
        "huggingface_hub", "huggingface_hub-0.34.4.dist-info",
        "safetensors", "safetensors-0.6.2.dist-info"], H100 + "/assets/runtime_overlay_r1"))
    receipt = {"run_id": "h100_w6_exact_environment_prepare_20261014_r1", "status": "PASS_TRANSFER_ONLY_NOT_IMPORT_OR_TRAINING", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "found_in_session": "01a0338b-2c56-70e1-a92f-67fb24f8453b", "source_receipt_expected_sha256": "9ca4ad8ca13b0a9e7397f6313e4eca225f10f52802e90423b06f7e786a58b989", "deepspeed_tree_expected_sha256": "b0cea87060d389e6e67c48c13d538dde77967be989ea40363d8f84b188e9d28a", "transfers": rows, "GPU": "NOT_RUN", "model_payload": "NOT_TRANSFERRED", "base_environment_changes": "NONE", "imports": "NOT_RUN"}
    with manifest.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    print(json.dumps({"manifest": str(manifest), "status": receipt["status"], "destination": H100, "compressed_bytes": sum(r["archive_stream_bytes"] for r in rows)}), flush=True)


if __name__ == "__main__":
    main()
