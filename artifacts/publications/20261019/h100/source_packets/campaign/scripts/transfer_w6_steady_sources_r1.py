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
    assert all(not p.endswith(".safetensors") for p in relative_paths)
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
                if time.monotonic() - started > 900:
                    raise TimeoutError("bounded source/cache transfer deadline")
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
    rows=[transfer("steady_source_helpers",A100,["zero3_200s_r2/steady_two_cycle_20261012_r1/scripts","zero3_200s_r2/pinned_host_registration_260s_r1/scripts","zero3_200s_r2/rms_norm_fusion_260s_r1/scripts","zero3_200s_r2/lora_cutlass_exact_round_260s_r1/scripts","zero3_200s_r2/selected_cache_h2d_prefetch_260s_r1/scripts","zero3_200s_r2/selected_reuse_block_fifo_260s_r1/scripts","zero3_200s_r2/selected_rollout_exact_reuse_260s_r1/scripts","zero3_200s_r2/continuation_h3_experience_20261011_r1/scripts","zero3_200s_r2/local_jit_cache_260s_r1/scripts","zero3_200s_r2/compatible_h3_260s_20261011_r1/scripts","zero3_200s_r2/packed_varlen_260s_r1/scripts","zero3_200s_r2/ngsafe_host_260s_r1/scripts","zero3_200s_r2/lora_epilogue_260s_r1/scripts"],H100+"/assets/steady_training_source_helpers_r1")]
    manifest=RUN/"manifests/w6_steady_source_helpers_stream_r1.json"
    with manifest.open("x") as f:
        json.dump({"status":"PASS_STREAM_TRANSFER_PENDING_PAYLOAD_GATE","model_weights":"ALREADY_ON_H100_NOT_TRANSFERRED","GPU_training":"NOT_RUN_BY_THIS_TRANSFER","local_payload_staging":"NONE","transfers":rows},f,indent=2)
    print(str(manifest),flush=True)

if __name__=="__main__":main()
