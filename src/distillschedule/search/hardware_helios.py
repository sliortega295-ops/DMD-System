"""Bind the existing H100 Helios native launcher; never invent a C2 executor.

The pinned 8fc70e8 program executes ONE Full5, not two continuous cycles.
This module compiles its real CLI and re-audits its actual native receipts.
It deliberately has no GPU-launch, SSH, cost-estimation, or auto-resume API.
The stopped campaign stays stopped. A controller may use the compiled command
only after separate authorization; this bridge cannot close a C2 claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Mapping

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from .allocation import ExecutionAllocation
from .contracts import require

NATIVE_HEAD = "8fc70e8ce9f6e0243604812120119c9f4c078e55"
LAUNCHER = "scripts/distillschedule/launch_helios_hardware_p0_node.sh"
SUMMARIZER = "scripts/distillschedule/summarize_helios_hardware_p0.py"
WORKER = "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py"
ARMS = ("best", "h100-nograd-retain", "h100-nograd-retain-nosync")
MODES = ("validate", "correctness", "timing")
RETENTION_SHA = "52f9b287011e681861731055b4b37ea7d65ab4cb39eb8ad0792025a7b0c23966"
FLASH_SHA = "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
NATIVE_FILES = {
    LAUNCHER: "4d14a1ff685161dac6909e96df1b709ab3ca33b29fbccc013114e33acc430211",
    SUMMARIZER: "ef8629fcaebf8714da9fa571f348fa9b18bf3b7162e865edfa91a5666acd61b6",
    WORKER: "65f7f85ccd12b6b6a779af2d7acae5e19a3641e760f79cf61a022afed2adc0c7",
    "scripts/distillschedule/probe_static_fsdp2_p0_world16.py": "e9c794287c8b2ad3dca8c9a51e666898d498e671ca950c9f615f6eb24ef7412c",
    "src/distillschedule/runtime/hardware_profile.py": "9d71603c3171ddc6a0277435d3ed5e4290240508755d876fc62b702f4371235d",
    "src/distillschedule/runtime/helios_attention_import.py": "5d70bc88a35c4f2bce82c1bf49d204cddd5deace89751c865bcf1e461aa35a41",
    "src/distillschedule/runtime/fsdp2_p0.py": "e706090caddb0e7e337cc7f9e57485785832563cba6de049f0f2d9201649831c",
    "src/distillschedule/runtime/fsdp2_static.py": "caed2c8f0d5109372e8631d0aeff7671d7a96076d4346f71b16d83055ac857bd",
    "configs/distillschedule/best_known/helios14b_nogan_best.json": "00716145197fa0ff9b0fce2557b732ef88d66864a1068d463bbf2d07f2c5783c",
    "configs/distillschedule/experiments/helios_h100_nograd_retain_v1.json": "e2d7aa8243c3d00afce3def178b43b00b281c6846e12bcdc43777630b57bde15",
    "configs/distillschedule/experiments/helios_h100_nograd_retain_nosync_v1.json": "ee8206d157ab9c3ffd8ea527cb1c1c4a34b4f94964121f8bdf3bae8f266b9f50",
}
TIMER = {"kind": "NATIVE_SINGLE_FULL5", "continuous_cycles": 1,
         "claimable_cycle2": False, "updates": {"generator": 1, "critic": 5, "ema": 5}}


def _absolute(value, label):
    require(isinstance(value, str) and Path(value).is_absolute()
            and ".." not in Path(value).parts and "\x00" not in value, f"{label}: absolute path required")
    return Path(value)


def _ref(value, label):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, f"{label}: exact path/hash required")
    _absolute(value["path"], label)
    require(isinstance(value["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
            f"{label}: SHA256 required")
    return dict(value)


def inspect_native_source(root: Path) -> dict[str, Any]:
    """Read local Git/source only. No Torch import, GPU query, or remote access."""
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    require(git("rev-parse", "HEAD") == NATIVE_HEAD, "unregistered H100 Helios native HEAD")
    require(git("rev-parse", "--show-toplevel") == str(root), "native source is not the checkout root")
    require(not git("status", "--porcelain", "--untracked-files=all"), "native checkout is dirty")
    for relative, expected in NATIVE_FILES.items():
        require(file_sha256(root / relative) == expected, f"pinned native source changed: {relative}")
    return {"head": NATIVE_HEAD, "files": dict(NATIVE_FILES), "scope": "LOCAL_SOURCE_ONLY_NO_GPU"}


def _native_control(root: Path, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Use the real stdlib native lowerer/auditor, isolated from planner imports."""
    require(action in ("inventory", "audit"), "unsupported CPU native operation")
    inspect_native_source(root)
    code = '''
import json, sys
from pathlib import Path
root, action = Path(sys.argv[1]), sys.argv[2]
sys.path[:0] = [str(root/'scripts/distillschedule'), str(root/'src'), str(root)]
import summarize_helios_hardware_p0 as native
data = json.load(sys.stdin)
if action == 'inventory':
    result = {}
    for arm in data['arms']:
        frozen = native.freeze_policy(root, variant='no-gan', arm=arm)
        profile = native.load_profile('helios14b_nogan_best', root=root)
        result[arm] = dict(frozen=frozen, policy_argv=native.policy_arguments(frozen['policy']),
                          workload_contract=profile['contract'])
else:
    result = native.summarize(Path(data['run_root']), expected_head=data['head'])
if 'torch' in sys.modules:
    raise RuntimeError('CPU control plane unexpectedly imported Torch')
print(json.dumps(result, allow_nan=False))
'''
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ("DISTILLSCHEDULE_", "NCCL_", "HELIOS_", "PYTHON"))}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1")
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(root.resolve()), action],
                            input=json.dumps(dict(payload)), text=True, capture_output=True,
                            cwd=root, env=env, timeout=120, check=True)
    inspect_native_source(root)
    return json.loads(result.stdout)


def validate_deployment(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"schema_version", "native_source", "site", "allocation", "asset_gate",
              "retention_api", "runtime_versions", "master_port", "deadline_seconds"}
    require(isinstance(value, Mapping) and set(value) == fields, "unknown/missing H100 Helios deployment field")
    d = json.loads(json.dumps(dict(value), allow_nan=False))
    require(d["schema_version"] == "distillschedule.hardware_helios_deployment.v1", "deployment schema mismatch")
    require(set(d["native_source"]) == {"head", "repo_root"}
            and d["native_source"]["head"] == NATIVE_HEAD, "native source must be registered 8fc70e8")
    _absolute(d["native_source"]["repo_root"], "native source")
    require(set(d["site"]) == {"project_root", "python", "helios_root", "asset_root", "hf_home"},
            "unknown/missing site fields; no free-form environment overrides")
    for key, path in d["site"].items():
        _absolute(path, key)
    require(Path(d["site"]["hf_home"]).is_relative_to(d["site"]["asset_root"]), "HF cache must be under asset root")
    allocation = ExecutionAllocation.from_dict(d["allocation"])
    require(allocation.world_size == 8 and allocation.node_count == 1
            and allocation.logical_global_batch == 16 and all(x.architecture == "sm90" for x in allocation.devices),
            "requires eight H100 physical ranks and sixteen logical B1 slots")
    require(set(allocation.role_slots) == {"generator", "critic"}
            and all(rows == tuple((r, r + 8) for r in range(8)) for rows in allocation.role_slots.values()),
            "native Helios requires exact generator/critic rank,r+8 slots")
    _ref(d["asset_gate"], "asset gate")
    _ref(d["retention_api"], "retention API")
    require(d["retention_api"]["sha256"] == RETENTION_SHA, "retention API proof is not the registered gate")
    versions = d["runtime_versions"]
    require(isinstance(versions, dict) and set(versions) == {"python", "packages", "driver"}, "runtime versions missing")
    require(all(isinstance(versions[k], str) and versions[k] for k in ("python", "driver")), "invalid runtime version")
    require(isinstance(versions["packages"], dict) and set(versions["packages"]) == {
        "torch", "triton", "deepspeed", "accelerate", "transformers", "diffusers", "peft", "safetensors"}
        and all(isinstance(v, str) and v for v in versions["packages"].values()), "exact native package inventory required")
    require(versions["packages"]["torch"].startswith("2.10."), "retention API is qualified for Torch 2.10")
    require(type(d["master_port"]) is int and 1024 <= d["master_port"] <= 65535, "invalid rendezvous port")
    require(type(d["deadline_seconds"]) is int and 0 < d["deadline_seconds"] <= 7200, "invalid bounded deadline")
    return d


def compile_native_plan(deployment, *, arm: str, mode: str, run_id: str,
                        source_root: Path | None = None, correctness=None) -> dict[str, Any]:
    d = validate_deployment(deployment)
    require(arm in ARMS and mode in MODES, "unregistered H100 Helios arm/mode")
    require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id), "unsafe run id")
    if mode == "timing":
        correctness = _ref(correctness, "fresh arm-specific correctness summary")
    else:
        require(correctness is None, "non-timing plan cannot inherit timing admission")
    root = Path(source_root or d["native_source"]["repo_root"])
    native = _native_control(root, "inventory", {"arms": [arm]})[arm]
    site = d["site"]
    env = dict(DISTILLSCHEDULE_PYTHON=site["python"], HELIOS_RUNTIME_ROOT=site["helios_root"],
               DISTILLSCHEDULE_ASSET_ROOT=site["asset_root"], HF_HOME=site["hf_home"],
               DISTILLSCHEDULE_MASTER_PORT=str(d["master_port"]),
               DISTILLSCHEDULE_RUN_DEADLINE_SECONDS=str(d["deadline_seconds"]))
    if arm != "best":
        env["DISTILLSCHEDULE_HELIOS_H100_RETENTION_API_RECEIPT"] = d["retention_api"]["path"]
    if correctness is not None:
        env.update(DISTILLSCHEDULE_HELIOS_H100_CORRECTNESS_SUMMARY=correctness["path"],
                   DISTILLSCHEDULE_HELIOS_H100_CORRECTNESS_SHA256=correctness["sha256"])
    argv = ["bash", str(Path(d["native_source"]["repo_root"]) / LAUNCHER), site["project_root"],
            run_id, "no-gan", mode, arm, NATIVE_HEAD, d["asset_gate"]["path"], d["asset_gate"]["sha256"]]
    value = {"schema_version": "distillschedule.hardware_helios_native_plan.v1", "deployment": d,
             "deployment_sha256": fingerprint(d), "arm": arm, "mode": mode, "run_id": run_id,
             "native_source": {"head": NATIVE_HEAD, "files": dict(NATIVE_FILES)},
             "native_policy": native, "argv": argv, "environment": env, "correctness": correctness,
             "run_root": str(Path(site["project_root"]) / "research_artifacts/five_workload_h100_20260930" / run_id),
             "timer": dict(TIMER), "execution_status": "NOT_RUN", "c2_status": "NOT_IMPLEMENTED_BY_PINNED_NATIVE",
             "numerical_status": "REVIEW_NOT_FULL_STATE_PARITY", "claimable_speedup": False,
             "authorization": "NO_GPU_AUTHORIZATION_CAMPAIGN_STOP_MUST_BE_PRESERVED"}
    return {**value, "plan_sha256": fingerprint(value)}


def validate_native_plan(plan, *, source_root=None):
    expected = compile_native_plan(plan["deployment"], arm=plan["arm"], mode=plan["mode"],
        run_id=plan["run_id"], source_root=source_root, correctness=plan["correctness"])
    require(plan == expected, "saved native command/policy/deployment was altered")
    return expected


def audit_mechanisms(rows, *, arm):
    """Additional exact controls absent from the older Full5 summarizer."""
    require(arm in ARMS and len(rows) == 8 and sorted(r.get("rank", -1) for r in rows) == list(range(8)),
            "all eight actual rank receipts required")
    retained, nosync = arm != "best", arm == "h100-nograd-retain-nosync"
    for row in rows:
        rank = row["rank"]
        attention = row.get("attention_backend", {})
        require(attention.get("status") == "PASS" and attention.get("requested") == attention.get("runtime") == "local-fa2"
                and attention.get("extension_sha256") == FLASH_SHA, "actual local-FA2 binding differs; FA3 is not admitted")
        scopes = row.get("generator_nograd_persistent_unshard", {})
        require(scopes.get("logical_b1_microbatch_count") == 2 and scopes.get("rollout_batch_count") == 10,
                "missing actual independent B1 rollout scopes")
        children = scopes.get("microbatches", [])
        require([c.get("logical_slot") for c in children] == [rank, rank + 8], "retention logical slots changed")
        for child in children:
            p = child.get("persistence", {})
            require(p.get("enabled") is retained and p.get("rollout_batch_count") == 5, "retention mechanism not executed")
            if retained:
                required = {"fsdp_module_count": 41, "set_no_reshard_count": 41,
                    "restore_reshard_policy_count": 41, "explicit_reshard_count": 41,
                    "fully_unsharded_before_exit_count": 41, "unsharded_after_exit_count": 0,
                    "restored": True, "flush_allocator_cache": True, "pre_empty_cache_count": 1, "post_empty_cache_count": 1}
                require(all(type(p.get(k)) is type(v) and p[k] == v for k, v in required.items()),
                        "retained FSDP units/allocator fences were not restored")
        fence = row.get("rollout_fence_policy", {})
        required = {"elide_explicit_rollout_sync": nosync, "generator_update_empty_cache_count": 0 if nosync else 2,
                    "generator_update_sync_count": 0 if nosync else 2, "critic_rollout_sync_count": 0 if nosync else 10,
                    "outer_full5_boundary_sync_preserved": True}
        require(set(fence) == set(required) and
                all(type(fence.get(k)) is type(v) and fence[k] == v for k, v in required.items()),
                "actual interior/outer fence execution differs from selected arm")
    return {"status": "PASS_ACTUAL_LOCAL_FA2_RETENTION_AND_FENCE_RECEIPTS", "physical_ranks": 8,
            "logical_slots": 16, "retained_units_per_scope": 41 if retained else 0}


def audit_completed_run(plan, *, source_root=None):
    """Run on the receipt host; raw paths are never translated or fabricated."""
    p = validate_native_plan(plan, source_root=source_root)
    d, run = p["deployment"], Path(p["run_root"])
    root = Path(source_root or d["native_source"]["repo_root"])
    observed = _native_control(root, "audit", {"run_root": str(run), "head": NATIVE_HEAD})
    require(observed.get("status") == "PASS" and observed.get("errors") == [], "native raw-evidence audit failed")
    saved = json.loads((run / "summary.json").read_text())
    require({k: v for k, v in saved.items() if k != "created_unix_ns"} ==
            {k: v for k, v in observed.items() if k != "created_unix_ns"}, "saved native summary differs from raw receipts")
    launch = json.loads((run / "manifests/launch.json").read_text())
    require(launch["frozen"] == p["native_policy"]["frozen"] and launch["asset_gate"] == d["asset_gate"]
            and (launch["run_id"], launch["arm"], launch["mode"]) == (p["run_id"], p["arm"], p["mode"]),
            "actual native launch differs from selected frozen plan")
    if p["mode"] == "timing":
        require({k: launch.get("correctness_gate", {}).get(k) for k in ("path", "sha256")} == p["correctness"],
                "actual timing did not use the selected correctness gate")
    if p["arm"] != "best":
        require({k: launch.get("retention_capability", {}).get(k) for k in ("path", "sha256")} == d["retention_api"],
                "actual native retention capability differs")
    command = shlex.split((run / "manifests/command.sh").read_text())
    prefix = [d["site"]["python"], "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=8", "--node-rank=0",
              "--master-addr=127.0.0.1", f"--master-port={d['master_port']}"]
    require(command == prefix + launch["worker_argv"] and launch["worker_argv"][0] ==
            str(Path(d["native_source"]["repo_root"]) / WORKER), "actual torchrun argv differs from selected native program")
    versions = json.loads((run / "manifests/environment_versions.json").read_text())
    require(versions.get("executable") == d["site"]["python"] and isinstance(versions.get("python"), str)
            and versions["python"].split()[:1] == [d["runtime_versions"]["python"]]
            and versions.get("packages") == d["runtime_versions"]["packages"], "actual runtime versions changed")
    inventory = observed["launcher_health"]["gpu_inventory"]
    allocation = ExecutionAllocation.from_dict(d["allocation"])
    for gpu, device in zip(inventory, allocation.devices, strict=True):
        require((gpu["index"], gpu["uuid"], gpu["memory_mib"] * 2**20, gpu["driver"]) ==
                (device.rank, device.uuid, device.capacity_bytes, d["runtime_versions"]["driver"]), "actual GPU allocation changed")
    mechanisms = ({"status": "NOT_RUN_VALIDATE_ONLY"} if p["mode"] == "validate" else
                  audit_mechanisms(observed["physical_receipts"], arm=p["arm"]))
    return {"status": "PASS_NATIVE_FULL5_EVIDENCE_ONLY", "plan_sha256": p["plan_sha256"],
            "summary": {"path": str(run / "summary.json"), "sha256": file_sha256(run / "summary.json")},
            "arm": p["arm"], "mode": p["mode"], "mechanisms": mechanisms, "timer": dict(TIMER),
            "full5_seconds_max_rank": observed["full5_seconds_max_rank"], "c2_status": "NOT_IMPLEMENTED_BY_PINNED_NATIVE",
            "numerical_status": "REVIEW_NOT_FULL_STATE_PARITY", "claimable_speedup": False}
