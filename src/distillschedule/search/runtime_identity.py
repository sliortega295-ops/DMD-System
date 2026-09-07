"""Observed two-node calibration identity, never an echo of expected digests.

Collection is CPU/driver metadata only, outside training timers. It is enabled
only for model-based native validation, not canonical or registered timing.
"""
from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from .contracts import require

SCHEMA = "distillschedule.model_runtime_identity.v1"


def collect_node_identity(*, source_head: str, node_rank: int) -> dict[str, Any]:
    require(type(node_rank) is int and node_rank in (0, 1), "invalid identity node")
    import torch
    require(not torch.cuda.is_initialized(), "identity probe must not own a CUDA context")
    packages = {}
    for name in ("torch", "triton", "diffusers", "transformers", "peft", "accelerate",
                 "flash-attn", "safetensors", "nvidia-nccl-cu12"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # The runtime values, executable and imported extension are observed, not
    # inferred from a venv directory's name. This is a version/selected-binary
    # identity, not a claim to hash every byte in a multi-GB environment.
    environment = {"python": platform.python_version(), "executable": str(Path(sys.executable).resolve()),
        "executable_sha256": file_sha256(Path(sys.executable)), "packages": packages,
        "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "nccl": list(torch.cuda.nccl.version()),
        "torch_extension_sha256": file_sha256(Path(torch._C.__file__)),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "scope": "OBSERVED_VERSIONS_AND_SELECTED_BINARIES_NOT_FULL_ENVIRONMENT_HASH"}
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id",
               "--format=csv,noheader,nounits"]
    raw = subprocess.check_output(command, text=True, timeout=15)
    gpus = [{"device": int(i), "uuid": uuid.strip(), "name": name.strip(),
             "total_bytes": int(total) * 2**20, "driver": driver.strip(), "pci_bus_id": bus.strip()}
            for i, uuid, name, total, driver, bus in csv.reader(io.StringIO(raw))]
    require(len(gpus) == 8 and {g["device"] for g in gpus} == set(range(8)), "eight GPU identities required")
    require(not torch.cuda.is_initialized(), "identity probe initialized CUDA")
    return {"schema_version": SCHEMA, "status": "PASS", "source_head": source_head,
        "node_rank": node_rank, "environment": environment,
        "hardware": {"gpus": sorted(gpus, key=lambda g: g["device"]),
                     "topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, timeout=15)},
        "cuda_context_created": False}


def wan_probe_environment(node: dict[str, Any]) -> dict[str, str]:
    from distillschedule.workloads.best_known.dispatch import clean_environment
    env = clean_environment(node)
    # Mirror the source-bound Wan launcher's import/library setup. Inherited
    # SSH PYTHONPATH must not make us inspect a different Torch than training.
    root, pyenv = env["DMD_SYSTEM_ROOT"], env["DISTILLSCHEDULE_PYENV"]
    require(root == node["repo_root"], "runtime root differs from node")
    env["PATH"] = pyenv + "/bin:" + env["PATH"]
    env["PYTHONPATH"] = ":".join((root + "/src", root, env["HELIOS_RUNTIME_ROOT"]))
    env["LD_LIBRARY_PATH"] = pyenv + "/lib" + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.update(OMP_NUM_THREADS="8", NCCL_ASYNC_ERROR_HANDLING="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def collect_planned_node_identity(receipt: dict[str, Any], *, node_rank: int) -> dict[str, Any]:
    from .execution import validated_profile
    from .wan_implementation import WAN
    from distillschedule.workloads.best_known.registry import git_head, repository_root
    require(type(node_rank) is int and node_rank in (0, 1), "invalid identity node")
    root = repository_root()
    validated_profile(receipt, root=root)
    require(receipt["profile_id"] == WAN, "runtime identity adapter not implemented for this workload")
    require(receipt["integration_head"] == git_head(root), "runtime probe source changed")
    node = receipt["nodes"][node_rank]
    require(Path(node["repo_root"]).resolve() == root.resolve(), "runtime probe on wrong source checkout")
    require(Path(node["python"]).resolve() == Path(sys.executable).resolve(), "runtime probe on wrong Python")
    code = ("import json; from distillschedule.search.runtime_identity import collect_node_identity; "
            f"print(json.dumps(collect_node_identity(source_head={git_head(root)!r}, node_rank={node_rank})))")
    result = subprocess.check_output([node["python"], "-c", code], cwd=root,
        env=wan_probe_environment(node), text=True, timeout=90)
    return json.loads(result)


def observed_binding(nodes: list[dict[str, Any]], *, source_head: str) -> dict[str, str]:
    require(len(nodes) == 2 and {n.get("node_rank") for n in nodes} == {0, 1},
            "both observed runtime node identities required")
    ordered = sorted(nodes, key=lambda n: n["node_rank"])
    uuids = []
    for node in ordered:
        require(node.get("schema_version") == SCHEMA and node.get("status") == "PASS"
                and node.get("source_head") == source_head and node.get("cuda_context_created") is False,
                "observed runtime identity failed/source changed")
        env, hardware = node.get("environment", {}), node.get("hardware", {})
        require(all(env.get(k) for k in ("python", "executable", "executable_sha256", "packages",
                "torch", "torch_cuda", "nccl", "torch_extension_sha256")), "observed environment incomplete")
        gpus = hardware.get("gpus", [])
        require(len(gpus) == 8 and {g.get("device") for g in gpus} == set(range(8))
                and bool(hardware.get("topology")), "observed hardware incomplete")
        require(all(g.get("uuid") and g.get("name") and g.get("driver")
                    and type(g.get("total_bytes")) is int and g["total_bytes"] > 0 for g in gpus),
                "invalid GPU identity")
        uuids.extend(g["uuid"] for g in gpus)
    require(len(set(uuids)) == 16, "duplicate GPU ownership across nodes")
    return {"environment_fingerprint": fingerprint([n["environment"] for n in ordered]),
            "hardware_fingerprint": fingerprint([n["hardware"] for n in ordered])}


def verify_observed_binding(nodes: list[dict[str, Any]], binding: Mapping[str, str]) -> None:
    actual = observed_binding(nodes, source_head=binding["source_head"])
    for key, value in actual.items():
        require(binding.get(key) == value, f"observed runtime {key} differs from calibration")


def verify_measurement_runtime(value: Mapping[str, Any], binding: Mapping[str, str]) -> None:
    evidence = value.get("runtime_identity_evidence", {})
    require(set(evidence) == {"before", "after"}, "missing observed runtime before/after evidence")
    for nodes in evidence.values():
        verify_observed_binding(nodes, binding)
