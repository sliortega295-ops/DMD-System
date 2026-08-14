"""Fail-closed rendering and validation for Accelerate/DeepSpeed launch configs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from distillschedule.core.contracts import ContractViolation

DEEPSPEED_CONFIG_PLACEHOLDER = "__DISTILLSCHEDULE_DEEPSPEED_CONFIG__"
NUM_MACHINES_PLACEHOLDER = "__DISTILLSCHEDULE_NUM_MACHINES__"
NUM_PROCESSES_PLACEHOLDER = "__DISTILLSCHEDULE_NUM_PROCESSES__"
MACHINE_RANK_PLACEHOLDER = "__DISTILLSCHEDULE_MACHINE_RANK__"


@dataclass(frozen=True)
class DeepSpeedContract:
    path: Path
    sha256: str
    zero_stage: int
    optimizer_offload: str
    parameter_offload: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "zero_stage": self.zero_stage,
            "optimizer_offload": self.optimizer_offload,
            "parameter_offload": self.parameter_offload,
        }


@dataclass(frozen=True)
class AccelerateContract:
    path: Path
    sha256: str
    distributed_type: str
    num_machines: int
    num_processes: int
    machine_rank: int
    multinode_launcher: str
    deepspeed: DeepSpeedContract

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "distributed_type": self.distributed_type,
            "num_machines": self.num_machines,
            "num_processes": self.num_processes,
            "machine_rank": self.machine_rank,
            "multinode_launcher": self.multinode_launcher,
            "deepspeed": self.deepspeed.to_dict(),
        }


def sha256_file(path: str | Path) -> str:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise ContractViolation("file does not exist", path=str(candidate))
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractViolation("expected a mapping", path=path)
    return value


def _offload_device(zero: Mapping[str, Any], key: str) -> str:
    value = zero.get(key)
    if value is None:
        return "none"
    if not isinstance(value, Mapping):
        raise ContractViolation(
            f"{key} must be a mapping when present",
            path=f"deepspeed.zero_optimization.{key}",
        )
    return str(value.get("device", "none")).strip().lower()


def inspect_deepspeed_config(path: str | Path) -> DeepSpeedContract:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise ContractViolation("DeepSpeed config does not exist", path=str(candidate))
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractViolation(
            f"invalid JSON: {exc.msg}", path=str(candidate)
        ) from exc
    root = _mapping(payload, path=str(candidate))
    zero = _mapping(
        root.get("zero_optimization"),
        path=f"{candidate}.zero_optimization",
    )
    stage = zero.get("stage")
    if isinstance(stage, bool) or not isinstance(stage, int):
        raise ContractViolation(
            "ZeRO stage must be an integer",
            path=f"{candidate}.zero_optimization.stage",
        )
    return DeepSpeedContract(
        path=candidate,
        sha256=sha256_file(candidate),
        zero_stage=stage,
        optimizer_offload=_offload_device(zero, "offload_optimizer"),
        parameter_offload=_offload_device(zero, "offload_param"),
    )


def require_routea_zero1_contract(path: str | Path) -> DeepSpeedContract:
    contract = inspect_deepspeed_config(path)
    if contract.zero_stage != 1:
        raise ContractViolation(
            f"Route-A requires ZeRO-1, observed ZeRO-{contract.zero_stage}",
            path="launcher.deepspeed.zero_stage",
        )
    if contract.optimizer_offload != "cpu":
        raise ContractViolation(
            "Route-A requires CPU optimizer offload",
            path="launcher.deepspeed.optimizer_offload",
        )
    if contract.parameter_offload != "none":
        raise ContractViolation(
            "Route-A forbids parameter offload",
            path="launcher.deepspeed.parameter_offload",
        )
    return contract


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - launcher environment provides it
        raise ContractViolation(
            "Accelerate config validation requires OmegaConf",
            path=str(path),
        ) from exc
    try:
        value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:  # pragma: no cover - OmegaConf error types are not stable
        raise ContractViolation(
            f"invalid Accelerate YAML: {exc}", path=str(path)
        ) from exc
    return _mapping(value, path=str(path))


def inspect_accelerate_config(
    path: str | Path,
    *,
    expected_deepspeed_config: str | Path | None = None,
    expected_num_machines: int | None = None,
    expected_num_processes: int | None = None,
    expected_machine_rank: int | None = None,
) -> AccelerateContract:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise ContractViolation("Accelerate config does not exist", path=str(candidate))
    payload = _load_yaml_mapping(candidate)
    distributed_type = str(payload.get("distributed_type", "")).strip().upper()
    if distributed_type != "DEEPSPEED":
        raise ContractViolation(
            "distributed_type must be DEEPSPEED",
            path=f"{candidate}.distributed_type",
        )
    deepspeed = _mapping(
        payload.get("deepspeed_config"), path=f"{candidate}.deepspeed_config"
    )
    launcher = str(deepspeed.get("deepspeed_multinode_launcher", "")).strip().lower()
    if launcher != "standard":
        raise ContractViolation(
            "multi-node launcher must be explicit 'standard'",
            path=f"{candidate}.deepspeed_config.deepspeed_multinode_launcher",
        )
    raw_ds_path = deepspeed.get("deepspeed_config_file")
    if not isinstance(raw_ds_path, str) or not raw_ds_path.strip():
        raise ContractViolation(
            "deepspeed_config_file must be a non-empty path",
            path=f"{candidate}.deepspeed_config.deepspeed_config_file",
        )
    ds_path = Path(raw_ds_path).expanduser()
    if not ds_path.is_absolute():
        ds_path = (candidate.parent / ds_path).resolve()
    else:
        ds_path = ds_path.resolve()
    if expected_deepspeed_config is not None:
        expected = Path(expected_deepspeed_config).expanduser().resolve()
        if ds_path != expected:
            raise ContractViolation(
                f"Accelerate references {ds_path}, expected {expected}",
                path=f"{candidate}.deepspeed_config.deepspeed_config_file",
            )

    def require_int(key: str) -> int:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractViolation(
                f"{key} must be an integer", path=f"{candidate}.{key}"
            )
        return value

    num_machines = require_int("num_machines")
    num_processes = require_int("num_processes")
    machine_rank = require_int("machine_rank")
    if num_machines <= 0 or num_processes <= 0:
        raise ContractViolation(
            "num_machines and num_processes must be positive",
            path=str(candidate),
        )
    if not 0 <= machine_rank < num_machines:
        raise ContractViolation(
            "machine_rank is outside [0, num_machines)",
            path=f"{candidate}.machine_rank",
        )
    for observed, expected, key in (
        (num_machines, expected_num_machines, "num_machines"),
        (num_processes, expected_num_processes, "num_processes"),
        (machine_rank, expected_machine_rank, "machine_rank"),
    ):
        if expected is not None and observed != expected:
            raise ContractViolation(
                f"observed {observed}, expected {expected}",
                path=f"{candidate}.{key}",
            )
    if num_processes % num_machines != 0:
        raise ContractViolation(
            "num_processes must be divisible by num_machines",
            path=str(candidate),
        )
    return AccelerateContract(
        path=candidate,
        sha256=sha256_file(candidate),
        distributed_type=distributed_type,
        num_machines=num_machines,
        num_processes=num_processes,
        machine_rank=machine_rank,
        multinode_launcher=launcher,
        deepspeed=require_routea_zero1_contract(ds_path),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def render_accelerate_config(
    *,
    template_path: str | Path,
    deepspeed_config_path: str | Path,
    output_path: str | Path,
    num_machines: int,
    num_processes: int,
    machine_rank: int,
) -> AccelerateContract:
    template = Path(template_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    ds_path = Path(deepspeed_config_path).expanduser().resolve()
    if not template.is_file():
        raise ContractViolation("Accelerate template does not exist", path=str(template))
    if num_machines <= 0 or num_processes <= 0:
        raise ContractViolation(
            "num_machines and num_processes must be positive",
            path="launcher.accelerate",
        )
    if num_processes % num_machines != 0:
        raise ContractViolation(
            "num_processes must be divisible by num_machines",
            path="launcher.accelerate",
        )
    if not 0 <= machine_rank < num_machines:
        raise ContractViolation(
            "machine_rank is outside [0, num_machines)",
            path="launcher.accelerate.machine_rank",
        )
    require_routea_zero1_contract(ds_path)
    text = template.read_text(encoding="utf-8")
    replacements = {
        DEEPSPEED_CONFIG_PLACEHOLDER: json.dumps(str(ds_path)),
        NUM_MACHINES_PLACEHOLDER: str(num_machines),
        NUM_PROCESSES_PLACEHOLDER: str(num_processes),
        MACHINE_RANK_PLACEHOLDER: str(machine_rank),
    }
    for placeholder, replacement in replacements.items():
        count = text.count(placeholder)
        if count != 1:
            raise ContractViolation(
                f"template must contain {placeholder!r} exactly once; found {count}",
                path=str(template),
            )
        text = text.replace(placeholder, replacement)
    _atomic_write_text(output, text)
    return inspect_accelerate_config(
        output,
        expected_deepspeed_config=ds_path,
        expected_num_machines=num_machines,
        expected_num_processes=num_processes,
        expected_machine_rank=machine_rank,
    )
