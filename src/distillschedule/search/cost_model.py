"""Source/precision-bound operation envelopes, never Nsight union-as-latency."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from typing import Any, Mapping

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from .contracts import require
from .plan_ir import nonnegative, validate_binding


class MissingCalibration(ValueError):
    pass


@dataclass(frozen=True)
class ComponentContract:
    """Reusable native operation identity, independent of a complete recipe.

    ``context`` must bind entry residency, initialized optimizer/cache state and
    concurrent resource usage. Serial calibration cannot predict contention by
    silently discounting NCCL. Source/hardware/assets remain bound by CostModel.
    Work includes actual shape/call counts; implementation contains only the
    choices consumed by this component. The adapter owns that projection.
    """
    family: str
    work: Mapping[str, Any]
    precision: Mapping[str, Any]
    placement: Mapping[str, Any]
    implementation: Mapping[str, Any]
    context: Mapping[str, Any]
    schema_version: str = "distillschedule.component_contract.v1"

    def __post_init__(self) -> None:
        require(self.schema_version == "distillschedule.component_contract.v1", "component schema mismatch")
        require(isinstance(self.family, str) and bool(self.family.strip()), "component family missing")
        for name in ("work", "precision", "placement", "implementation", "context"):
            value = getattr(self, name)
            require(isinstance(value, Mapping) and bool(value), f"component {name} contract missing")
            # Reject NaN, tensors and non-JSON values before hashing/calibration.
            object.__setattr__(self, name, json.loads(json.dumps(dict(value), allow_nan=False)))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "family": self.family,
                **{k: dict(getattr(self, k)) for k in
                   ("work", "precision", "placement", "implementation", "context")}}

    @property
    def digest(self) -> str:
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ComponentContract:
        require(set(value) == {"schema_version", "family", "work", "precision",
                               "placement", "implementation", "context"},
                "unknown/missing component field; recipe/occurrence IDs are not component costs")
        return cls(**value)


@dataclass(frozen=True)
class OperationCost:
    key: str
    samples_ns: tuple[int, ...]
    workspace_bytes_per_rank: int
    # Retained outputs are described by Lifetime, not counted again here.
    evidence: Mapping[str, Any]

    @property
    def duration_ns(self) -> int:
        return int(statistics.median(self.samples_ns))


@dataclass(frozen=True)
class CostModel:
    binding: Mapping[str, str]
    records: tuple[OperationCost, ...]
    scope: str = "MEASURED"
    schema_version: str = "distillschedule.cost_model.v1"

    def __post_init__(self) -> None:
        require(self.schema_version == "distillschedule.cost_model.v1", "cost schema mismatch")
        validate_binding(self.binding)
        require(self.scope in {"MEASURED", "CPU_FIXTURE_ONLY"}, "unknown cost evidence scope")
        require(len({r.key for r in self.records}) == len(self.records), "duplicate cost key")
        for record in self.records:
            require(record.key and record.samples_ns, "cost key/samples missing")
            for ns in record.samples_ns:
                nonnegative(ns, "cost sample")
            nonnegative(record.workspace_bytes_per_rank, "workspace")
            if self.scope == "MEASURED":
                evidence = record.evidence
                require(evidence.get("status") == "PASS" and evidence.get("profiled") is False,
                        "cost calibration must be passing and unprofiled")
                require(evidence.get("kind") in {"native_operation", "isolated_microbenchmark"},
                        "kernel union/heuristic is not a calibrated operation duration")
                require(evidence.get("operation_contract_sha256") and evidence.get("source_receipt"),
                        "calibration missing operation/source contract")
                sha = evidence.get("source_receipt_sha256", "")
                require(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "calibration receipt hash missing")

    def bind(self, binding: Mapping[str, str], *, problem_scope: str) -> None:
        require(dict(self.binding) == dict(binding), "cost source/environment/assets/hardware/precision mismatch")
        require(self.scope != "CPU_FIXTURE_ONLY" or problem_scope == "CPU_FIXTURE_ONLY",
                "fixture costs cannot predict or execute a real workload")

    def get(self, key: str) -> OperationCost:
        for record in self.records:
            if record.key == key:
                return record
        raise MissingCalibration(f"UNCALIBRATED: {key}")

    def get_component(self, contract: ComponentContract) -> OperationCost:
        record = self.get(contract.digest)
        require(record.evidence.get("component_contract") == contract.to_dict(),
                "component calibration identity missing or changed")
        require(self.scope == "CPU_FIXTURE_ONLY" or
                record.evidence.get("operation_contract_sha256") == contract.digest,
                "component operation contract hash mismatch")
        return record

    def verify_receipts(self) -> None:
        """Run beside the receipts (normally remotely), before using real costs.

        A caller-authored JSON claiming MEASURED is not enough. Reconcile its
        samples and identities with the content-hashed calibration receipt.
        """
        if self.scope == "CPU_FIXTURE_ONLY":
            return
        native_component_cache = {}
        for record in self.records:
            path = Path(record.evidence["source_receipt"])
            require(path.is_file(), f"calibration receipt missing: {path}")
            require(file_sha256(path) == record.evidence["source_receipt_sha256"], "calibration receipt hash mismatch")
            raw = json.loads(path.read_text())
            require(raw.get("schema_version") == "distillschedule.operation_calibration.v1", "unknown calibration receipt schema")
            require(raw.get("binding") == dict(self.binding), "calibration receipt identity mismatch")
            require(raw.get("key") == record.key and raw.get("samples_ns") == list(record.samples_ns), "calibration samples changed")
            require(raw.get("workspace_bytes_per_rank") == record.workspace_bytes_per_rank, "calibration memory changed")
            for key in ("status", "profiled", "kind", "operation_contract_sha256"):
                require(raw.get(key) == record.evidence[key], f"calibration evidence changed: {key}")
            if "component_contract" in record.evidence:
                contract = ComponentContract.from_dict(record.evidence["component_contract"])
                self.get_component(contract)
                require(raw.get("component_contract") == contract.to_dict(),
                        "component source receipt contract changed")
            if "component_sources" in raw:
                from .component_calibration import validate_pooled_receipt
                validate_pooled_receipt(raw)
            if record.evidence.get("component_contract", {}).get("family", "").startswith("h3."):
                from .h3_action_model_adapter import validate_component_receipt
                validate_component_receipt(raw, native_component_cache)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "binding": dict(self.binding), "scope": self.scope,
                "records": [{"key": r.key, "samples_ns": list(r.samples_ns),
                             "workspace_bytes_per_rank": r.workspace_bytes_per_rank,
                             "evidence": dict(r.evidence)} for r in self.records]}

    @property
    def digest(self) -> str:
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CostModel":
        require(set(value) == {"schema_version", "binding", "scope", "records"}, "unknown/missing cost model field")
        rows = tuple(OperationCost(**{**r, "samples_ns": tuple(r["samples_ns"])}) for r in value["records"])
        return cls(value["binding"], rows, value["scope"], value["schema_version"])
