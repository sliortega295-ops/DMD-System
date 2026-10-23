"""Pool reusable native components with a flat, content-addressed evidence chain.

This is metadata-only. It does not run workers, relabel old-source costs, or
turn synchronized calibration envelopes into performance measurements.
"""
from pathlib import Path

from distillschedule.workloads.best_known.registry import _object, file_sha256, write_new_json
from .contracts import require
from .cost_model import ComponentContract, CostModel, OperationCost


def _evidence(path, raw):
    return {**{k: raw[k] for k in ("status", "profiled", "kind",
              "operation_contract_sha256", "component_contract")},
            "source_receipt": str(path.resolve()), "source_receipt_sha256": file_sha256(path)}


def _flat_source(path, sha, binding):
    require(file_sha256(path) == sha, "component source hash changed")
    raw = _object(path)
    require("component_sources" not in raw, "pool requires flat native source receipts")
    contract = ComponentContract.from_dict(raw["component_contract"])
    require(raw["key"] == contract.digest and raw["kind"] == "native_operation",
            "not a native component source")
    row = OperationCost(raw["key"], tuple(raw["samples_ns"]),
                        raw["workspace_bytes_per_rank"], _evidence(path, raw))
    CostModel(binding, (row,)).verify_receipts()
    return row


def validate_pooled_receipt(raw):
    """Recheck every original receipt, not merely the aggregate JSON's hash."""
    require(raw.get("aggregation") == "POOLED_NATIVE_COMPONENT_SAMPLES"
            and raw.get("calibration_only") is True
            and raw.get("formal_timing_eligible") is False, "invalid component pool")
    sources = raw["component_sources"]
    require(sources and len({s["sha256"] for s in sources}) == len(sources),
            "missing/duplicate component sources")
    rows = [_flat_source(Path(s["path"]), s["sha256"], raw["binding"]) for s in sources]
    require(all(r.key == raw["key"] and r.evidence["component_contract"] == raw["component_contract"]
                for r in rows), "pooled component contracts differ")
    require(raw["samples_ns"] == [ns for r in rows for ns in r.samples_ns], "pooled component samples differ")
    require(raw["workspace_bytes_per_rank"] == max(r.workspace_bytes_per_rank for r in rows),
            "pooled component memory differs")


def pool_component_models(models, *, output: Path) -> CostModel:
    models = tuple(models)
    require(models and all(m.scope == "MEASURED" for m in models), "measured component models required")
    binding = dict(models[0].binding)
    grouped = {}
    for model in models:
        require(dict(model.binding) == binding, "component model binding differs")
        model.verify_receipts()
        for row in model.records:
            model.get_component(ComponentContract.from_dict(row.evidence["component_contract"]))
            path = Path(row.evidence["source_receipt"])
            sha = row.evidence["source_receipt_sha256"]
            _flat_source(path, sha, binding)
            grouped.setdefault(row.key, {}).setdefault(sha, (path, row))
    # Validate all inputs first. Never overwrite a previous collection.
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for key, sources in sorted(grouped.items()):
        first = next(iter(sources.values()))[1]
        raw = dict(schema_version="distillschedule.operation_calibration.v1", binding=binding,
            key=key, status="PASS", profiled=False, kind="native_operation",
            operation_contract_sha256=key, component_contract=first.evidence["component_contract"],
            samples_ns=[ns for _,r in sources.values() for ns in r.samples_ns],
            workspace_bytes_per_rank=max(r.workspace_bytes_per_rank for _,r in sources.values()),
            calibration_only=True, formal_timing_eligible=False,
            aggregation="POOLED_NATIVE_COMPONENT_SAMPLES",
            component_sources=[{"path":str(p.resolve()), "sha256":sha} for sha,(p,_) in sources.items()])
        validate_pooled_receipt(raw)
        path = output / f"{key}.json"
        write_new_json(path, raw)
        records.append(OperationCost(key, tuple(raw["samples_ns"]), raw["workspace_bytes_per_rank"],
                                     _evidence(path, raw)))
    result = CostModel(binding, tuple(records))
    result.verify_receipts()
    return result
