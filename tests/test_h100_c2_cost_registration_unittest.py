"""Central C2 cost admission: synthetic claims must never become native evidence.

Every model deliberately claims MEASURED to exercise the real central verifier,
not its CPU_FIXTURE_ONLY early return. No source checkout or GPU is required.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from distillschedule.search import hardware_helios_c2 as h100
from distillschedule.search.cost_model import ComponentContract, CostModel, OperationCost
from distillschedule.search.plan_ir import BINDING_KEYS
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint


class H100C2CostRegistrationTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(tempfile.gettempdir()) / "lyy-experiments/DMD-System/h100-c2-cost-registration"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="unit-", dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "synthetic-calibration.json"
        self.binding = {key: "a" * (40 if key == "source_head" else 64)
                        for key in BINDING_KEYS}
        self.binding["precision_fingerprint"] = fingerprint(h100.PRECISION)
        contract = ComponentContract(
            family=h100.FAMILY,
            work={"fixture_scope": "CPU_REJECTION_TEST_ONLY"},
            precision=h100.PRECISION,
            placement={"fixture_scope": "CPU_REJECTION_TEST_ONLY"},
            implementation={"fixture_scope": "CPU_REJECTION_TEST_ONLY"},
            context={"cycle": 2, "fixture_scope": "CPU_REJECTION_TEST_ONLY"},
        )
        self.raw = {
            "schema_version": "distillschedule.operation_calibration.v1",
            "binding": self.binding, "key": contract.digest,
            "samples_ns": [7], "workspace_bytes_per_rank": 16,
            "status": "PASS", "profiled": False, "kind": "native_operation",
            "operation_contract_sha256": contract.digest,
            "component_contract": contract.to_dict(), "evidence_scope": "MEASURED",
            "fixture_scope": "SYNTHETIC_INVALID_CLAIM_NOT_NATIVE_MEASUREMENT",
            "deployment": {},
        }

    def model(self, raw=None):
        raw = self.raw if raw is None else raw
        self.path.write_text(json.dumps(raw, sort_keys=True, allow_nan=False))
        evidence = {key: raw[key] for key in (
            "status", "profiled", "kind", "operation_contract_sha256", "component_contract")}
        evidence.update(source_receipt=str(self.path), source_receipt_sha256=file_sha256(self.path))
        record = OperationCost(raw["key"], tuple(raw["samples_ns"]),
                               raw["workspace_bytes_per_rank"], evidence)
        model = CostModel(self.binding, (record,), scope="MEASURED")
        self.assertEqual(model.scope, "MEASURED")
        return model

    def test_hash_consistent_claim_reaches_real_native_rejection(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, "unknown/missing C2 deployment field"):
            model.verify_receipts()

    def test_central_verifier_calls_registered_validator_with_raw_receipt(self):
        model = self.model()
        with patch.object(h100, "validate_component_receipt",
                          side_effect=RuntimeError("STRICT_C2_VALIDATOR_REACHED")) as validator:
            with self.assertRaisesRegex(RuntimeError, "STRICT_C2_VALIDATOR_REACHED"):
                model.verify_receipts()
        validator.assert_called_once_with(self.raw, {})

    def test_measured_model_rejects_fixture_or_missing_native_scope(self):
        for scope in ("CPU_FIXTURE_ONLY", None):
            raw = deepcopy(self.raw)
            if scope is None:
                raw.pop("evidence_scope")
            else:
                raw["evidence_scope"] = scope
            model = self.model(raw)
            with self.subTest(scope=scope), patch.object(h100, "validate_component_receipt") as validator:
                with self.assertRaisesRegex(ValueError, "fixture C2 costs cannot be relabeled measured"):
                    model.verify_receipts()
                validator.assert_not_called()

    def test_file_hash_is_checked_before_registered_native_validator(self):
        model = self.model()
        changed = deepcopy(self.raw)
        changed["samples_ns"] = [8]
        self.path.write_text(json.dumps(changed))
        with patch.object(h100, "validate_component_receipt") as validator:
            with self.assertRaisesRegex(ValueError, "calibration receipt hash mismatch"):
                model.verify_receipts()
            validator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
