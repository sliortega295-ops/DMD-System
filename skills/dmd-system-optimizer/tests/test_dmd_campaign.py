"""Synthetic contracts exercise persistence and rejection paths, not GPU performance."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import dmd_campaign as m
import dmd_perf as perf


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        receipt = self.root / "receipt.txt"
        receipt.write_text("SYNTHETIC test fixture; not a real checker result")
        self.ev = [{"path": str(receipt), "sha256": perf.sha_file(receipt), "locator": "fixture"}]
        self.path = self.root / "campaign.sqlite"
        self.c = {
            "schema_version": 1, "campaign_id": "fixture", "workload_id": "toy",
            "semantic_contract": "same-logical-work", "hardware_id": "cpu-fixture",
            "timer_scope": "complete-cycle", "claim_class": "strict",
            "baseline": {"id": "b", "source_identity": "b-source", "seconds": 100., "evidence": self.ev},
            "validators": self.ev, "budget": {"max_gpu_hours": 100., "max_full_runs": 100},
            "validation_by_risk": {"local": ["V0", "V1"], "stateful": ["V0", "V1", "V2"]},
            "timing": {"repeat_count": 6, "max_candidates": 1, "minimum_effect_seconds": .1,
                       "alpha": .05, "max_control_relative_drift": .02}}
        self.initialize()

    def initialize(self):
        return m.init(self.path, self.c, self.root)

    def state(self):
        with m.connection(self.path) as db:
            return m.status(db)

    def proposal(self, **values):
        return {"candidate_id": "c", "source_identity": "c-source", "control_id": "b",
                "risk": "local", "purpose": "timing", "gpu_hours": 1., "full_runs": 3, **values}

    def reserve(self, ident="run-0", **values):
        with m.connection(self.path) as db:
            return m.reserve(db, ident, self.proposal(**values))

    def settle(self, ident="run-0", **values):
        with m.connection(self.path) as db:
            return m.settle(db, ident, {"outcome": "pass", "gpu_hours": .9, "full_runs": 3,
                                       "evidence": self.ev, **values}, self.root)

    def ready(self, risk="local", **values):
        for i in range(6):
            self.reserve("run-" + str(i), risk=risk)
            self.settle("run-" + str(i))
        return {**{k: self.c[k] for k in ("workload_id", "semantic_contract", "hardware_id", "timer_scope", "claim_class")},
                "contract_sha256": self.state()["contract_sha256"],
                "candidate_id": "c", "source_identity": "c-source", "control_id": "b",
                "checks": {"V0": "PASS", "V1": "PASS"}, "validation_scope": "synthetic complete cycle",
                "evidence": self.ev, "unprofiled": True, "repeat_unit": "independent_processes",
                "compatibility": "PASS", "fallback": "PASS",
                "timings": [{"run_id": "run-" + str(i), "process_ids": [f"{i}-a", f"{i}-b", f"{i}-a2"],
                             "control_before": 100., "candidate": 99., "control_after": 100.}
                            for i in range(6)], **values}

    def assess(self, r):
        with m.connection(self.path) as db:
            return m.assess(db, r, self.root)

    def test_restart_keeps_reservations_and_actual_cost(self):
        self.reserve()
        self.initialize()
        self.assertEqual(self.state()["budget"]["reserved_full_runs"], 3)
        self.settle()
        s = self.state()["budget"]
        self.assertEqual(s["reserved_full_runs"], 0)
        self.assertEqual(s["used_full_runs"], 3)
        self.assertEqual(s["used_gpu_hours"], .9)

    def test_duplicate_reservation_and_settlement_idempotent(self):
        self.reserve(); self.reserve(); self.settle(); self.settle()
        self.assertEqual(len(self.state()["runs"]), 1)
        self.assertEqual(self.state()["budget"]["used_full_runs"], 3)
        with self.assertRaises(ValueError):
            self.settle(outcome="error")

    def test_contract_cannot_reset_budget_or_baseline(self):
        self.c["budget"]["max_gpu_hours"] += 1
        with self.assertRaises(ValueError):
            self.initialize()
        self.assertEqual(self.state()["contract"]["budget"]["max_gpu_hours"], 100)

    def test_impossible_confirmation_rejected(self):
        self.c["timing"]["repeat_count"] = 2
        with self.assertRaisesRegex(ValueError, "cannot resolve"):
            m.init(self.root / "impossible.sqlite", self.c, self.root)

    def test_overrun_is_recorded_and_blocks_future_work(self):
        self.reserve()
        self.settle(outcome="error", gpu_hours=101)
        self.assertTrue(self.state()["budget"]["exceeded"])
        with self.assertRaises(ValueError):
            self.reserve("next")

    def test_failed_runs_still_cost_resources(self):
        self.reserve(); self.settle(outcome="interrupted")
        self.assertEqual(self.state()["budget"]["used_full_runs"], 3)

    def test_concurrent_reservations_do_not_overspend(self):
        def attempt(i):
            try:
                self.reserve(str(i), purpose="probe", gpu_hours=60, full_runs=0)
                return True
            except ValueError:
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            accepted = list(pool.map(attempt, range(2)))
        self.assertEqual(sum(accepted), 1)
        self.assertEqual(self.state()["budget"]["reserved_gpu_hours"], 60)

    def test_one_percent_nonbitwise_local_gain_can_be_kept(self):
        r = self.ready(bitwise=False)
        d = self.assess(r)
        self.assertEqual(d["retention"], "KEEP")
        self.assertTrue(d["default_enable"])
        self.assertEqual(d["required_validation"], ["V0", "V1"])
        self.assertEqual(d["one_sided_sign_p"], 1/64)

    def test_negative_gain_is_inconclusive_not_promoted(self):
        r = self.ready()
        for t in r["timings"]:
            t["candidate"] = 101
        d = self.assess(r)
        self.assertEqual(d["performance"], "INCONCLUSIVE")
        self.assertFalse(d["default_enable"])

    def test_timing_does_not_override_gradient_failure(self):
        r = self.ready()
        r["checks"]["backward"] = "FAIL"
        d = self.assess(r)
        self.assertEqual(d["retention"], "DISCARD")
        self.assertEqual(d["performance"], "NOT_EVALUATED")

    def test_missing_continuation_stays_review(self):
        d = self.assess(self.ready(risk="stateful"))
        self.assertEqual(d["retention"], "REVIEW")
        self.assertFalse(d["default_enable"])

    def test_hardware_and_contract_mismatch_rejected(self):
        r = self.ready()
        r["hardware_id"] = "different"
        with self.assertRaisesRegex(ValueError, "hardware_id"):
            self.assess(r)

    def test_mutated_checker_is_rejected(self):
        r = self.ready()
        (self.root / "receipt.txt").write_text("changed checker")
        with self.assertRaises(ValueError):
            self.assess(r)

    def test_repeated_assessment_rechecks_settlement_receipts(self):
        other = self.root / "other.txt"; other.write_text("separate settlement")
        ref = [{"path": str(other), "sha256": perf.sha_file(other), "locator": "fixture"}]
        self.ev_original = copy.deepcopy(self.ev)
        self.ev = ref
        r = self.ready(evidence=self.ev_original)
        self.assess(r)
        other.write_text("changed")
        with self.assertRaises(ValueError):
            self.assess(r)

    def test_profiled_samples_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unprofiled"):
            self.assess(self.ready(unprofiled=False))

    def test_reused_invocation_ids_rejected(self):
        r = self.ready()
        r["timings"][1]["process_ids"] = r["timings"][0]["process_ids"]
        with self.assertRaisesRegex(ValueError, "reused"):
            self.assess(r)

    def test_control_drift_blocks_benefit(self):
        r = self.ready()
        r["timings"][0]["control_after"] = 110
        self.assertEqual(self.assess(r)["performance"], "INVALID_DRIFT")

    def test_cannot_select_only_favorable_samples(self):
        r = self.ready()
        r["timings"].pop()
        with self.assertRaisesRegex(ValueError, "repeat count"):
            self.assess(r)

    def test_no_optional_stopping_or_relabeling_candidate_source(self):
        self.ready()
        with self.assertRaisesRegex(ValueError, "repeat count"):
            self.reserve("seventh")
        with self.assertRaisesRegex(ValueError, "identity/risk"):
            self.reserve("changed", source_identity="new-source", purpose="probe")

    def test_assessment_is_immutable(self):
        r = self.ready(); self.assess(r)
        r["timings"][0]["candidate"] = 98
        with self.assertRaisesRegex(ValueError, "different evidence"):
            self.assess(r)

    def test_promote_requires_persisted_assessment(self):
        r = self.ready()
        with m.connection(self.path) as db, self.assertRaisesRegex(ValueError, "assess and persist"):
            m.promote(db, r, self.root)

    def test_promote_changes_control_not_baseline_and_is_idempotent(self):
        r = self.ready(); self.assess(r)
        with m.connection(self.path) as db:
            self.assertEqual(m.promote(db, r, self.root)["status"], "PROMOTED")
        with m.connection(self.path) as db:
            self.assertEqual(m.promote(db, r, self.root)["status"], "ALREADY_PROMOTED")
        self.assertEqual(self.state()["contract"]["baseline"]["id"], "b")
        self.assertEqual(self.state()["control"]["id"], "c")
        with self.assertRaisesRegex(ValueError, "stale control"):
            self.reserve("old-control", candidate_id="d")

    def test_compatibility_or_fallback_unknown_keeps_optional_default_off(self):
        d = self.assess(self.ready(fallback="UNKNOWN"))
        self.assertEqual(d["retention"], "KEEP")
        self.assertFalse(d["default_enable"])

    def test_diagnostic_claim_cannot_be_default(self):
        self.path = self.root / "diagnostic.sqlite"
        self.c["claim_class"] = "diagnostic_review"; self.initialize()
        d = self.assess(self.ready())
        self.assertEqual(d["claim_class"], "diagnostic_review")
        self.assertFalse(d["default_enable"])

    def test_capacity_value_requires_evidence(self):
        r = self.ready(capacity_evidence=self.ev)
        for t in r["timings"]:
            t["candidate"] = 101
        d = self.assess(r)
        self.assertEqual(d["retention"], "KEEP_OPTIONAL")
        self.assertFalse(d["default_enable"])

    def test_probe_does_not_consume_confirmation_slots(self):
        for i in range(3):
            self.reserve(f"probe-{i}", candidate_id=f"probe-{i}", purpose="probe", full_runs=0)
        self.ready()
        with self.assertRaisesRegex(ValueError, "confirmation budget"):
            self.reserve("other-confirmation", candidate_id="other")

    def test_gate_uses_persistent_budget_and_original_identity(self):
        self.reserve(gpu_hours=80)
        c = copy.deepcopy(self.c)
        c["control"] = copy.deepcopy(c["baseline"])
        c["budget"]["used_gpu_hours"] = 0
        bound = perf.bind_ledger(c, self.path)
        self.assertEqual(bound["budget"]["used_gpu_hours"], 80)
        c["baseline"]["source_identity"] = "invented"
        with self.assertRaisesRegex(ValueError, "baseline.source_identity"):
            perf.bind_ledger(c, self.path)

    def test_fresh_process_reads_same_ledger(self):
        self.reserve()
        p = subprocess.run([sys.executable, str(SCRIPTS / "dmd_campaign.py"), "status",
                            "--state", str(self.path)], capture_output=True, text=True, check=True,
                           cwd=self.root)
        self.assertEqual(json.loads(p.stdout)["budget"]["reserved_full_runs"], 3)


if __name__ == "__main__":
    unittest.main()
