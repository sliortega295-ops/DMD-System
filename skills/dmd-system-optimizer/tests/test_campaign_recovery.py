"""Review regressions: append evidence, bounded recovery, parsed receipts, outbox."""
import json
import subprocess
import sys
import unittest
from test_dmd_campaign import CampaignTests, SCRIPTS, m


class RecoveryTests(CampaignTests):
    # Reuse the fixture, not its test methods (see load_tests below).
    def test_review_can_add_v2_without_resampling(self):
        r = self.ready(risk="stateful")
        first = self.assess(r)
        self.assertEqual(first["retention"], "REVIEW")
        self.validation("v2", {"V2": "PASS"}, risk="stateful")
        r["checks"]["V2"] = "PASS"
        final = self.assess(r)
        self.assertEqual(final["retention"], "KEEP")
        self.assertEqual(final["decision_revision"], 2)
        self.assertEqual(len(self.state()["decisions"]), 2)
        self.assertEqual(self.state()["decisions"][0], first)
        self.assertEqual(self.assess(r), final)
        r["timings"][0]["candidate"] -= 1
        with self.assertRaisesRegex(ValueError, "different evidence"):
            self.assess(r)

    def invalidate(self, ident, reason="environment_interruption", origin="environment"):
        self.reserve(ident)
        sample = {"status": "invalid", "reason": reason, "failure_origin": origin, "valid_window": False}
        receipt = self.receipt(ident, "attempt-json", {"status": "interrupted", "sample": sample})
        self.settle(ident, outcome="interrupted", sample=sample, receipts=[receipt])

    def test_interruption_costs_budget_but_valid_samples_can_complete(self):
        self.invalidate("interrupted")
        r = self.ready()
        self.assertEqual(self.assess(r)["retention"], "KEEP")
        self.assertEqual(sum(x["proposal"]["purpose"] == "timing" for x in self.state()["runs"]), 7)
        self.assertEqual(self.state()["budget"]["used_full_runs"], 21)

    def test_replacement_attempt_cap_is_independent_of_valid_sample_count(self):
        for i in range(8):
            self.invalidate("invalid-" + str(i))
        with self.assertRaisesRegex(ValueError, "attempt cap"):
            self.reserve("ninth")

    def test_candidate_failure_cannot_be_replaced(self):
        with self.assertRaisesRegex(ValueError, "candidate failures"):
            self.invalidate("oom", origin="candidate")

    def test_valid_slow_sample_cannot_be_deleted_or_reclassified(self):
        r = self.ready(candidate_time=101)
        with self.assertRaisesRegex(ValueError, "repeat count"):
            self.reserve("replace-slow")
        with self.assertRaises(ValueError):
            self.settle("run-0", outcome="interrupted", sample={"status": "invalid"})
        self.assertEqual(self.assess(r)["performance"], "INCONCLUSIVE")

    def test_unregistered_failure_reason_cannot_unlock_samples(self):
        self.reserve("failed")
        sample = {"status": "invalid", "reason": "invalid_timer_window", "failure_origin": "environment", "valid_window": False}
        receipt = self.receipt("failed", "attempt-json", {"status": "interrupted", "sample": sample})
        with self.assertRaisesRegex(ValueError, "not predeclared"):
            self.settle("failed", outcome="interrupted", sample=sample, receipts=[receipt])

    def test_capacity_only_does_not_buy_timing_confirmation(self):
        self.validation(checks={"V0": "PASS", "V1": "PASS", "V2": "PASS", "capacity": "PASS"}, risk="stateful")
        r = {**{k: self.c[k] for k in ("workload_id", "semantic_contract", "hardware_id", "timer_scope", "claim_class")},
             "candidate_id": "c", "source_identity": "c-source", "control_id": "b", "contract_sha256": self.state()["contract_sha256"],
             "validation_scope": "synthetic complete cycle", "checks": {"V0": "PASS", "V1": "PASS", "V2": "PASS"},
             "evidence": self.ev, "capacity_evidence": self.ev, "latency_tradeoff": "UNKNOWN; default off; measure before speed-sensitive use"}
        d = self.assess(r)
        self.assertEqual(d["retention"], "KEEP_OPTIONAL")
        self.assertFalse(d["default_enable"])
        self.assertEqual(d["performance"], "NOT_EVALUATED")
        self.assertEqual(sum(x["proposal"]["purpose"] == "timing" for x in self.state()["runs"]), 0)

    def test_fail_body_with_pass_summary_rejected(self):
        r = self.ready(checks={"V0": "FAIL", "V1": "FAIL"}, compatibility="FAIL", fallback="FAIL", composition="FAIL")
        r["checks"] = {"V0": "PASS", "V1": "PASS"}
        with self.assertRaisesRegex(ValueError, "parsed checker body"):
            self.assess(r)

    def test_modified_timing_summary_rejected_even_before_first_decision(self):
        r = self.ready(); r["timings"][0]["candidate"] = 1
        with self.assertRaisesRegex(ValueError, "parsed runner body"):
            self.assess(r)

    def test_receipt_cannot_relabel_source_or_comparator(self):
        self.reserve("validation", purpose="validation")
        receipt = self.receipt("validation", "junit", '<testsuite><testcase name="V0"/></testsuite>')
        receipt["binding"]["source_identity"] = "wrong-source"
        with self.assertRaisesRegex(ValueError, "reserved run"):
            self.settle("validation", receipts=[receipt])

    def test_skip_is_not_pass(self):
        r = self.ready(checks={"V0": "PASS", "V1": "NOT_RUN"})
        self.assertEqual(self.assess(r)["retention"], "REVIEW")

    def test_new_failure_cannot_replay_old_keep(self):
        r = self.ready(); self.assess(r)
        self.validation("late-failure", {"V2": "FAIL"})
        self.assertEqual(self.assess(r)["retention"], "DISCARD")

    def test_failed_capacity_or_default_gate_does_not_invent_numerical_failure(self):
        r = self.ready(fallback="FAIL")
        d = self.assess(r)
        self.assertEqual(d["retention"], "KEEP")
        self.assertFalse(d["default_enable"])

    def test_final_combination_requires_original_baseline_comparison(self):
        with m.connection(self.path) as db:
            m.set_value(db, "control", {"id": "b2", "source_identity": "b2-source", "seconds": 100, "evidence": self.ev})
        original_proposal = self.proposal
        self.proposal = lambda **kw: original_proposal(**{"control_id": "b2", "comparator_id": "b2", "comparator_source_identity": "b2-source", **kw})
        r = self.ready(control_id="b2")
        d = self.assess(r)
        self.assertEqual(d["retention"], "KEEP")
        self.assertFalse(d["original_baseline_validated"])
        self.assertFalse(d["default_enable"])
        self.validation("original-baseline", {"V0": "PASS", "V1": "PASS"}, comparator_id="b", comparator_source_identity="b-source")
        d = self.assess(r)
        self.assertTrue(d["default_enable"])
        self.assertEqual(d["decision_revision"], 2)

    def test_outbox_survives_commit_and_fresh_process_replays_once(self):
        r = self.ready()
        bundle = {"result": r, "experience": {"summary": "参数上卡等待减少", "pattern": "parameter-residency",
                   "conditions": {"grad_mode": "no_grad"}, "correctness_conditions": {"grad_mode": "no_grad"},
                   "performance_conditions": {"exposed_gather": True}, "retry": "check changed gather overlap",
                   "failure_scope": "synthetic timing only"}}
        store = self.root / "memory"
        with m.connection(self.path) as db:
            out = m.finish(db, bundle, store, self.root)
        self.assertFalse(store.exists())  # Simulated crash after DB commit, before file publish.
        for _ in range(2):
            subprocess.run([sys.executable, str(SCRIPTS / "dmd_campaign.py"), "flush-memory", "--state", str(self.path)], check=True, capture_output=True)
        self.assertEqual(len(list(store.glob("*.json"))), 1)
        with m.connection(self.path) as db:
            self.assertEqual(m.finish(db, bundle, store, self.root)["memory_id"], out["memory_id"])
        self.assertEqual(self.state()["memory_pending"], 0)
        query = subprocess.run([sys.executable, str(SCRIPTS / "dmd_knowledge.py"), "recall", "--store", str(store),
                                "--query", "参数上卡总是在等，怎么优化？", "--brief"], capture_output=True, text=True, check=True)
        self.assertIn(out["memory_id"], query.stdout)

    def test_legacy_declarations_cannot_earn_new_keep(self):
        self.path = self.root / "legacy.sqlite"
        self.c["schema_version"] = 1; self.initialize()
        self.assertEqual(self.assess(self.ready())["retention"], "REVIEW")

    def test_missing_receipt_and_wrong_checker_version_rejected(self):
        self.reserve("validation", purpose="validation")
        receipt = self.receipt("validation", "junit", '<testsuite><testcase name="V0"/></testsuite>')
        receipt["checker"]["version"] = "unfrozen-v2"
        with self.assertRaisesRegex(ValueError, "frozen version"):
            self.settle("validation", receipts=[receipt])
        self.settle("validation")
        r = {**{k: self.c[k] for k in ("workload_id", "semantic_contract", "hardware_id", "timer_scope", "claim_class")},
             "candidate_id": "c", "source_identity": "c-source", "control_id": "b", "contract_sha256": self.state()["contract_sha256"],
             "validation_scope": "synthetic complete cycle", "checks": {"V0": "PASS", "V1": "PASS"}, "evidence": self.ev}
        with self.assertRaisesRegex(ValueError, "parsed checker body"):
            self.assess(r)

    def test_declared_failed_suite_does_not_pass_from_test_names(self):
        self.reserve("validation", purpose="validation")
        receipt = self.receipt("validation", "junit", '<testsuite failures="1"><testcase name="V0"/><testcase name="V1"/></testsuite>')
        self.assertEqual(receipt["checks"]["V0"], "FAIL")

    def test_review_cannot_freeze_invalid_process_sample_set(self):
        with self.assertRaisesRegex(ValueError, "reused"):
            self.assess(self.ready(risk="stateful", reused=True))
        self.assertEqual(self.state()["decisions"], [])

    def test_synthetic_timing_cannot_enter_observed_campaign(self):
        self.path = self.root / "observed.sqlite"
        self.c["measurement_kind"] = "observed"; self.initialize()
        with self.assertRaisesRegex(ValueError, "qualification differs"):
            self.assess(self.ready())

    def test_revision_writeback_links_previous_provisional_memory(self):
        r = self.ready(risk="stateful")
        metadata = {"summary": "参数上卡等待减少", "pattern": "parameter-residency", "conditions": {},
                    "correctness_conditions": {}, "performance_conditions": {}, "retry": "inspect new evidence", "failure_scope": "synthetic"}
        store = self.root / "memory"
        with m.connection(self.path) as db:
            old = m.finish(db, {"result": r, "experience": metadata}, store, self.root)
        m.flush_memory(self.path)
        self.validation("v2", {"V2": "PASS"}, risk="stateful")
        r["checks"]["V2"] = "PASS"
        with m.connection(self.path) as db:
            new = m.finish(db, {"result": r, "experience": metadata}, store, self.root)
        m.flush_memory(self.path)
        import dmd_knowledge
        rows = dmd_knowledge.search(store, "参数上卡", {}, limit=1)["matches"]
        self.assertEqual(rows[0]["id"], new["memory_id"])
        earlier = json.loads((store / (old["memory_id"] + ".json")).read_text())
        self.assertEqual(earlier["outcome"], "incomplete")

    def test_correctness_only_campaign_needs_no_timing_placeholder(self):
        self.path = self.root / "correctness-only.sqlite"
        self.c.update(timing=None, measurement_kind="observed")
        self.c["baseline"]["seconds"] = None
        self.initialize()
        self.validation(checks={"V0": "PASS", "V1": "PASS", "capacity": "PASS"})
        r = {**{k: self.c[k] for k in ("workload_id", "semantic_contract", "hardware_id", "timer_scope", "claim_class")},
             "candidate_id": "c", "source_identity": "c-source", "control_id": "b", "contract_sha256": self.state()["contract_sha256"],
             "validation_scope": "synthetic complete cycle", "checks": {"V0": "PASS", "V1": "PASS"}, "evidence": self.ev,
             "capacity_evidence": self.ev, "latency_tradeoff": "UNKNOWN"}
        self.assertEqual(self.assess(r)["retention"], "KEEP_OPTIONAL")
        self.assertEqual(self.state()["contract"]["baseline"]["seconds"], None)
        with self.assertRaisesRegex(ValueError, "disabled"):
            self.reserve("speed")


def load_tests(loader, suite, pattern):
    return unittest.TestSuite(RecoveryTests(name) for name in RecoveryTests.__dict__ if name.startswith("test_"))
