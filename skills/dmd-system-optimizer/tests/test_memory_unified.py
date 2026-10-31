"""Private learning must retain public retrieval semantics and visible corrections."""
import copy
import json
import subprocess
import sys
import unittest
from test_dmd_knowledge import KnowledgeTests, m
from dmd_perf import sha_file


class UnifiedMemoryTests(KnowledgeTests):
    def test_cpu_context_can_state_zero_gpus(self):
        context = {"hardware": "CPU", "nodes": 1, "gpus_per_node": 0, "physical_gpus": 0}
        self.assertEqual(m.recall("参数上卡", context)["normalized_context"], context)
        with self.assertRaises(ValueError):
            m.normalize_context({**context, "physical_gpus": -1})

    def record(self, ident, summary, **extra):
        r = {**copy.deepcopy(self.r), "id": ident, "summary": summary, "pattern": "parameter-residency",
             "correctness_conditions": {"grad_mode": "no_grad"}, "performance_conditions": {"exposed_gather": True}, **extra}
        self.put(r)

    def test_colloquial_chinese_and_grad_alias_share_condition_engine(self):
        self.record("waiting", "参数上卡等待减少")
        out = m.search(self.store, "参数上卡总是在等，怎么优化？", {"grad_mode": "no-grad", "dtype": "bf16"})
        row = out["matches"][0]
        self.assertNotIn("grad_mode", row["incompatible_conditions"])
        self.assertEqual(row["correctness_comparison"]["conflicting"], [])
        self.assertEqual(row["performance_comparison"]["unknown"], ["exposed_gather"])

    def test_public_and_private_cases_share_one_mechanism_packet(self):
        pattern = m.validate_pack()["patterns"][0]["id"]
        self.record("local-positive", "参数保留 residency", pattern=pattern, outcome="success")
        self.record("local-negative", "参数保留 residency", pattern=pattern, outcome="no_benefit")
        out = m.recall("参数保留 residency", {}, limit=20, stores=[self.store])
        packet = next(p for p in out["matches"] if p["pattern"]["id"] == pattern)
        self.assertTrue(any("record" in c for c in packet["cases"]["positive"]))
        self.assertTrue(any("record" in c for c in packet["cases"]["negative"]))
        self.assertTrue(any("record" not in c for group in packet["cases"].values() for c in group))

    def test_limit_one_cannot_hide_global_correction(self):
        self.record("a-old", "参数上卡等待", outcome="success")
        self.record("z-new", "修正了旧计时分母", outcome="no_benefit", supersedes=[{
            "id": "a-old", "kind": "correction", "conditions": {}, "reason": "denominator was wrong"}])
        row = m.search(self.store, "参数上卡", {}, limit=1)["matches"][0]
        self.assertEqual(row["id"], "z-new")
        old = next(c for c in m.search(self.store, "参数上卡", {})["matches"] if c["id"] == "a-old")
        self.assertEqual(old["correction"]["status"], "CORRECTED_FOR_CONTEXT")
        self.assertTrue(old["correction"]["current_successors"][0].endswith("z-new"))

    def test_context_update_does_not_blacklist_other_hardware(self):
        self.record("old", "参数上卡等待", outcome="success")
        self.record("new", "参数上卡等待", outcome="no_benefit", supersedes=[{
            "id": "old", "kind": "context_update", "conditions": {"gpu_family": "H100"}, "reason": "different exposed communication"}])
        for context, status in (({"gpu_family": "A100"}, "HISTORICAL_CONTEXT_REMAINS"),
                                ({"gpu_family": "H100"}, "CORRECTED_FOR_CONTEXT"), ({}, "SCOPE_UNKNOWN")):
            old = next(r for r in m.search(self.store, "参数上卡", context)["matches"] if r["id"] == "old")
            self.assertEqual(old["correction"]["status"], status)

    def test_chain_cycle_and_legacy_corrections_are_explicit(self):
        self.record("a", "参数上卡等待")
        self.record("b", "参数上卡等待", supersedes=["a"])
        old = next(r for r in m.search(self.store, "参数上卡", {})["matches"] if r["id"] == "a")
        self.assertEqual(old["correction"]["status"], "SCOPE_UNKNOWN")
        self.record("c", "参数上卡等待", supersedes=[{"id": "d", "kind": "correction", "conditions": {}, "reason": "fixture"}])
        self.record("d", "参数上卡等待", supersedes=[{"id": "c", "kind": "correction", "conditions": {}, "reason": "fixture"}])
        result = m.recall("参数上卡", {}, stores=[self.store])
        rows = [c for p in result["matches"] for group in p["cases"].values() for c in group]
        self.assertTrue(any(c["id"].endswith(":a") for c in rows))
        for ident in ("c", "d"):
            row = next(c for c in rows if c["id"].endswith(":" + ident))
            self.assertEqual(row["correction"]["status"], "CORRECTION_NEEDS_REVIEW")
            self.assertIn("cycle", row["correction"]["warning"])
            self.assertEqual(row["correction"]["current_successors"], [])

    def test_stale_correction_does_not_silently_replace_valid_evidence(self):
        self.record("a", "参数上卡等待")
        self.record("b", "参数上卡等待", supersedes=[{"id": "a", "kind": "correction", "conditions": {}, "reason": "fixture"}])
        self.source.write_text("changed")
        old = next(r for r in m.search(self.store, "参数上卡", {})["matches"] if r["id"] == "a")
        self.assertEqual(old["correction"]["status"], "CORRECTION_NEEDS_REVIEW")
        self.assertEqual(old["correction"]["current_successors"], [])

    def test_only_explicit_stores_are_read_and_brief_case_roundtrip(self):
        self.record("private", "参数上卡等待")
        self.assertNotIn(str(self.store), json.dumps(m.recall("参数上卡", {})))
        out = m.brief_recall(m.recall("参数上卡", {}, stores=[self.store], limit=20))
        item = next(c for p in out["matches"] for group in p["cases"].values() for c in group if "store" in c)
        expanded = m.read_case(item["id"], {}, stores=[self.store])
        self.assertEqual(expanded["record"]["id"], "private")
        self.assertIn("correctness_comparison", expanded)

    def test_private_locate_cli_uses_authorized_store_and_preserves_corrections(self):
        self.record("private", "参数上卡等待", reuse_decision={"implementation_id": "fsdp-phase-retention"},
                    source={"path": str(self.source), "sha256": sha_file(self.source)})
        self.record("corrected", "更正旧经验", supersedes=[{
            "id": "private", "kind": "correction", "conditions": {}, "reason": "fixture correction"}])
        ident = next(c["id"] for c in m.private_cases([self.store]) if c["record"]["id"] == "private")
        command = [sys.executable, m.__file__, "locate", "--case", ident]
        rejected = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("unknown case", rejected.stderr)
        accepted = subprocess.run(command + ["--store", str(self.store)], capture_output=True, text=True, check=True)
        result = json.loads(accepted.stdout)
        self.assertEqual(result["navigation"]["status"], "linked")
        self.assertEqual(result["code_navigation"]["source"]["sha256"], sha_file(self.source))
        self.assertEqual(result["correction"]["status"], "CORRECTED_FOR_CONTEXT")
        self.assertEqual(len(result["implementations"]), 1)
        self.assertFalse(result["automatic_application"])
        self.source.write_text("source drift")
        stale = m.locate_case(ident, {}, stores=[self.store])
        self.assertEqual(stale["evidence_status"], "STALE")
        self.assertEqual(stale["code_navigation"]["target_source_status"], "STALE")

    def test_native_label_roundtrip_and_partial_outcome_remain_inspectable(self):
        self.record("native", "参数上卡容量收益", outcome="success", decision="KEEP_OPTIONAL")
        self.record("partial", "参数上卡确认未完成", outcome="incomplete", decision={"retention": "REVIEW", "performance": "INCOMPLETE"})
        cmd = [sys.executable, m.__file__, "recall", "--query", "参数上卡", "--store", str(self.store), "--brief"]
        packet = json.loads(subprocess.check_output(cmd, text=True))
        rows = [c for p in packet["matches"] for group in p["cases"].values() for c in group if "store" in c]
        native = next(c for c in rows if c["id"].endswith(":native"))
        partial = next(c for c in rows if c["id"].endswith(":partial"))
        self.assertEqual(native["decision"]["retention"], "KEEP_OPTIONAL")
        self.assertEqual(native["decision"]["evidence_assurance"], "DECLARATION_ONLY")
        self.assertEqual(partial["outcome"], "incomplete")

    def test_record_returns_usable_qualified_id_without_extra_search(self):
        stored = self.put()
        command = [sys.executable, m.__file__, "case", "--id", stored["qualified_id"], "--store", str(self.store)]
        expanded = json.loads(subprocess.check_output(command, text=True))
        self.assertEqual(expanded["record"]["id"], self.r["id"])
        self.assertEqual(self.put()["qualified_id"], stored["qualified_id"])

    def test_malformed_legacy_decision_cannot_crash_brief_retrieval(self):
        self.record("old", "参数上卡等待")
        path = self.store / "old.json"
        legacy = json.loads(path.read_text()); legacy["decision"] = ["ambiguous", "old", "data"]
        path.write_text(json.dumps(legacy))
        result = m.brief_recall(m.recall("参数上卡", {}, stores=[self.store]))
        row = next(c for p in result["matches"] for group in p["cases"].values() for c in group if c["id"].endswith(":old"))
        self.assertEqual(row["decision"]["evidence_assurance"], "MALFORMED_LEGACY_DECISION")
        with self.assertRaisesRegex(ValueError, "decision must be an object"):
            self.put({**self.r, "id": "new", "decision": []})


def load_tests(loader, suite, pattern):
    return unittest.TestSuite(UnifiedMemoryTests(name) for name in UnifiedMemoryTests.__dict__ if name.startswith("test_"))
