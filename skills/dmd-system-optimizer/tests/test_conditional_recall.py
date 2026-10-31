"""Scoped retrieval and installation behavior; no GPU imports."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
REPO = SKILL.parents[1]
sys.path.insert(0, str(SKILL/"scripts"))
import dmd_knowledge as m


class RecallTests(unittest.TestCase):
    def recall(self, context=None, repo=None):
        return m.recall("参数保留 residency", context or {}, repo, 1)["matches"][0]

    def test_unknown_conditions_do_not_admit(self):
        r = self.recall()
        self.assertEqual(r["pattern"]["id"], "parameter-residency")
        self.assertEqual(r["next_action"], "CHECK_UNKNOWN_PRECONDITIONS")
        self.assertTrue(r["fresh_validation_required"])

    def test_correctness_conflict_blocks_as_is(self):
        r = self.recall({"weights_stable_during_reuse": False, "state_release_validated": True})
        self.assertEqual(r["next_action"], "DO_NOT_APPLY_AS_IS")

    def test_performance_conflict_is_not_semantic_rejection(self):
        r = self.recall({"weights_stable_during_reuse": True, "state_release_validated": True,
                         "exposed_gather": False, "memory_headroom": True, "repeated_forwards": True})
        self.assertEqual(r["next_action"], "ASSESS_CURRENT_BENEFIT")
        self.assertEqual(r["performance_conditions"]["conflicting"], ["exposed_gather"])
        self.assertTrue(r["prior_only"])

    def test_positive_and_negative_workloads_returned_together(self):
        r = self.recall({"workload": "W6", "ranks": 16})
        self.assertIn("helios-w2-retention-r36", {c["id"] for c in r["cases"]["positive"]})
        self.assertIn("helios-w6-retention-r1", {c["id"] for c in r["cases"]["negative"]})
        self.assertIn("workload", r["cases"]["positive"][0]["context_comparison"]["conflicting"])
        self.assertTrue(r["all_linked_cases_included"])

    def test_new_model_retrieves_mechanism_without_approval(self):
        r = self.recall({"model_family": "unseen-family"})
        self.assertTrue(r["cases"]["positive"])
        self.assertNotEqual(r["next_action"], "DO_NOT_APPLY_AS_IS")
        self.assertTrue(r["fresh_validation_required"])

    def test_numerical_review_not_upgraded_by_hash(self):
        r = self.recall(repo=REPO)
        c = next(c for c in r["cases"]["positive"] if c["id"] == "helios-w2-retention-r36")
        self.assertEqual(c["artifact_verification"], "HASH_CHECKED_NOT_TRUTH_PROVEN")
        self.assertEqual(c["evidence_status"], "HISTORICAL_REVIEW")
        self.assertFalse(c["automatic_application"])

    def test_missing_evidence_is_stale(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.recall(repo=Path(d))
        self.assertEqual(r["cases"]["positive"][0]["artifact_verification"], "STALE")

    def test_no_checkout_keeps_references_unchecked(self):
        self.assertEqual(self.recall()["cases"]["positive"][0]["artifact_verification"],
                         "PUBLIC_REFERENCES_NOT_LOCALLY_CHECKED")

    def test_unverified_leads_remain_separate(self):
        r = m.recall("pipeline 流水", {}, limit=1)["matches"][0]
        self.assertEqual(r["pattern"]["id"], "role-scheduling")
        self.assertTrue(r["cases"]["unresolved"])
        self.assertFalse(r["cases"]["positive"])
        self.assertEqual(r["cases"]["unresolved"][0]["artifact_verification"], "UNVERIFIED_CONVERSATION_LEAD")

    def test_no_match_does_not_invent_experience(self):
        self.assertEqual(m.recall("zzzzunmatchedqqqq", {})["matches"], [])

    def test_public_pack_has_no_private_history_or_paths(self):
        content = m.PACK.read_text()
        for value in ("/home/", "/share/", "/root/", "thread_id", "session_id", "raw_transcript"):
            self.assertNotIn(value, content)
        for case in m.validate_pack()["cases"]:
            for ref in case["evidence"]:
                self.assertTrue(ref["url"].startswith("https://github.com/sliortega295-ops/DMD-System/blob/"))

    def test_all_public_evidence_hashes_check(self):
        for c in m.validate_pack()["cases"]:
            if c["evidence"]:
                self.assertEqual(m.check_evidence(c["evidence"], REPO, "public evidence"), [], c["id"])

    def test_inventory_reads_sources_without_model_imports(self):
        r = m.inventory(REPO)
        self.assertEqual(len(r["records"]), 14)
        self.assertFalse(r["imports_executed"])
        self.assertEqual(r["gpu_validation"], "NOT_RUN")
        self.assertNotIn("torch", sys.modules)

    def test_installed_copy_recalls_without_original_checkout(self):
        spec = importlib.util.spec_from_file_location("installer", REPO/"tools/install_optimizer_skill.py")
        installer = importlib.util.module_from_spec(spec); spec.loader.exec_module(installer)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); dest = root/"installed/dmd-system-optimizer"
            self.assertEqual(installer.install(SKILL, dest)["status"], "INSTALLED")
            self.assertEqual(installer.install(SKILL, dest)["status"], "ALREADY_INSTALLED")
            p = subprocess.run([sys.executable, str(dest/"scripts/dmd_knowledge.py"), "recall",
                                "--query", "参数保留", "--limit", "1"], cwd=root, check=True,
                               capture_output=True, text=True)
            self.assertEqual(json.loads(p.stdout)["matches"][0]["pattern"]["id"], "parameter-residency")
            (dest/"SKILL.md").write_text("existing user version")
            with self.assertRaises(ValueError):
                installer.install(SKILL, dest)
            self.assertEqual((dest/"SKILL.md").read_text(), "existing user version")


if __name__ == "__main__":
    unittest.main()
