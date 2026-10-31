"""Scoped retrieval and installation behavior; no GPU imports."""
import importlib.util
import hashlib
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
        r = m.recall("FFN-up replay", {"workload": "W6", "ranks": 16}, limit=1)["matches"][0]
        self.assertIn("helios-w3-retention-r1", {c["id"] for c in r["cases"]["positive"]})
        self.assertIn("helios-w6-retention-r1", {c["id"] for c in r["cases"]["negative"]})
        w3 = next(c for c in r["cases"]["positive"] if c["id"] == "helios-w3-retention-r1")
        self.assertIn("workload", w3["context_comparison"]["conflicting"])
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
        self.assertEqual(r["cases"]["positive"][0]["artifact_verification"], "STALE_OR_UNAVAILABLE")

    def test_no_checkout_keeps_references_unchecked(self):
        self.assertEqual(self.recall()["cases"]["positive"][0]["artifact_verification"],
                         "PUBLIC_REFERENCES_NOT_LOCALLY_CHECKED")

    def test_unverified_leads_remain_separate(self):
        r = m.recall("pipeline 流水", {}, limit=1)["matches"][0]
        self.assertEqual(r["pattern"]["id"], "role-scheduling")
        self.assertTrue(r["cases"]["unresolved"])
        self.assertTrue(r["cases"]["positive"])
        self.assertTrue(r["cases"]["negative"])
        self.assertEqual(r["cases"]["unresolved"][0]["artifact_verification"], "UNVERIFIED_CONVERSATION_LEAD")

    def test_no_match_does_not_invent_experience(self):
        self.assertEqual(m.recall("zzzzunmatchedqqqq", {})["matches"], [])

    def test_public_pack_has_no_private_history_or_paths(self):
        content = m.PACK.read_text()
        for value in ("/home/", "/share/", "/root/", "thread_id", "session_id", "raw_transcript"):
            self.assertNotIn(value, content)
        for case in m.validate_pack()["cases"]:
            for ref in case["evidence"]:
                m.validate_reference(ref)

    def test_all_public_evidence_hashes_check(self):
        for c in m.validate_pack()["cases"]:
            if c["evidence"]:
                result = m.verify_references(c["evidence"], REPO)
                for row in result["evidence_checks"]:
                    # A portable install / shallow CI clone has no external checkout or old blobs.
                    # That is not a mismatch and must never be labeled checked.
                    self.assertIn(row["status"], {"HASH_MATCH", "NOT_LOCALLY_CHECKED", "PINNED_BLOB_UNAVAILABLE"}, c["id"])
                if any(row["status"] == "PINNED_BLOB_UNAVAILABLE" for row in result["evidence_checks"]):
                    self.assertEqual(result["artifact_verification"], "STALE_OR_UNAVAILABLE")

    def test_source_index_links_reach_cases_with_identical_identity(self):
        pack = m.validate_pack()
        cases = {c["id"]: c for c in pack["cases"]}
        sources = m.load(m.SOURCE_INDEX)["sources"]
        by_key = {(s["repository"], s["revision"], s["path"]): s for s in sources}
        self.assertEqual(len(by_key), len(sources))
        for s in sources:
            m.validate_reference(s)
            for ident in s["linked_case_ids"]:
                self.assertIn(ident, cases)
                self.assertIn(s["sha256"], {r["sha256"] for r in cases[ident]["evidence"]})
        for c in cases.values():
            for r in c["evidence"]:
                source = by_key[(r["repository"], r["revision"], r["path"])]
                self.assertIn(c["id"], source["linked_case_ids"])
                self.assertEqual(r["sha256"], source["sha256"])

    def test_discovered_sources_are_searchable_without_implied_curation(self):
        rows = m.source_search("DMD_LIVEWAN_POOL_DAG", limit=20)["matches"]
        target = next(r for r in rows if r["path"].endswith("DMD_LIVEWAN_POOL_DAG_20260907.md"))
        self.assertEqual(target["review_status"], "DISCOVERED_NOT_CURATED")
        self.assertFalse(target["automatic_application"])

    def test_h100_query_retrieves_h100_records(self):
        packets = m.recall("H100 pinned offload", {"training_mode": "training"})["matches"]
        cases = [c for p in packets for group in p["cases"].values() for c in group]
        self.assertTrue(any(c["provenance"]["collection"] == "Helios H100 history" for c in cases))

    def test_external_inference_scope_and_guidance_remain_visible(self):
        packets = m.recall("graph capture", {"training_mode": "training"})["matches"]
        cases = [c for p in packets for group in p["cases"].values() for c in group]
        c = next(c for c in cases if c["id"] == "rb-inference-graph-shape-miss")
        self.assertIn("training_mode", c["context_comparison"]["conflicting"])
        self.assertEqual(c["evidence_status"], "EXTERNAL_REPORTED_NOT_REPRODUCED")
        self.assertFalse(c["automatic_application"])

    def test_partial_h100_label_is_unknown_capacity_not_conflict(self):
        result = m.condition_match({"hardware": "8x H100-80GB"}, {"hardware": "H100", "physical_gpus": 8})
        self.assertEqual(result["conflicting"], [])
        self.assertEqual(result["unknown"], ["gpu_memory_gb"])
        self.assertEqual(set(result["matched"]), {"gpu_family", "physical_gpus"})

    def test_different_gpu_family_still_conflicts(self):
        result = m.condition_match({"hardware": "8x H100-80GB"}, {"gpu_family": "A100", "physical_gpus": 8})
        self.assertEqual(result["conflicting"], ["gpu_family"])

    def test_hardware_count_does_not_invent_nodes_or_ranks(self):
        normalized = m.normalize_context({"hardware": "8x H100"})
        self.assertEqual(normalized, {"gpu_family": "H100", "physical_gpus": 8})

    def test_explicit_two_node_topology_can_be_parsed(self):
        normalized = m.normalize_context({"hardware": "2x8 A100-40GB"})
        self.assertEqual(normalized, {"gpu_family": "A100", "gpu_memory_gb": 40,
                                     "physical_gpus": 16, "nodes": 2, "gpus_per_node": 8})

    def test_contradictory_explicit_hardware_facts_fail_closed(self):
        with self.assertRaises(m.InputError):
            m.recall("驻留", {"hardware": "H100", "gpu_family": "A100"})

    def test_contradictory_structured_allocation_is_not_a_partial_match(self):
        for label in ({}, {"hardware": "H100"}, {"hardware": "8x H100"}):
            with self.subTest(label=label), self.assertRaises(m.InputError):
                m.recall("驻留", {**label, "nodes": 2, "gpus_per_node": 8, "physical_gpus": 8})

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
