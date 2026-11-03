"""Current entrypoints must not inherit an archived run's numerical label."""
import json
from pathlib import Path
import sys
import unittest

SKILL = Path(__file__).resolve().parents[1]
REPO = SKILL.parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
import dmd_current


class CurrentValidationViewTests(unittest.TestCase):
    def test_official_navigation_uses_current_guide_without_inventing_pass(self):
        for query, expected in (("SD1.5 DMD2", "dmd2_sd15"),
                                ("官方Self-Forcing", "self_forcing_official")):
            result = dmd_current.lookup(query)
            row = result["matches"][0]
            self.assertEqual(row["id"], expected)
            self.assertTrue(row["profiles"])
            self.assertTrue((REPO / row["validation_guide"]).is_file())
            self.assertEqual(row["profile_history_source"], dmd_current.PROFILES)
            self.assertFalse(row["tests_run"])
            self.assertFalse(result["automatic_application"])
            for profile in row["profiles"]:
                self.assertNotIn("numerical_status", profile)
                self.assertNotIn("validation_status", profile)

    def test_related_capabilities_do_not_repeat_superseded_review_labels(self):
        for query in ("SD1.5 DMD2 resident tree fused Adam",
                      "Official Self-Forcing FSDP2 R6"):
            result = dmd_current.lookup(query)
            self.assertNotIn("REVIEW", json.dumps(result))
            capabilities = result["related_capabilities_require_recipe_check"]
            self.assertTrue(any(c.get("historical_evidence_source") == dmd_current.STRATEGIES
                                for c in capabilities))
        stored = json.loads(dmd_current.INDEX.read_text())
        self.assertTrue(any("HISTORICAL_NATIVE_GPU_REVIEW" in c["evidence"]
                            for c in stored["capabilities"]))

    def test_lookup_preserves_archived_profile_data(self):
        before = dmd_current.INDEX.read_bytes()
        data = json.loads(before)
        profiles = [p for row in data["workflows"] for p in row["profiles"]]
        self.assertTrue(any(p.get("numerical_status") == "REVIEW" for p in profiles))
        dmd_current.lookup("DMD2")
        self.assertEqual(before, dmd_current.INDEX.read_bytes())

    def test_generated_navigation_matches_the_real_source_tree(self):
        saved = json.loads(dmd_current.INDEX.read_text())
        self.assertEqual(saved, dmd_current.build_index(REPO, saved["source_revision"]))
        self.assertEqual(len(saved["workflows"]), 14)

    def test_summary_covers_all_workloads_and_keeps_timer_and_sources(self):
        guide = (REPO / "docs/guides/performance_validation.md").read_text()
        self.assertIn("Cycle 2 (s)", guide)
        self.assertIn("H100", guide)
        self.assertIn("G1/C5/EMA5 Full5", guide)
        self.assertIn("gan_nomerge_retime.json", guide)
        self.assertIn("artifacts/results/planner.json", guide)
        for workload in ("W1", "W2", "W3", "W4", "W5", "W6"):
            self.assertIn(workload, guide)
        for value in ("7.514", "3.475", "50.739", "30.959", "36.175",
                      "84.260", "42.450", "255.671", "116.254"):
            self.assertIn(value, guide)
        self.assertNotIn("767.020", guide)  # Retired approximate timing is not continuous evidence.

    def test_recipe_is_concise_and_continuous_results_do_not_relabel_old_timing(self):
        recipe = (REPO / "recipes/h3_fullseq/README.md").read_text()
        self.assertIn("exactly two fixed-contract cycles", recipe)
        self.assertIn("native numerical check", recipe)
        self.assertIn("training.json", recipe)
        evidence = json.loads((REPO / "artifacts/results/planner.json").read_text())
        self.assertEqual(evidence['original_summary_sha256'],
                         '6064d20602bc4e97863c76e5958949153345b0aac2e75304c81db3c37191c842')
        w5 = next(row for row in evidence['rows'] if row['id'] == 'W5')
        self.assertIn('Continuous', w5['name'])
        self.assertIsNone(w5['manual_C2'])
        self.assertIsNone(w5['selected_C2'])


if __name__ == "__main__":
    unittest.main()
