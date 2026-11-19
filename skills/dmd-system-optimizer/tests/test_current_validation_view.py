"""Current entrypoints must not inherit an archived run's numerical label."""
import csv
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

    def test_summary_matches_paper_workload_order_and_comparison_scope(self):
        guide = (REPO / "docs/guides/performance_validation.md").read_text()
        a100 = guide.split("## A100\n", 1)[1].split("## H100", 1)[0]
        h100 = guide.split("## H100\n", 1)[1].split("## Planning", 1)[0]
        for table in (a100, h100):
            for i in range(1, 7):
                self.assertIn(f"| W{i} |", table)
        self.assertIn("| W1 | 912 | OOM | 1054 | 767 | 1.19× |", a100)
        self.assertIn("| W4 | 55 | 45 | 53 | 45 | 1.22× |", h100)
        self.assertIn("artifacts/results/planner.json", guide)
        manifest = json.loads((REPO / "artifacts/results/manifest.json").read_text())
        for hardware, gpu_count in (("a100", 16), ("h100", 8)):
            scope = manifest["measurement_scope"][hardware]
            self.assertEqual(scope["window"], "complete_cycle")
            self.assertEqual(scope["physical_gpus"], gpu_count)
        result_files = {f["path"] for f in manifest["files"]}
        for retired in ("official.csv", "h100_sources.json", "gan_nomerge_retime.json"):
            self.assertNotIn(f"artifacts/results/{retired}", result_files)

    def test_readmes_use_the_result_workload_names(self):
        with (REPO / "artifacts/results/a100.csv").open(newline="") as source:
            names = {row["id"]: row["name"] for row in csv.DictReader(source)}
        catalog = json.loads((REPO / "src/distillschedule/workloads/catalog.json").read_text())
        readme = (REPO / "README.md").read_text()
        recipe_index = (REPO / "recipes/README.md").read_text()
        for item in catalog["workloads"]:
            identifier = item["paper_id"]
            name = names[identifier]
            recipe = item["recipe"]
            self.assertIn(f"[{identifier} {name}]({recipe})", readme)
            self.assertIn(f"| {identifier} | [{name}]({item['slug']}/README.md) |", recipe_index)
            self.assertEqual((REPO / recipe).read_text().splitlines()[0], f"# {identifier} — {name}")

    def test_full_sequence_recipe_stays_separate_from_paper_plan_pairs(self):
        recipe = (REPO / "recipes/h3_fullseq/README.md").read_text()
        self.assertIn("exactly two fixed-contract cycles", recipe)
        self.assertIn("native numerical check", recipe)
        self.assertIn("training.json", recipe)
        catalog = json.loads((REPO / "src/distillschedule/workloads/catalog.json").read_text())
        full_sequence = next(row for row in catalog["workloads"] if row["slug"] == "h3_fullseq")
        self.assertEqual((full_sequence["id"], full_sequence["paper_id"]), ("W5", "W1"))
        results = json.loads((REPO / "artifacts/results/planner.json").read_text())
        self.assertEqual(results["comparison_source"], "Figure 9")
        self.assertEqual([row["id"] for row in results["rows"]], ["W2", "W3", "W4", "W5", "W6"])


if __name__ == "__main__":
    unittest.main()
