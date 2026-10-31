"""Read-only helper regression tests. All timing inputs are synthetic."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import dmd_perf as perf


class ArithmeticTests(unittest.TestCase):
    def pair(self, **changes):
        return dict(timer_scope="synthetic cycle", semantic_contract="fixed-work",
                    hardware_id="CPU fixture", unprofiled=True, paired=True,
                    control_seconds=[100., 100.], candidate_seconds=[99., 102.], **changes)

    def test_target_keeps_original_baseline(self):
        result = perf.target(120, 100, 1.5)
        self.assertEqual(result["target_seconds"], 80)
        self.assertEqual(result["remaining_saving_seconds"], 20)
        self.assertAlmostEqual(result["target_latency_reduction_percent"], 100/3)

    def test_target_met(self):
        self.assertEqual(perf.target(120, 70, 1.5)["remaining_saving_seconds"], 0)

    def test_bad_numbers(self):
        for value in (None, True, float("nan"), float("inf"), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                perf.number(value, "fixture")

    def test_zero_runtime_rejected(self):
        with self.assertRaises(ValueError):
            perf.target(0, 100, 1.5)

    def test_descriptive_comparison_keeps_slow_sample(self):
        result = perf.compare(self.pair())
        self.assertEqual(result["paired_saving_seconds"], [1, -2])
        self.assertEqual(result["formal_acceptance"], "NOT_DECIDED")
        self.assertNotIn("default_enable", result)

    def test_one_percent_not_rejected_or_certified(self):
        data = self.pair(); data.update(control_seconds=[100.], candidate_seconds=[99.])
        result = perf.compare(data)
        self.assertAlmostEqual(result["latency_reduction_percent"], 1)
        self.assertEqual(result["formal_acceptance"], "NOT_DECIDED")

    def test_profiled_input_rejected(self):
        data = self.pair(); data["unprofiled"] = False
        with self.assertRaises(ValueError):
            perf.compare(data)

    def test_unequal_pairs_rejected(self):
        data = self.pair(); data["candidate_seconds"] = [99.]
        with self.assertRaises(ValueError):
            perf.compare(data)

    def test_dag_branch_switch(self):
        result = perf.dag_model({"nodes": [{"id": "a", "seconds": 10}, {"id": "b", "seconds": 8}],
                                 "candidate_seconds": {"a": 5}})
        self.assertEqual(result["base_model_seconds"], 10)
        self.assertEqual(result["candidate_model_seconds"], 8)
        self.assertEqual(result["candidate_critical_path"], ["b"])

    def test_dag_cycle_and_missing_dependency(self):
        for dep in ("a", "missing"):
            with self.subTest(dep=dep), self.assertRaises(ValueError):
                perf.dag_model({"nodes": [{"id": "a", "seconds": 1, "depends_on": [dep]}]})

    def test_old_permission_apis_absent(self):
        for name in ("gate", "rank", "bind_ledger", "campaign_target"):
            self.assertFalse(hasattr(perf, name), name)


class FileAndCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPTS / "dmd_perf.py"), *args],
                              text=True, capture_output=True, timeout=15)

    def test_json_duplicate_and_nonfinite_rejected(self):
        path = self.root / "bad.json"
        for body in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}', '[]'):
            path.write_text(body)
            with self.subTest(body=body), self.assertRaises(ValueError):
                perf.load(path)

    def test_evidence_hash_mismatch(self):
        path = self.root / "report.txt"; path.write_text("CPU fixture")
        ref = {"path": path.name, "sha256": perf.sha_file(path), "locator": "whole file"}
        self.assertEqual(perf.check_evidence([ref], self.root, "fixture"), [])
        path.write_text("changed")
        self.assertTrue(perf.check_evidence([ref], self.root, "fixture"))

    def test_cli_target(self):
        result = self.cli("target", "--baseline-seconds", "100", "--latency-reduction", ".5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["target_speedup"], 2)

    def test_cli_invalid_input(self):
        result = self.cli("compare", "--input", str(self.root / "missing.json"))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["status"], "INVALID")

    def test_help_has_only_lightweight_commands(self):
        result = self.cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("{inspect,target,dag,compare}", result.stdout)
        self.assertNotIn("--ledger", result.stdout)

    def test_removed_commands_fail_without_side_effects(self):
        before = list(self.root.iterdir())
        for command in ("rank", "gate"):
            self.assertEqual(self.cli(command).returncode, 2)
        self.assertEqual(list(self.root.iterdir()), before)

    def test_external_repository_inspection(self):
        def git(*args):
            subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True)
        git("init")
        (self.root / "README.md").write_text("external CPU fixture")
        git("add", "README.md")
        git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture")
        result = perf.inspect(self.root)
        self.assertEqual(result["mode"], "external-source")
        self.assertEqual(result["profiles"], [])
        self.assertFalse(result["gpu_validation_performed"])
        sub = self.root / "nested"; sub.mkdir()
        with self.assertRaises(ValueError):
            perf.inspect(sub)


if __name__ == "__main__":
    unittest.main()
