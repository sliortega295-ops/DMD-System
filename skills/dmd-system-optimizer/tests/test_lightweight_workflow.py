"""No-ledger CPU workflow and prompt contracts, not LLM/GPU performance tests.

The native scalar checker below really executes values and analytic derivatives.
It is deliberately plain JSON, not JUnit. Timing arrays are synthetic input for
our descriptive helper, never observations of training speed or significance.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
GOOD = "def forward(x): return (x + 1.) ** 2\ndef backward(x): return 2. * (x + 1.)\n"
BAD = "def forward(x): return (x + 1.) ** 2\ndef backward(x): return 0.\n"
CHECKER = '''import json, math, sys
from pathlib import Path
import workload
xs = [-2., -0.5, 0., 1., 3.]
forward = all(math.isclose(workload.forward(x), x*x + 2*x + 1., rel_tol=1e-12, abs_tol=1e-12) for x in xs)
backward = all(math.isclose(workload.backward(x), 2*x + 2., rel_tol=1e-12, abs_tol=1e-12) for x in xs)
result = {"forward": forward, "backward": backward, "status": "PASS" if forward and backward else "FAIL", "scope": "actual scalar CPU fixture only; not training/autograd validation"}
Path(sys.argv[1]).write_text(json.dumps(result))
print(json.dumps(result))
sys.exit(0 if forward and backward else 1)
'''


class LightweightWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "external-project"
        self.project.mkdir()
        (self.project / "workload.py").write_text(GOOD)
        (self.project / "check_native.py").write_text(CHECKER)
        self.commands = []

    def native(self, name):
        command = [sys.executable, "-B", "check_native.py", name + ".json"]
        self.commands.append(command)
        result = subprocess.run(command, cwd=self.project, text=True, capture_output=True, timeout=15)
        (self.project / (name + ".stdout")).write_text(result.stdout + result.stderr)
        return result, json.loads((self.project / (name + ".json")).read_text())

    def note(self, body):
        with (self.project / "WORK_LOG.md").open("a") as f:
            f.write(body + "\n")

    def assert_no_management_files(self):
        self.assertFalse(list(self.root.rglob("*.sqlite")))
        self.assertFalse(list(self.root.rglob("*.xml")))
        self.assertFalse(list(self.root.rglob("contract.json")))
        self.assertFalse(any("dmd_campaign.py" in " ".join(c) for c in self.commands))
        self.assertFalse(any("dmd_receipts.py" in " ".join(c) for c in self.commands))

    def compare(self, control, candidate, unprofiled=True):
        # Copy the actual helper to a different directory; no repository cwd needed.
        installed = self.root / "installed-helpers"
        installed.mkdir(exist_ok=True)
        shutil.copy2(SCRIPTS / "dmd_perf.py", installed / "dmd_perf.py")
        values = {"timer_scope": "synthetic complete cycle", "semantic_contract": "same-work fixture",
                  "hardware_id": "synthetic-not-measured", "unprofiled": unprofiled,
                  "control_seconds": control, "candidate_seconds": candidate, "paired": True}
        path = self.project / "descriptive-input.json"
        path.write_text(json.dumps(values))
        command = [sys.executable, str(installed / "dmd_perf.py"), "compare", "--input", str(path)]
        self.commands.append(command)
        return subprocess.run(command, cwd=self.project, text=True, capture_output=True, timeout=15)

    def test_native_checks_and_plain_log_need_no_ledger_or_adapter(self):
        run, result = self.native("quick")
        self.assertEqual(run.returncode, 0)
        self.assertEqual(result["status"], "PASS")
        self.note("Reused scalar implementation; quick.json passed within scalar scope. Training timing NOT_RUN.")
        final, result = self.native("final-combination")
        self.assertEqual(final.returncode, 0)
        self.assertTrue(result["backward"])
        self.assert_no_management_files()

    def test_loss_matching_does_not_hide_bad_gradient_and_can_be_reverted(self):
        (self.project / "workload.py").write_text(BAD)
        run, result = self.native("failed-candidate")
        self.assertEqual(run.returncode, 1)
        self.assertTrue(result["forward"])
        self.assertFalse(result["backward"])
        original_failure = (self.project / "failed-candidate.json").read_bytes()
        self.note("Reverted: same outputs but wrong derivative. Evidence: failed-candidate.json.")
        (self.project / "workload.py").write_text(GOOD)
        restored, _ = self.native("restored-reference")
        self.assertEqual(restored.returncode, 0)
        self.assertEqual((self.project / "failed-candidate.json").read_bytes(), original_failure)
        self.assert_no_management_files()

    def test_one_percent_single_pair_is_descriptive_not_rejected_or_certified(self):
        run = self.compare([100.], [99.])
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertAlmostEqual(result["latency_reduction_percent"], 1.)
        self.assertEqual(result["formal_acceptance"], "NOT_DECIDED")
        self.assertEqual(result["control_n"], 1)
        self.assertNotIn("default_enable", result)
        self.assert_no_management_files()

    def test_valid_slow_sample_remains_visible_in_descriptive_comparison(self):
        run = self.compare([100., 100.], [99., 102.])
        result = json.loads(run.stdout)
        self.assertEqual(result["paired_saving_seconds"], [1., -2.])
        self.assertEqual(result["formal_acceptance"], "NOT_DECIDED")
        self.assert_no_management_files()

    def test_profiled_timing_is_still_rejected(self):
        run = self.compare([100.], [90.], unprofiled=False)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(json.loads(run.stderr)["status"], "INVALID")

    def test_capacity_or_correctness_only_handoff_needs_no_timing_baseline(self):
        run, result = self.native("correctness-only")
        self.assertEqual(run.returncode, 0)
        self.note("Correctness-only scalar check passed. Capacity benefit NOT_MEASURED; latency UNKNOWN. No speed claim.")
        self.assertNotIn("seconds", result)
        self.assert_no_management_files()

    def test_stopped_handoff_preserves_existing_outputs_without_more_execution(self):
        self.native("completed-quick-check")
        before = (self.project / "completed-quick-check.json").read_bytes()
        launches = len(self.commands)
        # Tests the artifact handoff, not an autonomous agent obeying a stop.
        self.note("User stop / resource limit: no additional work. Preserve reference and candidate; final training validation NOT_RUN.")
        self.assertEqual(len(self.commands), launches)
        self.assertEqual((self.project / "completed-quick-check.json").read_bytes(), before)
        self.assert_no_management_files()


class LightweightInstructionTests(unittest.TestCase):
    def test_default_prompt_names_lightweight_native_workflow(self):
        text = (SKILL / "agents/openai.yaml").read_text()
        self.assertIn("lightweight default workflow", text)
        self.assertIn("WORK_LOG", text)
        self.assertIn("explicit resource limits", text)

    def test_skill_does_not_require_forms_or_automatic_full_validation(self):
        text = (SKILL / "SKILL.md").read_text()
        self.assertIn("No campaign", text)
        self.assertIn("advanced opt-in tools", text)
        self.assertIn("no default six-block", text.lower())
        self.assertIn("100 steps solely", text)
        self.assertIn("best verified version", text)

    def test_optional_tools_do_not_relax_existing_protocols(self):
        text = (SKILL / "SKILL.md").read_text()
        self.assertIn("Explicit user caps and stop requests are hard limits", text)
        self.assertIn("Do not remove required work/gradients", text)
        self.assertIn("Keep already adopted frozen protocols", text)
        self.assertIn("alter a checker to pass", text)

    def test_memory_failure_has_non_destructive_log_fallback(self):
        text = (SKILL / "references/reuse-memory.md").read_text()
        self.assertIn("A plain WORK_LOG is the default memory", text)
        self.assertIn("If lookup or indexing fails", text)
        self.assertIn("Do not delete", text)
        self.assertIn("optional, not a prerequisite", text)

    def test_default_effort_is_advisory_but_permission_is_not(self):
        text = (SKILL / "SKILL.md").read_text()
        self.assertIn("no default two-attempt ban", text.lower())
        self.assertIn("An unspecified", text)
        self.assertIn("budget is not unlimited permission", text)
        self.assertIn("Leave time for final checks", text)
        self.assertIn("not indefinitely block a useful small change", text)


if __name__ == "__main__":
    unittest.main()
