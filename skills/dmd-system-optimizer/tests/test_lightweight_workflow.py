"""Actual scalar CPU checks and lightweight delivery contracts, not GPU/LLM tests."""
import ast
import json
from pathlib import Path
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
f = all(math.isclose(workload.forward(x), x*x + 2*x + 1., rel_tol=1e-12, abs_tol=1e-12) for x in xs)
b = all(math.isclose(workload.backward(x), 2*x + 2., rel_tol=1e-12, abs_tol=1e-12) for x in xs)
r = {"forward": f, "backward": b, "status": "PASS" if f and b else "FAIL", "scope": "scalar CPU only; no training/autograd qualification"}
Path(sys.argv[1]).write_text(json.dumps(r))
print(json.dumps(r))
sys.exit(0 if f and b else 1)
'''


class LightweightWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)
        (self.project / "workload.py").write_text(GOOD)
        (self.project / "check_native.py").write_text(CHECKER)
        self.launches = 0

    def native(self, name):
        self.launches += 1
        run = subprocess.run([sys.executable, "-B", "check_native.py", name + ".json"],
                             cwd=self.project, capture_output=True, text=True, timeout=15)
        return run, json.loads((self.project / (name + ".json")).read_text())

    def note(self, text):
        with (self.project / "WORK_LOG.md").open("a") as f:
            f.write(text + "\n")

    def no_ledger(self):
        self.assertFalse(list(self.project.rglob("*.sqlite")))
        self.assertFalse(list(self.project.rglob("*.xml")))
        self.assertFalse((self.project / "contract.json").exists())

    def compare(self, control, candidate, unprofiled=True):
        p = self.project / "pair.json"
        p.write_text(json.dumps(dict(timer_scope="synthetic cycle", semantic_contract="fixed-work",
            hardware_id="synthetic", unprofiled=unprofiled, paired=True,
            control_seconds=control, candidate_seconds=candidate)))
        return subprocess.run([sys.executable, str(SCRIPTS / "dmd_perf.py"), "compare", "--input", str(p)],
                              cwd=self.project, capture_output=True, text=True, timeout=15)

    def test_native_checker_and_log_without_adapter(self):
        run, result = self.native("quick")
        self.assertEqual(run.returncode, 0)
        self.assertEqual(result["status"], "PASS")
        self.note("Scalar check passed; training timing NOT_RUN.")
        run, result = self.native("final-combination")
        self.assertEqual(run.returncode, 0)
        self.assertTrue(result["backward"])
        self.no_ledger()

    def test_equal_outputs_wrong_gradient_rejected_and_rollback(self):
        (self.project / "workload.py").write_text(BAD)
        run, result = self.native("failure")
        self.assertEqual(run.returncode, 1)
        self.assertTrue(result["forward"]); self.assertFalse(result["backward"])
        before = (self.project / "failure.json").read_bytes()
        (self.project / "workload.py").write_text(GOOD)
        self.assertEqual(self.native("restored")[0].returncode, 0)
        self.assertEqual((self.project / "failure.json").read_bytes(), before)
        self.no_ledger()

    def test_one_percent_is_descriptive_not_certified(self):
        run = self.compare([100.], [99.])
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertAlmostEqual(result["latency_reduction_percent"], 1)
        self.assertEqual(result["formal_acceptance"], "NOT_DECIDED")
        self.assertNotIn("default_enable", result)
        self.no_ledger()

    def test_slow_sample_stays_visible(self):
        result = json.loads(self.compare([100., 100.], [99., 102.]).stdout)
        self.assertEqual(result["paired_saving_seconds"], [1, -2])

    def test_profiled_timing_rejected(self):
        self.assertEqual(self.compare([100.], [90.], False).returncode, 2)

    def test_correctness_only_needs_no_fake_timing(self):
        run, result = self.native("correctness-only")
        self.assertEqual(run.returncode, 0)
        self.assertNotIn("seconds", result)
        self.note("Capacity and training performance NOT_MEASURED.")
        self.no_ledger()

    def test_stop_handoff_preserves_outputs_without_extra_run(self):
        self.native("completed")
        before = (self.project / "completed.json").read_bytes()
        launches = self.launches
        self.note("User stop: retain best version and candidate; final training checks NOT_RUN.")
        self.assertEqual(self.launches, launches)
        self.assertEqual((self.project / "completed.json").read_bytes(), before)
        self.no_ledger()


class LightweightInstructionTests(unittest.TestCase):
    def skill(self):
        return " ".join((SKILL / "SKILL.md").read_text().split())

    def test_entry_prompt_remains_lightweight(self):
        text = (SKILL / "agents/openai.yaml").read_text()
        for value in ("lightweight default workflow", "WORK_LOG", "explicit resource limits"):
            self.assertIn(value, text)

    def test_no_forms_or_mechanical_long_validation(self):
        text = self.skill()
        for value in ("No campaign", "no longer shipped", "no default six-block", "100 steps solely", "best verified version"):
            self.assertIn(value, text)

    def test_correctness_and_explicit_limits_stay_binding(self):
        text = self.skill()
        for value in ("Explicit user caps and stop requests are hard limits", "Preserve required work and gradients",
                      "Keep already adopted frozen protocols", "alter a checker to pass"):
            self.assertIn(value, text)

    def test_memory_failure_has_log_fallback(self):
        text = " ".join((SKILL / "references/reuse-memory.md").read_text().split())
        for value in ("A plain WORK_LOG is the durable experiment record", "If lookup or indexing fails", "Do not delete", "optional, not a prerequisite"):
            self.assertIn(value, text)

    def test_effort_is_guidance_not_a_permission_service(self):
        text = self.skill()
        for value in ("no default two-attempt ban", "budget is not unlimited permission",
                      "Leave time for final checks", "not indefinitely block a useful small change"):
            self.assertIn(value, text)

    def test_retired_management_modules_and_imports_absent(self):
        retired = {"dmd_campaign", "dmd_receipts", "dmd_reuse"}
        for name in retired:
            self.assertFalse((SCRIPTS / (name + ".py")).exists(), name)
        for path in SCRIPTS.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
                self.assertFalse(retired.intersection(names), str(path))
        for path in ("references/campaign-tools.md", "references/checker-receipts.md",
                     "assets/ledger-contract.template.json", "assets/run-reservation.template.json"):
            self.assertFalse((SKILL / path).exists(), path)
        self.assertTrue((SKILL / "references/helpers.md").is_file())


if __name__ == "__main__":
    unittest.main()
