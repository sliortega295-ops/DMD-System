"""CPU integration checks; no model construction, CUDA, SSH or training."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from distillschedule.workloads.best_known.registry import file_sha256, load_profile, resolve_profile
from distillschedule.workloads.best_known.evidence import summary_command, bind_livewan_gate

WORKLOAD = "livewan_crossscale_pipeline_best"
VARIANT = "c34-continuous-b16-residual-layout"
SOURCE = "4c994edca7d83ff553cab6025d8b93b279cd1634"


class LiveWanC34CanonicalTests(unittest.TestCase):
    def assets(self, directory: Path) -> Path:
        value = json.loads((ROOT / "configs/distillschedule/best_known/assets_examples" / (WORKLOAD + ".json")).read_text())
        source = directory / "source.json"
        source.write_text(json.dumps({"status": "PASS", "source_root": "/srv/prepared-wan"}))
        value["assets"]["source_receipt"]["path"] = str(source)
        value["correctness"] = {"path": "/srv/receipts/matched-gate.json", "sha256": "1" * 64}
        path = directory / "assets.json"
        path.write_text(json.dumps(value))
        return path

    def test_c34_identity_and_claim_boundary(self):
        p = load_profile(WORKLOAD)
        self.assertEqual(p["profile_version"], 2)
        self.assertEqual(p["contract"]["variant"], VARIANT)
        self.assertEqual(p["contract"]["continuous_cycles"], 2)
        self.assertEqual(p["contract"]["actual_global_batch"], 16)
        self.assertEqual(p["contract"]["final_versions"], {"generator": 2, "critic": 10, "ema": 0})
        self.assertEqual(p["source"]["performance_commit"], SOURCE)
        self.assertEqual(p["historical"]["strict_numeric"], "REVIEW")
        self.assertAlmostEqual(p["historical"]["seconds"], 47.01868359375)
        self.assertEqual(p["historical"]["previous_canonical"]["seconds"], 54.441603646)
        self.assertFalse(p["historical"]["formal_speedup_eligible"])

    def test_real_resolver_routes_two_cycle_worker_and_summarizer(self):
        for mode, native in [("correctness", "trajectory"), ("diagnostic-timing", "two-cycle-timing")]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                r = resolve_profile(WORKLOAD, assets_path=self.assets(Path(td)), mode=mode, run_id="c34-cpu-test")
                self.assertEqual(len(r["nodes"]), 2)
                for n in r["nodes"]:
                    self.assertEqual(n["command"][4], native)
                    env = n["environment"]
                    self.assertEqual(env["DISTILLSCHEDULE_RUN_KIND"], native)
                    self.assertEqual(env["LIVEWAN_PIPELINE_VARIANT"], VARIANT)
                    self.assertEqual(env["LIVEWAN_TWO_CYCLE_SEED"], "20260830")
                    self.assertEqual(env["LIVEWAN_TRAJECTORY_CYCLES"], "2")
                    self.assertEqual(env["LIVEWAN_TWO_CYCLE_ARM_ID"], WORKLOAD + "-v2")
                    self.assertEqual(env["LIVEWAN_TRAJECTORY_ARM_ID"], env["LIVEWAN_TWO_CYCLE_ARM_ID"])
                    self.assertTrue(n["resolved_native_plan"]["metadata"]["generator_residual_layout"]["enabled"])
                command = summary_command(r)
                expected = "summarize_livewan_trajectory_world16.py" if mode == "correctness" else "summarize_livewan_two_cycle_timing_world16.py"
                self.assertEqual(Path(command[1]).name, expected)
                self.assertNotIn("--run-kind", command)
                self.assertNotIn("--plan", command)

    def test_strict_timing_not_silently_admitted(self):
        with self.assertRaisesRegex(ValueError, "not authorized"):
            resolve_profile(WORKLOAD, mode="timing")

    def test_native_gate_binding_preserves_review_and_rejects_wrong_arm(self):
        with tempfile.TemporaryDirectory() as td:
            r = resolve_profile(WORKLOAD, assets_path=self.assets(Path(td)), run_id="correctness-run")
            gate = {
                "schema_version": "distillschedule.livewan_two_cycle_timing_gate.v2",
                "status": "PASS", "claimable_for_two_cycle_timing": True,
                "expected_head": r["integration_head"], "seed": 20260830,
                "numerical_review": {"state_comparisons": [{"status": "REVIEW"}]},
                "arms": {"pipeline": {"run_id": r["run_id"], "arm_id": WORKLOAD + "-v2",
                                      "pipeline_variant": VARIANT}},
            }
            bound = bind_livewan_gate(gate, r)
            self.assertEqual(bound["numerical_review"], gate["numerical_review"])
            self.assertNotIn("integration_binding", gate)
            self.assertEqual(bound["integration_binding"]["profile_sha256"], r["profile_sha256"])
            gate["arms"]["pipeline"]["arm_id"] = "another-arm"
            with self.assertRaisesRegex(ValueError, "arm binding differs"):
                bind_livewan_gate(gate, r)

    def test_actual_unified_dry_run_has_no_execution(self):
        with tempfile.TemporaryDirectory() as td:
            assets = self.assets(Path(td))
            proc = subprocess.run([
                sys.executable, str(ROOT / "scripts/distillctl.py"), "workloads", "run",
                "--workload", WORKLOAD, "--assets-config", str(assets),
                "--mode", "diagnostic-timing", "--run-id", "c34-no-gpu", "--dry-run",
            ], cwd=ROOT, capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            receipt = json.loads(proc.stdout)
            self.assertEqual(receipt["scope"], "RESOLVED_NOT_EXECUTED")
            self.assertFalse(receipt["historical_evidence"]["valid_for_integration_head"])

    def test_other_four_profiles_unchanged(self):
        # File hashes at merged PR26; also works in a shallow CI checkout.
        expected = {'wan13b_nogan_resident_best': '8a1bda0851c42f576b2c3358cc07ec2873fe641c82b7e020b876864fe8141775', 'helios14b_nogan_best': '00716145197fa0ff9b0fce2557b732ef88d66864a1068d463bbf2d07f2c5783c', 'helios14b_gan_best': '143896a4c51a062fba49ab223290e023f137ba3718941d7649954d2872a89e4b', 'minimax_h3_grid1000_best': 'f47e43a63867785bacdbf54fd9b7fe42dc4f34ff5acd2e5fa26b815b5ab07cac'}
        for name, sha in expected.items():
            path = ROOT / "configs/distillschedule/best_known" / (name + ".json")
            self.assertEqual(file_sha256(path), sha, name)



if __name__ == "__main__":
    unittest.main()
