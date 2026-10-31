"""External source reuse -> actual scalar checker -> plain log -> private Memory.

This synthetic scalar protocol fixture does not qualify FSDP/autograd/GPU speed.
No synthetic speed measurements, campaign ledger or receipt adapter are needed.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
REPO = SKILL.parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
import dmd_knowledge as knowledge
from dmd_perf import sha_file


class ExternalReuseTests(unittest.TestCase):
    def test_reuse_keep_or_rollback_then_recall_in_new_process(self):
        for inject_error in (False, True):
            with self.subTest(inject_error=inject_error), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                project = root / "user-project"
                shutil.copytree(SKILL / "tests/fixtures/external_cpu", project)
                retrieved = knowledge.recall("参数上卡总是在等，怎么优化？", {}, repo=REPO, limit=3)
                case = next(c for p in retrieved["matches"] for group in p["cases"].values() for c in group
                            if any(i["id"] == "fsdp-phase-retention" for i in c["navigation"]["implementations"]))
                located = knowledge.locate_case(case["id"], {}, REPO, show_code=True)
                component = next(i for i in located["implementations"] if i["id"] == "fsdp-phase-retention")
                ref = component["entrypoints"][0]
                self.assertEqual(ref["symbol_status"], "SYMBOL_AND_SPAN_CHECKED")
                source = REPO / ref["path"]
                self.assertEqual(sha_file(source), ref["sha256"])
                shutil.copyfile(source, project / "borrowed_phase.py")
                original = (project / "workload.py").read_text()
                (project / "workload.py").write_text(original + '''
from borrowed_phase import fsdp2_variable_forward_phase
def execute(engine, inputs, enabled=False):
    with fsdp2_variable_forward_phase(engine, enabled=enabled, with_grad=False,
                                     expected_units=1, module_type=Unit) as (view, receipt):
        outputs = [view(x) for x in inputs]
    return outputs, receipt
''')
                inputs = [[1., 2., 3.], [3., 1., 2.], [-1., 0., 2.], [2., 4., 6.]]
                (project / "input.json").write_text(json.dumps(inputs))
                (project / "config.json").write_text(json.dumps({"inject_error": inject_error}))
                check = subprocess.run([sys.executable, "-B", "check_workload.py"], cwd=project,
                                       capture_output=True, text=True, timeout=30)
                self.assertEqual(check.returncode, 1 if inject_error else 0, check.stderr)
                raw = project / "checker.xml"
                cpu_gate = subprocess.run([sys.executable, str(REPO / "scripts/check_cpu_receipt.py"), str(raw),
                                           "--require", "numerical", "--require", "path"],
                                          capture_output=True, text=True, timeout=30)
                self.assertEqual(cpu_gate.returncode == 0, not inject_error)
                # Reuse the actual project checker; do not translate it into a second schema.
                decision = "REVERT" if check.returncode else "KEEP_SCALAR_FIXTURE_ONLY"
                (project / "WORK_LOG.md").write_text(
                    f"{decision}: native checker.xml and path.json; no training or speed claim.\n")
                def evidence(path, locator="whole artifact"):
                    return {"path": str(path), "sha256": sha_file(path), "locator": locator}
                snapshot = root / "candidate_source.py"
                shutil.copyfile(project / "workload.py", snapshot)
                failure_bytes = raw.read_bytes()
                if inject_error:
                    (project / "workload.py").write_text(original)
                    rollback = subprocess.run([sys.executable, "-B", "-c",
                        "import json,workload; from pathlib import Path; "
                        "xs=json.loads(Path('input.json').read_text()); "
                        "got,receipt=workload.execute(workload.Engine(),xs); "
                        "assert got == [sum(w*v for w,v in zip([2.,-3.,4.],x)) for x in xs]; "
                        "assert receipt['enabled'] is False"], cwd=project, capture_output=True, timeout=30)
                    self.assertEqual(rollback.returncode, 0, rollback.stderr)
                    self.assertEqual(raw.read_bytes(), failure_bytes)
                store = root / "private-memory"
                record = {"id": "scalar-reuse", "kind": "experience",
                    "summary": "参数上卡等待：scalar phase reuse; native CPU checks only, no speed claim",
                    "pattern": "parameter-residency", "strength": "observation",
                    "outcome": "incorrect" if inject_error else "success",
                    "conditions": {"hardware": "CPU", "grad_mode": "no_grad", "measurement_kind": "synthetic"},
                    "correctness_conditions": {"same_frozen_parameters": True},
                    "performance_conditions": {}, "decision": decision,
                    "provenance": {"measurement_kind": "synthetic", "scope": "scalar protocol fixture, not training"},
                    "retry": "validate real target gradients, capacity and performance before reuse",
                    "failure_scope": "scalar negative control" if inject_error else "no GPU/autograd validation",
                    "evidence": [evidence(raw), evidence(project / "path.json"), evidence(project / "WORK_LOG.md")],
                    "source": {**evidence(snapshot, "execute adapter"), "symbol": "execute"},
                    "reuse_decision": {"implementation_id": "fsdp-phase-retention", "mode": "minimal_adaptation"}}
                written = knowledge.put(store, record, root)
                query = subprocess.run([sys.executable, str(SKILL / "scripts/dmd_knowledge.py"), "recall", "--brief",
                    "--store", str(store), "--query", "参数上卡总是在等，怎么优化？"],
                    check=True, capture_output=True, text=True, cwd=project, timeout=30)
                self.assertIn(written["qualified_id"], query.stdout)
                self.assertIn("SYNTHETIC_WORKFLOW_ONLY", query.stdout)
                self.assertIn("WORKFLOW_REFERENCE_ONLY", query.stdout)
                saved = json.loads(next(store.glob("*.json")).read_text())
                self.assertEqual(saved["outcome"], "incorrect" if inject_error else "success")
                self.assertEqual(saved["source"]["sha256"], sha_file(snapshot))
                self.assertFalse(list(root.rglob("*.sqlite")))
                self.assertFalse(list(root.rglob("contract.json")))


if __name__ == "__main__":
    unittest.main()
