"""External project -> real CPU checker -> synthetic decision -> private Memory.

This exercises the original parameter-phase Python code against a scalar protocol
stand-in. Neither the fixture nor its synthetic timings qualify FSDP/CUDA speed.
"""
import hashlib
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
import dmd_campaign as campaign
import dmd_receipts as receipts
import dmd_reuse as reuse
from dmd_perf import sha_file, digest


class ExternalReuseTests(unittest.TestCase):
    def test_reuse_keep_or_rollback_then_recall_in_new_process(self):
        for inject_error in (False, True):
            with self.subTest(inject_error=inject_error), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                project = root / "user-project"
                shutil.copytree(SKILL / "tests/fixtures/external_cpu", project)
                # Discover a relevant case, then inspect the actual indexed implementation.
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
                baseline_source = hashlib.sha256(original.encode()).hexdigest()
                # Minimal integration; no package installation or target migration.
                (project / "workload.py").write_text(original + '''
from borrowed_phase import fsdp2_variable_forward_phase
def execute(engine, inputs, enabled=False):
    with fsdp2_variable_forward_phase(engine, enabled=enabled, with_grad=False,
                                     expected_units=1, module_type=Unit) as (view, receipt):
        outputs = [view(x) for x in inputs]
    return outputs, receipt
''')
                (project / "input.json").write_text(json.dumps([[1., 2., 3.], [3., 1., 2.], [-1., 0., 2.], [2., 4., 6.]]))
                (project / "config.json").write_text(json.dumps({"inject_error": inject_error}))
                source_id = digest({p.name: sha_file(p) for p in (project / "workload.py", project / "borrowed_phase.py")})
                checker = {"path": str(project / "check_workload.py"), "sha256": sha_file(project / "check_workload.py"),
                           "locator": "CPU scalar checker", "version": "scalar-v1"}
                def evidence(path, locator="whole artifact"):
                    return {"path": str(path), "sha256": sha_file(path), "locator": locator}
                input_ref = evidence(project / "input.json")
                levels = {"V0": "source_guard", "V1": "numerical", "V2": "resume", "compatibility": "path",
                          "fallback": "fallback", "composition": "composition"}
                policy = {"checkers": {"junit": checker, "aba-json": checker},
                          "checks": {k: {"tests": [v], "scope": "scalar CPU protocol only: " + v} for k, v in levels.items()}}
                contract = {"schema_version": 2, "campaign_id": "external-scalar", "workload_id": "four-scalar-forwards",
                    "semantic_contract": "same outputs and release lifecycle", "hardware_id": "CPU", "timer_scope": "toy-cycle",
                    "claim_class": "strict", "measurement_kind": "synthetic",
                    "baseline": {"id": "baseline", "source_identity": baseline_source, "seconds": 100., "evidence": [input_ref]},
                    "validators": [checker], "receipt_policy": policy, "budget": {"max_gpu_hours": 0, "max_full_runs": 20},
                    "validation_by_risk": {"stateful": ["V0", "V1", "V2"]},
                    "timing": {"repeat_count": 6, "max_candidates": 1, "minimum_effect_seconds": 0, "alpha": .05,
                               "max_control_relative_drift": .01, "max_attempts": 6, "replaceable_failures": []}}
                state = root / "ledger.sqlite"
                campaign.init(state, contract, root)
                proposal = {"candidate_id": "candidate", "source_identity": source_id, "control_id": "baseline", "risk": "stateful",
                    "config_identity": sha_file(project / "config.json"), "input_identity": sha_file(project / "input.json"),
                    "comparator_id": "baseline", "comparator_source_identity": baseline_source,
                    "validation_scope": "scalar outputs and release; no distributed/autograd claim",
                    "purpose": "validation", "gpu_hours": 0, "full_runs": 1}
                with campaign.connection(state) as db:
                    campaign.reserve(db, "numerical", proposal)
                check = subprocess.run([sys.executable, "check_workload.py"], cwd=project, capture_output=True)
                self.assertEqual(check.returncode, 1 if inject_error else 0)
                raw_ref = evidence(project / "checker.xml")
                adapter = receipts.adapt({"format": "junit", "checker": checker, "raw": raw_ref,
                    "selection": policy["checks"], "binding": campaign.binding(proposal, "numerical")}, root)
                # The repository's existing checker consumes this same real CPU output.
                cpu_gate = subprocess.run([sys.executable, str(REPO / "scripts/check_cpu_receipt.py"), str(project / "checker.xml"),
                                           "--require", "numerical", "--require", "path"], capture_output=True, text=True)
                self.assertEqual(cpu_gate.returncode == 0, not inject_error)
                with campaign.connection(state) as db:
                    campaign.settle(db, "numerical", {"outcome": "incorrect" if inject_error else "pass", "gpu_hours": 0,
                        "full_runs": 1, "evidence": [raw_ref], "receipts": [adapter]}, root)
                times = []
                if not inject_error:
                    for i in range(6):
                        run_id = "timing-" + str(i)
                        timing_proposal = {**proposal, "purpose": "timing", "full_runs": 3}
                        with campaign.connection(state) as db:
                            campaign.reserve(db, run_id, timing_proposal)
                        timing = {"run_id": run_id, "process_ids": [run_id + suffix for suffix in ("-a", "-b", "-a2")],
                                  "control_before": 100., "candidate": 99., "control_after": 100.}
                        bound = campaign.binding(timing_proposal, run_id)
                        path = root / (run_id + ".json")
                        path.write_text(json.dumps({"status": "PASS", "binding": bound, "timing": timing,
                            "unprofiled": True, "repeat_unit": "independent_processes", "measurement_kind": "synthetic"}))
                        receipt = receipts.adapt({"format": "aba-json", "checker": checker, "raw": evidence(path), "binding": bound}, root)
                        with campaign.connection(state) as db:
                            campaign.settle(db, run_id, {"outcome": "pass", "gpu_hours": 0, "full_runs": 3,
                                "evidence": [evidence(path)], "receipts": [receipt]}, root)
                        times.append(timing)
                result = {**{k: contract[k] for k in ("workload_id", "semantic_contract", "hardware_id", "timer_scope", "claim_class")},
                    "contract_sha256": digest(contract), "candidate_id": "candidate", "source_identity": source_id,
                    "control_id": "baseline", "validation_scope": proposal["validation_scope"], "checks": adapter["checks"],
                    "unprofiled": True, "repeat_unit": "independent_processes", "timings": times, "evidence": [raw_ref]}
                reuse_decision = {"schema_version": 1, "target_project": "independent scalar project", "candidate_id": "candidate",
                    "source_identity": source_id, "implementation_id": "fsdp-phase-retention", "existing_equivalent": "per-call materialization; no phase reuse",
                    "mode": "minimal_adaptation", "dependencies": ["stdlib scalar stand-in; no real FSDP"], "conflicts": [],
                    "integration": "copy hash-checked phase helper; wrap existing execute", "expected_path": "one materialization for four forwards",
                    "rollback": "enabled=False restores original engine calls", "status": "ROLLED_BACK" if inject_error else "VERIFIED",
                    "path_evidence": [evidence(project / "path.json")], "numerical_evidence": [raw_ref], "rollback_evidence": [raw_ref]}
                reuse.validate(reuse_decision, root)
                snapshot = root / "candidate_source.py"
                shutil.copyfile(project / "workload.py", snapshot)
                experience = {"summary": "参数上卡等待：scalar phase reuse CPU integration; synthetic timing only", "pattern": "parameter-residency",
                    "conditions": {"hardware": "CPU", "grad_mode": "no_grad"}, "correctness_conditions": {"same_frozen_parameters": True},
                    "performance_conditions": {"exposed_gather": True}, "retry": "measure actual target hardware after numerical parity",
                    "failure_scope": "scalar negative control" if inject_error else "no GPU validation", "reuse_decision": reuse_decision,
                    "source": {**evidence(snapshot, "execute adapter"), "symbol": "execute"}}
                bundle = {"result": result, "experience": experience}
                store = root / "private-memory"
                with campaign.connection(state) as db:
                    finished = campaign.finish(db, bundle, store, root)
                self.assertEqual(finished["decision"]["retention"], "DISCARD" if inject_error else "KEEP")
                campaign.flush_memory(state)
                # Rollback through the existing flag was actually checked above; remove integration on failure.
                if inject_error:
                    (project / "workload.py").write_text(original)
                query = subprocess.run([sys.executable, str(SKILL / "scripts/dmd_knowledge.py"), "recall", "--brief",
                    "--store", str(store), "--query", "参数上卡总是在等，怎么优化？"], check=True, capture_output=True, text=True, cwd=project)
                self.assertIn(finished["memory_id"], query.stdout)
                self.assertIn("SYNTHETIC_WORKFLOW_ONLY", query.stdout)
                self.assertIn("WORKFLOW_REFERENCE_ONLY", query.stdout)
                record = json.loads(next(store.glob("*.json")).read_text())
                self.assertEqual(record["conditions"]["measurement_kind"], "synthetic")
                self.assertEqual(record["outcome"], "incorrect" if inject_error else "success")
                self.assertEqual(record["source"]["sha256"], sha_file(snapshot))


if __name__ == "__main__":
    unittest.main()
