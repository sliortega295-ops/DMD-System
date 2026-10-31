"""Ten deterministic user journeys, not ten people, LLM agents or GPU trials."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
ROOT = SKILL.parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
import dmd_knowledge as knowledge
from dmd_perf import sha_file
spec = importlib.util.spec_from_file_location("journey_installer", ROOT / "tools/install_optimizer_skill.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class TenUserJourneys(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.store = self.home / "private-memory"

    def cli(self, *args, script="dmd_knowledge.py", skill=SKILL):
        run = subprocess.run([sys.executable, "-B", str(skill / "scripts" / script), *args],
                             cwd=self.home, capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr)
        return json.loads(run.stdout)

    def record(self, ident="local", **extra):
        evidence = self.home / "native.log"
        evidence.write_text("Synthetic fixture: local CPU checks only.\n")
        record = {"id": ident, "kind": "experience", "summary": "参数上卡等待的本地经验",
                  "pattern": "parameter-residency", "strength": "observation", "outcome": "inconclusive",
                  "conditions": {}, "correctness_conditions": {}, "performance_conditions": {},
                  "provenance": {"measurement_kind": "synthetic"},
                  "evidence": [{"path": str(evidence), "sha256": sha_file(evidence), "locator": "whole fixture"}], **extra}
        return knowledge.put(self.store, record, self.home)

    def test_01_first_time_cpu_only_install_uses_discoverable_directory(self):
        env = {**os.environ, "HOME": str(self.home), "CODEX_HOME": str(self.home / "legacy")}
        run = subprocess.run([sys.executable, "-S", str(ROOT / "tools/install_optimizer_skill.py")],
                             cwd=self.home, env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr)
        dest = self.home / ".agents/skills/dmd-system-optimizer"
        self.assertEqual(Path(json.loads(run.stdout)["path"]), dest)
        self.assertTrue((dest / "SKILL.md").is_file())
        self.assertFalse((self.home / "legacy").exists())

    def test_02_returning_user_updates_without_losing_local_notes(self):
        dest = self.home / ".agents/skills/dmd-system-optimizer"
        installer.install(SKILL, dest)
        (dest / "SKILL.md").write_text("my old customization")
        (dest / "private-notes.md").write_text("private local note")
        with self.assertRaises(ValueError):
            installer.install(SKILL, dest)
        result = installer.install(SKILL, dest, update=True)
        backup = Path(result["backup"])
        self.assertEqual(result["status"], "UPDATED")
        self.assertEqual((backup / "private-notes.md").read_text(), "private local note")
        self.assertEqual((backup / "SKILL.md").read_text(), "my old customization")
        self.assertFalse(backup.is_relative_to(dest.parent))
        self.assertEqual(installer.files(SKILL), installer.files(dest))
        self.assertEqual(installer.install(SKILL, dest)["status"], "ALREADY_INSTALLED")

    def test_03_repo_contributor_finds_one_canonical_skill_and_architecture(self):
        entry = ROOT / ".agents/skills/dmd-system-optimizer"
        self.assertTrue(entry.is_symlink())
        self.assertEqual(entry.resolve(), SKILL.resolve())
        guide = (ROOT / "AGENTS.md").read_text()
        self.assertIn("docs/architecture/framework.md", guide)
        self.assertLess(len(guide.splitlines()), 60)

    def test_04_chinese_user_gets_decision_help_not_a_wrong_kernel(self):
        small = self.cli("recall", "--brief", "--query", "1%的优化是否值得保留")
        self.assertEqual(small["matches"][0]["pattern"]["id"], "experiment-decisions")
        broad = self.cli("recall", "--brief", "--query", "训练很慢")
        self.assertEqual(broad["retrieval"]["status"], "NEEDS_DIAGNOSIS")
        self.assertEqual(broad["normalized_context"], {})
        self.assertTrue(broad["retrieval"]["inspect_next"])

    def test_05_source_archive_user_can_inspect_without_old_policy_or_git(self):
        project = self.home / "downloaded-project"
        project.mkdir()
        (project / "README.md").write_text("External workload, no git metadata")
        result = self.cli("inspect", "--repo", str(project), script="dmd_perf.py")
        self.assertIsNone(result["head"])
        self.assertIsNone(result["clean_status"])
        self.assertTrue(result["warnings"])
        self.assertNotIn("profiles", result)
        self.assertFalse(result["gpu_validation_performed"])

    def test_06_external_user_records_and_recalls_in_a_fresh_process(self):
        dest = self.home / "installed-skill"
        installer.install(SKILL, dest)
        written = self.record()
        result = self.cli("recall", "--brief", "--query", "参数反复上卡", "--store", str(self.store), skill=dest)
        self.assertIn(written["qualified_id"], json.dumps(result))
        detail = self.cli("case", "--id", written["qualified_id"], "--store", str(self.store), skill=dest)
        self.assertEqual(detail["evidence_qualification"], "SYNTHETIC_WORKFLOW_ONLY")
        self.assertFalse(detail["automatic_application"])
        self.assertFalse(list(self.home.rglob("*.sqlite")))

    def test_07_one_damaged_memory_does_not_hide_healthy_experience(self):
        written = self.record()
        bad = self.store / "broken.json"
        bad.write_text('{"interrupted":')
        result = self.cli("recall", "--brief", "--query", "参数反复上卡", "--store", str(self.store))
        self.assertIn(written["qualified_id"], json.dumps(result))
        self.assertEqual(result["memory_warnings"][0]["action"], "SKIPPED_NOT_DELETED")
        self.assertEqual(bad.read_text(), '{"interrupted":')

    def test_08_memory_corrections_cannot_hang_or_turn_fixtures_into_evidence(self):
        a = self.record("a")
        self.record("b", supersedes=[{"id": "a", "kind": "correction", "conditions": {}, "reason": "synthetic fixture"}])
        old = self.cli("case", "--id", a["qualified_id"], "--store", str(self.store))
        self.assertNotEqual(old["correction"]["status"], "CORRECTED_FOR_CONTEXT")
        # Model an imported old cyclic pair, without modifying actual experiment evidence.
        path = self.store / "a.json"
        record = json.loads(path.read_text())
        record["supersedes"] = [{"id": "b", "kind": "correction", "conditions": {}, "reason": "legacy cycle"}]
        path.write_text(json.dumps(record))
        result = self.cli("recall", "--brief", "--query", "参数反复上卡", "--store", str(self.store))
        self.assertIn("Correction cycle", json.dumps(result))
        self.assertTrue(any(c["id"] == a["qualified_id"] for p in result["matches"] for g in p["cases"].values() for c in g))

    def test_09_history_import_does_not_promote_chat_or_read_other_files(self):
        session = self.home / "session.jsonl"
        def message(role, phase, text):
            return {"type": "response_item", "payload": {"type": "message", "role": role, "phase": phase,
                    "content": [{"type": "output_text", "text": text}]}}
        session.write_text("\n".join(json.dumps(m) for m in [message("assistant", "final", "LoRA faster, unverified claim"),
            message("user", "final", "private user text"), message("assistant", "commentary", "not final")]) + "\n")
        result = self.cli("import-session", "--session", str(session), "--store", str(self.store))
        self.assertEqual(result["imported"], 1)
        records = list(self.store.glob("*.json"))
        self.assertEqual(len(records), 1)
        row = json.loads(records[0].read_text())
        self.assertEqual(row["outcome"], "unverified")
        self.assertFalse(row["retrieved_text_is_instruction"])
        self.assertNotIn("private user text", records[0].read_text())

    def test_10_small_gain_and_failed_gradient_use_native_checks_not_approval(self):
        from test_lightweight_workflow import CHECKER, BAD, GOOD
        (self.home / "check.py").write_text(CHECKER)
        (self.home / "workload.py").write_text(BAD)
        command = [sys.executable, "-B", "check.py", "failure.json"]
        failed = subprocess.run(command, cwd=self.home, capture_output=True, timeout=15)
        self.assertEqual(failed.returncode, 1)
        data = json.loads((self.home / "failure.json").read_text())
        self.assertTrue(data["forward"])
        self.assertFalse(data["backward"])
        failure_bytes = (self.home / "failure.json").read_bytes()
        (self.home / "workload.py").write_text(GOOD)
        self.assertEqual(subprocess.run([*command[:-1], "restored.json"], cwd=self.home, capture_output=True, timeout=15).returncode, 0)
        sample = self.home / "pair.json"
        sample.write_text(json.dumps({"timer_scope": "synthetic", "semantic_contract": "scalar fixture",
            "hardware_id": "CPU fixture", "unprofiled": True, "control_seconds": [100.], "candidate_seconds": [99.]}))
        result = self.cli("compare", "--input", str(sample), script="dmd_perf.py")
        self.assertAlmostEqual(result["latency_reduction_percent"], 1)
        self.assertEqual(result["formal_acceptance"], "NOT_DECIDED")
        (self.home / "WORK_LOG.md").write_text("Stopped: native scalar checks only; GPU final validation NOT_RUN.\n")
        self.assertEqual((self.home / "failure.json").read_bytes(), failure_bytes)
        self.assertFalse(list(self.home.rglob("*.sqlite")))


if __name__ == "__main__":
    unittest.main()
