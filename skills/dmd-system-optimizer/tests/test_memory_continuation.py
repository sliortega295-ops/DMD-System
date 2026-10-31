"""Growing, copied and partly damaged local histories; CPU fixtures only."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
import dmd_knowledge as k


class MemoryContinuation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = self.root / "memory"

    def record(self, ident, **changes):
        row = {"id": ident, "kind": "experience", "summary": "uniquepipelinez probe",
               "outcome": "unverified", "strength": "unverified", "conditions": {},
               "provenance": {"measurement_kind": "synthetic"}, **changes}
        return k.put(self.store, row, self.root)

    def test_search_does_not_drop_history_after_twenty_mechanisms(self):
        for i in range(35):
            self.record(f"attempt-{i:02d}")
        result = k.search(self.store, "uniquepipelinez", {}, limit=100)
        self.assertEqual(len(result["matches"]), 35)
        self.assertEqual(result["total_matches"], 35)
        self.assertFalse(result["truncated"])
        limited = k.search(self.store, "uniquepipelinez", {}, limit=4)
        self.assertEqual(limited["total_matches"], 35)
        self.assertTrue(limited["truncated"])

    def test_late_failure_is_not_hidden_by_the_old_mechanism_cap(self):
        for i in range(30):
            self.record(f"a-{i:02d}")
        self.record("z-failed", outcome="incorrect", summary="uniquepipelinez wrong backward")
        result = k.search(self.store, "uniquepipelinez", {}, limit=100)
        row = next(r for r in result["matches"] if r["id"] == "z-failed")
        self.assertEqual(row["outcome"], "correctness_failure")
        self.assertFalse(row["automatic_application"])

    def test_invalid_correction_scope_refused_before_write(self):
        with self.assertRaises(ValueError):
            self.record("bad", supersedes=[{"id": "public:helios-crosskv-gan-positive",
                "kind": "correction", "conditions": {"nodes": []}, "reason": "fixture"}])
        self.assertFalse((self.store / "bad.json").exists())

    def test_imported_bad_scope_and_bad_reuse_do_not_break_healthy_recall(self):
        healthy = self.record("healthy", pattern="parameter-residency", summary="参数上卡等待")
        base = json.loads((self.store / "healthy.json").read_text())
        bad = [
            {"id": "bad-scope", "supersedes": [{"id": "public:helios-crosskv-gan-positive",
                "kind": "correction", "conditions": {"nodes": []}, "reason": "fixture"}]},
            {"id": "bad-reuse", "reuse_decision": {"implementation_id": []}},
        ]
        for change in bad:
            (self.store / (change["id"] + ".json")).write_text(json.dumps({**base, **change}))
        before = {p.name: p.read_bytes() for p in self.store.glob("*.json")}
        result = k.recall("参数上卡", {}, stores=[self.store])
        self.assertEqual(len(result["memory_warnings"]), 2)
        self.assertIn(healthy["qualified_id"], json.dumps(result))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.store.glob("*.json")})

    @staticmethod
    def message(text):
        return {"type": "response_item", "payload": {"type": "message", "role": "assistant",
            "phase": "final", "content": [{"type": "output_text", "text": text}]}}

    def test_copied_session_deduplicates_without_rewriting_original_provenance(self):
        original, copied = self.root / "session.jsonl", self.root / "copy.jsonl"
        original.write_text(json.dumps(self.message("uniquepipelinez claim, not measured")) + "\n")
        first = k.import_session(self.store, original)
        saved = {p.name: p.read_bytes() for p in self.store.glob("*.json")}
        shutil.copyfile(original, copied)
        second = k.import_session(self.store, copied)
        self.assertEqual(first["created"], 1)
        self.assertEqual(second["already_present"], 1)
        self.assertEqual(second["created"], 0)
        self.assertEqual(saved, {p.name: p.read_bytes() for p in self.store.glob("*.json")})

    def test_bad_session_rows_do_not_block_later_final_messages(self):
        session = self.root / "partial.jsonl"
        malformed = [
            {"type": "response_item", "payload": ["assistant", "message"]},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": False}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": []}]}},
        ]
        session.write_text("".join(json.dumps(r) + "\n" for r in
                                   [*malformed, self.message("uniquepipelinez final claim")]))
        result = k.import_session(self.store, session)
        self.assertEqual(result["imported"], 1)
        self.assertEqual(result["ignored"], 3)
        row = json.loads(next(self.store.glob("*.json")).read_text())
        self.assertEqual(row["strength"], "unverified")

    def test_modules_can_load_without_importing_the_cli(self):
        script = """import sys
sys.path.insert(0, sys.argv[1])
import dmd_memory, dmd_conditions, dmd_sources, dmd_navigation
assert 'dmd_knowledge' not in sys.modules
dmd_navigation.validate_catalog()
assert 'dmd_knowledge' not in sys.modules
assert not any(k.split('.')[0] in {'torch', 'numpy'} for k in sys.modules)
"""
        run = subprocess.run([sys.executable, "-S", "-c", script, str(SKILL / "scripts")],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()
