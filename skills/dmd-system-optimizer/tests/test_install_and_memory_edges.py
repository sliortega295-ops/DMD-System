"""Focused failures found during user journeys; stdlib only."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from test_user_journeys import installer, SKILL
sys.path.insert(0, str(SKILL / "scripts"))
import dmd_knowledge as k


class InstallerEdges(unittest.TestCase):
    def test_ancestor_symlink_cannot_install_back_inside_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source"; source.mkdir()
            (source / "SKILL.md").write_text("source")
            (root / "alias").symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "overlap"):
                installer.install(source, root / "alias/nested")
            self.assertFalse((source / "nested").exists())

    def test_failed_upgrade_restores_old_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source"; dest = root / "installed"
            source.mkdir(); (source / "SKILL.md").write_text("new")
            dest.mkdir(); (dest / "SKILL.md").write_text("local version")
            original = Path.rename
            def fail_staging(path, target):
                if path.name.startswith(".dmd-install-"):
                    raise OSError("injected publication failure")
                return original(path, target)
            with patch.object(Path, "rename", fail_staging), self.assertRaises(OSError):
                installer.install(source, dest, update=True)
            self.assertEqual((dest / "SKILL.md").read_text(), "local version")
            self.assertFalse(list(root.glob(".dmd-install-*")))

    def test_unknown_directory_never_replaced_by_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source"; dest = root / "user-data"
            source.mkdir(); (source / "SKILL.md").write_text("new")
            dest.mkdir(); (dest / "keep.txt").write_text("valuable")
            with self.assertRaises(ValueError):
                installer.install(source, dest, update=True)
            self.assertEqual((dest / "keep.txt").read_text(), "valuable")


class MemoryEdges(unittest.TestCase):
    def test_shared_descendants_are_not_revisited_exponentially(self):
        script = '''import sys
sys.path.insert(0, sys.argv[1])
import dmd_knowledge as k
cases=[]
for i in range(36):
    cases.append({"id":str(i), "title":str(i), "outcome":"inconclusive", "evidence_status":"HASH_CHECKED_NOT_TRUTH_PROVEN",
        "correction_edges":[{"id":str(j),"kind":"correction","conditions":{},"reason":"fixture"} for j in (i-1,i-2) if j>=0]})
k.corrections(cases,{})
assert cases[0]["correction"]["current_successors"] == ["35"]
'''
        result = subprocess.run([sys.executable, "-c", script, str(SKILL / "scripts")],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_self_correction_is_rejected_before_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = root / "memory"
            row = {"id":"self", "kind":"experience", "summary":"fixture", "outcome":"unverified",
                   "strength":"unverified", "conditions":{}, "provenance":{"scope":"synthetic"},
                   "supersedes":[{"id":"self","kind":"correction","conditions":{},"reason":"fixture"}]}
            with self.assertRaisesRegex(ValueError, "itself"):
                k.put(store, row, root)
            self.assertFalse((store / "self.json").exists())

    def test_bad_record_shapes_cannot_break_public_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {"kind":"experience", "summary":"fixture", "outcome":"unverified",
                    "strength":"unverified", "conditions":{}, "provenance":{}}
            bad = [{"pattern": []}, {"reuse_decision": []}, {"supersedes": [False]}, {"conditions": []}]
            for i, change in enumerate(bad):
                (root / f"bad{i}.json").write_text(json.dumps({**base,"id":f"bad{i}",**change}))
            before = {p: p.read_bytes() for p in root.glob("*.json")}
            result = k.recall("参数上卡", {}, stores=[root])
            self.assertEqual(len(result["memory_warnings"]), 4)
            self.assertTrue(result["matches"])
            self.assertEqual(before, {p: p.read_bytes() for p in root.glob("*.json")})


if __name__ == "__main__":
    unittest.main()
