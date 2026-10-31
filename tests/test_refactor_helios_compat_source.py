"""CPU-only integrity tests; no model modules or third-party packages are imported."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_helios_compat_source", ROOT / "scripts/distillschedule/verify_helios_compat_source.py"
)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class HeliosSourceRecoveryTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(tempfile.gettempdir()) / "lyy-experiments/DMD-System/helios-source-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        shutil.copytree(ROOT / "third_party/helios_compat", self.repo / "third_party/helios_compat")
        shutil.copytree(ROOT / "runtime", self.repo / "runtime")
        self.snapshot = self.repo / "third_party/helios_compat"

    def test_bytes_compile_and_both_import_graphs_pass(self):
        result = VERIFY.verify_source(self.repo)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["files_checked"], 83)
        self.assertEqual(result["gpu_execution"], "NOT_RUN")
        deps = {row["module"] for row in result["core_local_imports"]}
        self.assertIn("runtime.official_baseline", deps)
        self.assertIn("runtime.precast_checkpoint", deps)
        self.assertTrue(result["historical_reference_local_imports"])

    def test_transformer_tampering_is_rejected(self):
        path = self.snapshot / "source/helios/modules/transformer_helios.py"
        path.write_bytes(path.read_bytes() + b"\n# altered\n")
        with self.assertRaisesRegex(ValueError, "source identity mismatch"):
            VERIFY.verify_source(self.repo)

    def test_missing_scheduler_is_rejected(self):
        (self.snapshot / "source/helios/scheduler/scheduling_helios.py").unlink()
        with self.assertRaises(FileNotFoundError):
            VERIFY.verify_source(self.repo)

    def test_unregistered_shadow_module_is_rejected(self):
        (self.snapshot / "source/runtime.py").write_text("# accidental import shadow\n")
        with self.assertRaisesRegex(ValueError, "unregistered/missing"):
            VERIFY.verify_source(self.repo)

    def test_missing_dmd_runtime_export_is_rejected(self):
        (self.repo / "runtime/official_baseline.py").write_text("# missing exported functions\n")
        with self.assertRaisesRegex(ValueError, "missing runtime exports"):
            VERIFY.verify_source(self.repo)

    def test_manifest_cannot_relax_transformer_pin(self):
        path = self.snapshot / "source_manifest.json"
        data = json.loads(path.read_text())
        data["expected_transformer_sha256"] = "0" * 64
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "must not be relaxed"):
            VERIFY.verify_source(self.repo)

    def test_reviewed_rollout_alias_patch_retains_the_complete_original_receipt(self):
        manifest = json.loads((self.snapshot / "source_manifest.json").read_text())
        target = "source/helios/utils/utils_helios_post.py"
        row = next(row for row in manifest["files"] if row["path"] == target)
        overlay, = [row for row in manifest["reviewed_overlays"] if row["path"] == target]
        current = (self.snapshot / target).read_bytes()
        original = current
        for after, before in (
            (b"start_point_list = [noisy_model_input.detach()]", b"start_point_list = [noisy_model_input]"),
            (b"start_point_list.append(noisy_model_input.detach())", b"start_point_list.append(noisy_model_input)"),
        ):
            self.assertEqual(original.count(after), 1)
            original = original.replace(after, before)
        receipt = overlay["origin"]["original_manifest_row"]
        self.assertEqual(receipt, {
            "path": target,
            "source_path": "helios/utils/utils_helios_post.py",
            "source_git_blob": "706f6d9608e692bdd19f0c8d2d3dc4257f1074d4",
            "sha256": "e3ff9e5f8bf025ee740e436bc0c209c972f1e56eb0236b968fe6bd8284107004",
            "bytes": 184631,
            "usage": "integration_source",
        })
        for payload, identity in ((original, receipt), (current, row)):
            self.assertEqual(hashlib.sha256(payload).hexdigest(), identity["sha256"])
            self.assertEqual(len(payload), identity["bytes"])
            self.assertEqual(hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest(),
                             identity["source_git_blob"])
        self.assertEqual(overlay["current_sha256"], row["sha256"])
        self.assertEqual(overlay["current_git_blob"], row["source_git_blob"])
        self.assertEqual(overlay["origin"]["reviewed_live_git_blob"], row["source_git_blob"])
        self.assertEqual(overlay["origin"]["original_manifest_sha256"],
                         "a3d98faab5787f4f9f4e1155eb577235a885f14babc677004affb42877118bd1")


if __name__ == "__main__":
    unittest.main()
