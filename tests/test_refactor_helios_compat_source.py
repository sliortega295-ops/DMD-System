"""CPU-only integrity tests; no model modules or third-party packages are imported."""
from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
