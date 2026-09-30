import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from distillschedule.runtime.wan13b_upper_bound import Wan13BBootstrap


class BootstrapRelocationTests(unittest.TestCase):
    def test_relocation_keeps_provenance_and_requires_identical_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "transformer").mkdir()
            payload = root / "transformer/config.json"
            payload.write_text('{"test":1}')
            rows = [{"path": "transformer/config.json", "bytes": payload.stat().st_size,
                     "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}]
            identity = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            manifest = {"model_files": rows, "model_identity_sha256": identity,
                        "model_root": "/original/a100/model"}
            bootstrap = Wan13BBootstrap(root, manifest, "unused")
            receipt = bootstrap.validate_model_assets(root)
            self.assertEqual(receipt["source_model_root"], "/original/a100/model")
            self.assertEqual(receipt["content_validation"], "SHA256_PASS")
            payload.write_text('{"test":2}')
            with self.assertRaises(ValueError):
                bootstrap.validate_model_assets(root)

    def test_unsigned_or_escaping_inventory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                Wan13BBootstrap(root, {}, "unused").validate_model_assets(root)
            rows = [{"path": "../outside", "bytes": 0, "sha256": "0" * 64}]
            identity = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            manifest = {"model_files": rows, "model_identity_sha256": identity,
                        "model_root": "/original/a100/model"}
            with self.assertRaises(ValueError):
                Wan13BBootstrap(root, manifest, "unused").validate_model_assets(root)


if __name__ == "__main__":
    unittest.main()
