import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from distillschedule.runtime.asset_relocation import (
    RECEIPT_ENV, SHA_ENV, relocate_asset_path, relocation_identity,
)


class AssetRelocationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "owned/project"
        self.root.mkdir(parents=True)
        self.receipt = self.root / "relocation.json"
        self.value = {
            "schema_version": "distillschedule.asset_relocation.v1",
            "hardware_profile": "h100-8-b16", "owned_destination_root": str(self.root),
            "bindings": [{"source": "/original/asset", "destination": str(self.root / "asset"), "kind": "directory"}],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def env(self):
        self.receipt.write_text(json.dumps(self.value))
        return {RECEIPT_ENV: str(self.receipt), SHA_ENV: hashlib.sha256(self.receipt.read_bytes()).hexdigest(),
                "DISTILLSCHEDULE_HARDWARE_PROFILE": "h100-8-b16"}

    def test_default_is_original_path_resolution(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(relocate_asset_path("/original/asset/input.pt"), Path("/original/asset/input.pt"))
            self.assertIsNone(relocation_identity())

    def test_relocation_keeps_metadata_unmodified(self):
        spec = {"path": "/original/asset/input.pt", "sha256": "unchanged"}
        with patch.dict(os.environ, self.env(), clear=True):
            self.assertEqual(relocate_asset_path(spec["path"]), self.root / "asset/input.pt")
            self.assertEqual(spec, {"path": "/original/asset/input.pt", "sha256": "unchanged"})
            self.assertEqual(relocation_identity()["payload_identity"], "REQUIRES_EXISTING_CONTENT_GATES")

    def test_requires_opt_in_and_bound_hash(self):
        env = self.env()
        for updates in ({"DISTILLSCHEDULE_HARDWARE_PROFILE": "a100-16-b16"}, {SHA_ENV: ""}, {SHA_ENV: "0" * 64}):
            with self.subTest(updates=updates), patch.dict(os.environ, {**env, **updates}, clear=True):
                with self.assertRaises(ValueError):
                    relocate_asset_path("/original/asset/input.pt")

    def test_unmapped_path_and_prefix_collision_rejected(self):
        with patch.dict(os.environ, self.env(), clear=True):
            for path in ("/original/asset2/input.pt", "/other/input.pt", "/original/asset/../escape", "input.pt"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    relocate_asset_path(path)

    def test_longest_mapping_wins(self):
        self.value["bindings"].append({"source": "/original/asset/special", "destination": str(self.root / "special"), "kind": "directory"})
        with patch.dict(os.environ, self.env(), clear=True):
            self.assertEqual(relocate_asset_path("/original/asset/special/a.pt"), self.root / "special/a.pt")
            self.assertEqual(relocate_asset_path(self.root / "new.pt"), self.root / "new.pt")

    def test_targets_must_stay_owned(self):
        self.value["bindings"][0]["destination"] = "/other/user/project"
        with patch.dict(os.environ, self.env(), clear=True), self.assertRaises(ValueError):
            relocate_asset_path("/original/asset/input.pt")

    def test_file_mapping_does_not_map_children(self):
        self.value["bindings"][0]["kind"] = "file"
        with patch.dict(os.environ, self.env(), clear=True):
            self.assertEqual(relocate_asset_path("/original/asset"), self.root / "asset")
            with self.assertRaises(ValueError):
                relocate_asset_path("/original/asset/a.pt")

    def test_symlink_escape_rejected(self):
        (self.root / "asset").symlink_to(Path(self.tmp.name) / "outside")
        with patch.dict(os.environ, self.env(), clear=True), self.assertRaises(ValueError):
            relocate_asset_path("/original/asset/input.pt")

    def test_digest_without_receipt_rejected(self):
        with patch.dict(os.environ, {SHA_ENV: "f" * 64}, clear=True), self.assertRaises(ValueError):
            relocate_asset_path("/original/asset")


if __name__ == "__main__":
    unittest.main()
