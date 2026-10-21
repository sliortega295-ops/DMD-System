"""Publication metadata tests only; synthetic corruptions are not GPU evidence."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('publication_check', ROOT / 'scripts/distillschedule/check_source_publication.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PublicationIntegrityTest(unittest.TestCase):
    def test_real_archive_is_byte_exact_and_six_workloads_are_separate(self):
        result = module.verify(ROOT)
        self.assertEqual(result['status'], 'PASS_PUBLICATION_INTEGRITY_ONLY')
        self.assertEqual(result['GPU_started'], 0)
        self.assertEqual(result['source_branches'], 12)
        self.assertGreaterEqual(result['files'], 77)

    def fixture(self, root):
        path = root / 'artifacts/publications/20261019/source_manifest.json'
        path.parent.mkdir(parents=True)
        manifest = json.loads((ROOT / path.relative_to(root)).read_text())
        return path, manifest

    def test_hash_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, manifest = self.fixture(root)
            item = manifest['files'][0]
            target = root / item['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('SYNTHETIC_TEST_ONLY corrupted bytes')
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'bytes/hash'):
                module.verify(root)

    def test_escape_path_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, manifest = self.fixture(root)
            manifest['files'][0]['path'] = '../SYNTHETIC_TEST_ONLY'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                module.verify(root)


if __name__ == '__main__':
    unittest.main()
