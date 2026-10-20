"""Small stdlib-only tests; never load historical launchers or a GPU runtime."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('h100_publication', ROOT / 'scripts/distillschedule/check_h100_source_publication.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class H100PublicationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rel = Path('artifacts/publications/20261019/h100')
        shutil.copytree(ROOT / self.rel, self.root / self.rel)
        self.manifest = self.root / self.rel / 'source_manifest.json'
        self.data = json.loads(self.manifest.read_text())

    def update(self):
        self.manifest.write_text(json.dumps(self.data))

    def test_real_publication(self):
        self.assertEqual(MODULE.verify(ROOT)['workloads'], 6)

    def test_archive_byte_drift_rejected(self):
        p = self.root / self.data['files'][0]['path']
        p.write_bytes(p.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            MODULE.verify(self.root)

    def test_path_escape_rejected(self):
        self.data['files'][0]['path'] = '../outside.py'
        self.update()
        with self.assertRaisesRegex(ValueError, 'unsafe or duplicate'):
            MODULE.verify(self.root)

    def test_duplicate_workload_rejected(self):
        self.data['workloads'][-1]['id'] = 'W5'
        self.update()
        with self.assertRaisesRegex(ValueError, 'six H100 workloads'):
            MODULE.verify(self.root)

    def test_untimed_norm_promotion_rejected(self):
        self.data['workloads'][0]['latest_norm_end_to_end'] = 'PASS'
        self.update()
        with self.assertRaisesRegex(ValueError, 'untimed W1'):
            MODULE.verify(self.root)

    def test_grid_semantics_relabel_rejected(self):
        self.data['workloads'][4]['semantics'] = 'EXACT_CONTINUOUS'
        self.update()
        with self.assertRaisesRegex(ValueError, 'approximate Grid'):
            MODULE.verify(self.root)

    def test_unwaived_fa3_promotion_rejected(self):
        self.data['workloads'][5]['attention_backend'] = 'FA3'
        self.update()
        with self.assertRaisesRegex(ValueError, 'rejected FA3'):
            MODULE.verify(self.root)

    def test_no_automatic_resume(self):
        self.data['current_campaign_state'] = 'RUNNING'
        self.update()
        with self.assertRaisesRegex(ValueError, 'stop boundary'):
            MODULE.verify(self.root)

    def test_no_formal_claim_promotion(self):
        self.data['workloads'][0]['formal_speedup_eligible'] = True
        self.update()
        with self.assertRaisesRegex(ValueError, 'promote diagnostic'):
            MODULE.verify(self.root)


if __name__ == '__main__':
    unittest.main()
