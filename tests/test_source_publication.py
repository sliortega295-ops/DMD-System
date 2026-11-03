"""Publication metadata tests only; synthetic corruptions are not GPU evidence."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import io
import json
import shutil
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
        self.assertEqual(result['files'], 6)

    def fixture(self, root):
        path = root / 'artifacts/results/manifest.json'
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

    def test_invalid_current_results_rejected_even_with_matching_file_hashes(self):
        """Synthetic corruption reaches the data checks, not just the byte check."""
        cases = {'missing': 'complete source-bound', 'duplicate': 'unsafe or duplicate',
                 'ids': 'six measured', 'nonfinite': 'invalid measured',
                 'ratio': 'invalid measured', 'merge': 'unmerged weights'}
        for case, message in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shutil.copytree(ROOT / 'artifacts/results', root / 'artifacts/results')
                planner = Path('artifacts/publications/20261019/evidence/planner/six_workload_native_match.json')
                (root / planner).parent.mkdir(parents=True)
                shutil.copyfile(ROOT / planner, root / planner)
                path = root / 'artifacts/results/manifest.json'
                manifest = json.loads(path.read_text())
                if case == 'missing':
                    manifest['files'].pop()
                elif case == 'duplicate':
                    manifest['files'].append(manifest['files'][0].copy())
                else:
                    relative = ('artifacts/results/gan_nomerge_retime.json' if case == 'merge'
                                else 'artifacts/results/a100.csv')
                    target = root / relative
                    if case == 'merge':
                        value = json.loads(target.read_text())
                        value['merge_counts'][-1] = 1
                        target.write_text(json.dumps(value))
                    else:
                        rows = list(csv.DictReader(io.StringIO(target.read_text())))
                        if case == 'ids':
                            rows[-1]['id'] = rows[0]['id']
                        elif case == 'nonfinite':
                            rows[0]['system_s'] = 'nan'
                        else:
                            rows[0]['reference_ratio'] = '1'
                        with target.open('w', newline='') as stream:
                            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                            writer.writeheader()
                            writer.writerows(rows)
                    data = target.read_bytes()
                    binding = next(row for row in manifest['files'] if row['path'] == relative)
                    binding.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, message):
                    module.verify(root)


if __name__ == '__main__':
    unittest.main()
