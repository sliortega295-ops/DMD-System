"""Common-budget lowering tests; no fixture is native timing evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from distillschedule.search.helios_model_adapter import (
    _manual_match_lowering, gan_manual_incumbent,
)
from distillschedule.workloads.best_known.registry import file_sha256


class ManualMatchLoweringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'protocol.json'
        self.identity = {key: 'a' * (40 if key == 'source_head' else 64) for key in (
            'source_head', 'hardware_fingerprint', 'environment_fingerprint',
            'asset_fingerprint', 'semantic_fingerprint', 'precision_fingerprint')}
        self.limits = {i: 30 * 2**30 for i in range(16)}
        self.incumbent = gan_manual_incumbent()
        self.protocol = dict(scope='MANUAL_NATIVE_VS_SAVED_SELECTED_PLAN_NATIVE',
            identity=self.identity, registered_unix_ns=1, max_planner_overhead=0.03,
            manual_candidate=self.incumbent.candidate().to_dict(),
            capacity_policy=dict(gpu_metric='peak_allocated_bytes',
                memory_limits={str(k): v for k,v in self.limits.items()},
                host_limits={'0': 700 * 2**30, '1': 700 * 2**30},
                gpu_reserve_bytes=2**30, host_reserve_bytes=2**30))

    def lower(self, value=None):
        self.path.write_text(json.dumps(self.protocol if value is None else value))
        ref = dict(path=str(self.path), sha256=file_sha256(self.path))
        return _manual_match_lowering({'manual_match_protocol': ref},
            self.identity, self.limits, self.incumbent), ref

    def test_default_unchanged(self):
        self.assertEqual(_manual_match_lowering({}, {}, {}, None), {})

    def test_exact_reference_reaches_lowering(self):
        value, ref = self.lower()
        self.assertEqual(value, {'manual_match_protocol': ref})

    def test_source_identity_cannot_be_relabelled(self):
        bad = copy.deepcopy(self.protocol)
        bad['identity']['source_head'] = 'b' * 40
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.lower(bad)

    def test_declared_budget_cannot_change_planner_admission(self):
        bad = copy.deepcopy(self.protocol)
        bad['capacity_policy']['memory_limits']['9'] += 1
        with self.assertRaisesRegex(ValueError, 'budgets'):
            self.lower(bad)

    def test_incumbent_must_be_complete(self):
        bad = copy.deepcopy(self.protocol)
        bad['manual_candidate']['id'] = 'legacy-subset'
        with self.assertRaisesRegex(ValueError, 'complete'):
            self.lower(bad)

    def test_target_and_scope_fail_closed(self):
        for field, value in (('max_planner_overhead', 0.5), ('registered_unix_ns', True),
                             ('scope', 'OLD_TIMING')):
            bad = copy.deepcopy(self.protocol); bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.lower(bad)

    def test_stale_or_relative_reference_rejected(self):
        for ref in ({'path': 'relative.json', 'sha256': 'a' * 64},
                    {'path': str(self.path), 'sha256': 'a' * 64}):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                _manual_match_lowering({'manual_match_protocol': ref},
                    self.identity, self.limits, self.incumbent)


if __name__ == '__main__':
    unittest.main()
