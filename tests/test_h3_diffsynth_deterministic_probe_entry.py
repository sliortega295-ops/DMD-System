"""Pure admission tests; these do not replace GPU/FSDP numerical validation."""
import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / 'scripts/distillschedule/h3_diffsynth_deterministic_probe_entry.py'
spec = importlib.util.spec_from_file_location('w6_deterministic_diagnostic_entry', path)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class DiagnosticAdmissionTests(unittest.TestCase):
    def arguments(self, mode='correctness'):
        return ['--mode', mode, '--one-cycle-correctness', '--export-parity-artifacts',
                '--receipt-dir', '/not_created', '--placement', 'candidate']

    def test_accepts_only_explicit_one_cycle_correctness_evidence(self):
        args = entry.validate_diagnostic_arguments(self.arguments())
        self.assertEqual(args.receipt_dir, Path('/not_created'))
        self.assertTrue(args.one_cycle_correctness)

    def test_formal_timing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CORRECTNESS_CYCLE_AND_EVIDENCE'):
            entry.validate_diagnostic_arguments(self.arguments('timing'))

    def test_unbounded_or_no_evidence_probe_is_rejected(self):
        for removed in ('--one-cycle-correctness', '--export-parity-artifacts'):
            args = self.arguments()
            args.remove(removed)
            with self.assertRaisesRegex(ValueError, 'CORRECTNESS_CYCLE_AND_EVIDENCE'):
                entry.validate_diagnostic_arguments(args)
