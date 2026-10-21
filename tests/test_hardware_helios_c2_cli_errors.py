"""Public CPU CLI input errors; no native source or hardware operations."""
from contextlib import nullcontext, redirect_stderr
import io
import unittest
from unittest.mock import patch

from distillschedule.search import hardware_helios_c2_cli
from distillschedule.workloads.best_known.cli import main


class C2CLIErrorTests(unittest.TestCase):
    def test_missing_deployment_returns_public_error(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = main(['hardware-helios-c2', 'inspect', '--deployment',
                           '/nonexistent/c2-cli-test/deployment.json'])
        self.assertEqual(result, 2)
        self.assertTrue(stderr.getvalue().startswith('ERROR: '))

    def test_rejected_deployment_and_opt_in_return_public_error(self):
        for action, message in (('inspect', 'unknown/missing C2 deployment field'),
                                ('compile', 'C2 is default-off: explicit opt-in required')):
            argv = ['hardware-helios-c2', action, '--deployment', 'unused.json']
            if action == 'compile':
                argv += ['--arm', 'best', '--mode', 'correctness', '--run-id',
                         'CPU_ONLY', '--output', 'unused-output.json']
            # Compile's valid-deployment boundary is isolated so the actual
            # opt-in guard runs without any source/asset/hardware inspection.
            deployment = (patch.object(hardware_helios_c2_cli.adapter,
                          'validate_deployment', return_value={})
                          if action == 'compile' else nullcontext())
            stderr = io.StringIO()
            with self.subTest(action=action), redirect_stderr(stderr), \
                    patch.object(hardware_helios_c2_cli.c, 'read', return_value={}), \
                    deployment:
                self.assertEqual(main(argv), 2)
            self.assertEqual(stderr.getvalue(), f'ERROR: {message}\n')


if __name__ == '__main__':
    unittest.main()
