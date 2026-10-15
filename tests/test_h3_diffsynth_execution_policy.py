"""An optional common numerical policy is not a placement optimization."""
import importlib.util
import os
import unittest
from unittest import mock

from runtime.h3_diffsynth_execution_policy import (configure_deterministic_runtime,
                                                  validate_deterministic_timing_gate)


class ExecutionPolicyTests(unittest.TestCase):
    def test_default_timing_and_explicit_correctness_policy_remain_available(self):
        validate_deterministic_timing_gate(False, 'timing', False)
        validate_deterministic_timing_gate(True, 'correctness', False)

    def test_deterministic_timing_requires_actual_warmup_evidence(self):
        for capture, reference in ((False, None), (True, None), (False, 'actual')):
            with self.assertRaisesRegex(ValueError, 'REAL_WARMUP_REFERENCE_GATE'):
                validate_deterministic_timing_gate(True, 'timing', capture, reference)

    def test_reference_and_candidate_reference_scopes_are_explicit(self):
        validate_deterministic_timing_gate(True, 'timing', True, reference='C1_and_C2')
        validate_deterministic_timing_gate(True, 'timing', True,
                                          warmup_only_reference='frozen_C1_only')
        with self.assertRaisesRegex(ValueError, 'ONE_WARMUP_REFERENCE_SCOPE'):
            validate_deterministic_timing_gate(True, 'timing', True, 'C1_C2', 'C1')

    def test_warmup_only_reference_cannot_silently_skip_capture(self):
        with self.assertRaisesRegex(ValueError, 'REQUIRES_CAPTURE'):
            validate_deterministic_timing_gate(False, 'timing', False,
                                              warmup_only_reference='C1')

    def test_default_off_does_not_import_torch(self):
        original = __import__
        def guarded(name, *args, **kwargs):
            if name == 'torch':
                raise AssertionError('default policy must not import Torch')
            return original(name, *args, **kwargs)
        with mock.patch('builtins.__import__', side_effect=guarded):
            self.assertEqual(configure_deterministic_runtime(),
                             {'requested': False, 'configured': False})

    def test_missing_workspace_fails_before_torch(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'PRECEDE_PROCESS_START'):
                configure_deterministic_runtime(True)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'actual Torch CPU runtime required')
    def test_actual_torch_policy_before_cuda_and_restore(self):
        import torch
        previous = torch.are_deterministic_algorithms_enabled()
        previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        self.assertFalse(torch.cuda.is_initialized())
        try:
            with mock.patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
                receipt = configure_deterministic_runtime(True)
            self.assertTrue(receipt['deterministic_algorithms'])
            self.assertFalse(receipt['warn_only'])
            self.assertFalse(receipt['CUDA_initialized_before_worker'])
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=previous_warn)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'actual Torch CPU runtime required')
    def test_already_initialized_cuda_is_rejected_before_mutation(self):
        import torch
        with mock.patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}), \
             mock.patch.object(torch.cuda, 'is_initialized', return_value=True), \
             mock.patch.object(torch, 'use_deterministic_algorithms') as set_policy:
            with self.assertRaisesRegex(RuntimeError, 'PRECEDE_CUDA_INITIALIZATION'):
                configure_deterministic_runtime(True)
            set_policy.assert_not_called()
