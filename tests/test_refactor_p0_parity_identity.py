"""Equal finite values must not fail a numerical parity gate due to reduction rounding."""
import dataclasses
import importlib.util
import unittest
from unittest.mock import patch

from distillschedule.runtime.p0_parity import DEFAULT_P0_PARITY, tensor_parity_metrics


class ThresholdContractTests(unittest.TestCase):
    def test_original_strict_thresholds_are_unchanged(self):
        self.assertEqual(dataclasses.asdict(DEFAULT_P0_PARITY), {
            'mean_relative': 5e-3, 'normalized_l2': 5e-3, 'min_cosine': .9999,
            'max_abs_atol': .05, 'max_abs_rtol': .01})


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires an existing Torch environment')
class ExactValueParityTests(unittest.TestCase):
    def test_equal_values_do_not_use_inexact_cosine_reduction(self):
        import torch
        x = torch.tensor([.1, .2, .3, -.6], dtype=torch.float32)
        # A real large video prediction produced0.9998921156 against
        # itself. Reproduce that reduction error without a large test payload.
        with patch('torch.nn.functional.cosine_similarity', return_value=torch.tensor([.9998921])) as cosine:
            metric = tensor_parity_metrics(x, x.clone())
        self.assertEqual(metric['status'], 'PASS')
        self.assertEqual(metric['cosine'], 1.0)
        self.assertEqual(metric['normalized_l2'], 0.0)
        self.assertEqual(metric['max_abs'], 0.0)
        cosine.assert_not_called()

    def test_nonidentical_values_still_use_the_original_gate(self):
        import torch
        x = torch.tensor([.1, .2, .3, -.6], dtype=torch.float32)
        y = x.clone(); y[0] += .0001
        with patch('torch.nn.functional.cosine_similarity', return_value=torch.tensor([.9998])) as cosine:
            metric = tensor_parity_metrics(x, y)
        self.assertEqual(metric['status'], 'FAIL')
        cosine.assert_called_once()

    def test_nonfinite_and_dtype_mismatch_cannot_pass_via_identity(self):
        import torch
        for x in (torch.tensor([float('inf')]), torch.tensor([float('nan')])):
            self.assertEqual(tensor_parity_metrics(x, x.clone())['status'], 'FAIL')
        self.assertEqual(tensor_parity_metrics(torch.ones(4), torch.ones(4, dtype=torch.float64))['status'], 'FAIL')

    def test_zero_vector_policy_is_preserved(self):
        import torch
        metric = tensor_parity_metrics(torch.zeros(8), torch.zeros(8))
        self.assertEqual(metric['status'], 'PASS')
        self.assertEqual(metric['cosine_policy'], 'both_zero_defined_as_one')
