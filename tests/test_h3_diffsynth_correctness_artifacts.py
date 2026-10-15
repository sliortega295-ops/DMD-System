import importlib.util
from pathlib import Path
import tempfile
import unittest

from runtime.h3_diffsynth_correctness_artifacts import CorrectnessArtifacts, file_sha


class DisabledTests(unittest.TestCase):
    def test_disabled_does_not_read_tensors_or_create_directory(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'absent'
            writer = CorrectnessArtifacts(path, enabled=False, mode='timing')
            self.assertIsNone(writer.before_prediction(object(), identity={}))
            writer.after_prediction(None, object())
            self.assertIsNone(writer.export_trainable_state(None, None))
            self.assertFalse(path.exists())

    def test_timing_export_rejected_before_io(self):
        with self.assertRaisesRegex(ValueError, 'correctness-only'):
            CorrectnessArtifacts('/not_created', enabled=True, mode='timing')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
class TensorTests(unittest.TestCase):
    def test_real_inputs_outputs_rng_and_gradient_are_preserved(self):
        import torch
        with tempfile.TemporaryDirectory() as root:
            writer = CorrectnessArtifacts(Path(root) / 'actual', enabled=True, mode='correctness')
            x = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
            sigma = torch.tensor(.25)
            rng = torch.get_rng_state().clone()
            versions = (x._version, sigma._version)
            before = writer.before_prediction({'latent': x, 'sigma': sigma}, identity={'slot': 3})
            y = x.square(); writer.after_prediction(before, (y,))
            y.sum().backward()
            self.assertTrue(torch.equal(x.grad, 2 * x.detach()))
            self.assertEqual((x._version, sigma._version), versions)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            receipt = writer.receipt()['predictions'][0]
            self.assertEqual(receipt['sha256'], file_sha(receipt['path']))
            self.assertTrue(torch.equal(torch.load(receipt['path'], weights_only=True)[0], y.detach()))
            self.assertFalse(torch.cuda.is_initialized())

    def test_comparator_requires_actual_inputs_predictions_and_state(self):
        import torch
        from runtime.h3_diffsynth_correctness_artifacts import compare_artifacts
        with tempfile.TemporaryDirectory() as root:
            rows = []
            for name in ('reference', 'candidate'):
                writer = CorrectnessArtifacts(Path(root) / name, enabled=True, mode='correctness')
                x = torch.arange(4, dtype=torch.float32)
                before = writer.before_prediction(x, identity={'slot': 1})
                writer.after_prediction(before, (x.square(),))
                path = writer.directory / 'state.pt'
                torch.save({'parameters':{'LoRA':x}, 'Adam':{'student':[{'exp_avg':x}]}}, path)
                rows.append({'identities':{'slot':1}, 'predictions':writer.rows,
                             'trainable_state':{'path':str(path),'sha256':file_sha(path)}})
            equal = compare_artifacts(*rows)
            self.assertEqual(equal['status'], 'PASS')
            self.assertEqual(equal['comparisons'], equal['bitwise_equal_comparisons'])
            rows[1]['predictions'][0]['input_consumption']['sha256'] = 'wrong_input'
            self.assertEqual(compare_artifacts(*rows)['status'], 'FAIL')
            rows[1]['predictions'][0]['input_consumption'] = rows[0]['predictions'][0]['input_consumption']
            path = Path(rows[1]['trainable_state']['path'])
            torch.save({'parameters':{'LoRA':x+1}, 'Adam':{'student':[{'exp_avg':x}]}},path)
            rows[1]['trainable_state']['sha256'] = file_sha(path)
            self.assertEqual(compare_artifacts(*rows)['status'], 'FAIL')
