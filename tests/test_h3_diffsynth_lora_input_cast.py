"""CPU executable semantics; CUDA and full-model gates remain separate."""
from collections import Counter
import importlib.util
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from runtime.h3_diffsynth_lora_input_cast import install, select_input

try:
    import torch
except ImportError:
    torch = None


class AdmissionTests(unittest.TestCase):
    def test_disabled_does_not_inspect_or_mutate_module(self):
        self.assertIsNone(install(object(), enabled=False))

    def test_wrong_source_is_rejected_before_method_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'untrusted.py'; path.write_text('not the pinned source')
            with self.assertRaisesRegex(ValueError, 'SOURCE_MISMATCH'):
                install(SimpleNamespace(__file__=str(path)), enabled=True)


@unittest.skipUnless(torch is not None, 'actual Torch required')
class TorchCPUTests(unittest.TestCase):
    def test_no_autocast_preserves_original_cast(self):
        x = torch.ones(2, 3, dtype=torch.bfloat16, requires_grad=True)
        counts = Counter(); out = select_input(x, torch.ones(3), counts)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(counts, {'cpu_grad_original_cast': 1})
        out.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_bf16_autocast_keeps_existing_activation_and_fp32_weight(self):
        x = torch.ones(2, 3, dtype=torch.bfloat16)
        weight = torch.ones(4, 3, dtype=torch.float32)
        counts = Counter()
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
            self.assertIs(select_input(x, weight, counts), x)
        self.assertEqual(weight.dtype, torch.float32)
        self.assertEqual(counts['cpu_nograd_elided'], 1)
        self.assertEqual(counts['avoided_FP32_activation_bytes'], x.numel()*4)

    def test_other_activation_dtype_keeps_native_behavior(self):
        x = torch.ones(2, 3, dtype=torch.float32); counts = Counter()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            self.assertIs(select_input(x, torch.ones(3), counts), x)
        self.assertEqual(counts, {'cpu_grad_original_cast': 1})

    def test_actual_cpu_autocast_linear_output_and_all_gradients_bitwise(self):
        torch.manual_seed(37)
        base = torch.nn.Linear(24, 16, dtype=torch.bfloat16).requires_grad_(False)
        a = torch.nn.Linear(24, 4, bias=False, dtype=torch.float32)
        b = torch.nn.Linear(4, 16, bias=False, dtype=torch.float32)
        x = torch.randn(17, 24, dtype=torch.bfloat16, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        upstream = torch.randn(17, 16, dtype=torch.bfloat16)
        counts = Counter()
        with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
            ref = base(x) + b(a(x.to(a.weight.dtype)))
            got = base(y) + b(a(select_input(y, a.weight, counts)))
        rg = torch.autograd.grad(ref, (x, a.weight, b.weight), upstream)
        gg = torch.autograd.grad(got, (y, a.weight, b.weight), upstream)
        self.assertTrue(torch.equal(ref, got))
        for expected, observed in zip(rg, gg):
            self.assertTrue(torch.equal(expected, observed))
        self.assertEqual(counts['cpu_grad_elided'], 1)
        self.assertEqual((a.weight.dtype, b.weight.dtype), (torch.float32, torch.float32))


def private_source_present():
    if sys.modules.get('h3_dmd2') is not None:
        return True
    try:
        return importlib.util.find_spec('h3_dmd2') is not None
    except (ValueError, ModuleNotFoundError):
        return False


@unittest.skipUnless(torch is not None and private_source_present(), 'actual pinned private source required')
class PrivateNativeCPUTests(unittest.TestCase):
    def test_source_pinned_wrapper_executes_original_roles_and_module_hooks(self):
        import h3_dmd2
        torch.manual_seed(41)
        layer = h3_dmd2.DualLoRALinear(torch.nn.Linear(24, 16, dtype=torch.bfloat16), 4, 4)
        with torch.no_grad():
            for pair in layer.adapters.values():
                pair['B'].weight.normal_(std=0.03)
        x = torch.randn(17, 24, dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn(17, 16, dtype=torch.bfloat16)
        original = h3_dmd2.DualLoRALinear.forward
        expected = {}
        for role in ('teacher', 'student', 'fake_score'):
            layer.role = role
            v = x.detach().clone().requires_grad_(True)
            parameters = [v] if role == 'teacher' else [v, *layer.adapters[role].parameters()]
            with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
                out = layer(v)
            expected[role] = (out.detach(), torch.autograd.grad(out, parameters, dy))
        mechanism = install(h3_dmd2, enabled=True)
        calls = Counter(); handles = []
        for role, pair in layer.adapters.items():
            for name, module in pair.items():
                handles.append(module.register_forward_hook(
                    lambda m, a, o, key=role+name: calls.update([key])))
        try:
            with self.assertRaisesRegex(ValueError, 'ALREADY_INSTALLED'):
                install(h3_dmd2, enabled=True)
            for role in ('teacher', 'student', 'fake_score'):
                layer.role = role
                v = x.detach().clone().requires_grad_(True)
                parameters = [v] if role == 'teacher' else [v, *layer.adapters[role].parameters()]
                with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
                    out = layer(v)
                gradients = torch.autograd.grad(out, parameters, dy)
                self.assertTrue(torch.equal(out, expected[role][0]))
                for wanted, observed in zip(expected[role][1], gradients):
                    self.assertTrue(torch.equal(wanted, observed))
            self.assertEqual(calls, {'studentA': 1, 'studentB': 1, 'fake_scoreA': 1, 'fake_scoreB': 1})
            self.assertEqual(mechanism.finish_cycle()['cycle_delta']['cpu_grad_elided'], 2)
            self.assertEqual(mechanism.finish_cycle()['cycle_delta'], {})
        finally:
            for handle in handles: handle.remove()
            mechanism.restore()
        self.assertIs(h3_dmd2.DualLoRALinear.forward, original)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
