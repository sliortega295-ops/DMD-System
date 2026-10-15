"""Executable native CPU/checkpoint tests; not GPU or full-model proof."""
from collections import Counter
import importlib.util
import sys
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

try:
    private_present = 'h3_dmd2' in sys.modules or importlib.util.find_spec('h3_dmd2') is not None
except (ValueError, ModuleNotFoundError):
    private_present = False


@unittest.skipUnless(torch is not None and private_present, 'actual Torch and pinned private source required')
class NativeUnitLoRATests(unittest.TestCase):
    def make_layer(self):
        import h3_dmd2
        torch.manual_seed(73)
        layer = h3_dmd2.DualLoRALinear(torch.nn.Linear(24, 16, dtype=torch.bfloat16), 4, 4)
        with torch.no_grad():
            for pair in layer.adapters.values():
                pair['B'].weight.normal_(std=0.03)
        return layer

    def test_disabled_keeps_actual_native_function(self):
        import h3_dmd2
        from runtime.h3_diffsynth_lora_unit_inplace import UnitScaleInplaceLoRA
        original = h3_dmd2.DualLoRALinear.forward
        mechanism = UnitScaleInplaceLoRA(h3_dmd2, None, enabled=False)
        self.assertIs(h3_dmd2.DualLoRALinear.forward, original)
        mechanism.close()
        self.assertIs(h3_dmd2.DualLoRALinear.forward, original)

    def test_unit_scale_and_frozen_base_are_required(self):
        import h3_dmd2
        from runtime.h3_diffsynth_lora_unit_inplace import UnitScaleInplaceLoRA
        layer = self.make_layer()
        layer.scale = 2.0
        with self.assertRaisesRegex(ValueError, 'unit LoRA scale'):
            UnitScaleInplaceLoRA(h3_dmd2, SimpleNamespace(layers=[('actual', layer)]), enabled=True)
        layer.scale = 1.0
        layer.base.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen base'):
            UnitScaleInplaceLoRA(h3_dmd2, SimpleNamespace(layers=[('actual', layer)]), enabled=True)

    def test_native_direct_checkpoint_roles_and_cast_composition_bitwise(self):
        import h3_dmd2
        from torch.utils.checkpoint import checkpoint
        from runtime.h3_diffsynth_lora_input_cast import install, select_input
        from runtime.h3_diffsynth_lora_unit_inplace import UnitScaleInplaceLoRA

        for cast in (False, True):
            for replay in (False, True):
                with self.subTest(cast=cast, checkpoint=replay):
                    layer = self.make_layer()
                    x = torch.randn(17, 24, dtype=torch.bfloat16)
                    dy = torch.randn(17, 16, dtype=torch.bfloat16)
                    native_function = h3_dmd2.DualLoRALinear.forward
                    casting = install(h3_dmd2, enabled=cast)
                    baseline = h3_dmd2.DualLoRALinear.forward
                    counts = Counter()
                    handles = [layer.base.register_forward_hook(lambda *a: counts.update(['base']))]
                    for role, pair in layer.adapters.items():
                        for name, module in pair.items():
                            handles.append(module.register_forward_hook(
                                lambda *a, key=role+name: counts.update([key])))

                    def execute(role):
                        layer.role = role
                        value = x.detach().clone().requires_grad_(True)
                        params = [value] if role == 'teacher' else [value, *layer.adapters[role].parameters()]
                        with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
                            out = checkpoint(layer, value, use_reentrant=False) if replay else layer(value)
                        return out.detach(), torch.autograd.grad(out, params, dy)

                    selector = (None if casting is None else
                                lambda value, weight: select_input(value, weight, casting.counts))
                    mechanism = None
                    try:
                        expected = {}
                        counts.clear()
                        for role in ('student', 'fake_score', 'teacher'):
                            expected[role] = execute(role)
                        baseline_hooks = dict(counts)
                        mechanism = UnitScaleInplaceLoRA(h3_dmd2,
                            SimpleNamespace(layers=[('actual', layer)]), enabled=True, input_selector=selector)
                        counts.clear()
                        for role in ('student', 'fake_score', 'teacher'):
                            actual, grads = execute(role)
                            self.assertTrue(torch.equal(expected[role][0], actual))
                            for left, right in zip(expected[role][1], grads):
                                self.assertTrue(torch.equal(left, right))
                        self.assertEqual(dict(counts), baseline_hooks)
                        receipt = mechanism.finish_cycle()
                        self.assertTrue(receipt['counters']['physical_dual_lora_forward'] > 0)
                        self.assertEqual(receipt['input_selector_composed'], cast)
                        self.assertFalse(receipt['parameter_dtype_changed'])
                        self.assertTrue(all(p.dtype == torch.float32 for pair in layer.adapters.values()
                                            for p in pair.parameters()))
                    finally:
                        for h in handles:
                            h.remove()
                        if mechanism is not None:
                            mechanism.close()
                        self.assertIs(h3_dmd2.DualLoRALinear.forward, baseline)
                        if casting is not None:
                            casting.restore()
                        self.assertIs(h3_dmd2.DualLoRALinear.forward, native_function)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
