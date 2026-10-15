"""Placement/lifetime CPU tests; actual collective proof needs GPU receipts."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest import mock

from runtime.h3_diffsynth_rollout_residency import NativeNoGradParameterReuse


class FakeTorch:
    grad = True
    def is_grad_enabled(self):
        return self.grad
    @contextmanager
    def no_grad(self):
        previous = self.grad
        self.grad = False
        try:
            yield
        finally:
            self.grad = previous


class Block:
    def __init__(self):
        self.group = SimpleNamespace(is_unsharded=False, _reshard_after_forward=True)
        self.events = []
    def _get_fsdp_state(self):
        return SimpleNamespace(_fsdp_param_group=self.group)
    def set_reshard_after_forward(self, value, *, recurse):
        assert recurse is False
        self.events.append(('policy', value))
        self.group._reshard_after_forward = value
    def reshard(self):
        self.events.append(('reshard',))
        self.group.is_unsharded = False
    def parameters(self):
        return [SimpleNamespace(numel=lambda: 5, element_size=lambda: 2)]


class ResidencyTests(unittest.TestCase):
    def setup_controller(self, retained=1, raises=False):
        torch = FakeTorch()
        blocks = [Block(), Block()]
        adapter = SimpleNamespace(dit=SimpleNamespace(blocks=blocks),
                                  layers=[('lora', SimpleNamespace(role='student'))])
        def original(a, real, inputs, schedule, selected_step=None):
            self.assertFalse(torch.is_grad_enabled())
            for _ in range(8):
                for b in blocks:
                    b.group.is_unsharded = not b.group._reshard_after_forward
            if raises:
                raise ValueError('native failure')
            return real, inputs
        dmd = SimpleNamespace(rollout=original)
        c = NativeNoGradParameterReuse(dmd, adapter, retained_main_blocks=retained,
                logical_slot=lambda: 3, module_type=Block, torch_module=torch)
        return c, dmd, adapter, blocks, torch, original

    def test_default_does_not_import_torch_or_patch_anything(self):
        dmd = SimpleNamespace(rollout=object())
        before = dmd.rollout
        with mock.patch('builtins.__import__', side_effect=AssertionError('no import')):
            c = NativeNoGradParameterReuse(dmd, None)
            self.assertEqual(c.finish_cycle(), {'enabled':False, 'status':'DEFAULT_OFF_NO_PATCH'})
            c.close()
        self.assertIs(dmd.rollout, before)

    def test_only_selected_main_group_changes_policy_and_returns_original_values(self):
        c, dmd, adapter, blocks, torch, original = self.setup_controller()
        x, y = object(), object()
        self.assertEqual(dmd.rollout(adapter, x, y, ([0]*9, [0]*9), 3), (x, y))
        self.assertEqual(blocks[0].events, [('policy',False),('policy',True),('reshard',)])
        self.assertEqual(blocks[1].events, [])
        self.assertTrue(torch.is_grad_enabled())
        self.assertTrue(c.records[0]['restored'])
        self.assertEqual(c.records[0]['unsharded_before_exit'], 1)
        self.assertEqual(c.parameter_bytes, 10)
        c.close()
        self.assertIs(dmd.rollout, original)

    def test_native_failure_restores_policy_and_shards(self):
        c, dmd, adapter, blocks, _, _ = self.setup_controller(raises=True)
        with self.assertRaisesRegex(ValueError, 'native failure'):
            dmd.rollout(adapter, None, {}, ([0]*9, [0]*9), 0)
        self.assertFalse(blocks[0].group.is_unsharded)
        self.assertTrue(blocks[0].group._reshard_after_forward)
        self.assertTrue(c.records[0]['restored'])

    def test_scope_is_illegal_for_grad_path(self):
        c, *_ = self.setup_controller()
        with self.assertRaisesRegex(RuntimeError, 'GRAD_PATH'):
            with c.scope(0):
                pass

    def test_unexpected_entry_policy_fails_before_mutation(self):
        c, dmd, adapter, blocks, _, _ = self.setup_controller()
        blocks[0].group._reshard_after_forward = False
        with self.assertRaisesRegex(RuntimeError, 'UNEXPECTED_POLICY'):
            dmd.rollout(adapter, None, {}, ([0]*9, [0]*9), 3)
        self.assertEqual(blocks[0].events, [])

    def test_original_schedule_and_adapter_contract_fail_closed(self):
        _, dmd, adapter, blocks, _, _ = self.setup_controller()
        with self.assertRaisesRegex(RuntimeError, 'AR8_ROLLOUT'):
            dmd.rollout(adapter, None, {}, ([0]*8, [0]*8), 3)
        adapter.layers[0][1].role = 'fake_score'
        with self.assertRaisesRegex(RuntimeError, 'WRONG_ADAPTER_ROLE'):
            dmd.rollout(adapter, None, {}, ([0]*9, [0]*9), 3)
        self.assertTrue(blocks[0].group._reshard_after_forward)

    def test_complete_cycle_requires_two_independent_slot_scopes(self):
        c, *_ = self.setup_controller()
        c.records = [{'logical_slot':0, 'restored':True}]
        with self.assertRaisesRegex(RuntimeError, 'INDEPENDENT'):
            c.finish_cycle()
        c.records.append({'logical_slot':8, 'restored':True})
        self.assertEqual(len(c.finish_cycle()['records']), 2)
        c.begin_next_cycle()
        self.assertEqual(c.records, [])

    def test_missing_slot_and_wrapper_ownership_fail_closed(self):
        c, dmd, adapter, _, _, _ = self.setup_controller()
        dmd.rollout = object()
        with self.assertRaisesRegex(RuntimeError, 'OWNERSHIP'):
            c.close()

    def test_full_model_bound_not_exceeded(self):
        _, _, adapter, _, torch, _ = self.setup_controller()
        adapter.dit.blocks = [Block() for _ in range(50)]
        with self.assertRaisesRegex(ValueError, 'BOUNDED'):
            NativeNoGradParameterReuse(SimpleNamespace(rollout=lambda:None), adapter,
                retained_main_blocks=50, logical_slot=lambda:0, module_type=Block, torch_module=torch)


if __name__ == '__main__':
    unittest.main()
