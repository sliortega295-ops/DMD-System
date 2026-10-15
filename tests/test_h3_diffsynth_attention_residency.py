import importlib.util
import unittest

from runtime.h3_diffsynth_attention_residency import GPUCacheBudget


class BudgetTests(unittest.TestCase):
    def test_shared_budget_releases_and_rejects_overflow_atomically(self):
        budget = GPUCacheBudget(10)
        budget.acquire(6)
        with self.assertRaisesRegex(RuntimeError, 'BUDGET_EXCEEDED'):
            budget.acquire(5)
        self.assertEqual(budget.owned_bytes, 6)
        budget.acquire(4); budget.release(10)
        self.assertEqual((budget.owned_bytes, budget.peak_owned_bytes), (0, 10))
        with self.assertRaisesRegex(RuntimeError, 'accounting'):
            budget.release(1)

    def test_default_off_does_not_import_private_cache_or_torch(self):
        from runtime.h3_diffsynth_fsdp_attention_reuse import FSDPAttentionReuse
        pool = FSDPAttentionReuse(None, None, None, [], [], enabled=False)
        self.assertEqual(pool.finish_cycle(), {'enabled': False, 'status': 'DEFAULT_OFF_NO_PATCH'})
        with self.assertRaisesRegex(ValueError, 'requires'):
            FSDPAttentionReuse(None, None, None, [], [], enabled=False, gpu_tail_blocks=1)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('native_selected_h2d_prefetch_r1'),
                     'requires the pinned remote native cache environment')
class NativePoolMechanicsTests(unittest.TestCase):
    def make_pool(self, capacity=1 << 20):
        import torch
        import native_selected_h2d_prefetch_r1 as prefetch
        from runtime.h3_diffsynth_attention_residency import resident_pool_type
        budget = GPUCacheBudget(capacity)
        pool = resident_pool_type(prefetch, tail_blocks=1, budget=budget)(enabled=True)
        pool.cpu_test = True
        pool.target = torch.ops.aten.mm.default
        pool.blocks = {10: 0, 11: 1}
        return pool, budget

    def test_preserves_noncontiguous_storage_extent(self):
        import torch
        from runtime.h3_diffsynth_attention_residency import storage_bytes
        x = torch.empty_strided((2, 3), (10, 2))
        self.assertEqual(storage_bytes(x), 15 * x.element_size())
        self.assertEqual(storage_bytes(torch.empty(0, 3)), 0)

    def test_mixed_storage_original_and_replay_outputs_and_gradients(self):
        import torch
        pool, budget = self.make_pool()
        q = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        k = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 11
        expected = q @ k
        for block in range(2):
            with pool.capture(('studentG0', block), uses=2):
                for _ in range(2):
                    self.assertTrue(torch.equal(q @ k, expected))
        fake = [pool.fake_contexts(('fakeG0', block), lambda: (__import__('contextlib').nullcontext(),
                                                            __import__('contextlib').nullcontext()))
                for block in range(2)]
        for forward, _ in fake:
            with forward:
                for _ in range(2):
                    self.assertTrue(torch.equal(q @ k, expected))
        for _, replay in reversed(fake):
            with replay:
                for _ in range(2):
                    self.assertTrue(torch.equal(q @ k, expected))
        student = [pool.contexts_from_source(('studentG0', block)) for block in range(2)]
        for forward, _ in student:
            a = q.detach().clone().requires_grad_(); b = k.detach().clone().requires_grad_()
            with forward:
                result = [(a @ b) for _ in range(2)]
            self.assertTrue(all(torch.equal(value, expected) for value in result))
            sum(value.sum() for value in result).backward()
            ar = q.detach().clone().requires_grad_(); br = k.detach().clone().requires_grad_()
            (2 * (ar @ br).sum()).backward()
            self.assertTrue(torch.equal(a.grad, ar.grad)); self.assertTrue(torch.equal(b.grad, br.grad))
        for _, replay in reversed(student):
            with replay:
                for _ in range(2):
                    self.assertTrue(torch.equal(q @ k, expected))
        report = pool.finish_selected()
        self.assertEqual(report['GPU_cache']['source_entries'], 4)
        self.assertEqual(report['GPU_cache']['consumer_entries'], 6)
        self.assertEqual(report['GPU_cache']['released_entries'], 4)
        self.assertEqual(budget.owned_bytes, 0)
        self.assertEqual(report['counters']['actual_D2H_source_block_groups'], 2)
        self.assertEqual(report['counters']['source_block_groups_total'], 4)
        self.assertFalse(torch.cuda.is_initialized())

    def test_budget_failure_never_changes_to_cpu_silently(self):
        import torch
        pool, budget = self.make_pool(capacity=1)
        with self.assertRaisesRegex(RuntimeError, 'BUDGET_EXCEEDED'):
            with pool.capture(('studentG0', 1), uses=2):
                torch.ones(3, 4) @ torch.ones(4, 2)
        self.assertEqual(budget.owned_bytes, 0)
        self.assertFalse(pool.buffers)
