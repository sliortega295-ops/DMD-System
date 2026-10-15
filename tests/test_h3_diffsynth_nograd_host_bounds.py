"""CPU mechanics and real non-reentrant checkpoint tests, not GPU admission."""
from collections import Counter
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from runtime import h3_diffsynth_nograd_host_bounds as bounds

try:
    import torch
except ImportError:
    torch = None


class FakeBounds:
    def __init__(self, values=(0, 5, 8)):
        self.values = list(values)
        self._version = 0
        self.reads = 0

    def tolist(self):
        self.reads += 1
        return list(self.values)


class FakeEvent:
    def __init__(self, done=False):
        self.done = done
        self.waits = 0
        self.records = 0

    def query(self):
        return self.done

    def record(self):
        self.records += 1

    def synchronize(self):
        self.waits += 1
        self.done = True


class BoundsTests(unittest.TestCase):
    def test_default_off_does_not_import_or_mutate_native(self):
        native = SimpleNamespace(unchanged=True)
        before = vars(native).copy()
        with patch.dict('sys.modules', {'torch': None}):
            self.assertIsNone(bounds.install(native, None))
        self.assertEqual(vars(native), before)

    def test_once_per_object_and_once_per_call(self):
        tensor = FakeBounds()
        count = Counter()
        frame = bounds.BoundsFrame(count)
        self.assertEqual(frame.read(tensor), [0, 5, 8])
        self.assertEqual(frame.read(tensor), [0, 5, 8])
        self.assertEqual(tensor.reads, 1)
        frame.close()
        self.assertFalse(frame.entries)
        bounds.BoundsFrame(count).read(tensor)
        self.assertEqual(tensor.reads, 2)

    def test_different_equal_version_tensors_do_not_alias(self):
        frame = bounds.BoundsFrame(Counter())
        a, b = FakeBounds(), FakeBounds((0, 3, 9))
        self.assertNotEqual(frame.read(a), frame.read(b))
        self.assertEqual((a.reads, b.reads), (1, 1))

    def test_mutation_on_hit_fails_closed(self):
        frame = bounds.BoundsFrame(Counter()); tensor = FakeBounds()
        frame.read(tensor); tensor._version += 1
        with self.assertRaisesRegex(RuntimeError, 'MUTATED'):
            frame.read(tensor)

    def test_mutation_at_exit_fails_closed(self):
        frame = bounds.BoundsFrame(Counter()); tensor = FakeBounds()
        frame.read(tensor); tensor._version += 1
        with self.assertRaisesRegex(RuntimeError, 'MUTATED_AT_EXIT'):
            frame.close()

    def test_queue_bounded_and_waits_only_at_capacity(self):
        count = Counter(); queue = bounds.BlockQueue(FakeEvent, count)
        for _ in range(4): queue.before(); queue.after()
        self.assertEqual(count['oldest_nograd_block_host_waits'], 0)
        oldest = queue.events[0]
        queue.before(); queue.after()
        self.assertEqual(oldest.waits, 1)
        self.assertEqual(count['peak_queued_blocks'], 4)
        self.assertEqual(len(queue.events), 4)

    def test_completed_events_do_not_wait(self):
        count = Counter(); queue = bounds.BlockQueue(lambda: FakeEvent(True), count)
        for _ in range(9): queue.before(); queue.after()
        queue.retire()
        self.assertFalse(queue.events)
        self.assertEqual(count['oldest_nograd_block_host_waits'], 0)

    def test_no_unregistered_window_search(self):
        with self.assertRaises(ValueError): bounds.BlockQueue(FakeEvent, Counter(), window=8)

    def test_unknown_source_fails_before_torch(self):
        native = SimpleNamespace(__file__=__file__)
        with self.assertRaisesRegex(ValueError, 'SOURCE_MISMATCH'):
            bounds.install(native, None, enabled=True)


def fixture_attention(q, k, v, cu_seqlens, softmax_scale):
    bounds = cu_seqlens.tolist()
    return q.sin() + k * v + bounds[-1] * softmax_scale


@unittest.skipIf(torch is None, 'actual Torch unavailable locally; remote CPU gate required')
class TorchIntegrationTests(unittest.TestCase):
    def fixture(self):
        from torch.utils.checkpoint import checkpoint
        native = SimpleNamespace(__file__=__file__, _sdpa_varlen_attention=fixture_attention)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.projection = torch.nn.Linear(4, 4)

            def forward(self, x, cu):
                return native._sdpa_varlen_attention(self.projection(x), x, x, cu, 0.25)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.blocks = torch.nn.ModuleList([Block(), Block()])

            def forward(self, x, cu, use_checkpoint=False):
                for block in self.blocks:
                    x = checkpoint(block, x, cu, use_reentrant=False) if use_checkpoint else block(x, cu)
                return x

        native.MiniMaxH3DiT = Model
        return native, Model()

    def install(self, native, model):
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with patch.object(bounds, 'DIT_SHA', digest):
            return bounds.install(native, model, enabled=True)

    def test_nograd_outputs_and_call_counts_exact(self):
        native, model = self.fixture()
        x = torch.randn(3, 4); cu = torch.tensor([0, 1, 3], dtype=torch.int32)
        with torch.no_grad(): expected = model(x, cu)
        runtime = self.install(native, model)
        with patch.object(torch.cuda, 'Event', side_effect=lambda **kw: FakeEvent(True)), torch.no_grad():
            self.assertTrue(torch.equal(model(x, cu), expected))
            self.assertTrue(torch.equal(model(x, cu), expected))
        row = runtime.finish_cycle()
        self.assertEqual(row['counts']['actual_nograd_bounds_host_reads'], 2)
        self.assertEqual(row['counts']['cached_bounds_hits'], 2)
        self.assertEqual(row['counts']['nograd_main_blocks'], 4)
        self.assertEqual(row['counts']['nograd_transformer_calls'], 2)
        self.assertEqual(row['active_calls'], 0)

    def test_real_checkpoint_output_and_gradients_unchanged(self):
        native, model = self.fixture()
        original_x = torch.randn(3, 4)
        cu = torch.tensor([0, 1, 3], dtype=torch.int32)
        x = original_x.clone().requires_grad_()
        expected = model(x, cu, use_checkpoint=True); expected.sum().backward()
        dx = x.grad.clone(); dp = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        runtime = self.install(native, model)
        y = original_x.clone().requires_grad_()
        actual = model(y, cu, use_checkpoint=True); actual.sum().backward()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(y.grad, dx))
        self.assertTrue(all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), dp)))
        self.assertEqual(runtime.counts['actual_nograd_bounds_host_reads'], 0)
        self.assertEqual(runtime.counts['nograd_main_blocks'], 0)
        self.assertGreaterEqual(runtime.counts['original_grad_or_unscoped_bounds_reads'], 2)
        self.assertEqual(runtime.finish_cycle()['queued_blocks_after_terminal_sync'], 0)


if __name__ == '__main__': unittest.main()
