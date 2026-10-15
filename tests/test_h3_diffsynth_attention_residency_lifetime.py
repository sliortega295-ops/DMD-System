"""Control-flow checks, not CUDA allocator or full-model correctness evidence."""
import collections
import sys
import types
import unittest
from unittest.mock import patch

from runtime.h3_diffsynth_attention_residency import GPUCacheBudget, resident_pool_type


class ConsumerLifetimeTests(unittest.TestCase):
    def make_pool(self, *, prefetched=False, remaining=1, producer_stream=11):
        stream = types.SimpleNamespace(cuda_stream=22)
        records = []

        class Tensor:
            is_cuda = True

            def detach(self):
                # New metadata wrapper, same simulated storage lifetime ledger.
                return Tensor()

            def record_stream(self, consumer):
                records.append(consumer.cuda_stream)

        class Parent:
            def __init__(self, *, enabled):
                self.cpu_test = False
                self.prefetched = {}
                self.counters = collections.Counter()

            def restore(self, entry, args, kwargs):
                # The guard must precede the parent dropping the last owner.
                if records != [stream.cuda_stream] * (remaining - entry['remaining'] + 1):
                    raise AssertionError('consumer stream was not guarded before restore')
                entry['remaining'] -= 1
                return tuple(entry['values'])

        torch = types.ModuleType('torch')
        torch.Tensor = Tensor
        torch.cuda = types.SimpleNamespace(current_stream=lambda device: stream)
        pytree = types.ModuleType('torch.utils._pytree')
        pytree.tree_flatten = lambda value: (list(value), None)
        replay = types.ModuleType('native_sdpa_cpu_replay_r1')
        replay.signature = lambda value: value
        selected = types.ModuleType('selected_attention_pool_r1')
        selected._PHASE = types.SimpleNamespace(get=lambda: 'checkpoint_replay')
        modules = {'torch': torch, 'torch.utils._pytree': pytree,
                   'native_sdpa_cpu_replay_r1': replay,
                   'selected_attention_pool_r1': selected}
        budget = GPUCacheBudget(16)
        budget.acquire(16)
        entry = {'values': [Tensor(), 17], 'remaining': remaining, 'device': 'cuda:0',
                 'GPU_cache_owned_bytes': 16, 'GPU_cache_logical_bytes': 16,
                 'GPU_cache_producer_stream': producer_stream}
        with patch.dict(sys.modules, modules):
            pool_type = resident_pool_type(types.SimpleNamespace(PrefetchPool=Parent),
                                           tail_blocks=1, budget=budget)
            pool = pool_type(enabled=True)
        if prefetched:
            key = (id(entry), remaining, 'checkpoint_replay')
            pool.prefetched[key] = (entry, list(entry['values']), None, ('fakeG0', 49))
        return pool, entry, budget, records, modules

    def test_on_demand_head_is_guarded_before_final_release(self):
        pool, entry, budget, records, modules = self.make_pool()
        original_wrapper = entry['values'][0]
        with patch.dict(sys.modules, modules):
            restored = pool.restore(entry, (), {})
        self.assertIsNot(restored[0], original_wrapper)
        self.assertEqual(records, [22])
        self.assertEqual(pool.residency['consumer_stream_guards'], 1)
        self.assertEqual(pool.residency['on_demand_head_stream_guards'], 1)
        self.assertEqual(pool.residency['producer_consumer_stream_changes'], 1)
        self.assertEqual(pool.residency['released_entries'], 1)
        self.assertEqual(entry['values'], [None, None])
        self.assertEqual(budget.owned_bytes, 0)

    def test_prefetched_storage_and_same_stream_are_accounted(self):
        pool, entry, budget, records, modules = self.make_pool(prefetched=True, producer_stream=22)
        with patch.dict(sys.modules, modules):
            pool.restore(entry, (), {})
        self.assertEqual(records, [22])
        self.assertEqual(pool.residency['on_demand_head_stream_guards'], 0)
        self.assertEqual(pool.residency['producer_consumer_stream_changes'], 0)
        self.assertEqual(budget.owned_bytes, 0)

    def test_two_consumers_keep_owner_until_final_consume(self):
        pool, entry, budget, records, modules = self.make_pool(remaining=2)
        with patch.dict(sys.modules, modules):
            pool.restore(entry, (), {})
            self.assertEqual(budget.owned_bytes, 16)
            self.assertEqual(pool.residency['released_entries'], 0)
            pool.restore(entry, (), {})
        self.assertEqual(records, [22, 22])
        self.assertEqual(pool.residency['consumer_entries'], 2)
        self.assertEqual(pool.residency['released_entries'], 1)
        self.assertEqual(budget.owned_bytes, 0)
