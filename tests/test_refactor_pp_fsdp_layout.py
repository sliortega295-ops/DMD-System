"""PP stages are not extra samples; FSDP ranks are independent data lanes."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'src/distillschedule/runtime/pp_fsdp_layout.py'


class LayoutTests(unittest.TestCase):
    def layout(self, shard):
        import sys
        spec = importlib.util.spec_from_file_location('pp_fsdp_layout_test', PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.PPFSDPLayout(shard=shard)

    def test_two_waves_have_sixteen_unique_samples(self):
        for shard in (4, 8):
            layout = self.layout(shard)
            samples = [layout.source_rank(r, m) for r in range(8) for m in range(2)]
            self.assertEqual(sorted(samples), list(range(16)))
            for r in range(8):
                self.assertEqual(layout.source_rank(r, 0), layout.source_rank(r + 8, 0))
                self.assertEqual(layout.source_rank(r, 1), layout.source_rank(r + 8, 1))
            self.assertEqual(layout.receipt()['global_batch'], 16)
            self.assertEqual(layout.receipt()['microbatch_size'], 1)
            self.assertEqual(layout.receipt()['TP'], 1)

    def test_fsdp_collectives_never_span_pipeline_stages(self):
        for shard in (4, 8):
            layout = self.layout(shard)
            for group in layout.groups()['fsdp']:
                self.assertEqual(len(group), shard)
                self.assertEqual(len({r // 8 for r in group}), 1)
            self.assertEqual(layout.groups()['pp'], [[i, i + 8] for i in range(8)])

    def test_capacity_ladder_does_not_change_data_or_pipeline(self):
        a, b = self.layout(4), self.layout(8)
        self.assertEqual(a.replicate, 2)
        self.assertEqual(b.replicate, 1)
        for r in range(16):
            for m in range(2):
                self.assertEqual(a.source_rank(r, m), b.source_rank(r, m))
        self.assertEqual(a.block_ranges(30), ((0, 15), (15, 30)))
        self.assertEqual(a.block_ranges(50), ((0, 25), (25, 50)))

    def test_unsupported_shapes_and_ranks_fail_closed(self):
        layout = self.layout(4)
        for rank, wave in ((16, 0), (-1, 0), (0, 2)):
            with self.assertRaises(ValueError):
                layout.source_rank(rank, wave)
        with self.assertRaises(ValueError):
            layout.block_ranges(31)
        with self.assertRaises(ValueError):
            self.layout(16)


if __name__ == '__main__':
    unittest.main()
