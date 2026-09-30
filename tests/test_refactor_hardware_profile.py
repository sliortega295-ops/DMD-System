from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from distillschedule.runtime.hardware_profile import (
    MicrobatchCommitEngine,
    hardware_profile,
    runtime_physical_world_size,
)


class RecordingEngine:
    def __init__(self):
        self.global_steps = 2
        self.module = object()
        self.backward_values = []
        self.commits = 0
        self.boundaries = []

    def backward(self, loss):
        self.backward_values.append(loss)

    def step(self):
        self.global_steps += 1
        self.commits += 1

    def set_gradient_accumulation_boundary(self, *, is_boundary):
        self.boundaries.append(is_boundary)


class HardwareProfileTests(unittest.TestCase):
    def test_original_default_is_world16(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime_physical_world_size(), 16)
        p = hardware_profile("a100-16-b16")
        self.assertEqual(p.physical_world_size, 16)
        self.assertEqual(p.microbatch_count, 1)
        self.assertEqual(p.logical_slots(15), (15,))

    def test_h100_is_a_bijection_of_the_original_logical_slots(self):
        p = hardware_profile("h100-8-b16")
        self.assertEqual(p.global_batch, 16)
        self.assertEqual(p.microbatch_count, 2)
        self.assertEqual(p.logical_slots(3), (3, 11))
        slots = [slot for rank in range(8) for slot in p.logical_slots(rank)]
        self.assertEqual(sorted(slots), list(range(16)))
        self.assertEqual(p.mesh_shape("world16-fsdp"), (8,))
        self.assertEqual(p.mesh_shape("node-hsdp"), (2, 4))

    def test_fail_closed_on_unknown_hardware_and_rank(self):
        with self.assertRaises(ValueError):
            hardware_profile("h100-auto")
        with self.assertRaises(ValueError):
            hardware_profile("h100-8-b16").logical_slots(8)

    def test_environment_override_is_explicit(self):
        with patch.dict(os.environ, {"DISTILLSCHEDULE_HARDWARE_PROFILE": "h100-8-b16"}):
            self.assertEqual(runtime_physical_world_size(), 8)
        with patch.dict(os.environ, {"DISTILLSCHEDULE_HARDWARE_PROFILE": "unknown"}):
            with self.assertRaises(ValueError):
                runtime_physical_world_size()

    def test_two_b1_rows_make_one_commit_and_mean_loss(self):
        raw = RecordingEngine()
        engine = MicrobatchCommitEngine(raw, microbatch_count=2)
        with engine.microbatch(0):
            engine.backward(6.0)
            engine.step()
            self.assertEqual(raw.commits, 0)
        with engine.microbatch(1):
            engine.backward(10.0)
            engine.step()
        self.assertEqual(raw.backward_values, [3.0, 5.0])
        self.assertEqual(raw.commits, 1)
        self.assertEqual(engine.global_steps, 3)

    def test_inner_gan_boundary_cannot_commit_the_first_outer_microbatch(self):
        raw = RecordingEngine()
        engine = MicrobatchCommitEngine(raw, microbatch_count=2)
        for index in range(2):
            with engine.microbatch(index):
                engine.set_gradient_accumulation_boundary(is_boundary=False)
                engine.backward(2.0)
                engine.set_gradient_accumulation_boundary(is_boundary=True)
                engine.backward(4.0)
                engine.step()
        self.assertEqual(raw.backward_values, [1.0, 2.0, 1.0, 2.0])
        self.assertEqual(raw.commits, 1)

    def test_cannot_skip_reorder_or_duplicate_a_microbatch(self):
        engine = MicrobatchCommitEngine(RecordingEngine(), microbatch_count=2)
        with self.assertRaises(RuntimeError):
            with engine.microbatch(1):
                pass
        with engine.microbatch(0):
            engine.step()
        with self.assertRaises(RuntimeError):
            with engine.microbatch(0):
                pass

    def test_missing_step_poisoned_engine_cannot_be_reused(self):
        engine = MicrobatchCommitEngine(RecordingEngine(), microbatch_count=2)
        with self.assertRaises(RuntimeError):
            with engine.microbatch(0):
                engine.backward(1.0)
        with self.assertRaises(RuntimeError):
            with engine.microbatch(0):
                engine.step()


if __name__ == "__main__":
    unittest.main()
