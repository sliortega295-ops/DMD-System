"""CPU executable ordering/normalization gates for the H100 Helios port.

GPU FSDP, latent parity, and capacity remain separate admission gates.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from distillschedule.runtime.fsdp2_p0 import mesh_shape_for_policy
from distillschedule.runtime.hardware_profile import (
    MicrobatchCommitEngine,
    hardware_profile,
)
from distillschedule.runtime.helios_critic_scalar_logs import (
    TIMING_OMITTED,
    install_critic_scalar_log_elision,
    source_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"


def load_functions(namespace):
    names = {
        "_aggregate_logical_records", "_run_logical_generator_update_p0",
        "_precompute_logical_critic_rollouts", "_run_logical_critic_update_p0",
    }
    functions = [node for node in ast.parse(WORKER.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise AssertionError("H100 logical microbatch helpers are missing")
    tree = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *functions,
    ], type_ignores=[])
    ast.fix_missing_locations(tree)
    namespace.update(MicrobatchCommitEngine=MicrobatchCommitEngine,
                     nullcontext=nullcontext)
    exec(compile(tree, str(WORKER), "exec"), namespace)
    return namespace


def load_function(name, namespace):
    node = next(node for node in ast.parse(WORKER.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    tree = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        node,
    ], type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(WORKER), "exec"), namespace)
    return namespace[name]


class Engine:
    def __init__(self, role, version, events):
        self.module = object()
        self.role = role
        self.global_steps = version
        self.events = events
        self.backward_values = []
        self.boundaries = []

    def backward(self, value):
        self.backward_values.append(value)

    def step(self):
        self.events.append(("commit", self.role, self.global_steps))
        self.global_steps += 1

    def set_gradient_accumulation_boundary(self, *, is_boundary):
        self.boundaries.append(is_boundary)


class ScalarLog:
    def __init__(self, value):
        self.value = value
        self.item_count = 0

    def detach(self):
        return self

    def item(self):
        self.item_count += 1
        return self.value


def _critic_loss(
    *,
    fake_score_model,
    critic_engine,
    scalar_values,
    pipeline_micro_wave: int = None,
):
    """Legacy-signature fixture exercises the real source-pinned rewriter."""
    denoising_loss, gan_D_loss, r1_loss, r2_loss = scalar_values
    critic_engine.backward(6.0)
    critic_engine.step()
    return 6.0, {
        "denoising_loss": denoising_loss.detach().item(),
        "gan_D_loss": gan_D_loss.detach().item(),
        "r1_loss": r1_loss.detach().item(),
        "r2_loss": r2_loss.detach().item(),
        "critic_optimizer_committed": True,
    }


class HeliosH100AccumulationTests(unittest.TestCase):
    def _run_scalar_api_case(self, *, install_patch, collect):
        raw = Engine("C", 1000, [])
        values = [ScalarLog(i) for i in range(4)]
        module = SimpleNamespace(_critic_loss=_critic_loss)
        receipt = install_critic_scalar_log_elision(
            enabled=install_patch,
            module=module,
            expected_source_sha256=source_sha256(_critic_loss),
        )
        run = load_function("_run_gan_critic_precomputed", {
            "TIMING_OMITTED": TIMING_OMITTED,
            "critic_official_kwargs": lambda *args:
                {"critic_engine": raw, "scalar_values": values},
        })
        trace = SimpleNamespace(
            load_critic_entry=lambda **kwargs: SimpleNamespace(operation_trace=None),
            source=lambda **kwargs: None,
        )
        diagnostics = SimpleNamespace(
            scope=lambda *args: nullcontext(),
            add_tensor_tree=lambda *args, **kwargs: None,
        )
        with patch.dict(sys.modules, {"helios.utils.utils_helios_post": module}):
            result = run(
                rank=0, step=1000, x0=None, device=None, critic_engine=raw,
                trace=trace, cfg=None, collect_diagnostics=False,
                collect_official_scalar_logs=collect, diagnostics=diagnostics,
            )
        self.assertEqual(raw.backward_values, [6.0])
        self.assertEqual(raw.global_steps, 1001)
        self.assertTrue(result["critic_optimizer_committed"])
        return values, receipt

    def test_gan_legacy_scalar_api_default_collection_needs_no_keyword(self):
        values, receipt = self._run_scalar_api_case(install_patch=False, collect=True)
        self.assertEqual([value.item_count for value in values], [1] * 4)
        self.assertFalse(receipt["enabled"])

    def test_gan_patched_scalar_api_default_still_collects_all_four(self):
        values, receipt = self._run_scalar_api_case(install_patch=True, collect=True)
        self.assertEqual([value.item_count for value in values], [1] * 4)
        self.assertEqual(receipt["collected_scalar_count"], 4)
        self.assertEqual(receipt["elided_scalar_count"], 0)

    def test_gan_patched_scalar_api_explicit_elision_preserves_commit(self):
        values, receipt = self._run_scalar_api_case(install_patch=True, collect=False)
        self.assertEqual([value.item_count for value in values], [0] * 4)
        self.assertEqual(receipt["collected_scalar_count"], 0)
        self.assertEqual(receipt["elided_scalar_count"], 4)

    def test_gan_scalar_api_elision_without_registered_hook_fails_closed(self):
        with self.assertRaisesRegex(TypeError, "collect_scalar_logs"):
            self._run_scalar_api_case(install_patch=False, collect=False)

    def test_mesh8_and_hsdp2x4_require_explicit_profile(self):
        self.assertEqual(mesh_shape_for_policy("world16-fsdp"), (16,))
        self.assertEqual(mesh_shape_for_policy("node-hsdp"), (2, 8))
        with self.assertRaises(ValueError):
            mesh_shape_for_policy("world16-fsdp", world_size=8)
        self.assertEqual(mesh_shape_for_policy(
            "world16-fsdp", world_size=8, hardware_profile_name="h100-8-b16"
        ), (8,))
        self.assertEqual(mesh_shape_for_policy(
            "node-hsdp", world_size=8, hardware_profile_name="h100-8-b16"
        ), (2, 4))
        with self.assertRaises(ValueError):
            mesh_shape_for_policy("node-hsdp", world_size=16,
                                  hardware_profile_name="h100-8-b16")
        with patch.dict(os.environ, {"DISTILLSCHEDULE_HARDWARE_PROFILE": "h100-8-b16"}):
            self.assertEqual(mesh_shape_for_policy("world16-fsdp"), (8,))
            self.assertEqual(mesh_shape_for_policy("node-hsdp"), (2, 4))

    def _exercise(self, variant, profile_name):
        events = []
        profile = hardware_profile(profile_name)
        initial_g, initial_c = (2, 10) if variant == "no-gan" else (200, 1000)
        g, c = Engine("G", initial_g, events), Engine("C", initial_c, events)
        policy = SimpleNamespace(pack_real_score_pair=False,
                                 elide_official_generator_free_memory=False,
                                 elide_critic_scalar_logs=False)
        args = SimpleNamespace(variant=variant, timing_mode=False, profile_mode=False)

        def generator(**kwargs):
            wrapped = kwargs["generator_engine"]
            expected_commit = kwargs.get("expect_optimizer_commit", True)
            events.append(("G", kwargs["rank"], wrapped.global_steps, expected_commit))
            wrapped.backward(float(kwargs["rank"] + 1))
            wrapped.step()
            return {
                "source_rank": kwargs["rank"], "logical_step": 10 if variant == "no-gan" else 1000,
                "physical_generator_step_after": wrapped.global_steps,
                "local_trainable_before": "before", "local_trainable_after": "after",
                "rollout": {"call_count": 4, "stage_call_count": {0: 2, 1: 1, 2: 1}},
                "logical_score_requests": 3 if variant == "no-gan" else 4,
                "physical_score_forwards": 3 if variant == "no-gan" else 4,
            }

        def rollout(**kwargs):
            events.append(("rollout", kwargs["rank"], kwargs["generator_engine"].global_steps))
            records = [{"logical_steps": [step], "call_count": 6,
                        "stage_call_count": {0: 2, 1: 2, 2: 2},
                        "logical_call_count": 6, "logical_stage_call_count": {0: 2, 1: 2, 2: 2},
                        "dummy_call_count": 0} for step in kwargs["steps"]]
            return ({step: (kwargs["rank"], step) for step in kwargs["steps"]},
                    records, {"enabled": False, "rollout_batch_count": len(records)})

        def critic(**kwargs):
            wrapped = kwargs["critic_engine"]
            events.append(("C", kwargs["step"], kwargs["rank"], wrapped.global_steps))
            self.assertEqual(kwargs["x0"], (kwargs["rank"], kwargs["step"]))
            if variant == "gan":
                wrapped.set_gradient_accumulation_boundary(is_boundary=False)
                wrapped.backward(2.0)
                wrapped.set_gradient_accumulation_boundary(is_boundary=True)
                wrapped.backward(4.0)
            else:
                wrapped.backward(6.0)
            wrapped.step()
            return {"logical_step": kwargs["step"], "source_rank": kwargs["rank"],
                    "physical_critic_step_after": wrapped.global_steps, "score_forwards": 2 if variant == "gan" else 1}

        namespace = load_functions({
            "_run_nogan_generator_update_p0": generator,
            "_run_nogan_critic_precomputed": critic,
            "_run_gan_critic_precomputed": critic,
            "_precompute_critic_rollouts": rollout,
            "base": SimpleNamespace(_run_gan_generator_update=generator),
        })
        gen = namespace["_run_logical_generator_update_p0"](
            args=args, rank=3, hardware=profile, device=None, generator_engine=g,
            critic_engine=c, trace=None, cfg=None, negative=None,
            collect_diagnostics=False, diagnostics=None, policy=policy,
            score_pack_parity_check=False, generator_free_memory=None,
        )
        steps = tuple(range(initial_c, initial_c + 5))
        outputs, batches, persistence = namespace["_precompute_logical_critic_rollouts"](
            rank=3, hardware=profile, steps=steps, device=None, generator_engine=g,
            trace=None, policy=policy, collect_diagnostics=False,
            diagnostics=None, rollout_parity_check=False,
        )
        ema_updates = 0
        records = []
        for index, step in enumerate(steps):
            records.append(namespace["_run_logical_critic_update_p0"](
                args=args, rank=3, hardware=profile, step=step, rollout_outputs=outputs,
                device=None, critic_engine=c, trace=None, cfg=None,
                collect_diagnostics=False, diagnostics=None, policy=policy,
            ))
            if variant == "no-gan" or index == 0:
                ema_updates += 1
                events.append(("EMA", c.global_steps))
        return profile, g, c, gen, batches, persistence, records, ema_updates, events

    def test_nogan_two_b1_rows_keep_versions_until_one_commit(self):
        profile, g, c, gen, batches, _, records, ema_updates, events = self._exercise("no-gan", "h100-8-b16")
        self.assertEqual([event for event in events if event[0] == "G"],
                         [("G", 3, 2, False), ("G", 11, 2, True)])
        self.assertEqual([event for event in events if event[0] == "rollout"],
                         [("rollout", 3, 3), ("rollout", 11, 3)])
        for step in range(10, 15):
            self.assertEqual([event for event in events if event[0] == "C" and event[1] == step],
                             [("C", step, 3, step), ("C", step, 11, step)])
        self.assertEqual(g.backward_values, [2.0, 6.0])
        self.assertEqual(c.backward_values, [3.0] * 10)
        self.assertEqual((g.global_steps, c.global_steps, ema_updates), (3, 15, 5))
        self.assertEqual(gen["physical_score_forwards"], 6)
        self.assertEqual(gen["source_ranks"], [3, 11])
        self.assertEqual(len(batches), 10)
        self.assertEqual([row["logical_slot"] for row in batches], [3] * 5 + [11] * 5)
        self.assertEqual(len(records), 5)

    def test_gan_inner_two_backwards_cannot_commit_first_row(self):
        _, g, c, gen, _, _, _, ema_updates, events = self._exercise("gan", "h100-8-b16")
        self.assertEqual(g.backward_values, [2.0, 6.0])
        self.assertEqual(c.backward_values, [1.0, 2.0] * 10)
        self.assertEqual(len([event for event in events if event[0] == "commit" and event[1] == "G"]), 1)
        self.assertEqual(len([event for event in events if event[0] == "commit" and event[1] == "C"]), 5)
        self.assertEqual((g.global_steps, c.global_steps, ema_updates), (201, 1005, 1))
        self.assertEqual(gen["physical_score_forwards"], 8)

    def test_a100_default_still_executes_one_logical_row(self):
        _, g, c, gen, batches, _, records, ema_updates, events = self._exercise("no-gan", "a100-16-b16")
        self.assertNotIn("microbatches", gen)
        self.assertEqual(g.backward_values, [4.0])
        self.assertEqual(c.backward_values, [6.0] * 5)
        self.assertEqual(gen["physical_score_forwards"], 3)
        self.assertEqual(len(batches), 5)
        self.assertEqual((g.global_steps, c.global_steps, ema_updates), (3, 15, 5))
        self.assertEqual(len(records), 5)

    def test_world_gradient_mean_is_sixteen_independent_b1_losses(self):
        all_gradients = []
        for rank in range(8):
            raw = Engine("G", 2, [])
            engine = MicrobatchCommitEngine(raw, microbatch_count=2)
            for index, slot in enumerate(hardware_profile("h100-8-b16").logical_slots(rank)):
                with engine.microbatch(index):
                    engine.backward(float(slot + 1))
                    engine.step()
            all_gradients.append(sum(raw.backward_values))
        self.assertEqual(sum(all_gradients) / 8, sum(range(1, 17)) / 16)

    def test_nogan_helper_allows_unchanged_trainables_only_before_commit(self):
        events = []
        raw = Engine("G", 2, events)
        raw.module = SimpleNamespace(version=2)
        original_step = raw.step

        def step():
            original_step()
            raw.module.version += 1

        raw.step = step
        engine = MicrobatchCommitEngine(raw, microbatch_count=2)
        run = load_function("_run_nogan_generator_update_p0", {
            "TIMING_OMITTED": "omitted",
            "base": SimpleNamespace(_profile_fingerprint=lambda module, *args: str(module.version)),
            "run_generator_update_trace_rollout": lambda **kwargs: (object(), None, {}),
            "prepare_dmd_score_batch": lambda **kwargs: {"encoder_hidden_states": object()},
            "score_forward": lambda *args, **kwargs: object(),
            "construct_dmd_gradient": lambda **kwargs: (object(), {}),
            "detached_dmd_surrogate_loss": lambda *args: 6.0,
        })
        records = []
        for index, slot in enumerate((3, 11)):
            with engine.microbatch(index):
                records.append(run(
                    rank=slot, device=None, generator_engine=engine, critic_engine=raw,
                    trace=SimpleNamespace(load_generator_entry=lambda **kwargs: object()),
                    negative=None, collect_diagnostics=True,
                    diagnostics=SimpleNamespace(scope=lambda *args: nullcontext(),
                                                add_tensor_tree=lambda *args, **kwargs: None,
                                                diagnostic_scalar=lambda label, value: value),
                    policy=SimpleNamespace(pack_real_score_pair=False,
                                           elide_explicit_rollout_sync=False),
                    score_pack_parity_check=False, expect_optimizer_commit=index == 1,
                ))
        self.assertEqual(raw.backward_values, [3.0, 3.0])
        self.assertEqual([row["optimizer_committed"] for row in records], [False, True])
        self.assertEqual(records[0]["local_trainable_before"], records[0]["local_trainable_after"])
        self.assertNotEqual(records[1]["local_trainable_before"], records[1]["local_trainable_after"])

    def test_real_gan_critic_helper_distinguishes_inner_from_outer_commit(self):
        raw = Engine("C", 1000, [])
        engine = MicrobatchCommitEngine(raw, microbatch_count=2)

        def official_critic_loss(*, critic_engine, **kwargs):
            critic_engine.set_gradient_accumulation_boundary(is_boundary=False)
            critic_engine.backward(2.0)
            critic_engine.set_gradient_accumulation_boundary(is_boundary=True)
            critic_engine.backward(4.0)
            critic_engine.step()
            return 6.0, {"critic_optimizer_committed": True}

        run = load_function("_run_gan_critic_precomputed", {
            "TIMING_OMITTED": "omitted",
            "critic_official_kwargs": lambda cfg, operation_trace, source, step, x0, engine:
                {"critic_engine": engine},
        })
        trace = SimpleNamespace(load_critic_entry=lambda **kwargs: SimpleNamespace(operation_trace=None),
                                source=lambda **kwargs: None)
        diagnostics = SimpleNamespace(scope=lambda *args: nullcontext(),
                                      add_tensor_tree=lambda *args, **kwargs: None)
        module = SimpleNamespace(_critic_loss=official_critic_loss)
        records = []
        with patch.dict(sys.modules, {"helios.utils.utils_helios_post": module}):
            for index, slot in enumerate((3, 11)):
                with engine.microbatch(index):
                    records.append(run(
                        rank=slot, step=1000, x0=None, device=None, critic_engine=engine,
                        trace=trace, cfg=None, collect_diagnostics=False,
                        collect_official_scalar_logs=False, diagnostics=diagnostics,
                        expect_optimizer_commit=index == 1,
                    ))
        self.assertEqual(raw.backward_values, [1.0, 2.0, 1.0, 2.0])
        self.assertEqual(raw.global_steps, 1001)
        self.assertEqual([row["critic_optimizer_committed"] for row in records], [False, True])
        self.assertTrue(all(row["critic_logical_microbatch_boundary_reached"] for row in records))


if __name__ == "__main__":
    unittest.main()
