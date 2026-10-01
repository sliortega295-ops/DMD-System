"""CPU-only admission checks for the audited Wan runtime compatibility port.

These checks do not substitute for real Torch/FSDP or numeric parity gates.
They execute the pure census validators and check that the opt-in compatibility
surface does not remove newer Helios/GAN features or legacy defaults.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

from distillschedule.runtime import fsdp2_p0


ROOT = Path(__file__).resolve().parents[1]


def function(path: str, name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.parse((ROOT / path).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def keyword_defaults(path: str, name: str) -> dict[str, ast.expr | None]:
    node = function(path, name)
    return dict(zip(
        (argument.arg for argument in node.args.kwonlyargs),
        node.args.kw_defaults,
        strict=True,
    ))


def executable_function(path: str, name: str, namespace: dict):
    """Execute the actual function body without importing the Torch-only module."""
    tree = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        function(path, name),
    ], type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, path, "exec"), namespace)
    return namespace[name]


class WanRuntimeCompatibilityTests(unittest.TestCase):
    def test_exact_census_executes_and_rejects_incomplete_receipts(self) -> None:
        row = {
            "call_count": 4,
            "stage_call_count": {0: 2, 1: 1, 2: 1},
            "logical_call_count": 4,
            "logical_stage_call_count": {0: 2, 1: 1, 2: 1},
            "dummy_call_count": 0,
        }
        validator = fsdp2_p0.validate_exact_rollout_call_inventory
        self.assertEqual(validator(row, label="B1"), row)
        with self.assertRaises(ValueError):
            validator({key: value for key, value in row.items()
                       if key != "logical_call_count"}, label="B1")
        with self.assertRaises(AssertionError):
            validator(dict(row, dummy_call_count=1), label="B1")
        with self.assertRaises(TypeError):
            validator(dict(row, call_count=True), label="B1")

    def test_expected_census_comes_from_variable_trace_exits(self) -> None:
        validation = {
            "status": "PASS",
            "semantics": "official_helios_normal_rank0_broadcast_strict",
            "cross_rank_call_count_parity": True,
            "local_exception_entry_count": 0,
            "rank_count": 16,
            "operation_count": 1,
            "global_entry_count": 16,
            "operations": [{
                "operation": "step/12/critic/rollout", "rank_count": 16,
                "synchronized_exit": [1, 1, 0],
                "per_stage_call_counts": [2, 2, 1],
                "logical_call_count": 5, "physical_call_count": 5,
            }],
        }
        inventory = fsdp2_p0.expected_trace_rollout_call_inventory(
            validation, operation="step/12/critic/rollout", expected_rank_count=16
        )
        self.assertEqual(inventory["call_count"], 5)
        self.assertEqual(inventory["stage_call_count"], {0: 2, 1: 2, 2: 1})
        with self.assertRaises(ValueError):
            fsdp2_p0.expected_trace_rollout_call_inventory(
                validation, operation="step/12/critic/rollout", expected_rank_count=8
            )
        validation["operations"][0]["physical_call_count"] = 6
        with self.assertRaises(ValueError):
            fsdp2_p0.expected_trace_rollout_call_inventory(
                validation, operation="step/12/critic/rollout"
            )

    def test_clean_rollout_opt_in_preserves_sync_and_cache_features(self) -> None:
        gst = "runtime/gst_pool_helios.py"
        defaults = keyword_defaults(gst, "run_generator_update_trace_rollout")
        self.assertFalse(ast.literal_eval(defaults["clean_runtime"]))
        self.assertIsNone(ast.literal_eval(defaults["runtime_cache"]))
        self.assertTrue(ast.literal_eval(defaults["empty_cache_before_rollout"]))
        self.assertTrue(ast.literal_eval(defaults["synchronize_after_rollout"]))
        rollout = "runtime/static_fsdp2_p0_helios.py"
        defaults = keyword_defaults(rollout, "run_generator_trace_rollout_batch")
        self.assertFalse(ast.literal_eval(defaults["clean_runtime"]))
        self.assertIn("use_cross_kv_cache", defaults)
        self.assertIn("use_static_context_cache", defaults)
        self.assertTrue(ast.literal_eval(defaults["synchronize_after_rollout"]))
        source = (ROOT / rollout).read_text()
        self.assertIn("generator_nograd_cross_kv_cache_scope(", source)
        self.assertIn("generator_nograd_static_context_cache_scope(", source)
        self.assertIn("generator_nograd_residual_fusion_scope()", source)
        self.assertIn("validate_exact_rollout_call_inventory(", source)

    def test_deferred_validation_and_scalars_are_explicit_opt_ins(self) -> None:
        defaults = keyword_defaults("runtime/gst_pool_helios.py", "score_forward")
        self.assertTrue(ast.literal_eval(defaults["validate_finite"]))
        defaults = keyword_defaults("runtime/critic_pool_helios.py", "student_forward_loss")
        self.assertTrue(ast.literal_eval(defaults["validate_finite"]))
        self.assertIsNone(ast.literal_eval(defaults["deferred_finite_checks"]))
        path = "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
        for name in ("_run_nogan_generator_update_p0", "_run_nogan_critic_precomputed"):
            defaults = keyword_defaults(path, name)
            self.assertFalse(ast.literal_eval(defaults["collect_scalars"]))
            self.assertIsNone(ast.literal_eval(defaults["runtime_cache"]))
            self.assertIsNone(ast.literal_eval(defaults["deferred_finite_checks"]))

    def test_hardened_score_wrapper_preserves_finite_validation_option(self) -> None:
        path = "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py"
        model, condition, prediction = object(), object(), object()
        batch = {"hidden_states": SimpleNamespace(shape=(1, 16, 9, 48, 80))}
        calls, markers = [], []

        def original_score(model_arg, batch_arg, *, condition, adapter_enabled,
                           validate_finite=True):
            calls.append((model_arg, batch_arg, condition, adapter_enabled,
                          validate_finite))
            return prediction

        torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True,
            nvtx=SimpleNamespace(
                range_push=lambda label: markers.append(("push", label)),
                range_pop=lambda: markers.append(("pop",)),
            ),
        ))
        argv = []
        namespace = {"torch": torch, "sys": SimpleNamespace(argv=argv),
                     "_original_score_forward": original_score}
        executable_function(path, "_nvtx_call", namespace)
        run = executable_function(path, "_profiled_score_forward", namespace)
        for profile in (False, True):
            argv[:] = ["worker", "--profile-mode"] if profile else ["worker"]
            for option in ({}, {"validate_finite": True}, {"validate_finite": False}):
                with self.subTest(profile=profile, option=option):
                    calls.clear()
                    markers.clear()
                    result = run(model, batch, condition=condition,
                                 adapter_enabled=False, **option)
                    self.assertIs(result, prediction)
                    self.assertEqual(calls, [(model, batch, condition, False,
                                             option.get("validate_finite", True))])
                    self.assertEqual(markers, [
                        ("push", "P0_SCORE/nogan/B1/adapter-0"), ("pop",)
                    ] if profile else [])

    def test_persistent_unshard_and_triple_return_are_preserved(self) -> None:
        path = "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
        node = function(path, "_precompute_critic_rollouts")
        defaults = keyword_defaults(path, node.name)
        self.assertIsNone(ast.literal_eval(defaults["entry_outputs"]))
        self.assertFalse(ast.literal_eval(defaults["clean_runtime"]))
        self.assertIsInstance(node.body[-1], ast.Return)
        self.assertEqual([item.id for item in node.body[-1].value.elts],
                         ["outputs", "records", "persistence"])
        source = ast.get_source_segment((ROOT / path).read_text(), node)
        self.assertIn("generator_nograd_persistent_unshard_scope(", source)
        self.assertIn("use_cross_kv_cache=policy.generator_nograd_cross_kv_cache", source)
        self.assertIn("entry_outputs[int(step)] = entry", source)

    def test_missing_pyramid_initialization_is_seeded_and_opt_in(self) -> None:
        path = "runtime/critic_pool_helios.py"
        defaults = keyword_defaults(path, "build_helios_pool_model")
        self.assertIsNone(ast.literal_eval(defaults["missing_pyramid_patch_seed"]))
        source = (ROOT / path).read_text()
        self.assertIn("with torch.random.fork_rng(devices=[]):", source)
        self.assertIn("torch.manual_seed(int(missing_pyramid_patch_seed))", source)
        self.assertIn('module.to_empty(device="cpu")', source)
        self.assertIn("module.to(dtype=torch.bfloat16)", source)
        self.assertIn("pretrained Transformer left meta parameters", source)

    def test_deferred_finite_flag_does_not_force_host_boolean(self) -> None:
        class Flag:
            def __init__(self, finite: bool):
                self.finite = finite
                self.host_reads = 0

            def __bool__(self):
                self.host_reads += 1
                return self.finite

        class Model:
            def enable_adapters(self):
                pass

            def train(self):
                pass

        class Engine:
            module = Model()

            def __call__(self, **kwargs):
                return (object(),)

        loss = SimpleNamespace(requires_grad=True, grad_fn=object())
        flag = Flag(False)
        run = executable_function(
            "runtime/critic_pool_helios.py", "student_forward_loss",
            {"torch": SimpleNamespace(isfinite=lambda value: flag),
             "critic_denoising_loss": lambda *args: loss},
        )
        batch = {key: object() for key in (
            "hidden_states", "timestep", "encoder_hidden_states", "indices_hidden_states",
            "indices_latents_history_short", "indices_latents_history_mid",
            "indices_latents_history_long", "latents_history_short", "latents_history_mid",
            "latents_history_long", "critic_noise", "generated_x0",
        )}
        deferred = []
        _, observed_loss = run(
            Engine(), batch, validate_finite=False, deferred_finite_checks=deferred
        )
        self.assertIs(observed_loss, loss)
        self.assertEqual(deferred, [flag])
        self.assertEqual(flag.host_reads, 0)
        with self.assertRaisesRegex(AssertionError, "not finite"):
            run(Engine(), batch)
        self.assertEqual(flag.host_reads, 1)

    def test_preloaded_critic_entry_reuses_input_and_commits_once(self) -> None:
        observed = []

        class Engine:
            global_steps = 10

            def backward(self, loss):
                observed.append(("backward", loss))

            def step(self):
                observed.append(("step",))
                self.global_steps += 1

        class Trace:
            def load_critic_entry(self, **kwargs):
                raise AssertionError("preloaded entry must not be reloaded")

        entry, cache, loss = object(), {}, object()

        def prepare(**kwargs):
            self.assertIs(kwargs["loaded"], entry)
            self.assertIs(kwargs["runtime_cache"], cache)
            return {"entry": kwargs["loaded"]}

        def forward(engine, batch, **kwargs):
            self.assertIs(kwargs["deferred_finite_checks"], deferred)
            self.assertFalse(kwargs["validate_finite"])
            deferred.append(True)
            return object(), loss

        deferred = []
        run = executable_function(
            "scripts/distillschedule/probe_static_fsdp2_p0_world16.py",
            "_run_nogan_critic_precomputed",
            {"prepare_student_batch": prepare, "student_forward_loss": forward,
             "TIMING_OMITTED": "omitted"},
        )
        result = run(
            rank=3, step=10, x0=object(), device=object(), critic_engine=Engine(),
            trace=Trace(), collect_diagnostics=False,
            diagnostics=SimpleNamespace(scope=lambda *args: nullcontext(),
                                        add_tensor_tree=lambda *args, **kwargs: None),
            entry=entry, runtime_cache=cache, deferred_finite_checks=deferred,
        )
        self.assertEqual(observed, [("backward", loss), ("step",)])
        self.assertEqual(result["physical_critic_step_after"], 11)
        self.assertEqual(result["loss"], "omitted")
        self.assertEqual(deferred, [True])


if __name__ == "__main__":
    unittest.main()
