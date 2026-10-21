"""Stdlib-only routing/telemetry tests, NOT CUDA/FSDP/numerical proof."""
from __future__ import annotations

import ast
import math
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from distillschedule.runtime import helios_lora_epilogue as helper


class Tensor:
    def __init__(self, shape, *, dtype="torch.bfloat16", cuda=True, contiguous=True, value=1.0):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype, self.is_cuda, self.device = dtype, cuda, "cuda:0"
        self.contiguous, self.value = contiguous, value

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def to(self, dtype):
        return Tensor(self.shape, dtype=dtype, value=self.value)

    def __add__(self, other):
        return Tensor(self.shape, dtype=self.dtype, value=self.value + other.value)

    def __mul__(self, factor):
        return Tensor(self.shape, dtype=self.dtype, value=self.value * factor)

    def add_(self, other):
        self.value += other.value
        return self


class PlainLinear:
    def __init__(self, rank=2, width=8):
        self.weight = Tensor((width, rank))
        self.bias = None
        self.call_count = self.pre_hook_count = self.post_hook_count = 0
        self.original_forward_count = 0

    def __call__(self, value, *args, **kwargs):
        self.call_count += 1
        self.pre_hook_count += 1
        out = self.forward(value, *args, **kwargs)
        self.post_hook_count += 1
        return out

    def forward(self, value, *args, **kwargs):
        self.original_forward_count += 1
        return Tensor((*value.shape[:-1], self.weight.shape[0]), value=value.value)


class FakeLinear:
    disable_adapters = False
    merged = False
    active_adapters = ["default"]
    lora_variant = {}

    def __init__(self):
        self.base_layer = PlainLinear(8, 8)
        self.lora_A = {"default": PlainLinear(8, 2)}
        self.lora_B = {"default": PlainLinear(2, 8)}
        self.lora_dropout = {"default": lambda value: value}
        self.scaling = {"default": 1.0}

    def _check_forward_args(self, value, *args, **kwargs):
        pass

    def _cast_input_dtype(self, value, dtype):
        return value.to(dtype)

    def __call__(self, value, **kwargs):
        return self.forward(value, **kwargs)

    def forward(self, value, **kwargs):
        base = self.base_layer(value)
        if self.disable_adapters:
            return base
        update = self.lora_B["default"](self.lora_A["default"](value))
        return base + update * self.scaling["default"]


class Model:
    def __init__(self):
        self.linear = FakeLinear()
        self.saved_replay = None

    def named_modules(self):
        yield "", self
        yield "linear", self.linear

    def __call__(self, value):
        return self.forward(value)

    def forward(self, value):
        if self.saved_replay == "CREATE":
            self.saved_replay = self._distillschedule_lora_checkpoint_function_wrapper(self.linear)
            return self.saved_replay(value)
        return self.linear(value)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.grad_enabled = True
        self.torch = types.SimpleNamespace(
            bfloat16="torch.bfloat16", float32="torch.float32",
            is_grad_enabled=lambda: self.grad_enabled,
        )
        self.patch = patch.object(helper, "_torch", return_value=self.torch)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.kernel_calls = []

    def install(self, model, *, candidate=True, test_shapes=((4, 8),)):
        def fused_b(h, weight, base):
            self.kernel_calls.append((h.shape, weight.shape, self.grad_enabled))
            update = Tensor(base.shape, value=h.value)
            return base + update if self.grad_enabled else base.add_(update)

        return helper.install_generator_nograd_lora_fused_epilogue(
            model, enabled=True, role="critic", linear_type=FakeLinear,
            variant_kwarg_keys=(), fused_epilogue=lambda base, update, scale: base.add_(update),
            all_role_scale1=True, checkpoint_replay_scope=True,
            lora_b_gemm_epilogue=candidate,
            lora_b_gemm_epilogue_impl=fused_b if candidate else None,
            lora_b_gemm_test_shapes=test_shapes if candidate else None,
        )

    def test_default_off_preserves_forward_identity_and_no_kernel_import(self):
        model = Model()
        before = model.forward.__func__
        receipt = helper.install_generator_nograd_lora_fused_epilogue(
            model, enabled=False, role="generator", linear_type=FakeLinear,
        )
        self.assertIs(before, model.forward.__func__)
        self.assertFalse(receipt["lora_b_gemm_epilogue_requested"])
        self.assertFalse(receipt["lora_b_gemm_epilogue_enabled"])
        self.assertNotIn("lora_b_gemm_runtime_identity", receipt)

    def test_optin_requires_all_role_and_checkpoint_scope(self):
        for enabled, all_role, cp in ((False, True, True), (True, False, False), (True, True, False)):
            with self.assertRaises(ValueError):
                helper.install_generator_nograd_lora_fused_epilogue(
                    Model(), enabled=enabled, role="generator", linear_type=FakeLinear,
                    all_role_scale1=all_role, checkpoint_replay_scope=cp,
                    lora_b_gemm_epilogue=True,
                )

    def test_existing_enabled_epilogue_default_does_not_wrap_B(self):
        model = Model()
        receipt = self.install(model, candidate=False)
        b = model.linear.lora_B["default"]
        original = b.forward.__func__
        got = model(Tensor((4, 8), value=2.0))
        self.assertEqual(got.value, 4.0)
        self.assertIs(original, b.forward.__func__)
        self.assertFalse(getattr(b, "_distillschedule_lora_b_gemm_wrapper_installed", False))
        self.assertFalse(receipt["lora_b_gemm_epilogue_enabled"])
        self.assertEqual(receipt["fused_grad_invocation_count"], 1)
        self.assertEqual(receipt["lora_b_gemm_selected_invocation_count"], 0)

    def test_injected_shapes_cannot_enter_production(self):
        with self.assertRaises(ValueError):
            helper.install_generator_nograd_lora_fused_epilogue(
                Model(), enabled=False, role="generator", lora_b_gemm_test_shapes=((4, 8),),
            )

    def test_selected_path_retains_B_module_call_and_hooks(self):
        model = Model()
        receipt = self.install(model)
        output = model(Tensor((4, 8), value=2.0))
        b = model.linear.lora_B["default"]
        self.assertEqual(output.value, 4.0)
        self.assertEqual((b.call_count, b.pre_hook_count, b.post_hook_count), (1, 1, 1))
        self.assertEqual(b.original_forward_count, 0)
        self.assertEqual(len(self.kernel_calls), 1)
        self.assertEqual(receipt["lora_b_gemm_selected_invocation_count"], 1)
        self.assertEqual(receipt["lora_b_gemm_kernel_call_count"], 1)
        self.assertEqual(receipt["lora_b_gemm_selected_shape_counts"], {"M4_N8_R2": 1})
        self.assertEqual(receipt["fused_grad_invocation_count"], 1)

    def test_unselected_shape_keeps_old_path_without_old_fallback(self):
        model = Model()
        receipt = self.install(model)
        output = model(Tensor((3, 8), value=2.0))
        self.assertEqual(output.value, 4.0)
        self.assertEqual(model.linear.lora_B["default"].original_forward_count, 1)
        self.assertEqual(receipt["lora_b_gemm_fallback_reasons"], {"NOT_SELECTED_SHAPE": 1})
        self.assertEqual(receipt["reference_fallback_count"], 0)
        self.assertEqual(receipt["fused_grad_invocation_count"], 1)
        self.assertEqual(self.kernel_calls, [])

    def test_exact_measured_shapes_only_not_threshold(self):
        weight = Tensor((5120, 256))
        for m in (11040, 13680):
            self.assertTrue(helper._lora_b_gemm_selection(Tensor((m, 256)), Tensor((m, 5120)), weight, self.torch)[0])
        for m in (512, 12000, 14000):
            selected, reason, _ = helper._lora_b_gemm_selection(Tensor((m, 256)), Tensor((m, 5120)), weight, self.torch)
            self.assertFalse(selected)
            self.assertEqual(reason, "NOT_SELECTED_SHAPE")
        self.assertEqual(helper.LORA_B_GEMM_VALIDATED_SHAPES, ((11040, 5120), (13680, 5120), (11040, 13824)))

    def test_metadata_guards_fail_closed_or_original_fallback(self):
        base, weight = Tensor((11040, 5120)), Tensor((5120, 256))
        for h, expected in ((Tensor((11040, 128)), "RANK_OR_WEIGHT_SHAPE"),
                            (Tensor((11040, 256), dtype="torch.float32"), "DTYPE"),
                            (Tensor((11040, 256), cuda=False), "DEVICE_NOT_CUDA"),
                            (Tensor((11040, 256), contiguous=False), "NONCONTIGUOUS")):
            self.assertEqual(helper._lora_b_gemm_selection(h, base, weight, self.torch)[1], expected)

    def test_grad_and_nograd_counters_are_separate(self):
        model = Model()
        receipt = self.install(model)
        model(Tensor((4, 8)))
        self.grad_enabled = False
        model(Tensor((4, 8)))
        self.assertEqual(receipt["lora_b_gemm_selected_grad_invocation_count"], 1)
        self.assertEqual(receipt["lora_b_gemm_selected_nograd_invocation_count"], 1)
        self.assertEqual(receipt["lora_b_gemm_completed_invocation_count"], 2)
        self.assertEqual(receipt["fused_invocation_count"], 2)

    def test_adapter_disabled_bypasses_new_kernel(self):
        model = Model()
        receipt = self.install(model)
        model.linear.disable_adapters = True
        model(Tensor((4, 8)))
        self.assertEqual(receipt["adapter_disabled_bypass_count"], 1)
        self.assertEqual(self.kernel_calls, [])
        self.assertEqual(model.linear.lora_B["default"].call_count, 0)

    def test_checkpoint_captured_scope_is_reentered_and_finally_restored(self):
        model = Model()
        receipt = self.install(model)
        model.saved_replay = "CREATE"
        value = Tensor((4, 8))
        model(value)
        # Pure CPU scope fixture: a replay called outside model-forward still
        # selects the same path; this is not actual PyTorch backward evidence.
        model.saved_replay(value)
        self.assertEqual(receipt["checkpoint_wrap_count"], 1)
        self.assertEqual(receipt["checkpoint_scope_execution_count"], 2)
        self.assertEqual(receipt["checkpoint_scope_restore_count"], 2)
        self.assertEqual(receipt["lora_b_gemm_selected_invocation_count"], 2)
        with self.assertRaises(RuntimeError):
            model.linear.lora_B["default"](Tensor((4, 2)), _distillschedule_lora_base_result=Tensor((4, 8)))

    def test_selected_bias_and_scale_are_not_silent_fallbacks(self):
        for mode in ("bias", "scale"):
            model = Model()
            self.install(model)
            if mode == "bias":
                model.linear.lora_B["default"].bias = object()
            else:
                model.linear.scaling["default"] = 0.5
            with self.assertRaises(RuntimeError):
                model(Tensor((4, 8)))


class ConditionalBackwardTests(unittest.TestCase):
    """Execute the actual Function methods with tensor-operation spies, no Torch."""
    def setUp(self):
        path = Path(helper.__file__).with_name("helios_lora_b_gemm_kernel.py")
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_BAdd")
        methods = []
        for node in cls.body:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
                methods.append(node)
        self.calls = []
        self.env = {"torch": types.SimpleNamespace(is_grad_enabled=lambda: False),
                    "_raw": lambda *args, **kwargs: "out"}
        code_tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods], type_ignores=[])
        ast.fix_missing_locations(code_tree)
        exec(compile(code_tree, str(path), "exec"), self.env)
        calls = self.calls

        class Spy:
            def __init__(self, label):
                self.label = label

            def __matmul__(self, other):
                calls.append((self.label, "@", other.label))
                return (self.label, other.label)

            def transpose(self, *_args):
                return Spy(self.label + ".T")

        self.Spy = Spy

    def run_case(self, needs):
        ctx = types.SimpleNamespace(needs_input_grad=(*needs, False))
        ctx.save_for_backward = lambda *tensors: setattr(ctx, "saved_tensors", tensors)
        h, b, base, g = [self.Spy(x) for x in ("H", "B", "base", "g")]
        self.env["forward"](ctx, h, b, base, None)
        got = self.env["backward"](ctx, g)
        return ctx, got, g

    def test_frozen_B_does_not_save_H_or_compute_dB(self):
        ctx, got, g = self.run_case((True, False, True))
        self.assertEqual([t.label for t in ctx.saved_tensors], ["B"])
        self.assertEqual(self.calls, [("g", "@", "B")])
        self.assertIsNone(got[1])
        self.assertIs(got[2], g)

    def test_only_B_grad_does_not_save_B_or_compute_dH(self):
        ctx, got, _ = self.run_case((False, True, False))
        self.assertEqual([t.label for t in ctx.saved_tensors], ["H"])
        self.assertEqual(self.calls, [("g.T", "@", "H")])
        self.assertIsNone(got[0])
        self.assertIsNone(got[2])

    def test_base_only_is_identity_no_saved_inputs_or_mm(self):
        ctx, got, g = self.run_case((False, False, True))
        self.assertEqual(ctx.saved_tensors, ())
        self.assertEqual(self.calls, [])
        self.assertEqual(got, (None, None, g, None))

    def test_all_grads_use_reference_mm_and_first_order_guard(self):
        ctx, got, _ = self.run_case((True, True, True))
        self.assertEqual([t.label for t in ctx.saved_tensors], ["H", "B"])
        self.assertEqual(self.calls, [("g", "@", "B"), ("g.T", "@", "H")])
        self.env["torch"].is_grad_enabled = lambda: True
        with self.assertRaises(RuntimeError):
            self.env["backward"](ctx, self.Spy("g"))

    def test_kernel_rounding_and_launch_callback_are_before_apply_hook_return(self):
        source = Path(helper.__file__).with_name("helios_lora_b_gemm_kernel.py").read_text()
        self.assertLess(source.index("rounded = acc.to(tl.bfloat16).to(tl.float32)"), source.index('"add.rn.f32'))
        self.assertIn("enable_fp_fusion=False", source)
        self.assertIn("on_kernel_launch()", source)
        self.assertIn("ctx.needs_input_grad[:3]", source)


if __name__ == "__main__":
    unittest.main()
