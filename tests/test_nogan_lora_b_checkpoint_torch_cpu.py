"""Real Torch CPU checkpoint/replay gate with an injected reference executor.

This exercises the production public-call and replay wrappers, not CUDA,
Triton numerical accumulation, native HSDP, or actual device eligibility.
The CPU selector override is explicit and test-only; production is unchanged.
"""
from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from distillschedule.runtime import helios_lora_epilogue as helper
    from distillschedule.runtime.fsdp2_static import fsdp2_metadata_tolerant_checkpoint


@unittest.skipUnless(TORCH_AVAILABLE, "existing Torch environment required; no installation")
class RealCheckpointTests(unittest.TestCase):
    def models(self, candidate):
        class Linear(torch.nn.Module):
            disable_adapters = False
            merged = False
            active_adapters = ["default"]
            lora_variant = {}

            def __init__(self):
                super().__init__()
                self.base_layer = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)
                self.base_layer.requires_grad_(False)
                self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(8, 2, bias=False).to(torch.bfloat16)})
                self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 8, bias=False).to(torch.bfloat16)})
                self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(0.0)})
                self.scaling = {"default": 1.0}

            def _check_forward_args(self, *_args, **_kwargs):
                pass

            def _cast_input_dtype(self, value, dtype):
                return value.to(dtype)

            def forward(self, value):
                base = self.base_layer(value)
                update = self.lora_B["default"](self.lora_A["default"](value))
                return base + update * self.scaling["default"]

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear()

            def forward(self, value):
                return self.linear(value).sin().square()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()

            def forward(self, value):
                return fsdp2_metadata_tolerant_checkpoint(self.block, value,
                    function_wrapper=getattr(self, "_distillschedule_lora_checkpoint_function_wrapper", None))

        torch.manual_seed(13)
        model = Model()
        counts = dict(base=0, A=0, B=0, block=0)
        handles = []
        for label, module in (("base", model.block.linear.base_layer),
                              ("A", model.block.linear.lora_A["default"]),
                              ("B", model.block.linear.lora_B["default"]),
                              ("block", model.block)):
            handles.append(module.register_forward_pre_hook(
                lambda _module, _args, label=label: counts.__setitem__(label, counts[label] + 1)))
        receipt = helper.install_generator_nograd_lora_fused_epilogue(
            model, enabled=True, role="critic", linear_type=Linear, variant_kwarg_keys=(),
            fused_epilogue=lambda base, update, scale: base + update * scale,
            all_role_scale1=True, checkpoint_replay_scope=candidate,
            lora_b_gemm_epilogue=candidate,
            lora_b_gemm_epilogue_impl=(
                lambda hidden, weight, base: base + torch.nn.functional.linear(hidden, weight)
            ) if candidate else None,
            lora_b_gemm_test_shapes=((4, 8),) if candidate else None,
        )
        return model, receipt, counts, handles

    def test_actual_nonreentrant_replay_outputs_gradients_adam_and_rng(self):
        reference, _, ref_counts, _ = self.models(False)
        candidate, receipt, got_counts, _ = self.models(True)
        self.assertEqual(set(reference.state_dict()), set(candidate.state_dict()))
        candidate.load_state_dict(reference.state_dict())
        opts = [torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
                for model in (reference, candidate)]
        torch.manual_seed(19)
        inputs = [torch.randn(4, 8, dtype=torch.bfloat16) for _ in range(2)]
        for value in inputs:
            arm_results = []
            for model, opt in zip((reference, candidate), opts):
                opt.zero_grad(set_to_none=True)
                input = value.detach().clone().requires_grad_()
                before = torch.get_rng_state().clone()
                # Actual wrapper is exercised; only device selector/executor
                # are CPU-reference surrogates, not production GPU evidence.
                with patch.object(helper, "_lora_b_gemm_selection", return_value=(True, "SELECTED", "M4_N8_R2")):
                    output = model(input)
                    output.float().sum().backward()
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                gradients = [p.grad.clone() for p in model.parameters() if p.requires_grad]
                arm_results.append((output.detach(), input.grad.clone(), gradients))
                opt.step()
            for old, new in zip(arm_results[0][:2], arm_results[1][:2]):
                self.assertTrue(torch.equal(old, new))
            for old, new in zip(arm_results[0][2], arm_results[1][2]):
                self.assertTrue(torch.equal(old, new))
            for old, new in zip(reference.parameters(), candidate.parameters()):
                self.assertTrue(torch.equal(old, new))
            for old_p, new_p in zip(opts[0].param_groups[0]["params"], opts[1].param_groups[0]["params"]):
                for key in opts[0].state[old_p]:
                    self.assertTrue(torch.equal(opts[0].state[old_p][key], opts[1].state[new_p][key]))
        self.assertEqual(ref_counts, got_counts)
        self.assertEqual(receipt["checkpoint_wrap_count"], 2)
        self.assertEqual(receipt["checkpoint_scope_execution_count"], 4)
        self.assertEqual(receipt["checkpoint_scope_execution_count"], receipt["checkpoint_scope_restore_count"])
        self.assertEqual(receipt["checkpoint_missing_active_scope_count"], 0)
        self.assertGreater(receipt["lora_b_gemm_kernel_call_count"], 0)
        self.assertEqual(receipt["lora_b_gemm_kernel_call_count"], receipt["lora_b_gemm_selected_invocation_count"])


if __name__ == "__main__":
    unittest.main()
