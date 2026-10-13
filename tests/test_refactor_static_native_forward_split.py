"""No-torch source-splitting gate; native GPU model parity is a separate gate."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import unittest


class _TorchFixture:
    grad = True

    @classmethod
    def is_grad_enabled(cls):
        return cls.grad


torch = _TorchFixture


class _ModelFixture:
    def __init__(self):
        self.blocks = [lambda x, c, k=k: x * (k + 1) + c for k in range(4)]
        self.gradient_checkpointing = True
        self.original_checkpoint_calls = 0

    def _gradient_checkpointing_func(self, function, *args):
        self.original_checkpoint_calls += 1
        return function(*args)

    def forward(self, hidden_states, condition, *, return_dict=True):
        hidden_states = hidden_states + condition * 2
        temb = condition * 3
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for iidx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, condition)
        else:
            for iidx, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, condition)
        output = hidden_states + temb
        if not return_dict:
            return (output,)
        return {"sample": output}


class NativeForwardSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "src/distillschedule/runtime/static_native_forward.py"
        spec = importlib.util.spec_from_file_location("static_native_forward_gate", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.factory = staticmethod(module.create_admitted_forward_split)
        cls.source_sha = hashlib.sha256(Path(inspect.getsourcefile(_ModelFixture)).read_bytes()).hexdigest()

    def test_original_checkpoint_call_and_math_preserved_both_modes(self):
        for grad in (True, False):
            torch.grad = grad
            reference = _ModelFixture()
            expected = reference.forward(2.5, 0.7)
            model = _ModelFixture()
            split = self.factory(model, expected_source_sha256=self.source_sha, expected_blocks=4)
            context = split.prefix(model, 2.5, 0.7)
            context = split.blocks(model, context, 0, 2)
            context = split.blocks(model, context, 2, 4)
            observed = split.suffix(model, context)
            self.assertEqual(expected, observed)
            self.assertEqual(reference.original_checkpoint_calls, model.original_checkpoint_calls)
            self.assertEqual(model.original_checkpoint_calls, 4 if grad else 0)
            self.assertNotIn("self", context)
            self.assertNotIn("block", context)

    def test_source_mismatch_fails_before_execution(self):
        with self.assertRaisesRegex(ValueError, "source"):
            self.factory(_ModelFixture(), expected_source_sha256="0" * 64, expected_blocks=4)

    def test_block_count_mismatch_and_invalid_range_fail(self):
        model = _ModelFixture()
        with self.assertRaisesRegex(ValueError, "block"):
            self.factory(model, expected_source_sha256=self.source_sha, expected_blocks=30)
        split = self.factory(model, expected_source_sha256=self.source_sha, expected_blocks=4)
        context = split.prefix(model, 2.5, 0.7, return_dict=False)
        with self.assertRaisesRegex(ValueError, "range"):
            split.blocks(model, context, 2, 5)
        context = split.blocks(model, context, 0, 4)
        self.assertIsInstance(split.suffix(model, context), tuple)


if __name__ == "__main__":
    unittest.main()
