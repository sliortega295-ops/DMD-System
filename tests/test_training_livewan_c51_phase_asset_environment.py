"""C51 phase metadata admission only; no Torch/model/native qualification."""
from copy import deepcopy
import unittest

from distillschedule.training.livewan_reference_assets import _environment, _c51_phase_environment, environment_snapshot
from distillschedule.workloads.best_known.registry import fingerprint


FLAG = "DISTILLSCHEDULE_LIVEWAN_C51_PHASE_RECEIPTS"
RUN = "DISTILLSCHEDULE_RUN_ID"


class PhaseAssetEnvironment(unittest.TestCase):
    def ordinary(self):
        return {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "USE_TF": "0", "USE_FLAX": "0"}

    def diagnostic(self):
        return {**self.ordinary(), FLAG: "v1", RUN: "SYNTHETIC_TEST_ONLY-c51-assets"}

    def test_default_reference_and_c51_environment_are_unmodified(self):
        value = self.ordinary()
        original = deepcopy(value)
        _environment(value)
        _c51_phase_environment(value)
        self.assertEqual(value, original)

    def test_reference_still_rejects_even_valid_c51_diagnostics(self):
        with self.assertRaisesRegex(ValueError, "reference cannot inherit"):
            _environment(self.diagnostic())

    def test_c51_retains_both_values_and_full_environment_identity(self):
        value = environment_snapshot(self.diagnostic())
        original, digest = deepcopy(value), fingerprint(value)
        _c51_phase_environment(value)
        self.assertEqual(value, original)
        self.assertEqual(fingerprint(value), digest)
        self.assertNotEqual(digest, fingerprint(self.ordinary()))

    def test_missing_invalid_or_non_v1_pair_is_rejected(self):
        for key, replacement in [(FLAG, None), (RUN, None), (FLAG, "0"), (FLAG, "v2"),
                                 (RUN, ""), (RUN, "."), (RUN, "../escape"), (RUN, "x" * 129), (RUN, 1)]:
            with self.subTest(key=key, replacement=replacement):
                value = self.diagnostic()
                if replacement is None:
                    value.pop(key)
                else:
                    value[key] = replacement
                with self.assertRaises(ValueError):
                    _c51_phase_environment(value)

    def test_diagnostics_are_not_a_generic_override_or_credential_escape(self):
        for key in ("DISTILLSCHEDULE_EXECUTION_PLAN", "DISTILLSCHEDULE_IMPLEMENTATION", "LIVEWAN_FORCE",
                    "DMD_FORCE", "NSYS_FLAGS", "NCU_FLAGS", "TORCH_COMPILE_DEBUG", "TORCH_LOGS",
                    "LOCAL_KERNELS", "LD_PRELOAD", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "HF_TOKEN"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                _c51_phase_environment({**self.diagnostic(), key: "SYNTHETIC_TEST_ONLY"})


if __name__ == "__main__":
    unittest.main()
