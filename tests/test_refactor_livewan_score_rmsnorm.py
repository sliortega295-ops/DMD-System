from types import SimpleNamespace
import inspect
import pytest

from distillschedule.runtime.livewan_score_rmsnorm import (
    validate_frozen_rms_inputs,
    build_frozen_score_rmsnorm,
)


def tensors():
    torch = SimpleNamespace(bfloat16="bf16")
    x = SimpleNamespace(is_cuda=True, device="cuda:0", dtype="bf16", ndim=3,
                        shape=(2, 19320, 1536), is_contiguous=lambda: True)
    w = SimpleNamespace(is_cuda=True, device="cuda:0", dtype="bf16", shape=(1536,),
                        is_contiguous=lambda: True, requires_grad=False)
    return torch, x, w


def test_registered_metadata():
    validate_frozen_rms_inputs(*tensors(), 1e-6)


@pytest.mark.parametrize("attribute,value", [
    ("dtype", "fp32"), ("is_cuda", False), ("requires_grad", True),
    ("device", "cuda:1"), ("shape", (5120,)),
])
def test_affine_contract_fail_closed(attribute, value):
    torch, x, w = tensors()
    setattr(w, attribute, value)
    with pytest.raises(RuntimeError, match="contract changed"):
        validate_frozen_rms_inputs(torch, x, w, 1e-6)


def test_epsilon_fail_closed():
    with pytest.raises(RuntimeError):
        validate_frozen_rms_inputs(*tensors(), 1e-5)


def test_not_installed_by_default():
    source = inspect.getsource(build_frozen_score_rmsnorm)
    assert "enable_fp_fusion=False" in source
    assert "CANARY_ONLY_NOT_PRODUCTION_INSTALLED" in source
    assert "torch.cuda.synchronize" not in source
    assert ".cpu()" not in source
