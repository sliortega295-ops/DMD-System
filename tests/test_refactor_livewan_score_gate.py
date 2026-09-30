"""Canary metadata and fail-closed contract; real CUDA checks are separate."""
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from distillschedule.runtime.livewan_score_gate import build, validate


def inputs(b=2, d=1536, layout="channel"):
    def tensor(shape, strides, **kw):
        return SimpleNamespace(is_cuda=True, device="cuda:0", dtype="bf16",
            ndim=len(shape), shape=shape, requires_grad=False,
            stride=lambda i=None: strides if i is None else strides[i], **kw)
    s = 19320
    st = (s*d, 1, s) if layout == "channel" else (s*d, d, 1)
    x = tensor((b, s, d), st)
    y = tensor((b, s, d), (s*d, d, 1))
    g = tensor((b, 1, d), (6*d, d, 1))
    return SimpleNamespace(bfloat16="bf16"), x, y, g


@pytest.mark.parametrize("b,d,layout", [(b, d, layout) for b in (1, 2)
    for d in (1536, 5120) for layout in ("channel", "row")])
def test_registered_dense_layouts(b, d, layout):
    validate(*inputs(b, d, layout))


@pytest.mark.parametrize("which,attr,value", [
    (1,"dtype","fp32"), (2,"dtype","fp32"), (3,"dtype","fp32"),
    (1,"is_cuda",False), (2,"device","cuda:1"), (3,"device","cuda:1"),
    (3,"requires_grad",True), (3,"shape",(3, 1, 1536)),
    (1,"shape",(2,19320,1280)), (2,"shape",(1,19320,1536)),
])
def test_metadata_fail_closed(which, attr, value):
    args = inputs()
    setattr(args[which], attr, value)
    with pytest.raises(RuntimeError, match="contract changed"):
        validate(*args)


def test_no_import_time_install_or_host_round_trip():
    source = inspect.getsource(build)
    assert "CANARY_ONLY_NOT_PRODUCTION_INSTALLED" in source
    assert "enable_fp_fusion=False" in source
    assert source.count("mul.rn.f32") == 2 and source.count("add.rn.f32") == 1
    assert ").to(tl.bfloat16).to(tl.float32)" in source
    for token in (".cpu(", ".item(", ".tolist(", "synchronize(", "tl.sum("):
        assert token not in source


def test_plan_only_needs_no_torch_or_assets(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "never_created.json"
    p = subprocess.run([sys.executable,
        str(root / "scripts/distillschedule/canary_livewan_score_gate.py"),
        "--output",str(out),"--plan-only"], capture_output=True,text=True,
        check=True,timeout=15)
    receipt = json.loads(p.stdout)
    assert receipt["status"] == receipt["world16"] == "NOT_RUN"
    assert receipt["gate_tolerance"] == "BITWISE_OUTPUT_DX_DY"
    assert not receipt["production_installed"] and not out.exists()
