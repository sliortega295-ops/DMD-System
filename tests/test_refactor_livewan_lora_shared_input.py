"""C40 is not a production policy; GPU parity lives in its bounded canary."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_import_is_lazy_and_default_worker_is_unchanged():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c",
        "import sys; from distillschedule.runtime import livewan_lora_shared_input; "
        "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    worker = (ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py").read_text()
    assert "livewan_lora_shared_input" not in worker


def test_two_outputs_share_payload_but_not_bf16_gradient_accumulation():
    source = (ROOT / "src/distillschedule/runtime/livewan_lora_shared_input.py").read_text()
    assert source.count("x.to(torch.bfloat16)") == 1
    assert "return y.view_as(y), y.view_as(y)" in source
    assert 'gradient_accumulation_dtype="float32"' in source
    assert "add.rn.f32" in source
    assert "ctx.set_materialize_grads(False)" in source
    assert ".cpu(" not in source
    assert "synchronize(" not in source
