"""Plan-gated exact runtime specializations for pinned LiveWan LoRA."""

from __future__ import annotations

import hashlib
import inspect
import textwrap
from typing import Any


EXPECTED_FORWARD_SHA256 = (
    "70d00d496aa2bd8ea06d64e6b04af873644eff52539396f74ec586a4cbbe97a3"
)


def _source_sha256(value: Any) -> str:
    source = textwrap.dedent(inspect.getsource(value)).strip()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def install_scale_one_lora_epilogue(torch: Any) -> dict[str, Any]:
    """Remove only the identity multiply in the pinned scale-one LoRA path.

    The FP32 LoRA GEMMs, FP32-to-BF16 cast, and BF16 residual add remain in the
    same order.  Unknown or already-modified upstream code fails closed.
    """

    from wanstreamer.lora import LoRALinear

    if getattr(LoRALinear, "_distillschedule_scale_one_epilogue", False):
        return {
            "enabled": True,
            "name": "scale_one_identity_epilogue_v1",
            "source_forward_sha256": EXPECTED_FORWARD_SHA256,
            "class_forward_patched": True,
            "install_status": "ALREADY_INSTALLED",
        }
    actual = _source_sha256(LoRALinear.forward)
    if actual != EXPECTED_FORWARD_SHA256:
        raise RuntimeError(
            "LiveWan LoRALinear.forward source changed: "
            f"expected {EXPECTED_FORWARD_SHA256}, observed {actual}"
        )
    original = LoRALinear.forward

    def scale_one_forward(self: Any, x: Any) -> Any:
        y = self.base(x)
        if not self.enabled:
            return y
        h = torch.nn.functional.linear(x.to(self.a.dtype), self.a)
        update = torch.nn.functional.linear(h, self.b).to(y.dtype)
        if self.scale == 1.0:
            return y + update
        return y + update * self.scale

    LoRALinear._distillschedule_original_forward = original
    LoRALinear.forward = scale_one_forward
    LoRALinear._distillschedule_scale_one_epilogue = True
    return {
        "enabled": True,
        "name": "scale_one_identity_epilogue_v1",
        "source_forward_sha256": actual,
        "class_forward_patched": True,
        "install_status": "PASS",
    }


def uninstall_scale_one_lora_epilogue() -> None:
    """Test-only restoration; production workers exit with the patch installed."""

    from wanstreamer.lora import LoRALinear

    original = getattr(LoRALinear, "_distillschedule_original_forward", None)
    if original is None:
        return
    LoRALinear.forward = original
    delattr(LoRALinear, "_distillschedule_original_forward")
    delattr(LoRALinear, "_distillschedule_scale_one_epilogue")
