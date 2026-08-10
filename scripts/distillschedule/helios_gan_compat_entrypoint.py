#!/usr/bin/env python3
"""Run the pinned Helios GAN entrypoint with diagnostics-only runtime hooks.

The wrapper never reimplements model math.  With diagnostics disabled it simply
executes ``$HELIOS_RUNTIME_ROOT/train_helios.py`` as ``__main__``.  Exact Nsight
profiling is installed by wrapping the already-imported upstream Generator and
Critic loss call boundaries; the underlying upstream functions and their
optimizer/EMA semantics remain unchanged.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path
from typing import Any

from distillschedule.runtime.nsys_range import GanFull5NsightRange


def _step(kwargs: dict[str, Any], name: str) -> int:
    value = kwargs.get("global_step")
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} did not receive integer global_step")
    return value


def _install_nsys_hooks() -> GanFull5NsightRange:
    """Patch only call boundaries; upstream functions remain the math owner."""

    gate = GanFull5NsightRange.from_env()
    if not gate.enabled:
        return gate

    import helios.utils.utils_helios_post as post

    original_generator = post._generator_loss
    original_critic = post._critic_loss

    def generator_loss(*args: Any, **kwargs: Any):
        logical_step = _step(kwargs, "_generator_loss")
        if logical_step == 1000:
            gate.before_model_compute(logical_step)
        return original_generator(*args, **kwargs)

    def critic_loss(*args: Any, **kwargs: Any):
        logical_step = _step(kwargs, "_critic_loss")
        result = original_critic(*args, **kwargs)
        # In the frozen low-VRAM DeepSpeed GAN path the joint denoising +
        # GAN/R1/R2 engine step is completed inside upstream _critic_loss before
        # it returns.  Step1004 has no EMA update under the post-750 predicate,
        # so this is the exact end of the profiled model/optimizer window.
        if logical_step == 1004:
            gate.after_critic_commit(logical_step)
        return result

    post._generator_loss = generator_loss
    post._critic_loss = critic_loss
    return gate


def main() -> None:
    root = Path(os.environ["HELIOS_RUNTIME_ROOT"]).expanduser().resolve()
    upstream = root / "train_helios.py"
    if not upstream.is_file():
        raise FileNotFoundError(f"pinned Helios entrypoint missing: {upstream}")

    # Import/patch the upstream helper module before train_helios executes its
    # ``from ...utils_helios_post import _generator_loss, _critic_loss`` lines.
    gate = _install_nsys_hooks()
    try:
        runpy.run_path(str(upstream), run_name="__main__")
    finally:
        # A profiling run that crashes or skips either boundary must not leave a
        # plausible-looking exact-range claim behind.
        if gate.enabled:
            gate.assert_complete()


if __name__ == "__main__":
    main()
