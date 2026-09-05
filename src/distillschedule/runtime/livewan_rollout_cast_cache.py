"""Opt-in no-grad autocast-cache lifetime; no effective dtype policy changes.

An outer *disabled* autocast context keeps original nested contexts' cache alive
until this one rollout ends. In particular, it must not encompass checkpoint
replay, a trainable forward, or an optimizer update. Production is not enabled
by importing this module.
"""
from contextlib import contextmanager


@contextmanager
def rollout_cast_cache_scope(torch, *, enabled=False, device_type="cuda"):
    if not enabled:
        yield
        return
    if torch.is_grad_enabled():
        raise RuntimeError("rollout cast cache requires no_grad")
    if torch.is_autocast_enabled(device_type):
        raise RuntimeError("rollout cast cache requires original outer autocast disabled")
    if not torch.is_autocast_cache_enabled():
        raise RuntimeError("rollout cast cache requires original cache enabled")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("rollout cast cache is registered for no_grad, not inference_mode")
    # Do not supply dtype: the thread's original fast dtype is preserved. The
    # actual linear compute policy remains controlled by existing inner scopes.
    with torch.autocast(device_type, enabled=False):
        yield

