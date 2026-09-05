"""Opt-in no-grad autocast-cache lifetime; no effective dtype policy changes.

An outer *disabled* autocast context keeps original nested contexts' cache alive
until this one rollout ends. In particular, it must not encompass checkpoint
replay, a trainable forward, or an optimizer update. Production is not enabled
by importing this module.
"""
from contextlib import contextmanager


NAME = "generator_nograd_autocast_weight_cache_lifetime_v1"
ATTRIBUTE = "_distillschedule_rollout_cast_cache"
PLAN_CONTRACT = {
    "enabled": True, "name": NAME, "scope": "single_prepare_rollout_generator_only",
    "outer_autocast_enabled": False, "changes_effective_dtype": False,
    "cache_cross_optimizer": False, "parameter_dtype": "float32",
    "preserve_native_inner_autocast": True,
}


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


class RolloutCastCache:
    def __init__(self):
        self.entries = self.exits = self.failures = 0
        self.active = False

    @contextmanager
    def scope(self, torch):
        if self.active:
            self.failures += 1
            raise RuntimeError("recursive rollout cast-cache scope")
        self.active = True
        self.entries += 1
        try:
            with rollout_cast_cache_scope(torch, enabled=True):
                yield
        except BaseException:
            self.failures += 1
            raise
        finally:
            self.active = False
            self.exits += 1

    def receipt(self):
        return {"name": NAME, "execution_status": "PASS", "scope_entries": self.entries,
                "scope_exits": self.exits, "active": self.active, "contract_failures": self.failures,
                "effective_dtype_changed": False, "cache_cross_optimizer": False}


def install(torch, student):
    if str(torch.__version__) != "2.10.0+cu128":
        raise RuntimeError("C28 requires the validated Torch 2.10 CUDA 12.8 cache lifecycle")
    if (hasattr(student, ATTRIBUTE) or hasattr(student, "causal_layers")
            or len(student.blocks) != 30):
        raise RuntimeError("C28 requires one unwrapped resident 30-block Generator")
    if not all(p.is_cuda and p.dtype == torch.float32 and p.requires_grad
               and not hasattr(p, "to_local") for p in student.parameters()):
        raise RuntimeError("C28 Generator parameter dtype/trainability/placement changed")
    helper = RolloutCastCache()
    setattr(student, ATTRIBUTE, helper)
    return helper


@contextmanager
def model_rollout_cast_cache_scope(torch, student):
    helper = getattr(student, ATTRIBUTE, None)
    if helper is None:
        yield
    else:
        with helper.scope(torch):
            yield


def runtime_receipt_errors(r, *, rank, cycles):
    if rank >= 8:
        return [] if r.get("execution_status") == "NOT_APPLICABLE" else ["C28 changed Score scope"]
    if (r.get("name") != NAME or r.get("execution_status") != "PASS"
        or r.get("scope_entries") != 10 * cycles or r.get("scope_exits") != 10 * cycles
        or r.get("active") is not False or r.get("contract_failures") != 0
        or r.get("effective_dtype_changed") is not False or r.get("cache_cross_optimizer") is not False):
        return ["C28 actual scope/lifecycle census failed"]
    return []
