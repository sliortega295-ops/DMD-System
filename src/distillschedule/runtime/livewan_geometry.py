"""Immutable worker geometry: installed once before model/CUDA/thread setup.

Legacy metadata helpers may read the frozen default without installation.
A real worker must install its exact plan binding, including for the default.
There is no reset, context toggle, environment fallback or reinstallation API.
"""
from dataclasses import dataclass
import re
from threading import Lock

from distillschedule.search.livewan_request import LiveWanRequest


@dataclass(frozen=True)
class Geometry:
    request: LiveWanRequest = LiveWanRequest()
    plan_sha256: str | None = None

    @property
    def latent_shape(self):
        return self.request.latent_shape


_default = Geometry()
_installed = None
_lock = Lock()


def geometry():
    # Immutable object publication; readers never observe partial state.
    with _lock:
        return _installed if _installed is not None else _default


def install_geometry(request, *, plan_sha256):
    global _installed
    if not isinstance(request, LiveWanRequest) or not isinstance(plan_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", plan_sha256):
        raise RuntimeError("LiveWan geometry requires typed request and exact plan identity")
    with _lock:
        if _installed is not None:
            raise RuntimeError("LiveWan geometry is installed once; reinstallation is forbidden")
        _installed = Geometry(request, plan_sha256)
        return _installed


def latent_shape():
    return geometry().latent_shape


def receipt_errors(raw, request, *, expected_plan_sha256=None):
    """Check observed tensor metadata, never infer actual shape from a config."""
    if request == LiveWanRequest():
        return []  # Preserve historical default receipt compatibility.
    errors = []
    if raw.get("workload_request") != request.to_dict():
        errors.append("actual LiveWan workload request differs")
    shapes = raw.get("actual_latent_shapes", [])
    if (not shapes or any(not isinstance(s, list) or len(s) != 5 or s[0] not in (1, 2)
                         or tuple(s[1:]) != request.latent_shape[1:] for s in shapes)):
        errors.append("actual LiveWan latent shape differs or is unobserved")
    installed = raw.get("installed_plan_sha256", "")
    if not isinstance(installed, str) or not re.fullmatch(r"[0-9a-f]{64}", installed):
        errors.append("actual LiveWan installed plan identity is missing")
    if expected_plan_sha256 is not None and installed != expected_plan_sha256:
        errors.append("actual LiveWan installed plan differs from selected launch plan")
    return errors


def launch_binding(run_root):
    """Read request launch identities without installing worker process state."""
    from pathlib import Path
    from distillschedule.search.livewan_request import request_from_plan
    from distillschedule.workloads.best_known.registry import _object, fingerprint
    paths = [Path(run_root) / f"launch-native-plan.node{n}.json" for n in (0, 1)]
    if not any(p.exists() for p in paths):
        return LiveWanRequest(), {}
    plans = [_object(p) for p in paths]
    requests = [request_from_plan(p) for p in plans]
    if requests[0] != requests[1] or requests[0] == LiveWanRequest():
        raise ValueError("LiveWan request launch plans differ or are default templates")
    return requests[0], {n: fingerprint(p) for n, p in enumerate(plans)}
