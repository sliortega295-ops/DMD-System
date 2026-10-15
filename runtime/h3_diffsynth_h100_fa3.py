"""Opt-in H100 FA3 binding for the pinned native W6 attention dispatcher.

No packages or private source files are edited. The original segmentation,
layouts, scale, autograd, checkpoint and native model remain in control. The
existing aten-Flash cache is deliberately rejected for this backend-only arm.
"""
from __future__ import annotations

import hashlib
import importlib
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace


_PREDICTION_ROLE = ContextVar('h100_fa3_prediction_role', default=None)


@contextmanager
def prediction_role_scope(role):
    """Explicit native prediction identity, never inferred from thread ownership."""
    if role not in ('student', 'fake_score', 'teacher'):
        raise ValueError('EXPLICIT_NATIVE_PREDICTION_ROLE_REQUIRED')
    token = _PREDICTION_ROLE.set(role)
    try:
        yield
    finally:
        _PREDICTION_ROLE.reset(token)


NATIVE_DISPATCH_SHA256 = "1126b2e19184299cbfc611dcf66e46d0fd1a7bfa1d616f2ed38bc60ba19679ae"
FA3_ASSET_SHA256 = {
    "__init__.py": "29756640933e2a159cd94a9625eb2ea4268ffa6a80bd3c7bc86b9e1404e09471",
    "_flash_attn3_cuda_373a45c.abi3.so": "8aa9bb49e174872be4e2cb8c2608f2d66de29b513f704fb8d91771f9cd39df56",
    "_ops.py": "7f392b365cc945a3b3a420e24802ef776a79316c068e6e99e4da682e8927a1f6",
    "flash_attn3/__init__.py": "0c560f96b857c188c4a8297ff27d1299619957c3459a5e433e132329fbfdf066",
    "flash_attn_config.py": "bb1a3e7b231c0e2b7fffc603692e11b64e414cc2b34d22bdb4e2938dd5957bfc",
    "flash_attn_interface.py": "cb46c3d89606305817ea20eb7b7cc9376b013eeabfd7637a328e7967e1d008d4",
    "metadata.json": "f1b4b0ef991c0966f7d848777154c0591c4580968fb6da33f6b2a2829a109309",
}


def validate_backend_probe(*, backend, placement, asset_dir, mode, export,
                           one_cycle, deterministic, deterministic_backward,
                           common_native_warmup, reference, retained_blocks):
    """CPU-only admission for this bounded backend correctness experiment."""
    if reference is not None and not common_native_warmup:
        raise ValueError('C1_REFERENCE_REQUIRES_COMMON_NATIVE_WARMUP')
    if deterministic_backward and not deterministic:
        raise ValueError('FA3_EXPLICIT_DETERMINISM_REQUIRES_COMMON_TORCH_POLICY')
    if common_native_warmup and not (
            mode == 'correctness' and export and not one_cycle and deterministic):
        raise ValueError('BACKEND_SWITCH_IS_TWO_CYCLE_CORRECTNESS_ONLY')
    if backend != 'native':
        if placement == 'candidate':
            raise ValueError('FA3_CANNOT_REUSE_NATIVE_ATEN_NINE_RESULT_CACHE')
        if retained_blocks:
            raise ValueError('NO_COMBINED_FA3_AND_WEIGHT_RETENTION_EXPERIMENT')
        if not (common_native_warmup and deterministic_backward and reference is not None):
            raise ValueError('FA3_REQUIRES_VERIFIED_NATIVE_C1_AND_EXPLICIT_DETERMINISTIC_BACKWARD')
    if (backend != 'native') != (asset_dir is not None):
        raise ValueError('EXPLICIT_ISOLATED_FA3_ASSET_IDENTITY_REQUIRED')


def snapshot_slot_rng(states):
    """Hash already-saved CPU RNG states; never query CUDA or copy a tensor."""
    import torch
    import json
    rows = {}
    for slot, state in states.items():
        row = {}
        for name in ('torch', 'cuda'):
            value = state[name]
            if value is None:
                row[name] = None
            else:
                if value.device.type != 'cpu':
                    raise ValueError('RNG_RECEIPT_REQUIRES_ALREADY_SAVED_CPU_STATE')
                row[name] = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        numpy = state['numpy']
        row['numpy'] = hashlib.sha256(numpy[1].tobytes() +
            repr((numpy[0], str(numpy[1].dtype), numpy[1].shape, *numpy[2:])).encode()).hexdigest()
        row['python'] = hashlib.sha256(repr(state['python']).encode()).hexdigest()
        row['combined_sha256'] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        rows[str(slot)] = row
    return rows


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_native_attention(backend="native", *, placement, asset_dir=None, record_calls=False,
                               deterministic_backward=False):
    if backend not in ("native", "fa3", "fa3-nograd", "fa3-rollout"):
        raise ValueError("unknown W6 attention backend")
    if placement not in ("resident", "fsdp", "candidate"):
        raise ValueError("unknown W6 placement")
    if backend != "native" and placement == "candidate":
        raise ValueError("FA3 cannot use the aten Flash nine-result attention cache; validate a separate cache port first")
    if (backend != "native") != (asset_dir is not None):
        raise ValueError("an explicit isolated FA3 asset is required only for the FA3 arm")
    attention = importlib.import_module("diffsynth.core.attention.attention")
    if _sha(attention.__file__) != NATIVE_DISPATCH_SHA256:
        raise ValueError("pinned private attention dispatcher source mismatch")
    if attention.ATTENTION_IMPLEMENTATION != "torch":
        raise ValueError("unexpected initial native attention backend; no silent environment override")
    receipt = {"requested": backend, "dispatcher_sha256": NATIVE_DISPATCH_SHA256,
               "selected_implementation": "torch", "patch_applied": False,
               "attention_cache": "aten_flash" if placement == "candidate" else "disabled",
               "actual_segment_calls": {}, "call_counters_enabled": bool(record_calls and backend != "native")}
    receipt['FA3_deterministic_backward_requested'] = bool(deterministic_backward)
    if backend == "native":
        return receipt

    root = Path(asset_dir).resolve(strict=True)
    for name, expected in FA3_ASSET_SHA256.items():
        if _sha(root / name) != expected:
            raise ValueError(f"isolated FA3 asset mismatch: {name}")
    from kernels.utils import get_local_kernel
    kernel = get_local_kernel(root, "flash_attn3")
    interface = kernel.flash_attn_interface
    if Path(interface.__file__).resolve() != root / "flash_attn_interface.py":
        raise ValueError("FA3 interface did not resolve to the verified isolated asset")
    if not callable(interface.flash_attn_func) or not callable(interface._flash_attn_backward):
        raise ValueError("FA3 forward/backward API is incomplete")
    import torch
    if deterministic_backward and not torch.are_deterministic_algorithms_enabled():
        raise ValueError('FA3_DETERMINISTIC_BACKWARD_REQUIRES_COMMON_TORCH_POLICY')
    # This extension has its own deterministic keyword, default False. A global
    # Torch policy alone does not configure its backward. Bind process-locally,
    # without mutating the shared imported interface or its nine-result peer.
    original = interface.flash_attn_func
    counters = receipt["actual_segment_calls"]

    def observed_call(q, k, v, **kwargs):
        if 'deterministic' in kwargs and kwargs['deterministic'] != bool(deterministic_backward):
            raise ValueError('CONFLICTING_FA3_DETERMINISTIC_ARGUMENT')
        kwargs['deterministic'] = bool(deterministic_backward)
        if record_calls:
            key = f"grad{int(torch.is_grad_enabled())}:B{q.shape[0]}:S{q.shape[1]}:H{q.shape[2]}:D{q.shape[3]}"
            if backend == 'fa3-rollout':
                key += f":role{_PREDICTION_ROLE.get()}"
            counters[key] = counters.get(key, 0) + 1
        return original(q, k, v, **kwargs)

    interface = SimpleNamespace(flash_attn_func=observed_call)
    attention.flash_attn_interface = interface
    attention.ATTENTION_IMPLEMENTATION = "flash_attention_3"
    if backend in ("fa3-nograd", "fa3-rollout"):
        import torch
        native_fa3 = attention.flash_attention_3
        native_dispatch = attention.attention_forward
        grad_calls = receipt["native_grad_segment_calls"] = {}
        score_calls = receipt["native_score_segment_calls"] = {}

        def no_grad_only(q, k, v, *args, **kwargs):
            grad_enabled = torch.is_grad_enabled()
            role = _PREDICTION_ROLE.get() if backend == 'fa3-rollout' else None
            if backend == 'fa3-rollout' and not grad_enabled and role is None:
                raise ValueError('FA3_ROLLOUT_REQUIRES_EXPLICIT_NATIVE_ROLE')
            if grad_enabled or (backend == 'fa3-rollout' and role != 'student'):
                if record_calls:
                    key = f"B{q.shape[0]}:H{q.shape[1]}:S{q.shape[2]}:D{q.shape[3]}"
                    if grad_enabled:
                        grad_calls[key] = grad_calls.get(key, 0) + 1
                    else:
                        key += f":role{role}"
                        score_calls[key] = score_calls.get(key, 0) + 1
                # Existing native dispatcher owns mask/scale/layout handling.
                # Compatibility mode selects its original Torch SDPA path.
                return native_dispatch(q, k, v, *args, compatibility_mode=True, **kwargs)
            return native_fa3(q, k, v, *args, **kwargs)

        attention.flash_attention_3 = no_grad_only
        receipt["grad_and_checkpoint_replay_backend"] = "native_torch_SDPA"
        receipt["no_grad_backend"] = "isolated_FA3"
        if backend == 'fa3-rollout':
            receipt['no_grad_backend'] = 'isolated_FA3_for_student_only'
            receipt['fake_teacher_score_backend'] = 'native_torch_SDPA'
            receipt['missing_prediction_role'] = 'FAIL_CLOSED_NO_FALLBACK'
    receipt.update(selected_implementation="flash_attention_3", patch_applied=True,
                   isolated_asset_root=str(root), asset_sha256=dict(FA3_ASSET_SHA256),
                   native_segment_function_unchanged=True, native_FA3_layout_wrapper_unchanged=True,
                   cache_compatibility="NO_ATEN_CACHE_IN_THIS_BACKEND_ONLY_ARM")
    return receipt
