"""No-copy saved-tensor observation without bypassing mutation checks."""


def pack_saved_tensor(tensor, *, release_storage=False):
    # Hooks replace SavedVariable's native version check. A detached wrapper
    # shares storage AND the version counter (including views/detached aliases),
    # without retaining the original tensor's grad_fn and creating a cycle.
    version = tensor._version
    saved = tensor.detach()
    if release_storage:
        import torch
        # Keep only the detached alias's shared version counter. set_data
        # preserves that counter while releasing this wrapper's payload; it
        # does not change the source tensor or advance/reset its version.
        # This token must never be used as the saved backward value.
        saved.data = torch.empty(0, device=saved.device, dtype=saved.dtype)
    return saved, version


def unpack_saved_tensor(packed):
    tensor, version = packed
    current = tensor._version
    if current != version:
        raise RuntimeError(
            'one of the variables needed for gradient computation has been '
            'modified by an inplace operation: observed saved tensor is at '
            f'version {current}; expected version {version} instead.'
        )
    return tensor
