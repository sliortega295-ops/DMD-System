"""No-copy saved-tensor observation without bypassing mutation checks."""


def pack_saved_tensor(tensor):
    # Hooks replace SavedVariable's native version check. A detached wrapper
    # shares storage AND the version counter (including views/detached aliases),
    # without retaining the original tensor's grad_fn and creating a cycle.
    return tensor.detach(), tensor._version


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
