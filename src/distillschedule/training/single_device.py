"""Small shared CPU/CUDA placement scopes; no distributed or offload policy."""
from contextlib import contextmanager, nullcontext


def execution_device(value, *, require):
    import torch
    require(isinstance(value, str), 'device must be cpu or a single CUDA device')
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError):
        require(False, 'invalid device')
    require(device.type in ('cpu', 'cuda') and (device.type != 'cpu' or device.index is None),
            'device must be cpu or a single CUDA device')
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'requested CUDA device is unavailable')
        index = device.index if device.index is not None else torch.cuda.current_device()
        require(0 <= index < torch.cuda.device_count(), 'requested CUDA device index is unavailable')
        device = torch.device('cuda', index)
    return device


def device_scope(device):
    import torch
    return torch.cuda.device(device) if device.type == 'cuda' else nullcontext()


@contextmanager
def construction_rng(device, seed):
    """Initialize only owned generators, restoring the caller even on failure."""
    from .checkpoint import rng_state, restore_rng
    from .session import _seed
    with device_scope(device):
        outer = rng_state(device=device)
        try:
            _seed(seed, device=device)
            yield
        finally:
            restore_rng(outer)
