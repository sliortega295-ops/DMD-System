"""Small shared CPU/CUDA placement scopes; no distributed or offload policy."""
from contextlib import contextmanager, nullcontext


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def admit_fp32(parameters, *, require=_require):
    """Reject unsupported process/precision ownership before loading assets."""
    import os
    import torch
    require(not torch.distributed.is_initialized(), 'distributed ownership is not admitted')
    require(all(os.environ.get(key, default) == default for key, default in
                (('WORLD_SIZE', '1'), ('LOCAL_WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0'))),
            'one actual process/rank required')
    require(torch.get_default_dtype() == torch.float32 and not torch.is_autocast_enabled('cpu')
            and not torch.is_autocast_enabled('cuda'), 'explicit FP32 with no autocast required')
    return execution_device(parameters.get('device', 'cpu'), require=require)


def place_module(module, device, *, require=_require):
    """Own native plain tensors, then move devices without a dtype conversion.

    Registration follows strict native asset loading. In particular Qwen's
    complex RoPE must NOT pass through ``Module.to(dtype=...)`` or ``float()``
    after registration: those would discard its imaginary component.
    """
    import torch
    from torch.utils._pytree import tree_leaves
    for child in module.modules():
        for name, value in tuple(vars(child).items()):
            if name in ('_parameters', '_buffers', '_modules'):
                continue
            if type(value) is torch.Tensor:
                require(not value.requires_grad, 'unregistered native trainable tensor: ' + name)
                delattr(child, name)
                child.register_buffer(name, value)
            elif isinstance(value, (dict, list, tuple)):
                require(not any(isinstance(item, torch.Tensor) for item in tree_leaves(value)),
                        'unowned native tensor container requires an explicit placement adapter: ' + name)
    module.to(device)
    validate_placement(module, torch.device(device), require=require)
    return module


def validate_placement(module, device, *, require=_require):
    import torch
    from torch.utils._pytree import tree_leaves
    require(all(value.device == device and not value.is_meta
                and (not value.is_floating_point() or value.dtype == torch.float32)
                for value in (*module.parameters(), *module.buffers())),
            'native parameters, buffers and owned plain tensors require one FP32 device')
    require(not any(isinstance(item, torch.Tensor)
                    for child in module.modules() for name, value in vars(child).items()
                    if name not in ('_parameters', '_buffers', '_modules')
                    for item in tree_leaves(value)),
            'unowned native plain tensor requires an explicit placement adapter')


class ResidentEMA(dict):
    """Mapping-only serialization; common DMD still owns EMA math and clocks."""
    _require = staticmethod(_require)

    def state_dict(self):
        return dict(self)

    def runtime_tensors(self):
        yield from self.values()

    def validate_state_dict(self, value):
        import torch
        from .checkpoint import _validate_layout
        _validate_layout(dict(self), value, 'resident flow EMA')
        self._require(all(type(tensor) is torch.Tensor and not tensor.is_meta
                          and tensor.layout == torch.strided and not tensor.requires_grad
                          and bool(torch.isfinite(tensor).all()) for tensor in value.values()),
                      'resident flow EMA requires finite detached dense tensors')

    def load_state_dict(self, value):
        import torch
        self.validate_state_dict(value)
        with torch.no_grad():
            for name, live in self.items():
                live.copy_(value[name])


def resident_flow_state(state, device, *, ema_class=ResidentEMA):
    """Keep native Adam step and common EMA on the already-owned CUDA device.

    CPU reference behavior is unchanged. This configures native AdamW, never
    reimplements updates or moves Parameters after optimizer construction.
    """
    if device.type == 'cuda':
        if 'student_ema' in state:
            state['student_ema'] = ema_class(state['student_ema'])
        for role in ('student', 'fake'):
            optimizer = state[role + '_optimizer']
            optimizer.defaults['capturable'] = True
            for group in optimizer.param_groups:
                group['capturable'] = True
    return state


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
