"""Measured implementation helper, relocated from r1/scripts/sf_fsdp2_mutable_cache_abi.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
from collections.abc import MutableMapping

class MutableKVState(MutableMapping):

    def __init__(self, storage):
        if type(storage) is not dict:
            raise TypeError('MUTABLE_KV_REQUIRES_NATIVE_DICT')
        self.storage = storage

    def __getitem__(self, key):
        return self.storage[key]

    def __setitem__(self, key, value):
        self.storage[key] = value

    def __delitem__(self, key):
        del self.storage[key]

    def __iter__(self):
        return iter(self.storage)

    def __len__(self):
        return len(self.storage)

def wrap_native_cache(cache):
    if isinstance(cache, MutableKVState):
        return cache
    if type(cache) is dict:
        return MutableKVState(cache)
    if type(cache) is list:
        return [wrap_native_cache(c) for c in cache]
    if cache is None:
        return None
    raise TypeError('UNSUPPORTED_NATIVE_MUTABLE_CACHE')

def install_native_mutable_cache_abi(module):
    """Root pre-hook precedes FSDP2's input hook; actual cache Tensor IDs unchanged."""
    receipt = dict(mechanism='OPAQUE_ALIAS_MUTABLE_KV_ABI', calls=0, grad_calls=0, nograd_calls=0, tensor_clones=0, tensor_detaches=0)

    def before(module, args, kwargs):
        if kwargs.get('kv_cache') is None:
            return None
        import torch
        updated = dict(kwargs)
        updated['kv_cache'] = wrap_native_cache(kwargs['kv_cache'])
        receipt['calls'] += 1
        receipt['grad_calls' if torch.is_grad_enabled() else 'nograd_calls'] += 1
        return (args, updated)
    handle = module.register_forward_pre_hook(before, with_kwargs=True, prepend=True)
    return (handle, receipt)
