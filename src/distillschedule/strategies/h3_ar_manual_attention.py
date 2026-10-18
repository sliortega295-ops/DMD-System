"""Manual-only H56 no-grad rollout backend; canonical grad/cache remain native."""
from collections import Counter
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
import functools
import hashlib

POLICY_ID = 'uncached-rollout-fa3-varlen-v1'
_MAIN = ContextVar('h3_ar_manual_attention_main', default=None)
_ROOT = ContextVar('h3_ar_manual_attention_root', default=None)
PINS = {
    'metadata.json': 'f1b4b0ef991c0966f7d848777154c0591c4580968fb6da33f6b2a2829a109309',
    'flash_attn_interface.py': 'cb46c3d89606305817ea20eb7b7cc9376b013eeabfd7637a328e7967e1d008d4',
    '__init__.py': '29756640933e2a159cd94a9625eb2ea4268ffa6a80bd3c7bc86b9e1404e09471',
    '_ops.py': '7f392b365cc945a3b3a420e24802ef776a79316c068e6e99e4da682e8927a1f6',
    '_flash_attn3_cuda_373a45c.abi3.so': '8aa9bb49e174872be4e2cb8c2608f2d66de29b513f704fb8d91771f9cd39df56',
}


def validate_backend_root(root):
    path = Path(root)
    if (not path.is_absolute() or path.name != 'torch210-cxx11-cu128-x86_64-linux'
            or 'acb0f42e6901d3e9effa8a83bd32bc2bf0cfdfc2' not in path.parts):
        raise ValueError('EXPLICIT_PINNED_EXISTING_FA3_BUILD_REQUIRED')
    for name, expected in PINS.items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest() != expected:
            raise ValueError('FA3_SOURCE_OR_BINARY_CHANGED:' + name)
    return path


def tensor_token(value):
    return (str(value.device), str(value.dtype), tuple(value.shape), tuple(value.stride()),
            value.data_ptr(), value.storage_offset(), int(value._version))


def eligible(*, grad_enabled, root, call, main):
    return (not grad_enabled and root == ('student', False) and call is None and main is not None)


class ManualNoGradAttention:
    id = POLICY_ID

    def __init__(self, kernel_root):
        import torch
        if torch.__version__ != '2.10.0+cu128':
            raise RuntimeError('PINNED_TORCH210_REQUIRED')
        root = validate_backend_root(kernel_root)
        from kernels.utils import get_local_kernel
        self.backend = get_local_kernel(root, 'flash_attn3')
        if not callable(self.backend.flash_attn_varlen_func):
            raise RuntimeError('PINNED_VARLEN_ENTRY_MISSING')
        self.kernel_root = str(root)
        self.counts, self.previous = Counter(), Counter()
        self.cu_token = self.bounds = None

    def attach(self, module, adapter, pool, positive):
        import torch
        import native_selected_rollout_reuse_r1 as routing
        if pool is None or len(adapter.dit.blocks) != 50:
            raise RuntimeError('ORIGINAL_FIFTY_BLOCK_AP_CACHE_REQUIRED')
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != 'bd617a4312c70405e916707e1908100b484b2259dfbeba4af7de5bb145206d80':
            raise RuntimeError('ORIGINAL_NATIVE_DIT_REQUIRED')
        cu = positive['packed']['cu_seqlens']
        if not cu.is_cuda or cu.dtype != torch.int32 or tuple(cu.shape) != (3,):
            raise RuntimeError('NATIVE_THREE_BOUND_CUDA_CU_REQUIRED')
        # One setup-only value read. Training routing uses storage metadata only.
        bounds = cu.tolist()
        if not (bounds[0] == 0 < bounds[1] < bounds[2] and bounds[1] >= bounds[2]-bounds[1]):
            raise RuntimeError('ORIGINAL_MAIN_PLUS_NONEMPTY_PADDING_REQUIRED')
        self.bounds, self.cu_token = bounds, tensor_token(cu)
        self.call_context = routing._CALL
        self.install_dispatch(module, adapter, torch)

    def install_dispatch(self, module, adapter, torch):
        """Compose the actual RMS-generated forward globals; no norm/RoPE rewrite."""
        before = [(id(p), getattr(p, 'ds_numel', p.numel()), p.dtype, p.requires_grad)
                  for p in adapter.parameters()]
        original = module._sdpa_varlen_attention
        tables = {}
        for index, block in enumerate(adapter.dit.blocks):
            attention = block.attn
            if (attention.num_heads, attention.head_dim) != (56, 128):
                raise RuntimeError('ACTUAL_H56_D128_REQUIRED_NOT_HIDDEN_DERIVED_H42')
            bound = attention.forward
            function = getattr(bound, '__func__', bound)
            table = function.__globals__
            if table.get('_sdpa_varlen_attention') is not original:
                raise RuntimeError('UNSUPPORTED_RMS_NATIVE_DISPATCH_COMPOSITION')
            tables[id(table)] = table
            def forward(*args, _original=bound, _index=index, **kwargs):
                maximum = kwargs.get('max_seqlen')
                if type(maximum) is not int or maximum != self.bounds[1]:
                    raise RuntimeError('NATIVE_HOST_MAXIMUM_CHANGED')
                token = _MAIN.set((_index, maximum))
                self.counts['main_forward_calls'] += 1
                try:
                    return _original(*args, **kwargs)
                finally:
                    _MAIN.reset(token)
            attention.forward = forward

        def dispatch(q, k, v, cu_seqlens, softmax_scale):
            main = _MAIN.get()
            if not eligible(grad_enabled=torch.is_grad_enabled(), root=_ROOT.get(),
                            call=self.call_context.get(), main=main):
                self.counts['main_native_fallback' if main is not None else 'refiner_native_fallback'] += 1
                return original(q, k, v, cu_seqlens, softmax_scale)
            if (tuple(q.shape) != (self.bounds[-1], 56, 128) or k.shape != q.shape or v.shape != q.shape
                    or q.dtype != torch.bfloat16 or any(x.dtype != q.dtype or not x.is_cuda for x in (q, k, v))
                    or softmax_scale != 128**-.5 or tensor_token(cu_seqlens) != self.cu_token):
                raise RuntimeError('EXACT_NATIVE_UNCACHED_ROLLOUT_INPUT_CONTRACT_CHANGED')
            output = self.backend.flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens,
                main[1], main[1], softmax_scale=softmax_scale, causal=False,
                deterministic=False, return_attn_probs=False)
            if (output.shape != q.shape or output.dtype != q.dtype or not output.is_contiguous()
                    or output.requires_grad):
                raise RuntimeError('NO_GRAD_ORIGINAL_CONTIGUOUS_OUTPUT_REQUIRED')
            self.counts['actual_FA3_varlen_calls'] += 1
            return output

        for table in tables.values():
            table['_sdpa_varlen_attention'] = dispatch
        old_predict = adapter.predict
        @functools.wraps(old_predict)
        def predict(*args, **kwargs):
            roles = {layer.role for _, layer in adapter.layers}
            if len(roles) != 1:
                raise RuntimeError('ORIGINAL_ADAPTER_ROLE_RACE')
            token = _ROOT.set((next(iter(roles)), torch.is_grad_enabled()))
            self.counts['application_predict_calls'] += 1
            try:
                return old_predict(*args, **kwargs)
            finally:
                _ROOT.reset(token)
        adapter.predict = predict
        if before != [(id(p), getattr(p, 'ds_numel', p.numel()), p.dtype, p.requires_grad)
                      for p in adapter.parameters()]:
            raise RuntimeError('ATTENTION_OPTIN_MUTATED_REGISTERED_PARAMETERS')

    def finish_cycle(self, cycle):
        if _ROOT.get() is not None or _MAIN.get() is not None:
            raise RuntimeError('UNCLOSED_ATTENTION_SCOPE')
        delta = self.counts - self.previous
        expected = dict(main_forward_calls=700, application_predict_calls=12,
                        actual_FA3_varlen_calls=350, main_native_fallback=350,
                        refiner_native_fallback=24)
        if dict(delta) != expected:
            raise RuntimeError('FAIL_ACTUAL_UNCACHED_ROLLOUT_BACKEND_CENSUS:' + repr(dict(delta)))
        self.previous = self.counts.copy()
        return dict(policy_id=self.id, actual_cycle=cycle, per_cycle_counts=dict(delta),
                    kernel_root=self.kernel_root, backend_pins=PINS, heads=56, head_dim=128,
                    bounds=self.bounds, grad_and_selected_cache_paths_unchanged=True,
                    params_relocated=False, weights_merged=False, automatic_planning_eligible=False,
                    hot_path_host_value_reads=0)
