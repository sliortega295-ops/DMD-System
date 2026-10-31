"""Measured implementation helper, relocated from r2_fsdp2_baselines/scripts/sf_grad_rope.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
import hashlib
import inspect
from pathlib import Path

def require(value, reason):
    if not value:
        raise ValueError('SF_FP64_GRAD_ROPE:' + reason)

def build(torch):
    import triton
    import triton.language as tl

    @triton.jit
    def rotate(X, F, Y, PAIRS: tl.constexpr, HEADS: tl.constexpr, D: tl.constexpr, LENGTH: tl.constexpr, H: tl.constexpr, W: tl.constexpr, START: tl.constexpr, FS: tl.constexpr, CONJ: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < PAIRS
        token = i // (HEADS * D)
        col = i % D
        first = D - 2 * (D // 3)
        second = first + D // 3
        position = tl.where(col < first, token // (H * W) + START, tl.where(col < second, token // W % H, token % W))
        a = tl.load(X + 2 * i, valid, other=0).to(tl.float64)
        b = tl.load(X + 2 * i + 1, valid, other=0).to(tl.float64)
        active = valid & (token < LENGTH)
        c = tl.load(F + position * FS + 2 * col, active, other=1.0).to(tl.float64)
        s = tl.load(F + position * FS + 2 * col + 1, active, other=0.0).to(tl.float64)
        if CONJ:
            s = -s
        real = a * c - b * s
        imag = a * s + b * c
        tl.store(Y + 2 * i, tl.where(token < LENGTH, real, a), valid)
        tl.store(Y + 2 * i + 1, tl.where(token < LENGTH, imag, b), valid)

    def op(x, grid, freqs, start=0, conjugate=False):
        require(not torch.is_grad_enabled() and x.is_cuda and (x.dtype == torch.bfloat16) and (x.ndim == 4) and (x.shape[0] == 1) and (x.shape[2] in (12, 40)) and (x.shape[3] == 128) and x.is_contiguous() and (grid.device.type == 'cpu') and (tuple(grid.shape) == (1, 3)) and (freqs.device == x.device) and (freqs.dtype == torch.complex128) and freqs.is_contiguous() and (freqs.ndim == 2) and (freqs.shape[1] == 64) and (type(start) is int) and (start >= 0) and (type(conjugate) is bool) and (not freqs.requires_grad), 'NATIVE_BF16_FP64_ROPE_ABI')
        f, h, w = [int(v) for v in grid[0]]
        length = f * h * w
        require(min(f, h, w) > 0 and length <= x.shape[1] and (max(start + f, h, w) <= freqs.shape[0]), 'NATIVE_ROPE_GEOMETRY')
        out = torch.empty_like(x)
        values = torch.view_as_real(freqs)
        pairs = x.numel() // 2
        rotate[triton.cdiv(pairs, 256),](x, values, out, PAIRS=pairs, HEADS=x.shape[2], D=64, LENGTH=length, H=h, W=w, START=start, FS=values.stride(0), CONJ=conjugate, BLOCK=256, num_warps=4, enable_fp_fusion=False)
        return out
    return op

class AllModeRoPE:

    def __init__(self, trainer, *, op=None, cpu_test_only=False):
        import torch
        from wan.modules import model, causal_model
        self.trainer = trainer
        self.cpu = cpu_test_only
        self.enabled = True
        self.saved = []
        self.tested = set()
        self.grad_tested = set()
        self.qualifying = False
        self.op = build(torch) if op is None else op
        self.record = dict(mechanism='NATIVE_FP64_ROPE_FORWARD_AND_LINEAR_ADJOINT_FUSION', calls={}, grad_reference_calls=0, fused_grad_forward_calls=0, fused_grad_backward_calls=0, fp64_arithmetic=True, original_complex128_frequencies=True, output_dtype='torch.bfloat16', gradient_implementation_changed=True, gradient_math_changed=False, precision_changed=False, canaries=[], grad_canaries=[], contract_failures=0)
        helper = self

        class Rotation(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, grid, freqs, start):
                require(not freqs.requires_grad and x.dtype is torch.bfloat16, 'FROZEN_FREQUENCIES_BF16_INPUT_REQUIRED')
                ctx.save_for_backward(freqs)
                ctx.grid, ctx.start = (grid, start)
                ctx.qualifying = helper.qualifying
                if not ctx.qualifying:
                    helper.record['fused_grad_forward_calls'] += 1
                return helper.op(x, grid, freqs, start, False)

            @staticmethod
            def backward(ctx, gradient):
                require(not torch.is_grad_enabled(), 'HIGHER_ORDER_GRADIENT_NOT_ADMITTED')
                freqs, = ctx.saved_tensors
                if not ctx.qualifying:
                    helper.record['fused_grad_backward_calls'] += 1
                return (helper.op(gradient.contiguous(), ctx.grid, freqs, ctx.start, True), None, None, None)
        self.rotation = Rotation
        for module, name, causal, sha in [(model, 'rope_apply', False, 'f15214485a5297e6f9e79684bbad80acbcdc88c3272596b256a739a734529449'), (causal_model, 'causal_rope_apply', True, 'ae2644311ad212977260549e4b6034618647b82e2ee47a244bdaac50d8ff0778')]:
            original = getattr(module, name)
            require(hashlib.sha256(Path(inspect.getsourcefile(original)).read_bytes()).hexdigest() == sha, 'PINNED_NATIVE_ROPE_SOURCE')

            def dispatch(x, grid_sizes, freqs, *args, original=original, causal=causal, **kwargs):
                require(self.enabled or self.cpu, 'GPU_RUNTIME_DISABLE_NOT_ADMITTED')
                if not self.enabled:
                    return original(x, grid_sizes, freqs, *args, **kwargs)
                require(len(args) <= 1 and set(kwargs) <= {'start_frame'}, 'NATIVE_ROPE_ARGUMENTS')
                start = args[0] if args else kwargs.get('start_frame', 0)
                require(causal or start == 0, 'NONCAUSAL_START')
                key = ('causal' if causal else 'full', x.shape[2], x.shape[-1], str(x.dtype))
                if torch.is_grad_enabled():
                    if not x.requires_grad:
                        self.record['grad_reference_calls'] += 1
                        return original(x, grid_sizes, freqs, *args, **kwargs)
                    require(self.cpu or key in self.grad_tested, 'UNQUALIFIED_GRADIENT_GEOMETRY')
                    return self.rotation.apply(x, grid_sizes, freqs, start)
                if not self.cpu and key not in self.tested:
                    require(trainer.step < 5, 'UNQUALIFIED_NOGRAD_GEOMETRY_AFTER_C1')
                    test = x[:, :64].contiguous()
                    grid = torch.tensor([[1, 1, test.shape[1]]], dtype=torch.long)
                    expected = original(test, grid, freqs, start_frame=start) if causal else original(test, grid, freqs)
                    actual = self.op(test, grid, freqs, start, False)
                    torch.testing.assert_close(actual, expected)
                    self.record['canaries'].append(dict(key=str(key), relative_l2=float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)), scope='C1_UP_TO64_TOKEN_ORIGINAL_HEADS_FP64'))
                    self.tested.add(key)
                label = ('causal' if causal else 'full') + '-heads' + str(x.shape[2])
                self.record['calls'][label] = self.record['calls'].get(label, 0) + 1
                return self.op(x, grid_sizes, freqs, start, False)
            setattr(module, name, dispatch)
            self.saved.append((module, name, original, dispatch))
        if not self.cpu:
            try:
                self.qualify_gradients()
            except BaseException:
                self.close()
                raise

    def qualify_gradients(self):
        """Initialization-time component tests, outside checkpoint saved hooks."""
        import torch
        self.qualifying = True
        native = self.trainer.model.generator.model
        frequencies = native.freqs.to(device=self.trainer.device)
        require(frequencies.dtype is torch.complex128 and (not frequencies.requires_grad), 'ORIGINAL_FP64_FREQUENCIES_REQUIRED')
        grid = torch.tensor([[1, 1, 64]], dtype=torch.long)
        values = torch.arange(64 * 12 * 128, device=frequencies.device).reshape(1, 64, 12, 128).float()
        base = values.sin().bfloat16()
        upstream = values.cos().bfloat16()
        try:
            for module, name, original, dispatch in self.saved:
                causal = name == 'causal_rope_apply'
                start = 7 if causal else 0
                with torch.enable_grad():
                    a = base.clone().requires_grad_()
                    b = base.clone().requires_grad_()
                    expected = original(a, grid, frequencies, start_frame=start) if causal else original(a, grid, frequencies)
                    actual = self.rotation.apply(b, grid, frequencies, start)
                    ga = torch.autograd.grad(expected, a, upstream)[0]
                    gb = torch.autograd.grad(actual, b, upstream)[0]
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(gb, ga)
                key = ('causal' if causal else 'full', 12, 128, 'torch.bfloat16')
                self.grad_tested.add(key)
                self.record['grad_canaries'].append(dict(key=str(key), forward_relative_l2=float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)), backward_relative_l2=float((gb.float() - ga.float()).norm() / ga.float().norm().clamp_min(1e-30)), scope='INIT_COMPONENT_CANARY_UNCHANGED_DTYPE_DEFAULTS_NOT_C2_TIMER'))
        finally:
            self.qualifying = False

    def receipt(self):
        return dict(self.record, canary_keys=len(self.tested), grad_canary_keys=len(self.grad_tested), no_frequency_gradient=True, cpu_oracle_only=self.cpu)

    def close(self):
        for module, name, original, installed in reversed(self.saved):
            require(getattr(module, name) is installed, 'GLOBAL_ROPE_PATCH_OWNERSHIP')
            setattr(module, name, original)
