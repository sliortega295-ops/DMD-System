"""Measured implementation helper, relocated from r4_researchboost_transfer/scripts/sf_grad_pointwise.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
import types

def require(ok, reason):
    if not ok:
        raise ValueError('SF_GRAD_POINTWISE:' + reason)

def geometry(x, condition):
    d = x.shape[-1]
    frames = condition.numel() // d
    require(x.ndim == 3 and x.shape[0] == 1 and x.is_contiguous() and (condition.stride(-1) == 1) and (frames > 0) and (x.shape[1] % frames == 0), 'CONDITION_GEOMETRY')
    return (frames, x.reshape(1, frames, -1, d), condition.reshape(1, frames, 1, d))

def cpu_ops(torch):
    """Same arithmetic and save/reduce boundaries; CPU oracle, not kernel proof."""

    def mod(x, s, b):
        _, view, sv = geometry(x, s)
        return (view * (1 + sv) + b.reshape_as(sv)).reshape_as(x)

    def gate(x, y, g):
        _, view, gv = geometry(y, g)
        return x + (view * gv).reshape_as(x)

    def pair(g, x, s, plus_one):
        _, gv, sv = geometry(g, s)
        return ((gv * (1 + sv if plus_one else sv)).reshape_as(x), g * x)
    return (mod, gate, pair)

def cuda_ops(torch):
    import triton
    import triton.language as tl

    @triton.jit
    def forward(X, S, B, Y, D: tl.constexpr, T: tl.constexpr, SS: tl.constexpr, BS: tl.constexpr, N: tl.constexpr, GATE: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        off = i // D // T
        x = tl.load(X + i, i < N, other=0).to(tl.float32)
        if GATE:
            v = tl.load(S + i, i < N, other=0).to(tl.float32)
            g = tl.load(B + off * BS + i % D, i < N, other=0).to(tl.float32)
            y = x + (v * g).to(tl.bfloat16).to(tl.float32)
        else:
            s = tl.load(S + off * SS + i % D, i < N, other=0).to(tl.float32)
            b = tl.load(B + off * BS + i % D, i < N, other=0).to(tl.float32)
            scale = (1.0 + s).to(tl.bfloat16).to(tl.float32)
            y = (x * scale).to(tl.bfloat16).to(tl.float32) + b
        tl.store(Y + i, y.to(tl.bfloat16), i < N)

    @triton.jit
    def backward_pair(G, X, S, DX, DS, D: tl.constexpr, T: tl.constexpr, SS: tl.constexpr, N: tl.constexpr, PLUS_ONE: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        g = tl.load(G + i, i < N, other=0).to(tl.float32)
        x = tl.load(X + i, i < N, other=0).to(tl.float32)
        s = tl.load(S + i // D // T * SS + i % D, i < N, other=0).to(tl.float32)
        if PLUS_ONE:
            s = (1.0 + s).to(tl.bfloat16).to(tl.float32)
        tl.store(DX + i, (g * s).to(tl.bfloat16), i < N)
        tl.store(DS + i, (g * x).to(tl.bfloat16), i < N)

    def mod(x, s, b):
        frames, _, _ = geometry(x, s)
        d, n = (x.shape[-1], x.numel())
        sv, bv = (s.reshape(-1, d), b.reshape(-1, d))
        y = torch.empty_like(x)
        forward[triton.cdiv(n, 1024),](x, sv, bv, y, D=d, T=x.shape[1] // frames, SS=sv.stride(0), BS=bv.stride(0), N=n, GATE=False, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
        return y

    def gate(x, y, g):
        frames, _, _ = geometry(x, g)
        d, n = (x.shape[-1], x.numel())
        gv = g.reshape(-1, d)
        out = torch.empty_like(x)
        forward[triton.cdiv(n, 1024),](x, y, gv, out, D=d, T=x.shape[1] // frames, SS=0, BS=gv.stride(0), N=n, GATE=True, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
        return out

    def pair(g, x, s, plus_one):
        frames, _, _ = geometry(x, s)
        d, n = (x.shape[-1], x.numel())
        sv = s.reshape(-1, d)
        dx, ds = (torch.empty_like(x), torch.empty_like(x))
        backward_pair[triton.cdiv(n, 1024),](g.contiguous(), x, sv, dx, ds, D=d, T=x.shape[1] // frames, SS=sv.stride(0), N=n, PLUS_ONE=plus_one, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
        return (dx, ds)
    return (mod, gate, pair)

class GradPointwise:

    def __init__(self, trainer, *, cpu_test_only=False):
        import torch
        from wan.modules.model import WanAttentionBlock
        from wan.modules.causal_model import CausalWanAttentionBlock
        from .sf_nograd_modulation import rewrite
        require(not cpu_test_only or not torch.cuda.is_initialized(), 'CPU_TEST_ONLY')
        self.originals = []
        self.qualifying = False
        self.record = dict(mechanism='NATIVE_LN_BF16_POINTWISE_FORWARD_ADJOINT_FUSION', cpu_test_only=cpu_test_only, modules=0, grad_blocks={}, no_grad_delegations=0, modulation_forward=0, residual_forward=0, modulation_backward=0, residual_backward=0, canaries=[], contract_failures=0, source_proofs={}, native_layernorm_preserved=True, native_gradient_reductions=True, checkpoint_policy_changed=False, master_storage_changed=False, precision_changed=False, rng_draws_added=0)
        helper = self
        mod, gate, pair = cpu_ops(torch) if cpu_test_only else cuda_ops(torch)

        def validate(x, s):
            require(x.dtype == s.dtype == torch.bfloat16 and x.device == s.device, 'BF16_NATIVE_ABI')
            require(cpu_test_only or (x.is_cuda and x.shape[-1] == 1536), 'GPU_WIDTH')
            geometry(x, s)

        def reduce(g, condition):
            _, view, cv = geometry(g, condition)
            return view.sum_to_size(cv.shape).reshape_as(condition)

        class Modulate(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, s, b):
                validate(x, s)
                require(s.shape == b.shape and b.dtype == x.dtype, 'BIAS_ABI')
                ctx.save_for_backward(x, s)
                ctx.qualifying = helper.qualifying
                if not ctx.qualifying:
                    helper.record['modulation_forward'] += 1
                return mod(x, s, b)

            @staticmethod
            def backward(ctx, g):
                require(not torch.is_grad_enabled(), 'HIGHER_ORDER_NOT_ADMITTED')
                x, s = ctx.saved_tensors
                dx, product = pair(g, x, s, True)
                if not ctx.qualifying:
                    helper.record['modulation_backward'] += 1
                return (dx, reduce(product, s), reduce(g, s))

        class Residual(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, y, g):
                validate(x, g)
                require(x.shape == y.shape and y.dtype == x.dtype and y.is_contiguous(), 'RESIDUAL_ABI')
                ctx.save_for_backward(y, g)
                ctx.qualifying = helper.qualifying
                if not ctx.qualifying:
                    helper.record['residual_forward'] += 1
                return gate(x, y, g)

            @staticmethod
            def backward(ctx, incoming):
                require(not torch.is_grad_enabled(), 'HIGHER_ORDER_NOT_ADMITTED')
                y, g = ctx.saved_tensors
                dy, product = pair(incoming, y, g, False)
                if not ctx.qualifying:
                    helper.record['residual_backward'] += 1
                return (incoming, dy, reduce(product, g))
        self.modulate, self.residual = (Modulate, Residual)

        def norm_op(x, norm, scale, bias):
            require(not norm.elementwise_affine and norm.eps == 1e-06, 'NATIVE_NO_AFFINE_LN')
            return Modulate.apply(norm(x), scale, bias)

        def gated_op(x, y, gate):
            return Residual.apply(x, y, gate)
        for role in ('generator', 'fake_score'):
            for module in getattr(trainer.model, role).modules():
                cls = next((c for c in type(module).__mro__ if c in (WanAttentionBlock, CausalWanAttentionBlock)), None)
                if cls is None:
                    continue
                original = module.forward
                function, proof = rewrite(cls.forward, norm_op, gated_op)
                self.originals.append((module, original))
                self.record['source_proofs'][cls.__name__] = proof

                def forward(this, *args, original=original, fn=function, role=role, **kwargs):
                    if not torch.is_grad_enabled():
                        helper.record['no_grad_delegations'] += 1
                        return original(*args, **kwargs)
                    helper.record['grad_blocks'][role] = helper.record['grad_blocks'].get(role, 0) + 1
                    return fn(this, *args, **kwargs)
                module.forward = types.MethodType(forward, module)
        self.record['modules'] = len(self.originals)
        require(len(self.originals) > 0 if cpu_test_only else len(self.originals) == 60, 'TRAINABLE_BLOCK_SCOPE')
        if not cpu_test_only:
            try:
                self.qualify(torch, trainer.device)
            except BaseException:
                self.close()
                raise

    def qualify(self, torch, device):
        """Actual BF16 fwd/adjoint before C1, no training RNG or parameters."""
        self.qualifying = True
        try:
            for frames in (1, 3):
                z = torch.arange(frames * 128 * 1536, device=device, dtype=torch.float32).reshape(1, -1, 1536)
                h, incoming = (z.sin().bfloat16(), z.cos().bfloat16())
                e = torch.arange(frames * 6 * 1536, device=device).reshape(1, frames, 6, 1536).float().sin().bfloat16()
                s, b = (e[:, :, 1:2], e[:, :, 3:4])
                for kind in ('modulate', 'residual'):
                    a = [v.detach().clone().requires_grad_() for v in (h, s, b) if kind == 'modulate']
                    if kind == 'residual':
                        a = [v.detach().clone().requires_grad_() for v in (h, h * 0.7, s)]
                    v = [t.detach().clone().requires_grad_() for t in a]
                    with torch.enable_grad():
                        reference = cpu_ops(torch)[0 if kind == 'modulate' else 1](*a)
                        actual = (self.modulate if kind == 'modulate' else self.residual).apply(*v)
                        ga = torch.autograd.grad(reference, a, incoming)
                        gb = torch.autograd.grad(actual, v, incoming)
                    for x, y in [(actual, reference), *zip(gb, ga)]:
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
                    self.record['canaries'].append(dict(kind=kind, frames=frames, forward_bitwise=True, all_input_gradients_bitwise=True))
        finally:
            self.qualifying = False

    def receipt(self):
        return dict(self.record)

    def close(self):
        for module, original in reversed(self.originals):
            module.forward = original
