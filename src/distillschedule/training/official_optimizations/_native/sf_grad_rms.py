"""Measured implementation helper, relocated from r3_target2x/scripts/sf_grad_rms.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
import hashlib
import inspect
from pathlib import Path
import types

def require(ok, reason):
    if not ok:
        raise ValueError('SF_GRAD_RMS:' + reason)

def cpu_ops(torch):
    """Actual adjoint arithmetic for CPU integration tests, not GPU evidence."""

    def forward(x, w, eps):
        f = x.float()
        r = torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)
        n = (f * r).to(x.dtype)
        return (n * w, n, r)

    def backward(g, x, w, n, r):
        f = x.float()
        v = (g * w).float()
        dr = (v * f).sum(-1, keepdim=True)
        dv = -0.5 * dr * r.pow(3)
        dx = (v * r + dv / x.shape[-1] * (2.0 * f)).to(x.dtype)
        return (dx, (g * n).sum((0, 1)))
    return (forward, backward)

def cuda_ops(torch):
    import triton
    import triton.language as tl

    @triton.jit
    def fwd(X, W, Y, NORMAL, RSTD, D: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        c = tl.arange(0, B)
        x = tl.load(X + row * D + c, c < D, other=0).to(tl.float32)
        r = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        n = (x * r).to(tl.bfloat16)
        w = tl.load(W + c, c < D, other=0).to(tl.float32)
        y = n.to(tl.float32) * w
        tl.store(NORMAL + row * D + c, n, c < D)
        tl.store(Y + row * D + c, y, c < D)
        tl.store(RSTD + row, r)

    @triton.jit
    def bwd(G, X, W, NORMAL, RSTD, DX, GW, D: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        c = tl.arange(0, B)
        x = tl.load(X + row * D + c, c < D, other=0).to(tl.float32)
        g = tl.load(G + row * D + c, c < D, other=0).to(tl.float32)
        w = tl.load(W + c, c < D, other=0).to(tl.float32)
        n = tl.load(NORMAL + row * D + c, c < D, other=0).to(tl.float32)
        r = tl.load(RSTD + row)
        v = (g * w).to(tl.bfloat16).to(tl.float32)
        dr = tl.sum(v * x, 0)
        dv = -0.5 * dr * (r * r * r)
        dx = v * r + dv / D * (2.0 * x)
        tl.store(DX + row * D + c, dx, c < D)
        tl.store(GW + row * D + c, g * n, c < D)

    def forward(x, w, eps):
        d = x.shape[-1]
        y = torch.empty_like(x)
        n = torch.empty_like(x)
        r = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
        fwd[x.numel() // d,](x, w, y, n, r, D=d, EPS=eps, B=triton.next_power_of_2(d), num_warps=4, enable_fp_fusion=False)
        return (y, n, r)

    def backward(g, x, w, n, r):
        d = x.shape[-1]
        dx = torch.empty_like(x)
        gw = torch.empty_like(x)
        bwd[x.numel() // d,](g.contiguous(), x, w, n, r, dx, gw, D=d, B=triton.next_power_of_2(d), num_warps=4, enable_fp_fusion=False)
        return (dx, gw.sum((0, 1)))
    return (forward, backward)

class GradRMS:

    def __init__(self, trainer, *, cpu_test_only=False):
        import torch
        from wan.modules.model import WanRMSNorm
        require(hashlib.sha256(Path(inspect.getsourcefile(WanRMSNorm)).read_bytes()).hexdigest() == 'f15214485a5297e6f9e79684bbad80acbcdc88c3272596b256a739a734529449', 'NATIVE_SOURCE')
        require(not cpu_test_only or not torch.cuda.is_initialized(), 'CPU_TEST_ONLY')
        self.originals = []
        self.qualifying = False
        self.cpu = cpu_test_only
        self.record = dict(mechanism='NATIVE_BF16_RMS_FORWARD_ADJOINT_FUSION', modules=0, forward_calls=0, backward_calls=0, no_grad_delegations=0, contract_failures=0, roles={}, canaries=[], master_storage_changed=False, checkpoint_policy_changed=False, semantic_math_changed=False, reduction_tree_changed=True, gradient_implementation_changed=True, affine_product_before_weight_reduction='BF16_THEN_NATIVE_SUM', legacy_no_grad_grad_reference_calls='DELEGATION_TO_THIS_EXPLICIT_GRAD_HANDLER', cpu_test_only=cpu_test_only)
        fw, bw = cpu_ops(torch) if cpu_test_only else cuda_ops(torch)
        helper = self

        class RMS(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, w, eps):
                require(x.ndim == 3 and x.shape[0] == 1 and x.is_contiguous() and (x.dtype == w.dtype == torch.bfloat16) and w.is_contiguous() and (x.device == w.device) and (tuple(w.shape) == (x.shape[-1],)) and (eps == 1e-06), 'BF16_NATIVE_ABI')
                require(cpu_test_only or (x.is_cuda and x.shape[-1] == 1536), 'NATIVE_WIDTH')
                y, n, r = fw(x, w, eps)
                ctx.save_for_backward(x, w, n, r)
                ctx.qualifying = helper.qualifying
                if not ctx.qualifying:
                    helper.record['forward_calls'] += 1
                return y

            @staticmethod
            def backward(ctx, gradient):
                require(not torch.is_grad_enabled(), 'HIGHER_ORDER_NOT_ADMITTED')
                x, w, n, r = ctx.saved_tensors
                dx, dw = bw(gradient, x, w, n, r)
                if not ctx.qualifying:
                    helper.record['backward_calls'] += 1
                return (dx, dw, None)
        self.function = RMS
        for role in ('generator', 'fake_score'):
            for module in getattr(trainer.model, role).modules():
                if type(module) is not WanRMSNorm:
                    continue
                original = module.forward
                self.originals.append((module, original))

                def forward(this, x, original=original, role=role):
                    if not torch.is_grad_enabled():
                        self.record['no_grad_delegations'] += 1
                        return original(x)
                    self.record['roles'][role] = self.record['roles'].get(role, 0) + 1
                    return self.function.apply(x, this.weight, this.eps)
                module.forward = types.MethodType(forward, module)
        self.record['modules'] = len(self.originals)
        require(len(self.originals) > 0 if self.cpu else len(self.originals) == 240, 'ROLE_SCOPE')
        if not self.cpu:
            try:
                self.qualify(torch, trainer.device, WanRMSNorm)
            except BaseException:
                self.close()
                raise

    def qualify(self, torch, device, native_class):
        """No training RNG, no model state, no checkpoint hooks; before C1 timer."""
        self.qualifying = True
        try:
            native = native_class(1536, eps=1e-06).to(device=device, dtype=torch.bfloat16)
            z = torch.arange(64 * 1536, device=device, dtype=torch.float32).reshape(1, 64, 1536)
            for scale in (1.0, 0.01):
                x = (z.sin() * scale).bfloat16()
                grad = z.cos().bfloat16()
                with torch.no_grad():
                    native.weight.copy_((torch.arange(1536, device=device).float().sin() * 0.2 + 1).bfloat16())
                with torch.enable_grad():
                    a = x.clone().requires_grad_()
                    b = x.clone().requires_grad_()
                    w = native.weight.detach().clone().requires_grad_()
                    expected = native(a)
                    actual = self.function.apply(b, w, native.eps)
                    ga = torch.autograd.grad(expected, (a, native.weight), grad)
                    gb = torch.autograd.grad(actual, (b, w), grad)
                for v, u in ((actual, expected), *zip(gb, ga)):
                    torch.testing.assert_close(v, u)
                self.record['canaries'].append(dict(scale=scale, forward_relative_l2=float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)), input_gradient_relative_l2=float((gb[0].float() - ga[0].float()).norm() / ga[0].float().norm().clamp_min(1e-30)), weight_gradient_relative_l2=float((gb[1].float() - ga[1].float()).norm() / ga[1].float().norm().clamp_min(1e-30)), scope='INIT_UNCHANGED_TORCH_DTYPE_DEFAULTS'))
        finally:
            self.qualifying = False

    def receipt(self):
        return dict(self.record)

    def close(self):
        for module, original in reversed(self.originals):
            module.forward = original
