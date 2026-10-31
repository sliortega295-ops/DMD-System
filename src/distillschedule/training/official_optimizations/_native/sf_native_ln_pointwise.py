"""Measured implementation helper, relocated from r2_fsdp2_baselines/scripts/sf_native_ln_pointwise.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
def compose(pointwise, gated):

    def mod(x, norm, scale, bias):
        return pointwise(norm(x), scale, bias)
    return (mod, gated)

def build(torch):
    import triton
    import triton.language as tl
    from .sf_nograd_modulation import build as original_build, require
    unused, gated = original_build(torch)

    @triton.jit
    def kernel(X, S, B, Y, D: tl.constexpr, T: tl.constexpr, SS: tl.constexpr, BS: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + i, i < N, other=0).to(tl.float32)
        s = tl.load(S + i // D // T * SS + i % D, i < N, other=0).to(tl.float32)
        b = tl.load(B + i // D // T * BS + i % D, i < N, other=0).to(tl.float32)
        scale = (1.0 + s).to(tl.bfloat16).to(tl.float32)
        product = (x * scale).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + i, (product + b).to(tl.bfloat16), i < N)

    def pointwise(x, scale, bias):
        require(not torch.is_grad_enabled() and x.is_cuda and (x.ndim == 3) and (x.shape[0] == 1) and (x.dtype == scale.dtype == bias.dtype == torch.bfloat16) and (x.shape[-1] in (1536, 5120)) and x.is_contiguous() and (x.device == scale.device == bias.device), 'NATIVE_LN_POINTWISE_ABI')
        d = x.shape[-1]
        s = scale.reshape(-1, d)
        b = bias.reshape(-1, d)
        require(s.shape == b.shape and x.shape[1] % s.shape[0] == 0, 'NATIVE_LN_CONDITION_GEOMETRY')
        out = torch.empty_like(x)
        n = x.numel()
        kernel[triton.cdiv(n, 1024),](x, s, b, out, D=d, T=x.shape[1] // s.shape[0], SS=s.stride(0), BS=b.stride(0), N=n, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
        return out
    return compose(pointwise, gated)
