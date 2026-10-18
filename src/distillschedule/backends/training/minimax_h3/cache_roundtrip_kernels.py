"""Read-only diagnostic fingerprints, not a replacement training kernel."""
import triton
import triton.language as tl


@triton.jit
def _mix(x):
    x = x ^ (x >> 16)
    x = x * 0x7FEB352D
    x = x ^ (x >> 15)
    x = x * 0x846CA68B
    return x ^ (x >> 16)


@triton.jit
def _words(X, OUT, N: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr,
           S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
           BITS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offset = ((i // (D1*D2*D3))*S0 + ((i // (D2*D3)) % D1)*S1
              + ((i // D3) % D2)*S2 + (i % D3)*S3)
    raw = tl.load(X + offset, i < N, other=0)
    if BITS == 16:
        bits = raw.to(tl.uint16, bitcast=True).to(tl.uint32)
    else:
        bits = raw.to(tl.uint32, bitcast=True)
    index = i.to(tl.uint32)
    first = _mix(bits ^ _mix(index + 0x9E3779B9))
    second = _mix(bits + _mix(index ^ 0x85EBCA6B))
    a = tl.xor_sum(tl.where(i < N, first, 0), 0)
    b = tl.sum(tl.where(i < N, second, 0), 0, dtype=tl.uint32)
    tl.store(OUT + tl.program_id(0)*2, a)
    tl.store(OUT + tl.program_id(0)*2 + 1, b)


@triton.jit
def _fold(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    a = tl.load(X + i*2, i < N, other=0).to(tl.uint32)
    b = tl.load(X + i*2+1, i < N, other=0).to(tl.uint32)
    tl.store(OUT + tl.program_id(0)*2, tl.xor_sum(a,0))
    tl.store(OUT + tl.program_id(0)*2+1, tl.sum(b,0,dtype=tl.uint32))


def fingerprint(torch, value):
    if (value.device.type != 'cuda' or value.dtype not in (torch.bfloat16,torch.float32)
            or value.ndim not in (3,4) or not 0 < value.numel() <= 2**29):
        raise ValueError('ROUNDTRIP_UNSUPPORTED_OR_UNBOUNDED_TENSOR')
    shape=(1,)*(4-value.ndim)+tuple(value.shape)
    strides=(0,)*(4-value.ndim)+tuple(value.stride())
    count=triton.cdiv(value.numel(),4096)
    scratch=torch.empty((count,2),device=value.device,dtype=torch.int64)
    with torch.no_grad():
        _words[(count,)](value.detach(),scratch,value.numel(),*shape[1:],*strides,
                        16 if value.dtype==torch.bfloat16 else 32,4096,num_warps=4)
        while count>1:
            following=triton.cdiv(count,1024)
            out=torch.empty((following,2),device=value.device,dtype=torch.int64)
            _fold[(following,)](scratch,out,count,1024,num_warps=4)
            scratch,count=out,following
    return scratch.view(2)
