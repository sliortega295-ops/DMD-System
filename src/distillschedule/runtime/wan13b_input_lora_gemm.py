"""One owned GPU: BF16 input -> FP32 LoRA-A GEMM, FP32 weights unchanged.
Primitive diagnostic only; outputs and benchmark records do not grant E2E parity.
"""
import torch,triton,triton.language as tl
@triton.jit
def mm(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,PREC:tl.constexpr):
 m=tl.program_id(0)*BM+tl.arange(0,BM)
 n=tl.program_id(1)*BN+tl.arange(0,BN)
 k=tl.arange(0,BK)
 acc=tl.zeros((BM,BN),tl.float32)
 for start in range(tl.cdiv(K,BK)):
  kk=start*BK+k
  x=tl.load(X+m[:,None]*K+kk[None,:],mask=(m[:,None]<M)&(kk[None,:]<K),other=0).to(tl.float32)
  w=tl.load(W+n[None,:]*K+kk[:,None],mask=(n[None,:]<N)&(kk[:,None]<K),other=0)
  if PREC == 'rn':
   bits=tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r,f", [w], dtype=tl.int32, is_pure=True, pack=1)
   w=bits.to(tl.float32,bitcast=True)
   acc=tl.dot(x,w,acc,input_precision='tf32')
  else:
   acc=tl.dot(x,w,acc,input_precision=PREC)
 tl.store(Y+m[:,None]*N+n[None,:],acc,mask=(m[:,None]<M)&(n[None,:]<N))
def fused(x,w,bm=32,bn=64,bk=32,warps=4,precision='rn'):
 m,k=x.shape;n=w.shape[0];out=torch.empty((m,n),device=x.device,dtype=torch.float32)
 mm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](x,w,out,m,n,k,bm,bn,bk,precision,num_warps=warps)
 return out

def fused_input_lora_a(x,weight):
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.float32:
        raise ValueError('BF16_INPUT_FP32_LORA_STORAGE_REQUIRED')
    if not x.is_contiguous() or not weight.is_contiguous() or weight.shape != (256,1536):
        raise ValueError('QUALIFIED_LORA_A_LAYOUT_REQUIRED')
    # No host value read, RNG, weight merge, or parameter/cache mutation.
    shape=x.shape[:-1]
    return fused(x.reshape(-1,1536),weight,64,64,64,4,'rn').reshape(*shape,256)
