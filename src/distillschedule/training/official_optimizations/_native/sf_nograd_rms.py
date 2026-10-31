"""Measured implementation helper, relocated from r2_fsdp2_baselines/scripts/sf_nograd_rms.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
import hashlib
import inspect
from pathlib import Path
import types
MODEL_SHA256 = 'f15214485a5297e6f9e79684bbad80acbcdc88c3272596b256a739a734529449'

def require(value, reason):
    if not value:
        raise ValueError(reason)

def build(torch):
    import triton
    import triton.language as tl

    @triton.jit
    def rms(X, W, Y, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
        square = tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', '=f,f,f', [x, x], dtype=tl.float32, is_pure=True, pack=1)
        mean = tl.sum(square, 0) / D
        normalized = (x * tl.rsqrt(mean + EPS)).to(tl.bfloat16).to(tl.float32)
        weight = tl.load(W + col, col < D, other=0).to(tl.float32)
        value = tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', '=f,f,f', [normalized, weight], dtype=tl.float32, is_pure=True, pack=1)
        tl.store(Y + row * D + col, value, col < D)

    def op(x, weight, eps):
        require(not torch.is_grad_enabled() and x.is_cuda and (x.device == weight.device) and (x.dtype == weight.dtype == torch.bfloat16) and x.is_contiguous() and (x.ndim == 3) and (x.shape[0] == 1) and (x.shape[-1] in (1536, 5120)) and (tuple(weight.shape) == (x.shape[-1],)) and weight.is_contiguous() and (eps == 1e-06), 'NATIVE_NOGRAD_BF16_RMS_CONTRACT')
        d = x.shape[-1]
        out = torch.empty_like(x)
        rms[x.numel() // d,](x, weight, out, D=d, EPS=eps, BLOCK=triton.next_power_of_2(d), num_warps=4 if d == 1536 else 8, enable_fp_fusion=False)
        return out
    return op

class NoGradRMS:

    def __init__(self, trainer, *, op=None, device_type='cuda'):
        import torch
        from wan.modules.model import WanRMSNorm
        self.trainer = trainer
        self.originals = []
        self.canaries = set()
        self.device_type = device_type
        path = Path(inspect.getsourcefile(WanRMSNorm)).resolve()
        require(hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256, 'NATIVE_RMS_SOURCE_CHANGED')
        self.op = build(torch) if op is None else op
        self.record = dict(mechanism='NATIVE_BF16_NOGRAD_RMS_FUSION', native_model_sha256=MODEL_SHA256, intermediate_bf16_rounding=True, affine_dtype='bfloat16', output_dtype='bfloat16', reduction_tree_changed=True, semantic_math_changed=False, grad_path_changed=False, calls={}, grad_reference_calls={}, contract_failures=0, canaries=[], provenance='Port of existing LiveWan C51 RMS template; BF16 affine preserves this native source')
        for role in ('generator', 'fake_score', 'real_score'):
            for module in getattr(trainer.model, role).modules():
                if type(module) is not WanRMSNorm:
                    continue
                original = module.forward
                self.originals.append((module, original))

                def forward(this, x, original=original, role=role):
                    if torch.is_grad_enabled():
                        self.record['grad_reference_calls'][role] = self.record['grad_reference_calls'].get(role, 0) + 1
                        return original(x)
                    require(x.device.type == self.device_type, 'RMS_DEVICE_SCOPE')
                    key = (role, tuple(x.shape), str(x.dtype), str(this.weight.dtype))
                    result = self.op(x, this.weight, this.eps)
                    if key not in self.canaries:
                        require(trainer.step < 5, 'UNQUALIFIED_RMS_SHAPE_AFTER_WARMUP')
                        expected = original(x)
                        torch.testing.assert_close(result, expected)
                        rel = float((result.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30))
                        self.record['canaries'].append(dict(key=str(key), relative_l2=rel, scope='C1_ONLY_DTYPE_DEFAULT_OPERATOR_CHECK', bitwise=torch.equal(result, expected)))
                        self.canaries.add(key)
                    self.record['calls'][role] = self.record['calls'].get(role, 0) + 1
                    return result
                module.forward = types.MethodType(forward, module)
        require(len(self.originals) == 400 if device_type == 'cuda' else len(self.originals) > 0, 'NATIVE_RMS_MODULE_CENSUS')

    def receipt(self):
        return dict(self.record, modules=len(self.originals), canary_keys=len(self.canaries))

    def close(self):
        for module, original in self.originals:
            module.forward = original
