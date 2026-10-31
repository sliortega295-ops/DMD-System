"""Measured implementation helper, relocated from r2_fsdp2_baselines/scripts/sf_nograd_modulation.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
import types

def require(value, reason):
    if not value:
        raise ValueError('SF_NOGRAD_MODULATION:' + reason)

def build(torch):
    import triton
    import triton.language as tl

    @triton.jit
    def norm_mod(X, S, B, Y, D: tl.constexpr, T: tl.constexpr, SS: tl.constexpr, BS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        value = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
        mean = tl.sum(value, 0) / D
        delta = value - mean
        variance = tl.sum(tl.where(col < D, delta * delta, 0.0), 0) / D
        normal = (delta * tl.rsqrt(variance + EPS)).to(tl.bfloat16).to(tl.float32)
        scale = tl.load(S + row // T * SS + col, col < D, other=0).to(tl.float32)
        bias = tl.load(B + row // T * BS + col, col < D, other=0).to(tl.float32)
        scale = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
        product = (normal * scale).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + row * D + col, (product + bias).to(tl.bfloat16), col < D)

    @triton.jit
    def gated(X, V, G, Y, D: tl.constexpr, T: tl.constexpr, GS: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
        at = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + at, at < N, other=0).to(tl.float32)
        v = tl.load(V + at, at < N, other=0).to(tl.float32)
        g = tl.load(G + at // D // T * GS + at % D, at < N, other=0).to(tl.float32)
        product = (v * g).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + at, (x + product).to(tl.bfloat16), at < N)

    def validate(x, e):
        require(not torch.is_grad_enabled() and x.is_cuda and (x.dtype == torch.bfloat16) and (x.ndim == 3) and (x.shape[0] == 1) and (x.shape[-1] in (1536, 5120)) and x.is_contiguous() and (e.device == x.device) and (e.dtype == x.dtype) and (e.shape[-1] == x.shape[-1]) and (e.stride(-1) == 1), 'NATIVE_BF16_B1_ABI')
        view = e.reshape(-1, x.shape[-1])
        frames = view.shape[0]
        require(frames > 0 and x.shape[1] % frames == 0, 'PER_FRAME_CONDITION_GEOMETRY')
        return (view, x.shape[1] // frames)

    def mod(x, norm, scale, bias):
        s, t = validate(x, scale)
        b, tb = validate(x, bias)
        require(t == tb and norm.eps == 1e-06 and (not norm.elementwise_affine), 'NATIVE_NO_AFFINE_LN')
        out = torch.empty_like(x)
        d = x.shape[-1]
        norm_mod[x.shape[1],](x, s, b, out, D=d, T=t, SS=s.stride(0), BS=b.stride(0), EPS=norm.eps, BLOCK=triton.next_power_of_2(d), num_warps=4 if d == 1536 else 8, enable_fp_fusion=False)
        return out

    def residual(x, y, gate):
        g, t = validate(x, gate)
        require(x.shape == y.shape and x.dtype == y.dtype and (x.device == y.device) and y.is_contiguous(), 'RESIDUAL_ABI')
        out = torch.empty_like(x)
        n = x.numel()
        gated[triton.cdiv(n, 1024),](x, y, g, out, D=x.shape[-1], T=t, GS=g.stride(0), N=n, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
        return out
    return (mod, residual)

def rewrite(original, mod, residual):
    """Exact AST matching, not textual/heuristic replacement of arbitrary ops."""
    function = original.__func__ if hasattr(original, '__func__') else original
    path = Path(inspect.getsourcefile(function))
    expected = {'model.py': 'f15214485a5297e6f9e79684bbad80acbcdc88c3272596b256a739a734529449', 'causal_model.py': 'ae2644311ad212977260549e4b6034618647b82e2ee47a244bdaac50d8ff0778'}
    require(path.name in expected and hashlib.sha256(path.read_bytes()).hexdigest() == expected[path.name], 'NATIVE_BLOCK_SOURCE')
    causal = path.name == 'causal_model.py'
    expressions = {}
    for norm, scale, bias, gate in [('norm1', 1, 0, 2), ('norm2', 4, 3, 5)]:
        n = f'self.{norm}(x)'
        if causal:
            n += ' .unflatten(dim=1, sizes=(num_frames, frame_seqlen))'
        old = f'{n} * (1 + e[{scale}]) + e[{bias}]'
        if causal:
            old = f'({old}).flatten(1, 2)'
        expressions[ast.unparse(ast.parse(old, mode='eval').body)] = f'_sf_norm_mod(x, self.{norm}, e[{scale}], e[{bias}])'
        term = f'y * e[{gate}]' if not causal else f'(y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[{gate}]).flatten(1, 2)'
        expressions[ast.unparse(ast.parse('x + ' + term, mode='eval').body)] = f'_sf_gated(x, y, e[{gate}])'
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    changes = []

    class Transform(ast.NodeTransformer):

        def visit(self, node):
            key = ast.unparse(node) if isinstance(node, (ast.BinOp, ast.Call)) else None
            if key in expressions:
                changes.append(key)
                return ast.copy_location(ast.parse(expressions[key], mode='eval').body, node)
            return super().visit(node)
    tree = Transform().visit(tree)
    require(len(changes) == 4 and set(changes) == set(expressions), 'FOUR_EXACT_NATIVE_EXPRESSIONS')
    scope = dict(function.__globals__, _sf_norm_mod=mod, _sf_gated=residual)
    exec(compile(ast.fix_missing_locations(tree), '<sf-nograd-bf16-fusion>', 'exec'), scope)
    return (scope[function.__name__], dict(native_source_sha256=expected[path.name], ast_sha256=hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(), expressions=changes))

class NoGradModulation:

    def __init__(self, trainer, *, ops=None, cpu_test_only=False):
        import torch
        from wan.modules.causal_model import CausalWanAttentionBlock
        from wan.modules.model import WanAttentionBlock
        self.originals = []
        self.tested = set()
        self.record = dict(mechanism='NATIVE_BF16_NOGRAD_MODULATION_RESIDUAL_FUSION', calls={}, grad_reference_calls={}, norm_mod_calls=0, gated_residual_calls=0, all_bf16_intermediate_rounding_preserved=True, grad_path_changed=False, contract_failures=0, canaries=[], source_proofs={})
        mod, gated = build(torch) if ops is None else ops

        def norm_op(x, norm, scale, bias):
            actual = mod(x, norm, scale, bias)
            self.record['norm_mod_calls'] += 1
            key = ('norm', tuple(x.shape), tuple(scale.shape), str(x.dtype))
            if not cpu_test_only and key not in self.tested:
                require(trainer.step < 5, 'UNQUALIFIED_NORM_GEOMETRY_AFTER_C1')
                n = norm(x).reshape(1, scale.numel() // x.shape[-1], -1, x.shape[-1])
                s = scale.reshape(1, scale.numel() // x.shape[-1], 1, x.shape[-1])
                b = bias.reshape_as(s)
                expected = (n * (1 + s) + b).reshape_as(x)
                torch.testing.assert_close(actual, expected)
                self.record['canaries'].append(dict(key=str(key), relative_l2=float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30))))
                self.tested.add(key)
            return actual

        def gated_op(x, y, gate):
            actual = gated(x, y, gate)
            self.record['gated_residual_calls'] += 1
            key = ('gated', tuple(x.shape), tuple(gate.shape), str(x.dtype))
            if not cpu_test_only and key not in self.tested:
                require(trainer.step < 5, 'UNQUALIFIED_GATED_GEOMETRY_AFTER_C1')
                frames = gate.numel() // x.shape[-1]
                expected = x + (y.reshape(1, frames, -1, x.shape[-1]) * gate.reshape(1, frames, 1, x.shape[-1])).reshape_as(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.record['canaries'].append(dict(key=str(key), bitwise=True))
                self.tested.add(key)
            return actual
        for role in ('generator', 'fake_score', 'real_score'):
            for module in getattr(trainer.model, role).modules():
                native = next((c for c in type(module).__mro__ if c in (CausalWanAttentionBlock, WanAttentionBlock)), None)
                if native is None:
                    continue
                require(not module.norm1.elementwise_affine and (not module.norm2.elementwise_affine), 'AFFINE_LN_UNQUALIFIED')
                original = module.forward
                function, proof = rewrite(native.forward, norm_op, gated_op)
                self.originals.append((module, original))
                self.record['source_proofs'][native.__name__] = proof

                def forward(this, *args, original=original, function=function, role=role, **kwargs):
                    name = 'grad_reference_calls' if torch.is_grad_enabled() else 'calls'
                    self.record[name][role] = self.record[name].get(role, 0) + 1
                    return original(*args, **kwargs) if torch.is_grad_enabled() else function(this, *args, **kwargs)
                module.forward = types.MethodType(forward, module)
        require(len(self.originals) == 100 if not cpu_test_only else len(self.originals) > 0, 'NATIVE_BLOCK_CENSUS')

    def receipt(self):
        return dict(self.record, blocks=len(self.originals), canary_keys=len(self.tested))

    def close(self):
        for module, original in self.originals:
            module.forward = original
