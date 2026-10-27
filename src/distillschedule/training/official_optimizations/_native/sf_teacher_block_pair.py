"""Measured implementation helper, relocated from r6_target2x_continuation/scripts/sf_teacher_block_pair.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap
import types

def require(ok, why):
    if not ok:
        raise ValueError('SF_TEACHER_BLOCK_PAIR:' + why)

def coroutine(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    fn = tree.body[0]
    require(isinstance(fn, ast.FunctionDef) and (not fn.decorator_list), 'NATIVE_FORWARD')
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and ast.unparse(n) == 'x = block(x, **kwargs)']
    require(len(calls) == 1, 'EXACT_NOGRAD_BLOCK_SITE')
    old = copy.deepcopy(calls[0].value)
    before = ast.dump(tree, include_attributes=False)
    calls[0].value = ast.Yield(value=ast.Tuple(elts=[ast.Name(id=k, ctx=ast.Load()) for k in ('block', 'x', 'kwargs')], ctx=ast.Load()))
    changed = copy.deepcopy(tree)
    calls[0].value = old
    require(ast.dump(tree, include_attributes=False) == before, 'NON_BLOCK_AST_CHANGED')
    scope = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(changed), inspect.getsourcefile(function), 'exec'), scope)
    return (scope[fn.name], dict(native_ast_sha256=hashlib.sha256(before.encode()).hexdigest(), coroutine_ast_sha256=hashlib.sha256(ast.dump(changed, include_attributes=False).encode()).hexdigest(), reversed_body_exact=True, edit='one non-grad block call replaced with yield(block,x,kwargs)'))

class TeacherBlockPair:

    def __init__(self, trainer, *, cpu_test_only=False):
        import torch
        from torch.distributed.fsdp import FSDPModule
        from wan.modules.model import WanModel
        require(hashlib.sha256(Path(inspect.getsourcefile(WanModel)).read_bytes()).hexdigest() == 'f15214485a5297e6f9e79684bbad80acbcdc88c3272596b256a739a734529449', 'PINNED_NATIVE_SOURCE')
        self.torch, self.cpu, self.trainer = (torch, cpu_test_only, trainer)
        self.teacher = trainer.model.real_score
        self.native = self.teacher.model
        require(isinstance(self.teacher, FSDPModule) and all((isinstance(b, FSDPModule) for b in self.native.blocks)), 'ACTUAL_FSDP2_REQUIRED')
        require(all((not p.requires_grad for p in self.teacher.parameters())), 'FROZEN_TEACHER_REQUIRED')
        require(self.native.model_type == 't2v' and (not self.native.gradient_checkpointing), 'TEACHER_NOGRAD_ONLY')
        require(cpu_test_only or (self.native.dim == 5120 and len(self.native.blocks) == 40), 'PINNED_TEACHER_GEOMETRY')
        require(not cpu_test_only or not torch.cuda.is_initialized(), 'CPU_ONLY')
        self.original_kl = trainer.model._compute_kl_grad
        self.original_forward = self.native._forward
        self.iterator, self.proof = coroutine(WanModel._forward)
        self.active = None
        self.pending = None
        self.held = None
        self.record = dict(mechanism='TEACHER_CFG_B1_BLOCK_INTERLEAVE_AND_PAIR_RETENTION', pairs=0, logical_teacher_forwards=0, block_calls=0, block_hold_entries=0, block_releases=0, staged_branch_outputs=0, consumed_branch_outputs=0, contract_failures=0, precision_changed=False, batch_changed=False, logical_calls_changed=False, full_teacher_retained=False, cpu_test_only=cpu_test_only, source_proof=self.proof)
        signature = inspect.signature(self.original_kl)

        def kl(_model, *args, **kwargs):
            require(self.active is None and self.pending is None, 'REENTRANCY')
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            require(not torch.is_grad_enabled(), 'FROZEN_DMD_TARGET')
            cond, uncond = (a['conditional_dict'], a['unconditional_dict'])
            require(set(cond) == set(uncond) == {'prompt_embeds'}, 'TEXT_ONLY_CONTRACT')
            require(cond['prompt_embeds'].shape == uncond['prompt_embeds'].shape and cond['prompt_embeds'].shape[0] == 1, 'CONDITION_GEOMETRY')
            self.active = dict(cond=cond['prompt_embeds'], uncond=uncond['prompt_embeds'], step=int(trainer.step), calls=0)
            try:
                result = self.original_kl(*args, **kwargs)
                require(self.active['calls'] == 2 and self.pending is None, 'INCOMPLETE_CFG_PAIR')
                self.record['pairs'] += 1
                return result
            finally:
                if self.held is not None:
                    self.held.set_reshard_after_forward(True, recurse=False)
                    self.held.reshard()
                    self.held = None
                self.active = None

        def forward(_native, *args, **kwargs):
            require(self.active is not None and (not torch.is_grad_enabled()), 'TEACHER_OUTSIDE_CFG_SCOPE')
            require(int(trainer.step) == self.active['step'], 'STEP_CHANGED')
            bound = inspect.signature(WanModel._forward).bind(self.native, *args, **kwargs)
            bound.apply_defaults()
            a = dict(bound.arguments)
            a.pop('self')
            require(not a['classify_mode'] and (not a['concat_time_embeddings']) and all((a[k] is None for k in ('register_tokens', 'cls_pred_branch', 'gan_ca_blocks', 'clip_fea', 'y'))), 'NATIVE_T2V_ONLY')
            require(a['x'].shape[0] == 1 and a['t'].shape == (1,), 'NATIVE_B1')
            slot = self.active['calls']
            require(slot in (0, 1), 'EXTRA_TEACHER_CALL')
            context = self.active['cond' if slot == 0 else 'uncond']
            require(a['context'].data_ptr() == context.data_ptr() and a['context'].shape == context.shape, 'CONTEXT_IDENTITY')
            self.active['calls'] += 1
            self.record['logical_teacher_forwards'] += 1
            if slot == 1:
                require(self.pending is not None, 'MISSING_UNCONDITIONAL_OUTPUT')
                for name in ('x', 't'):
                    v, p = (a[name], self.active[name])
                    require(v.data_ptr() == p.data_ptr() and v.shape == p.shape and (v.stride() == p.stride()) and (v.dtype == p.dtype), 'INPUT_IDENTITY_' + name)
                result = self.pending
                self.pending = None
                self.record['consumed_branch_outputs'] += 1
                return result
            self.active.update(x=a['x'], t=a['t'])
            b = dict(a, context=self.active['uncond'])
            left = self.iterator(self.native, **a)
            right = self.iterator(self.native, **b)
            ls, rs = (next(left), next(right))
            for index, block in enumerate(self.native.blocks):
                require(ls[0] is rs[0] is block, 'BLOCK_ORDER')
                require(ls[1].shape == rs[1].shape and ls[1].shape[0] == 1, 'BLOCK_B1_GEOMETRY')
                self.held = block
                block.set_reshard_after_forward(False, recurse=False)
                self.record['block_hold_entries'] += 1
                try:
                    lc = block(ls[1], **ls[2])
                    self.record['block_calls'] += 1
                    block.set_reshard_after_forward(True, recurse=False)
                    rc = block(rs[1], **rs[2])
                    self.record['block_calls'] += 1
                finally:
                    block.set_reshard_after_forward(True, recurse=False)
                    block.reshard()
                    self.held = None
                    self.record['block_releases'] += 1
                last = index == len(self.native.blocks) - 1
                out = []
                for iterator, value in ((left, lc), (right, rc)):
                    try:
                        nxt = iterator.send(value)
                        require(not last, 'EXTRA_BLOCK_YIELD')
                        out.append(nxt)
                    except StopIteration as e:
                        require(last, 'EARLY_BLOCK_EXIT')
                        out.append(e.value)
                del lc, rc
                if not last:
                    ls, rs = out
            self.pending = out[1]
            self.record['staged_branch_outputs'] += 1
            return out[0]
        self.installed_kl = types.MethodType(kl, trainer.model)
        self.installed_forward = types.MethodType(forward, self.native)
        trainer.model._compute_kl_grad = self.installed_kl
        self.native._forward = self.installed_forward

    def receipt(self):
        require(self.active is None and self.pending is None and (self.held is None), 'LIVE_STATE_FINAL')
        return dict(self.record, live_state=False)

    def close(self):
        self.receipt()
        require(self.trainer.model._compute_kl_grad is self.installed_kl and self.native._forward is self.installed_forward, 'PATCH_OWNERSHIP')
        self.trainer.model._compute_kl_grad = self.original_kl
        self.native._forward = self.original_forward
