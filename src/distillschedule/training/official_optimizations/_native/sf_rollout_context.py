"""Measured implementation helper, relocated from r2_fsdp2_baselines/scripts/sf_rollout_context.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
import hashlib
import inspect
from pathlib import Path
import types

def require(value, reason):
    if not value:
        raise ValueError('SF_ROLLOUT_CONTEXT:' + reason)

class RolloutContext:

    def __init__(self, trainer, *, cpu_test_only=False):
        import torch
        from wan.modules.model import WanT2VCrossAttention
        self.trainer = trainer
        self.cpu = cpu_test_only
        self.generator = trainer.model.generator
        self.native = self.generator.model
        self.active = False
        self.condition = None
        self.version = 0
        self.projected = None
        self.kv = {}
        self.checked = set()
        self.originals = []
        self.handles = []
        require(self.native.model_type == 't2v' and self.native.gradient_checkpointing is True, 'T2V_NATIVE_GRAD_PATH')
        require(hashlib.sha256(Path(inspect.getsourcefile(WanT2VCrossAttention)).read_bytes()).hexdigest() == 'f15214485a5297e6f9e79684bbad80acbcdc88c3272596b256a739a734529449', 'NATIVE_TEXT_KV_SOURCE')
        self.record = dict(mechanism='C_ROLLOUT_STATIC_TEXT_CONTEXT_HOISTING', epochs=0, releases=0, generator_commits=0, grad_simulation_bypasses=0, active_root_calls=0, text_builds=0, text_hits=0, kv_builds=0, kv_hits=0, canaries=0, gradient_path_changed=False, tensor_detaches=0, contract_failures=0)
        self.sim_original = trainer.model._consistency_backward_simulation

        def simulate(_model, *args, **kwargs):
            require(not self.active, 'NESTED_CONTEXT_SCOPE')
            if torch.is_grad_enabled():
                self.record['grad_simulation_bypasses'] += 1
                return self.sim_original(*args, **kwargs)
            self.active = True
            self.record['epochs'] += 1
            version = self.version
            try:
                result = self.sim_original(*args, **kwargs)
                require(version == self.version, 'WEIGHT_VERSION_CHANGED_IN_CONTEXT_SCOPE')
                return result
            finally:
                self.active = False
                self.projected = None
                self.kv.clear()
                self.condition = None
                self.record['releases'] += 1
        self.sim_installed = types.MethodType(simulate, trainer.model)
        trainer.model._consistency_backward_simulation = self.sim_installed

        def before(module, args, kwargs):
            if not self.active:
                return
            require(not torch.is_grad_enabled(), 'CONTEXT_SCOPE_GRADIENT')
            tensor = kwargs['conditional_dict']['prompt_embeds']
            key = (id(tensor), tensor._version, tuple(tensor.shape), tensor.dtype, tensor.device, self.version)
            if self.condition is None:
                self.condition = key
            else:
                require(self.condition == key, 'CONDITION_OR_WEIGHT_VERSION_CHANGED')
            self.record['active_root_calls'] += 1
        self.handles.append(self.generator.register_forward_pre_hook(before, with_kwargs=True, prepend=True))
        module = self.native.text_embedding
        original = module.forward
        self.originals.append((module, original))

        def text(this, x):
            if not self.active:
                return original(x)
            require(not torch.is_grad_enabled(), 'TEXT_CACHE_GRADIENT')
            if self.projected is None:
                self.projected = original(x)
                self.record['text_builds'] += 1
            else:
                self.record['text_hits'] += 1
                if not self.cpu and 'text' not in self.checked:
                    require(trainer.step < 5, 'TEXT_CANARY_AFTER_C1')
                    torch.testing.assert_close(self.projected, original(x), rtol=0, atol=0)
                    self.checked.add('text')
                    self.record['canaries'] += 1
            return self.projected
        module.forward = types.MethodType(text, module)
        for index, block in enumerate(self.native.blocks):
            module = block.cross_attn
            require(type(module) is WanT2VCrossAttention, 'NATIVE_CROSS_ATTENTION_REQUIRED')
            original_cross = module.forward
            self.originals.append((module, original_cross))

            def cross(this, x, context, context_lens, crossattn_cache=None, index=index, original_cross=original_cross):
                if not self.active:
                    return original_cross(x, context, context_lens, crossattn_cache=crossattn_cache)
                require(not torch.is_grad_enabled() and context_lens is None, 'NATIVE_STATIC_CONTEXT_ONLY')
                state = self.kv.setdefault(index, dict(is_init=False))
                if not state['is_init']:
                    self.record['kv_builds'] += 1
                else:
                    self.record['kv_hits'] += 1
                    if not self.cpu and index not in self.checked:
                        require(trainer.step < 5, 'KV_CANARY_AFTER_C1')
                        k = this.norm_k(this.k(context)).view(x.shape[0], -1, this.num_heads, this.head_dim)
                        v = this.v(context).view(x.shape[0], -1, this.num_heads, this.head_dim)
                        torch.testing.assert_close(state['k'], k, rtol=0, atol=0)
                        torch.testing.assert_close(state['v'], v, rtol=0, atol=0)
                        self.checked.add(index)
                        self.record['canaries'] += 1
                return original_cross(x, context, context_lens, crossattn_cache=state)
            module.forward = types.MethodType(cross, module)

        def pre(opt, args, kwargs):
            require(not self.active, 'G_COMMIT_IN_CACHE_SCOPE')

        def post(opt, args, kwargs):
            self.version += 1
            self.record['generator_commits'] += 1
        self.handles.extend([trainer.generator_optimizer.register_step_pre_hook(pre), trainer.generator_optimizer.register_step_post_hook(post)])

    def receipt(self):
        require(not self.active and self.projected is None and (not self.kv), 'LIVE_CONTEXT_AT_END')
        return dict(self.record, cache_live=False, generator_version=self.version)

    def close(self):
        require(not self.active, 'CLOSE_IN_CONTEXT_SCOPE')
        current = self.trainer.model._consistency_backward_simulation
        if current is self.sim_installed:
            self.trainer.model._consistency_backward_simulation = self.sim_original
        else:
            require(current.__func__ is getattr(type(self.trainer.model), '_consistency_backward_simulation'), 'PARENT_CLEANUP_CHANGED')
        for module, original in reversed(self.originals):
            module.forward = original
        for handle in self.handles:
            handle.remove()
