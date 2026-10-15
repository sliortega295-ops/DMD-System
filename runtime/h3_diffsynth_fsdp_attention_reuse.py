"""Opt-in native attention return reuse with FSDP checkpoint contexts.

The exact, frozen A100 cache classes are reused. This adapter replaces their
ZeRO-leaf-specific installer, not their tensor transport or attention backward.
Each logical B1 slot has a separate pool and version view. No cached DiT output,
loss, sigma, model forward or native backward graph is removed.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import functools
import hashlib
from pathlib import Path

ACTIVE = ContextVar('h100_w6_selected_attention_slot', default=None)
CALL = ContextVar('h100_w6_selected_attention_call', default=None)

def _sha(module):
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


class FSDPAttentionReuse:
    def __init__(self, dmd, native, adapter, trainers, slots, *, enabled=False,
                 gpu_tail_blocks=0, gpu_budget_bytes=24 * (1 << 30)):
        self.enabled = bool(enabled); self.pools = {}; self.views = {}
        self.gpu_budget = None
        if gpu_tail_blocks < 0 or (gpu_tail_blocks and not enabled):
            raise ValueError('GPU cache residency requires the enabled native attention cache')
        if not enabled: return
        import torch
        import native_selected_h2d_prefetch_r1 as prefetch
        import selected_attention_pool_r1 as generic
        from native_selected_rollout_reuse_r1 import token
        from native_selected_cycle_scope_r1 import RelativeVersionView
        if _sha(dmd) != '77744f5ec391be813d42ebaf6a2692c18ea58f5ca7fb41398fd287d2a177ca62':
            raise ValueError('native DMD source mismatch')
        if _sha(native) != 'bd617a4312c70405e916707e1908100b484b2259dfbeba4af7de5bb145206d80':
            raise ValueError('native DiT source mismatch')
        if _sha(prefetch) != '53a6e6962005981d22ed35b4ef6c33ff819f1bdb22fd7c1c9ae4007de39bc68a' or _sha(generic) != '20bed542097baf40a97cd80e45c11af26fd25d5b18f76dcbb0183c9a09c2e728':
            raise ValueError('exact attention cache helper source mismatch')
        if len(slots) != len(trainers) or len(set(slots)) != len(slots):
            raise ValueError('independent logical pool identities required')
        blocks = {id(b): i for i,b in enumerate(adapter.dit.blocks)}
        if len(blocks) not in (2,50): raise ValueError('native tiny2 or real50 blocks required')
        pool_type = prefetch.PrefetchPool
        if gpu_tail_blocks:
            if gpu_tail_blocks > len(blocks): raise ValueError('GPU cache tail exceeds model blocks')
            from runtime.h3_diffsynth_attention_residency import GPUCacheBudget, resident_pool_type
            self.gpu_budget = GPUCacheBudget(gpu_budget_bytes)
            pool_type = resident_pool_type(prefetch, tail_blocks=gpu_tail_blocks, budget=self.gpu_budget)
        for slot,trainer in zip(slots,trainers):
            pool = pool_type(enabled=True); pool.blocks = blocks
            self.pools[slot] = pool; self.views[slot] = RelativeVersionView(trainer)
        original_checkpoint = native.gradient_checkpoint_forward
        original_predict = adapter.predict
        original_rollout = dmd.rollout

        def current():
            slot = ACTIVE.get()
            if slot not in self.pools: raise RuntimeError('explicit logical B1 cache scope missing')
            return self.pools[slot],self.views[slot]

        @functools.wraps(original_checkpoint)
        def checkpoint(model, enabled, offload, *args, **kwargs):
            pool,_=current(); index=pool.blocks.get(id(model)); kind=CALL.get()
            if index is None or kind is None:
                return original_checkpoint(model,enabled,offload,*args,**kwargs)
            if offload: raise RuntimeError('boundary offload is not this attention cache policy')
            if kind=='SELECTED_NG':
                if enabled or torch.is_grad_enabled():raise RuntimeError('selected source must be native no-grad')
                with pool.capture(('studentG0',index),uses=2):
                    return original_checkpoint(model,enabled,offload,*args,**kwargs)
            if not enabled or not torch.is_grad_enabled():raise RuntimeError('native grad checkpoint required')
            if kind=='STUDENT_GRAD':
                contexts=pool.contexts_from_source(('studentG0',index))
            elif kind=='FAKE_GRAD':
                contexts=pool.fake_contexts(('fakeG0',index),lambda:(nullcontext(),nullcontext()))
            else:raise RuntimeError('unsupported native checkpoint call')
            # Same native non-reentrant custom forward, args, kwargs and default
            # RNG preservation. Only the explicit cache contexts are added.
            from diffsynth.core.gradient.gradient_checkpoint import create_custom_forward
            return torch.utils.checkpoint.checkpoint(create_custom_forward(model),*args,**kwargs,
                    use_reentrant=False,context_fn=lambda:contexts)
        native.gradient_checkpoint_forward=checkpoint

        @functools.wraps(original_predict)
        def predict(latents,sigmas,inputs,checkpoint=False):
            pool,view=current(); roles={layer.role for _,layer in adapter.layers}
            if len(roles)!=1:raise RuntimeError('uniform native adapter role required')
            role=next(iter(roles));kind=None;plan=pool.rollout_plan
            if plan is not None:
                if role!='student' or torch.is_grad_enabled() or checkpoint or inputs is not plan['inputs']:
                    raise RuntimeError('native rollout identity changed')
                if view.generator_step!=0 or view.critic_step!=0:raise RuntimeError('rollout version changed')
                index=plan['calls'];plan['calls']+=1;pool.routing['rollout_predictions']+=1
                if index==plan['step']:
                    if pool.selected is not None:raise RuntimeError('duplicate selected source')
                    pool.selected={'step':index,'sigmas':tuple(token(x) for x in sigmas),'inputs':inputs,
                                   'inputs_versions':{k:token(v) for k,v in inputs.items() if isinstance(v,torch.Tensor)}}
                    pool.routing['rollout_selected_sources']+=1;kind='SELECTED_NG'
            elif torch.is_grad_enabled():
                if not checkpoint:raise RuntimeError('native grad checkpoint disabled')
                if role=='fake_score':
                    if view.generator_step!=view.critic_step or view.generator_step!=0:raise RuntimeError('Fake source version')
                    pool.routing['fake_grad_predictions']+=1;kind='FAKE_GRAD'
                else:
                    if role!='student' or view.generator_step!=0 or view.critic_step!=1:raise RuntimeError('Student consumer version')
                    selected=pool.selected
                    if not selected or tuple(token(x) for x in latents)!=selected['selected_latents'] or tuple(token(x) for x in sigmas)!=selected['sigmas']:
                        raise RuntimeError('selected latent/timestep identity mismatch')
                    if inputs is not selected['inputs'] or {k:token(v) for k,v in inputs.items() if isinstance(v,torch.Tensor)}!=selected['inputs_versions']:
                        raise RuntimeError('selected condition identity mismatch')
                    pool.routing['student_grad_predictions']+=1;kind='STUDENT_GRAD'
            key=CALL.set(kind)
            try:return original_predict(latents,sigmas,inputs,checkpoint)
            finally:CALL.reset(key)
        adapter.predict=predict

        @functools.wraps(original_rollout)
        def rollout(a,real,inputs,schedule,selected_step=None):
            pool,view=current()
            if a is not adapter or pool.rollout_plan is not None or selected_step is None or len(schedule[0])!=9 or view.config.critic_updates!=1:
                raise RuntimeError('native rollout scope/schedule mismatch')
            pool.rollout_plan={'step':selected_step,'inputs':inputs,'calls':0}
            try:
                result=original_rollout(a,real,inputs,schedule,selected_step)
                if pool.rollout_plan['calls']!=8 or pool.selected is None:raise RuntimeError('native rollout census')
                pool.selected['selected_latents']=tuple(token(x) for x in result[1]);return result
            finally:pool.rollout_plan=None
        dmd.rollout=rollout

    @contextmanager
    def scope(self,slot):
        if not self.enabled:
            yield;return
        if ACTIVE.get() is not None or slot not in self.pools:raise RuntimeError('cache scope nesting or identity')
        key=ACTIVE.set(slot)
        try:yield
        finally:ACTIVE.reset(key)

    def finish_cycle(self):
        if not self.enabled:return {'enabled':False,'status':'DEFAULT_OFF_NO_PATCH'}
        from native_selected_cycle_scope_r1 import finish_cycle
        result = {'enabled':True,'backend':'FSDP2_EXPLICIT_NONREENTRANT_CONTEXT_NOT_ZERO3_LEAF',
                'logical_slots':{str(slot):finish_cycle(pool,self.views[slot]) for slot,pool in self.pools.items()},
                'physical_DiT_not_skipped':True,'native_backward_unchanged':True}
        if self.gpu_budget is not None and self.gpu_budget.owned_bytes:
            raise RuntimeError('GPU cache source storage survived complete epoch')
        return result

    def begin_next_cycle(self):
        if not self.enabled:return
        from native_selected_cycle_scope_r1 import begin_next_cycle
        for slot,pool in self.pools.items():
            begin_next_cycle(pool,self.views[slot])
            if hasattr(pool,'reset_residency_epoch'):pool.reset_residency_epoch()
