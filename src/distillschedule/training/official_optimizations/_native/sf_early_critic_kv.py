"""Measured implementation helper, relocated from r2_fsdp2_baselines/scripts/sf_early_critic_kv.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
import types
from .sf_preencoded_lifecycle import DeadKVController, storage_info, tensors

def require(value, reason):
    if not value:
        raise ValueError('SF_EARLY_CRITIC_KV:' + reason)

class EarlyCriticKV(DeadKVController):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.early = None
        self.record = dict(early_critic_releases=0, original_generator_releases=0, critic_entries_with_dead_kv_absent=0, returned_tensor_aliases=0, tensor_detaches=0, contract_failures=0)
        self.original_simulation = None
        self.installed_simulation = None
        self.handle = None

    def release_no_grad_critic_kv(self, result):
        require(not self.torch.is_grad_enabled(), 'NO_GRAD_SIMULATION_REQUIRED')
        require(self.pending == (self.trainer.step, 'C') and (not self.backward_complete), 'WRONG_ROLE_VERSION_OR_PHASE')
        require(self.early is None, 'DUPLICATE_EARLY_RELEASE')
        pipeline = self.trainer.model.inference_pipeline
        cache = pipeline.kv_cache1
        require(cache is not None and pipeline.crossattn_cache is not None, 'MISSING_NATIVE_KV')
        owned = {}
        for tensor in tensors([pipeline.kv_cache1, pipeline.kv_cache2, pipeline.crossattn_cache]):
            info = storage_info(tensor)
            if info:
                owned[info[0]] = info[1]['storage_bytes']
        for tensor in tensors(result):
            require(not tensor.requires_grad and tensor.grad_fn is None, 'RETURNED_AUTOGRAD_GRAPH')
            info = storage_info(tensor)
            require(info is None or info[0] not in owned, 'RETURNED_KV_STORAGE_ALIAS')
        row = dict(step=int(self.trainer.step), role='C', blocks=len(cache), key_shapes=[list(item['k'].shape) for item in cache], key_dtypes=sorted({str(item['k'].dtype) for item in cache}), global_end_indices=[item['global_end_index'].detach() for item in cache], local_end_indices=[item['local_end_index'].detach() for item in cache], released_owner_storage_bytes=sum(owned.values()), release_phase='AFTER_NOGRAD_ROLLOUT_BEFORE_SCORE')
        self.audit.record('C.rollout.end.before_KV_release')
        pipeline.kv_cache1 = pipeline.kv_cache2 = pipeline.crossattn_cache = None
        self.early = row
        self.record['early_critic_releases'] += 1
        self.audit.record('C.rollout.end.after_KV_release')

    def install(self):
        super().install()
        self.original_simulation = self.trainer.model._consistency_backward_simulation

        def simulate(_model, *args, **kwargs):
            result = self.original_simulation(*args, **kwargs)
            if not self.torch.is_grad_enabled():
                self.release_no_grad_critic_kv(result)
            return result
        self.installed_simulation = types.MethodType(simulate, self.trainer.model)
        self.trainer.model._consistency_backward_simulation = self.installed_simulation

        def critic_entry(module, args, kwargs):
            if self.pending != (self.trainer.step, 'C'):
                return
            require(self.early is not None and self._no_live_kv(), 'CRITIC_STARTED_BEFORE_EARLY_RELEASE')
            self.record['critic_entries_with_dead_kv_absent'] += 1
        self.handle = self.trainer.model.fake_score.register_forward_pre_hook(critic_entry, with_kwargs=True, prepend=True)

    def _no_live_kv(self):
        pipeline = self.trainer.model.inference_pipeline
        return all((getattr(pipeline, name) is None for name in ('kv_cache1', 'kv_cache2', 'crossattn_cache')))

    def release_after_commit(self, role):
        if role == 'G':
            require(self.early is None, 'CRITIC_STATE_IN_GENERATOR_COMMIT')
            super().release_after_commit(role)
            self.releases[-1]['release_phase'] = 'AFTER_GENERATOR_BACKWARD_AND_COMMIT'
            self.record['original_generator_releases'] += 1
            return
        require(role == 'C' and self.pending == (self.trainer.step, 'C') and self.backward_complete and (self.early is not None), 'CRITIC_COMMIT_WITHOUT_MATCHED_BACKWARD_AND_RELEASE')
        require(self.early['step'] == int(self.trainer.step) and self._no_live_kv(), 'KV_REBUILT_OR_WRONG_VERSION_BEFORE_COMMIT')
        self.releases.append(self.early)
        self.early = None
        self.pending = None
        self.backward_complete = False

    def summary(self):
        require(self.early is None, 'UNCOMMITTED_EARLY_RELEASE')
        result = super().summary()
        result.update(mechanism='C_KV_RELEASE_AFTER_NOGRAD_ROLLOUT_G_AFTER_COMMIT', early_critic_release=dict(self.record), generator_autograd_lifetime_changed=False)
        return result

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        current = self.trainer.model._consistency_backward_simulation
        if current is self.installed_simulation:
            self.trainer.model._consistency_backward_simulation = self.original_simulation
