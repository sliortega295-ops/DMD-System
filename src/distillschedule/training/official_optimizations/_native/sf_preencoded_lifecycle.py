"""Measured implementation helper, relocated from r1/scripts/sf_preencoded_lifecycle.py.

The public owner validates capabilities and lifecycle; this module does not
launch experiments or claim framework GPU acceptance. Source lock records all
original and relocated executable definitions.
"""
def require(ok, why):
    if not ok:
        raise ValueError('Self-Forcing cache: ' + why)
from contextlib import nullcontext

def local_tensor(t):
    from torch.distributed.tensor import DTensor
    return t.to_local() if isinstance(t, DTensor) else t

def storage_info(t):
    """Metadata only; neither tensor values nor CUDA synchronization."""
    t = local_tensor(t)
    if t.device.type == 'meta':
        return None
    s = t.untyped_storage()
    size = s.nbytes()
    if not size:
        return None
    key = (str(t.device), s.data_ptr(), size)
    return (key, dict(device=str(t.device), address=s.data_ptr(), storage_bytes=size, shape=list(t.shape), dtype=str(t.dtype), logical_bytes=t.numel() * t.element_size()))

def tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)

class DeadKVController:

    def __init__(self, trainer, torch, observer, audit):
        self.trainer, self.torch, self.observer, self.audit = (trainer, torch, observer, audit)
        self.pending = None
        self.backward_complete = False
        self.releases = []

    def release_after_commit(self, role):
        require(self.pending == (self.trainer.step, role) and self.backward_complete, 'KV_RELEASE_BEFORE_COMPLETE_BACKWARD_OR_WRONG_ROLE')
        p = self.trainer.model.inference_pipeline
        cache = p.kv_cache1
        require(cache is not None and p.crossattn_cache is not None, 'MISSING_LIVE_NATIVE_KV')
        row = dict(step=int(self.trainer.step), role=role, blocks=len(cache), key_shapes=[list(x['k'].shape) for x in cache], key_dtypes=sorted({str(x['k'].dtype) for x in cache}), global_end_indices=[x['global_end_index'].detach() for x in cache], local_end_indices=[x['local_end_index'].detach() for x in cache])
        unique = {}
        for tensor in tensors([p.kv_cache1, p.kv_cache2, p.crossattn_cache]):
            info = storage_info(tensor)
            if info:
                unique[info[0]] = info[1]['storage_bytes']
        del tensor
        row['released_owner_storage_bytes'] = sum(unique.values())
        self.audit.record(role + '.commit.before_KV_release')
        p.kv_cache1 = None
        p.kv_cache2 = None
        p.crossattn_cache = None
        del cache
        self.releases.append(row)
        self.pending = None
        self.backward_complete = False
        self.audit.record(role + '.commit.after_KV_release')

    def install(self):
        trainer = self.trainer
        original = trainer.fwdbwd_one_step

        def fwdbwd(batch, train_generator):
            role = 'G' if train_generator else 'C'
            require(self.pending is None, 'PREVIOUS_PHASE_KV_NOT_RELEASED')
            self.pending = (trainer.step, role)
            self.backward_complete = False
            self.audit.record(role + '.fwdbwd.begin')
            context = self.torch.autograd.graph.saved_tensors_hooks(self.audit.tracker.pack, self.audit.tracker.unpack) if self.audit.enabled else nullcontext()
            with context:
                result = original(batch, train_generator)
            self.backward_complete = True
            self.audit.record(role + '.backward.end')
            return result
        trainer.fwdbwd_one_step = fwdbwd
        for method, role in [('generator_loss', 'G'), ('critic_loss', 'C')]:
            original_loss = getattr(trainer.model, method)

            def loss(*a, _original=original_loss, _role=role, **kw):
                result = _original(*a, **kw)
                self.audit.record(_role + '.forward.end_before_backward')
                if trainer.step == 0 and self.audit.enabled:
                    self.audit.snapshot('allocator-' + _role + '-forward-end')
                return result
            setattr(trainer.model, method, loss)
        for name, role in [('generator_optimizer', 'G'), ('critic_optimizer', 'C')]:
            optimizer = getattr(trainer, name)
            original_step = optimizer.step

            def step(*a, _original=original_step, _role=role, **kw):
                self.audit.record(_role + '.optimizer.begin')
                result = _original(*a, **kw)
                self.release_after_commit(_role)
                return result
            optimizer.step = step

    def summary(self):
        rows = []
        for r in self.releases:
            out = dict(r)
            for name in ('global_end_indices', 'local_end_indices'):
                out[name] = [int(x.item()) for x in r[name]]
            rows.append(out)
        return dict(mechanism='DROP_DEAD_NATIVE_SELF_AND_CROSS_KV_AFTER_BACKWARD_AND_COMMIT', releases=rows, release_calls=len(rows), pending=self.pending, live_cache_at_end=any((getattr(self.trainer.model.inference_pipeline, n, None) is not None for n in ('kv_cache1', 'kv_cache2', 'crossattn_cache'))))
