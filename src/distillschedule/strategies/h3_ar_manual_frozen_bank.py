"""Manual experiment primitive. No planner, training math or collective implementation."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import functools
import threading
import weakref

import torch
from torch import nn

_KEY=ContextVar('manual_frozen_adaln_bank_key',default=None)

@dataclass(frozen=True)
class Key:
    kind:str
    index:int
    cycle:int=0
    def __post_init__(self):
        if not(type(self.index)is int and type(self.cycle)is int and
            ((self.kind=='fixed'and 0<=self.index<8 and self.cycle==0)or
             (self.kind=='dynamic'and self.index in(0,1)and self.cycle>=1))):
            raise ValueError('EXPLICIT_ADMITTED_KEY_REQUIRED')

@contextmanager
def key_scope(key):
    if not isinstance(key,Key):raise ValueError('MISSING_OR_INCOMPLETE_CAPTURED_KEY')
    token=_KEY.set(key)
    try:yield
    finally:_KEY.reset(token)

def current_key():return _KEY.get()

class Proxy(nn.Module):
    def __init__(self,owner,index):
        super().__init__();object.__setattr__(self,'_owner',weakref.ref(owner));self.index=index
    def forward(self,x):
        owner=self._owner()
        if owner is None:raise RuntimeError('BANK_OWNER_GONE')
        return owner.lookup(self.index,x)

class FrozenAdaLNBank:
    """Relocate existing modules; cache only frozen values, never merge weights."""
    def __init__(self,model,*,expected_blocks,hidden_size,input_size):
        if hasattr(model,'_manual_frozen_adaln_bank'):raise ValueError('BANK_ALREADY_PRESENT')
        blocks=list(model.blocks)
        if len(blocks)!=expected_blocks:raise ValueError('BLOCK_COUNT_CHANGED')
        self.hidden_size=hidden_size;self.input_size=input_size
        originals=[block.adaln_proj for block in blocks]
        if len({id(x)for x in originals})!=len(originals):raise ValueError('SHARED_PROJECTION_NOT_ADMITTED')
        for projection in originals:
            if(type(projection).__name__!='MiniMaxH3AdalnProj'or
               (projection.expand_ratio,projection.modality_num,projection.hidden_size)!=(6,3,hidden_size)or
               (projection.linear.in_features,projection.linear.out_features)!=(input_size,18*hidden_size)or
               any(p.requires_grad for p in projection.parameters())):
                raise ValueError('REQUIRES_ORIGINAL_FROZEN_PROJECTION')
        before={id(p):p for p in model.parameters()}
        old_names={id(p):name for name,p in model.named_parameters()}
        bank=nn.ModuleList(originals)
        for i,block in enumerate(blocks):block.adaln_proj=Proxy(self,i)
        model.add_module('_manual_frozen_adaln_bank',bank)
        after={id(p):p for p in model.parameters()}
        if set(before)!=set(after):raise RuntimeError('REGISTERED_PARAMETERS_LOST_OR_COPIED')
        if any(list(block.adaln_proj.parameters())for block in blocks):raise RuntimeError('PROXY_REGISTERS_PARAMETERS')
        self._bank=weakref.ref(bank);self._model=weakref.ref(model)
        self.original_canonical_names=old_names
        self.frozen_ids={id(p)for projection in bank for p in projection.parameters()}
        self.cache={};self.pending_equal=[];self.counts=Counter();self._versions=None;self._cp_active=False
        self.scope='SAME_PARAMETERS_REGISTERED_TOP_LEVEL_BANK_NOT_ELIMINATED'
    def bind_frozen_versions(self):
        self._versions={id(p):(id(getattr(p,'ds_tensor',p)),int(getattr(p,'ds_tensor',p)._version))for projection in self._bank()for p in projection.parameters()}
        if set(self._versions)!=self.frozen_ids:raise RuntimeError('FROZEN_IDENTITY_CHANGED')
    def _check_frozen(self):
        if self._versions is None:raise RuntimeError('BIND_AFTER_ENGINE_SETUP_REQUIRED')
        for projection in self._bank():
            for p in projection.parameters():
                value=getattr(p,'ds_tensor',p)
                if p.requires_grad or self._versions.get(id(p))!=(id(value),int(value._version)):
                    raise RuntimeError('FROZEN_PARAMETER_OR_PARTITION_CHANGED')
    def lookup(self,index,x):
        key=current_key()
        if not isinstance(key,Key):raise RuntimeError('MISSING_CAPTURED_KEY_NO_FALLBACK')
        if x.ndim!=2 or x.shape[0]not in(1,2)or x.shape[1]!=self.input_size or x.requires_grad:
            raise ValueError('NATIVE_FROZEN_TIMESTEP_INPUT_CHANGED')
        self._check_frozen();slot=(index,key)
        self.counts['proxy_calls']+=1
        if slot in self.cache:
            old,outputs=self.cache[slot]
            if x.shape!=old.shape or x.dtype!=old.dtype or x.device!=old.device:raise ValueError('KEY_INPUT_METADATA_MISMATCH')
            # No host-visible GPU read in the hot path. Validation after fence.
            self.pending_equal.append(torch.eq(x,old).all())
            self.counts['cache_reads']+=1
            return outputs
        if len(self.cache)>=len(self._bank())*10:raise RuntimeError('BOUNDED_CACHE_EXCEEDED')
        outputs=self._bank()[index](x)
        if not(isinstance(outputs,tuple)and len(outputs)==6 and all(not v.requires_grad and v.dtype==x.dtype for v in outputs)):
            raise RuntimeError('ORIGINAL_FROZEN_OUTPUT_CONTRACT_CHANGED')
        # Tiny timestep copy, not weights or hidden activations; output views
        # retain the SAME original BF16 projection storage, no data conversion.
        self.cache[slot]=(x.detach().clone(),outputs)
        self.counts['bank_forward_calls']+=1
        self.counts['cache_writes']+=1
        return outputs
    @contextmanager
    def capture_checkpoints(self):
        """Wrap stored block function; preserve actual module and CP kwargs."""
        import torch.utils.checkpoint as cp
        if self._cp_active:raise RuntimeError('NESTED_CHECKPOINT_PATCH_SCOPE')
        if current_key()is None:raise RuntimeError('ROOT_KEY_REQUIRED')
        original=cp.checkpoint;owner_thread=threading.get_ident();self._cp_active=True
        def wrapped(function,*args,**kwargs):
            if threading.get_ident()!=owner_thread:raise RuntimeError('CONCURRENT_CHECKPOINT_CALL_NOT_ADMITTED')
            key=current_key()
            if not isinstance(key,Key)or kwargs.get('use_reentrant')is not False:raise ValueError('NONREENTRANT_CAPTURED_KEY_REQUIRED')
            self.counts['checkpoint_wraps']+=1
            @functools.wraps(function)
            def scoped(*a,**kw):
                self.counts['scoped_execution_entry']+=1
                try:
                    with key_scope(key):return function(*a,**kw)
                finally:self.counts['scoped_execution_finally']+=1
            return original(scoped,*args,**kwargs)
        cp.checkpoint=wrapped
        try:yield
        finally:
            owned=cp.checkpoint is wrapped;cp.checkpoint=original;self._cp_active=False
            if not owned:raise RuntimeError('CHECKPOINT_PATCH_OWNERSHIP_CHANGED')
    def finish_cycle(self,*,synchronized):
        if synchronized is not True or self._cp_active or current_key()is not None:raise RuntimeError('ORIGINAL_FENCED_BOUNDARY_REQUIRED')
        if self.pending_equal and not bool(torch.stack(self.pending_equal).all().item()):
            raise RuntimeError('CACHE_KEY_INPUT_VALUES_DIFFER_NO_TIMING_ADMISSION')
        self.pending_equal.clear();self._check_frozen()
        static=sum(key.kind=='fixed'for _,key in self.cache)
        dynamic=sum(key.kind=='dynamic'for _,key in self.cache)
        for slot in list(self.cache):
            if slot[1].kind=='dynamic':del self.cache[slot]
        return dict(counts=dict(self.counts),static_entries=static,dynamic_entries_released=dynamic,
                    registered_frozen_parameters_retained=True,weights_merged=False,
                    values_equal=True,host_value_reads='ONLY_FINISH_AFTER_ORIGINAL_FENCE')


POLICY_ID = 'exact-frozen-adaln-bank-v1'
PROJECTION_NUMEL = 260209152
ORIGINAL_LEAF_NUMEL = 657499392
BANK_LEAF_NUMEL = ORIGINAL_LEAF_NUMEL - PROJECTION_NUMEL


def validate_manual_policy(policy_id, *, mode, implementation, plan_path):
    """Explicit manual-only opt-in. Never admit calibration or saved-plan use."""
    if policy_id is None:
        return
    if (policy_id not in ('off', POLICY_ID) or mode != 'manual' or plan_path is not None
            or getattr(implementation, 'id', None) != 'zero3-ap'):
        raise ValueError('MANUAL_ONLY_ZERO3_AP_NO_SAVED_PLAN_REQUIRED')


def tensor_identity(x):
    """Host metadata only. Strong references prevent allocator-address reuse."""
    if x.ndim != 0 or x.requires_grad:
        raise ValueError('ORIGINAL_SCALAR_FROZEN_SIGMA_REQUIRED')
    return (str(x.device), str(x.dtype), x.data_ptr(), tuple(x.shape),
            tuple(x.stride()), x.storage_offset(), int(x._version))


class NativeSigmaKeys:
    def __init__(self, schedule):
        if len(schedule) != 2 or any(x.ndim != 1 or len(x) != 9 for x in schedule):
            raise ValueError('ORIGINAL_EIGHT_STEP_SCHEDULE_REQUIRED')
        self.schedule = tuple(schedule)
        self.schedule_identity = tuple((id(x), int(x._version)) for x in schedule)
        self.fixed = [tuple(s[i] for s in schedule) for i in range(8)]
        self.fixed_map = {tuple(tensor_identity(x) for x in pair): Key('fixed', i)
                          for i, pair in enumerate(self.fixed)}
        if len(self.fixed_map) != 8:
            raise ValueError('UNIQUE_ORIGINAL_FIXED_SIGMA_STORAGE_REQUIRED')
        self.dynamic = []
        self.cycle = 0
        self.active = False
        self.calls = 0
        self.fixed_seen = []

    def begin(self, cycle):
        if self.active or self.dynamic or cycle != self.cycle + 1 or cycle not in (1, 2):
            raise RuntimeError('NATIVE_TWO_CONTIGUOUS_CYCLES_REQUIRED')
        self.check_schedule()
        self.cycle, self.active, self.calls, self.fixed_seen = cycle, True, 0, []

    def check_schedule(self):
        if tuple((id(x), int(x._version)) for x in self.schedule) != self.schedule_identity:
            raise RuntimeError('ORIGINAL_SCHEDULE_MUTATED')

    def resolve(self, pair):
        if not self.active or len(pair) != 2:
            raise RuntimeError('ACTIVE_CYCLE_AND_NATIVE_SIGMA_PAIR_REQUIRED')
        self.check_schedule()
        identity = tuple(tensor_identity(x) for x in pair)
        self.calls += 1
        if identity in self.fixed_map:
            key = self.fixed_map[identity]
            self.fixed_seen.append(key.index)
            return key
        for old_pair, old_identity, key in self.dynamic:
            if tuple(tensor_identity(x) for x in old_pair) != old_identity:
                raise RuntimeError('HELD_DYNAMIC_SIGMA_MUTATED')
            if identity == old_identity:
                return key
        if len(self.dynamic) >= 2:
            raise RuntimeError('MORE_THAN_TWO_ORIGINAL_CONTINUOUS_DRAWS')
        key = Key('dynamic', len(self.dynamic), self.cycle)
        self.dynamic.append((tuple(pair), identity, key))
        return key

    def finish(self):
        self.check_schedule()
        if (not self.active or self.calls != 12 or len(self.dynamic) != 2
                or self.fixed_seen[:8] != list(range(8)) or len(self.fixed_seen) != 9):
            raise RuntimeError('ORIGINAL_8_ROLLOUT_C1_T_F_G1_KEY_CENSUS_REQUIRED')
        result = dict(application_calls=self.calls, fixed_calls=len(self.fixed_seen),
                      dynamic_draws=len(self.dynamic), key_kind='EXACT_ORIGINAL_SIGMA_STORAGE_NOT_GRID')
        self.dynamic.clear()
        self.active = False
        return result


class ManualFrozenAdaLNPolicy:
    """Experimental collaborator for the unchanged native W6 Session only."""
    id = POLICY_ID
    leaf_numel = BANK_LEAF_NUMEL

    def install_model(self, model):
        self.bank = FrozenAdaLNBank(model, expected_blocks=50, hidden_size=5376, input_size=2688)
        total = sum(getattr(p, 'ds_numel', p.numel()) for p in model.parameters()
                    if id(p) in self.bank.frozen_ids)
        if total != 50 * PROJECTION_NUMEL:
            raise RuntimeError('ALL_ORIGINAL_FROZEN_ADALN_PARAMETERS_REQUIRED')
        self.registered_frozen_numel = total
        self.previous_counts = Counter()

    def bind_engine(self, engine):
        registered = {id(p) for p in engine.module.parameters()}
        if not self.bank.frozen_ids <= registered:
            raise RuntimeError('ENGINE_DROPPED_ORIGINAL_FROZEN_PARAMETERS')
        if any(getattr(p, 'ds_z3_leaf_module', None) is not None
               for projection in self.bank._bank() for p in projection.parameters()):
            raise RuntimeError('BANK_PARAMETERS_STILL_IN_BLOCK_LEAF')
        self.bank.bind_frozen_versions()

    def attach(self, adapter, trainer):
        self.keys = NativeSigmaKeys(trainer.schedule)
        original = adapter.predict
        def predict(latents, sigmas, inputs, checkpoint=False):
            key = self.keys.resolve(sigmas)
            with key_scope(key), self.bank.capture_checkpoints():
                return original(latents, sigmas, inputs, checkpoint)
        adapter.predict = predict

    def begin_cycle(self, cycle):
        self.keys.begin(cycle)

    def finish_cycle(self, cycle):
        key_receipt = self.keys.finish()
        receipt = self.bank.finish_cycle(synchronized=True)
        counts = Counter(receipt['counts'])
        delta = counts - self.previous_counts
        expected = dict(proxy_calls=700, checkpoint_wraps=100,
                        scoped_execution_entry=200, scoped_execution_finally=200,
                        bank_forward_calls=500 if cycle == 1 else 100,
                        cache_writes=500 if cycle == 1 else 100,
                        cache_reads=200 if cycle == 1 else 600)
        if (dict(delta) != expected or receipt['static_entries'] != 400
                or receipt['dynamic_entries_released'] != 100):
            raise RuntimeError('FAIL_EXACT_NATIVE_ADALN_BANK_CENSUS:' + repr(dict(delta)))
        self.previous_counts = counts
        return dict(policy_id=self.id, exact_input_gate=receipt, per_cycle_counts=dict(delta),
                    key_resolution=key_receipt, registered_frozen_numel=self.registered_frozen_numel,
                    parameters_eliminated=False, weights_merged=False, automatic_planning_eligible=False)
