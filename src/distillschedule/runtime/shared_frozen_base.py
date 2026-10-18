"""Explicit setup-only storage sharing for two equal frozen Resident bases.

Not an adapter merge, model swap, sharding policy, optimizer or planner. Callers
must supply equal checkpoint identities and audit the returned handle outside
the performance window. No caller or canonical default is changed by import.
"""
from dataclasses import dataclass
from typing import Any

def canonical_name(name: str) -> str:
    return name.replace('.base_layer.', '.')

def _storage(t):
    s=t.untyped_storage()
    return (str(t.device),int(s.data_ptr()),int(s.nbytes()))

def physical_parameter_bytes(*models):
    return sum(k[2]for k in {_storage(p)for m in models for p in m.parameters()if p.numel()})

def _indexed(model):
    out={}
    for name,p in model.named_parameters():
        key=canonical_name(name)
        if key in out:raise ValueError('Ambiguous canonical parameter name: '+key)
        out[key]=(name,p)
    return out

def _check_unmerged(model):
    if any(bool(getattr(m,'merged',False))for m in model.modules()):
        raise ValueError('Merged adapters are outside frozen-sharing contract')

@dataclass
class FrozenAliasAudit:
    reference: Any
    candidate: Any
    pairs: tuple
    receipt: dict

    def audit(self):
        _check_unmerged(self.reference);_check_unmerged(self.candidate)
        for name,left,right,left_version,right_version,left_id,right_id,pointer in self.pairs:
            if left.requires_grad or right.requires_grad or id(left)!=left_id or id(right)!=right_id:
                raise RuntimeError('Frozen parameter identity/grad changed: '+name)
            if left._version!=left_version or right._version!=right_version:
                raise RuntimeError('Shared frozen parameter mutated: '+name)
            if left.data_ptr()!=pointer or right.data_ptr()!=pointer or left.dtype!=right.dtype or left.device!=right.device or left.shape!=right.shape:
                raise RuntimeError('Shared frozen storage/layout changed: '+name)
        return dict(status='PASS_FROZEN_ALIAS_POINTER_VERSION_AUDIT',pairs=len(self.pairs),scope='HOST_METADATA_ONLY_NOT_A_PROOF_AGAINST_UNTRACKED_DATA_WRITES')

def share_frozen_parameters(reference,candidate,*,reference_identity,candidate_identity):
    import torch
    if not reference_identity or reference_identity!=candidate_identity:
        raise ValueError('Same explicit frozen checkpoint identity required')
    if reference is candidate:raise ValueError('Distinct role models required')
    _check_unmerged(reference);_check_unmerged(candidate)
    left,right=_indexed(reference),_indexed(candidate)
    all_params=[p for m in(reference,candidate)for p in m.parameters()]
    if any(hasattr(p,'device_mesh')or p.is_meta for p in all_params):
        raise ValueError('Full materialized Resident parameters required; no DTensor/meta')
    left_train={_storage(p)for p in reference.parameters()if p.requires_grad}
    right_train={_storage(p)for p in candidate.parameters()if p.requires_grad}
    if left_train&right_train:raise ValueError('Role trainables must already be independent')
    trainable_ids=[(id(p),p.data_ptr(),p.dtype)for p in all_params if p.requires_grad]
    pending=[];unmatched=[]
    # Validate the WHOLE eligible set before changing any candidate storage.
    for key,(name,q)in right.items():
        if q.requires_grad:continue
        if key not in left:
            unmatched.append(name);continue
        _,p=left[key]
        if p.requires_grad:raise ValueError('Frozen/trainable ownership differs: '+key)
        if p.dtype!=q.dtype or p.device!=q.device or p.shape!=q.shape or p.stride()!=q.stride():
            raise ValueError('Frozen dtype/device/shape/stride differs: '+key)
        if not torch.equal(p.detach(),q.detach()):
            raise ValueError('Frozen values differ: '+key)
        if _storage(p)in(left_train|right_train)or _storage(q)in(left_train|right_train):
            raise ValueError('Frozen/trainable storage overlaps: '+key)
        pending.append((name,p,q))
    if not pending:raise ValueError('No identical frozen parameters to share')
    before=physical_parameter_bytes(reference,candidate);pairs=[]
    with torch.no_grad():
        for name,p,q in pending:
            # Keep BOTH Parameter objects and all role-local trainables intact.
            # Setup-only: no saved autograd tensors/graphs may predate this call.
            q.data=p.detach()
            pairs.append((name,p,q,p._version,q._version,id(p),id(q),p.data_ptr()))
    after=physical_parameter_bytes(reference,candidate)
    if trainable_ids!=[(id(p),p.data_ptr(),p.dtype)for p in all_params if p.requires_grad]:
        raise RuntimeError('Unexpected role-trainable change')
    receipt=dict(schema='distillschedule.shared_frozen_resident_base.v1',enabled=True,scope='SETUP_ONLY_FROZEN_STORAGE_ALIAS_NOT_WEIGHT_MERGE',checkpoint_identity=reference_identity,parameter_objects_preserved=True,role_trainables_independent=True,optimizer_untouched=True,shared_parameter_tensors=len(pending),shared_logical_numel=sum(p.numel()for _,p,_ in pending),physical_parameter_storage_before_bytes=before,physical_parameter_storage_after_bytes=after,physical_parameter_storage_saved_bytes=before-after,unmatched_frozen_candidate_parameters=unmatched,default_or_scheduler_installation=False)
    audit=FrozenAliasAudit(reference,candidate,tuple(pairs),receipt);audit.audit()
    return audit
