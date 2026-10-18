import copy
import pytest
import torch
from distillschedule.runtime.shared_frozen_base import share_frozen_parameters,physical_parameter_bytes

class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__();self.base_layer=torch.nn.Linear(8,8,dtype=torch.float64);self.base_layer.requires_grad_(False)
        self.lora_A=torch.nn.Linear(8,3,bias=False,dtype=torch.float64);self.lora_B=torch.nn.Linear(3,8,bias=False,dtype=torch.float64)
    def forward(self,x):return self.base_layer(x)+self.lora_B(self.lora_A(x))

def roles():
    torch.manual_seed(20);a=Linear();b=copy.deepcopy(a)
    with torch.no_grad():b.lora_A.weight.mul_(.8);b.lora_B.weight.mul_(1.2)
    return a,b

def share(a,b):return share_frozen_parameters(a,b,reference_identity='same-sha',candidate_identity='same-sha')

def test_shared_base_preserves_two_role_outputs_gradients_optimizer_and_rng():
    a,b=roles();refa,refb=copy.deepcopy(a),copy.deepcopy(b);before=physical_parameter_bytes(a,b)
    ids=[id(p)for m in(a,b)for p in m.parameters()];rng=torch.random.get_rng_state().clone();audit=share(a,b)
    assert ids==[id(p)for m in(a,b)for p in m.parameters()]
    assert a.base_layer.weight.data_ptr()==b.base_layer.weight.data_ptr()
    assert a.lora_A.weight.data_ptr()!=b.lora_A.weight.data_ptr()
    assert audit.receipt['physical_parameter_storage_saved_bytes']==8*(8*8+8)
    assert physical_parameter_bytes(a,b)==before-8*(8*8+8)
    x=torch.arange(24,dtype=torch.float64).reshape(3,8)/20
    for left,right in((a,refa),(b,refb)):
        opt=torch.optim.AdamW([p for p in left.parameters()if p.requires_grad],lr=1e-3);ref=torch.optim.AdamW([p for p in right.parameters()if p.requires_grad],lr=1e-3)
        y,z=left(x),right(x);assert torch.equal(y,z);y.square().mean().backward();z.square().mean().backward()
        for p,q in zip(left.parameters(),right.parameters()):
            if p.requires_grad:assert torch.equal(p.grad,q.grad)
        opt.step();ref.step()
        for p,q in zip(left.parameters(),right.parameters()):assert torch.equal(p,q)
        for p,q in zip(opt.param_groups[0]['params'],ref.param_groups[0]['params']):
            for name in('exp_avg','exp_avg_sq','step'):assert torch.equal(opt.state[p][name],ref.state[q][name])
    assert torch.equal(rng,torch.random.get_rng_state());assert audit.audit()['pairs']==2

def test_mismatch_fails_before_any_alias():
    a,b=roles();old=b.base_layer.weight.data_ptr()
    with torch.no_grad():b.base_layer.bias.add_(1)
    with pytest.raises(ValueError,match='values differ'):share(a,b)
    assert b.base_layer.weight.data_ptr()==old and old!=a.base_layer.weight.data_ptr()

def test_identity_and_trainable_ownership_fail_closed():
    a,b=roles()
    with pytest.raises(ValueError):share_frozen_parameters(a,b,reference_identity='a',candidate_identity='b')
    b.lora_A.weight=a.lora_A.weight
    with pytest.raises(ValueError,match='trainables'):share(a,b)

def test_native_inplace_mutation_is_audited():
    a,b=roles();r=share(a,b)
    with torch.no_grad():b.base_layer.weight.add_(1)
    with pytest.raises(RuntimeError,match='mutated'):r.audit()

def test_no_merged_adapter_no_sharded_or_meta():
    a,b=roles();b.merged=True
    with pytest.raises(ValueError,match='Merged'):share(a,b)
    b.merged=False;b.base_layer.weight.device_mesh='forbidden'
    with pytest.raises(ValueError,match='Resident'):share(a,b)

def test_same_role_model_rejected():
    a,b=roles()
    with pytest.raises(ValueError,match='Distinct'):share(a,a)
