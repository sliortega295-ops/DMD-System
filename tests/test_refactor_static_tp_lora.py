"""Real CPU/Gloo TP4 output and autograd, no GPU or timing surrogate."""
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F


def _rank_test(rank,rendezvous):
    from distillschedule.runtime.static_tp_lora import (
        StaticTPLoRALinear, StaticTPRMSNorm, bind_peft_linear_tp,
    )
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=4,
                            timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(1904)
        w=torch.randn(20,16,dtype=torch.float64)
        bias=torch.randn(20,dtype=torch.float64)
        a=torch.randn(4,16,dtype=torch.float32)
        b=torch.randn(20,4,dtype=torch.float32)
        x=torch.randn(3,16,dtype=torch.float64)
        target=torch.randn(3,20,dtype=torch.float64)
        for style in ('column','row'):
            ref_x=x.clone().requires_grad_();ref_a=a.clone().requires_grad_();ref_b=b.clone().requires_grad_()
            ref=F.linear(ref_x,w,bias)+(F.linear(F.linear(ref_x.float(),ref_a),ref_b)*0.25).to(w.dtype)
            ref.backward(target)
            module=StaticTPLoRALinear(w,a,b,base_bias=bias,group=dist.group.WORLD,
                style=style,scale=0.25,cpu_reference_dtype=True)
            actual_x=(x if style=='column' else x.chunk(4,dim=1)[rank]).clone().requires_grad_()
            actual=module(actual_x)
            wanted=ref if style=='row' else ref.chunk(4,dim=1)[rank]
            torch.testing.assert_close(actual,wanted,rtol=1e-5,atol=2e-6)
            actual.backward(target if style=='row' else target.chunk(4,dim=1)[rank].contiguous())
            torch.testing.assert_close(actual_x.grad,ref_x.grad if style=='column' else ref_x.grad.chunk(4,dim=1)[rank],rtol=1e-5,atol=3e-6)
            torch.testing.assert_close(module.lora_A.grad,ref_a.grad if style=='column' else ref_a.grad.chunk(4,dim=1)[rank],rtol=2e-5,atol=3e-6)
            torch.testing.assert_close(module.lora_B.grad,ref_b.grad.chunk(4,dim=0)[rank] if style=='column' else ref_b.grad,rtol=2e-5,atol=3e-6)
            assert module.base_weight.grad is None
            assert not module.parameter_layout()['merged_execution']
        # Across-head normalization needs a global statistic AND its backward.
        rms_weight=torch.rand(16,dtype=torch.float64)
        norm=StaticTPRMSNorm(rms_weight,group=dist.group.WORLD,eps=1e-6,
                            cpu_reference_dtype=True)
        dense=x.clone().requires_grad_()
        expected=dense*torch.rsqrt(dense.square().mean(-1,keepdim=True)+1e-6)*rms_weight
        gradient=torch.randn_like(expected)
        expected.backward(gradient)
        local=x.chunk(4,dim=1)[rank].clone().requires_grad_()
        actual=norm(local)
        torch.testing.assert_close(actual,expected.chunk(4,dim=1)[rank],rtol=1e-10,atol=1e-10)
        actual.backward(gradient.chunk(4,dim=1)[rank].contiguous())
        torch.testing.assert_close(local.grad,dense.grad.chunk(4,dim=1)[rank],rtol=1e-10,atol=1e-10)
        assert norm.weight.grad is None
        # Keep real PEFT adapter-control logic instead of impersonating a tuner.
        import copy
        from peft.tuners.lora.layer import Linear
        from torch import nn
        from torch.utils.checkpoint import checkpoint
        for style in ('column','row'):
            native=Linear(nn.Linear(16,20,bias=True,dtype=torch.float64),
                          'default',r=4,lora_alpha=1,lora_dropout=0,init_lora_weights=False)
            native.base_layer.weight.data.copy_(w)
            native.base_layer.bias.data.copy_(bias)
            native.base_layer.requires_grad_(False)
            native.lora_A['default'].to(torch.float32)
            native.lora_B['default'].to(torch.float32)
            native.lora_A['default'].weight.data.copy_(a)
            native.lora_B['default'].weight.data.copy_(b)
            candidate=bind_peft_linear_tp(copy.deepcopy(native),group=dist.group.WORLD,
                                          style=style,cpu_reference_dtype=True)
            dense=x.clone().requires_grad_()
            expected=checkpoint(native,dense,use_reentrant=False)
            expected.backward(target)
            local=(x if style=='column' else x.chunk(4,dim=1)[rank]).clone().requires_grad_()
            actual=checkpoint(candidate,local,use_reentrant=False)
            wanted=expected.chunk(4,dim=1)[rank] if style=='column' else expected
            torch.testing.assert_close(actual,wanted,rtol=1e-5,atol=2e-6)
            actual.backward(target.chunk(4,dim=1)[rank].contiguous() if style=='column' else target)
            torch.testing.assert_close(local.grad,dense.grad if style=='column' else dense.grad.chunk(4,dim=1)[rank],rtol=2e-5,atol=3e-6)
            torch.testing.assert_close(candidate.lora_A['default'].weight.grad,
                native.lora_A['default'].weight.grad if style=='column' else native.lora_A['default'].weight.grad.chunk(4,dim=1)[rank],rtol=2e-5,atol=3e-6)
            torch.testing.assert_close(candidate.lora_B['default'].weight.grad,
                native.lora_B['default'].weight.grad.chunk(4,dim=0)[rank] if style=='column' else native.lora_B['default'].weight.grad,rtol=2e-5,atol=3e-6)
            candidate.enable_adapters(False)
            native.enable_adapters(False)
            teacher=native(x)
            torch.testing.assert_close(candidate(local.detach()),teacher.chunk(4,dim=1)[rank] if style=='column' else teacher,rtol=1e-10,atol=1e-10)
            candidate.enable_adapters(True)
            assert not candidate.disable_adapters and not candidate.merged
            assert candidate._static_tp_layout['teacher_adapter_disable']=='ORIGINAL_PEFT_SHELL'
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_cpu_gloo_tp4_outputs_and_lora_and_input_gradients(tmp_path:Path):
    mp.spawn(_rank_test,args=(str(tmp_path/'gloo_rendezvous'),),nprocs=4,join=True)
