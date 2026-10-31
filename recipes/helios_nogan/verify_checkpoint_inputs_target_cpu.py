"""Explicit existing-environment Torch 2.10 CPU entry; never a native GPU gate.

Run from an already provisioned target environment. Missing stack/version is a
nonzero NOT_RUN, never an installation or a skip counted as successful proof.
"""
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests'),str(ROOT)]


def verify_production_hidden_shapes(torch, cls, policy):
    """Original production-width preprocessing on meta, no physical capacity proof."""
    with torch.device('meta'):
        model=cls(has_multi_term_memory_patch=True).to(dtype=torch.bfloat16).eval()
    assert len(model.blocks)==40 and all(p.is_meta for p in model.parameters())
    histories=[torch.empty(1,16,t,48,80,device='meta',dtype=torch.bfloat16) for t in (2,2,16)]
    observed=[]
    with torch.no_grad():
        for h,w in ((12,20),(24,40),(48,80)):
            hidden,*_=model.process_input_hidden_states(
                latents=torch.empty(1,16,9,h,w,device='meta',dtype=torch.bfloat16),
                indices_hidden_states=torch.arange(9).reshape(1,-1),
                indices_latents_history_short=torch.arange(2).reshape(1,-1),
                indices_latents_history_mid=torch.arange(2).reshape(1,-1),
                indices_latents_history_long=torch.arange(16).reshape(1,-1),
                latents_history_short=histories[0],latents_history_mid=histories[1],latents_history_long=histories[2])
            assert hidden.is_meta and hidden.dtype==torch.bfloat16
            observed.append(list(hidden.shape))
    assert observed==policy.native_contract()['shapes'], 'original production preprocessing hidden-shape mismatch'
    return dict(scope='META_ORIGINAL_PREPROCESSOR_NO_PHYSICAL_CAPACITY',shapes=observed,blocks=40)


def main():
    import importlib.util
    import torch
    if not torch.__version__.startswith('2.10.'):
        print(json.dumps(dict(status='NOT_RUN',reason='target Torch 2.10 required',observed=torch.__version__)))
        return 2
    missing=[name for name in ('pytest','einops','peft','diffusers','transformers') if importlib.util.find_spec(name) is None]
    if missing:
        print(json.dumps(dict(status='NOT_RUN',reason='existing original Helios stack unavailable',missing=missing)))
        return 2
    import pytest
    code=pytest.main(['-q',str(ROOT/'tests/test_training_helios_checkpoint_inputs_cpu.py')])
    if code:return int(code)
    from copy import deepcopy
    from contextlib import nullcontext
    from unittest.mock import patch
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from distillschedule.runtime.fsdp2_static import configure_fsdp2_checkpointing
    from distillschedule.runtime.helios_checkpoint_inputs_cpu import GCheckpointInputSession
    from distillschedule.training import helios_checkpoint_cpu as policy
    from test_training_end_to_end import equal
    torch.set_num_threads(1);torch.manual_seed(121)
    cls=load_reference_model_class(backend='torch-sdpa')
    production_shapes=verify_production_hidden_shapes(torch,cls,policy)
    base=cls(patch_size=(1,2,2),has_multi_term_memory_patch=True,num_attention_heads=2,
        attention_head_dim=16,in_channels=16,out_channels=16,text_dim=4,freq_dim=16,ffn_dim=64,
        num_layers=40,rope_dim=(4,6,6)).double().train()
    base.enable_gradient_checkpointing();configure_fsdp2_checkpointing(base)
    spec=dict(schema_version='CPU_SYNTHETIC_TEST_ONLY_ACTUAL_ORIGINAL_40_BLOCK_CLASS',
        shapes=[[1,49,32],[1,76,32],[1,184,32]],dtype='torch.float64',checkpoint_blocks=40,max_connected_grad_stages=3)

    def run(model,offload,exits):
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        noise=[torch.randn(1,16,9,h,w,dtype=torch.float64,requires_grad=True) for h,w in ((2,2),(4,4),(8,8))]
        text=torch.randn(1,2,4,dtype=torch.float64,requires_grad=True)
        histories=[torch.randn(1,16,t,8,8,dtype=torch.float64) for t in (2,2,16)]
        count=[0];hook=model.register_forward_pre_hook(lambda *args:count.__setitem__(0,count[0]+1))
        registration=GCheckpointInputSession(owner=model,blocks=tuple(model.blocks),capacity=policy.capacity_template(),
            identity=dict(source='CPU_SYNTHETIC_TEST_ONLY',task_sha256='a'*64,generator_version=2),device='cpu') if offload else None
        try:
            with registration if offload else nullcontext():
                previous=None
                for x,exit_index in zip(noise,exits,strict=True):
                    if previous is not None:
                        x=x+torch.nn.functional.interpolate(previous,size=x.shape[2:],mode='nearest')
                    # Genuine native graph pattern: every non-exit forward is
                    # no-grad and replaces the stage input, severing earlier
                    # stages. Keep the sole final-output objective unchanged.
                    for index in range(exit_index+1):
                        with torch.set_grad_enabled(index==exit_index):
                            previous=model(hidden_states=x,timestep=torch.tensor([500.],dtype=torch.float64),encoder_hidden_states=text,
                                indices_hidden_states=torch.arange(9).reshape(1,-1),
                                indices_latents_history_short=torch.arange(2).reshape(1,-1),
                                indices_latents_history_mid=torch.arange(2).reshape(1,-1),
                                indices_latents_history_long=torch.arange(16).reshape(1,-1),
                                latents_history_short=histories[0],latents_history_mid=histories[1],latents_history_long=histories[2],
                                return_dict=False,is_first_denoising_step=False)[0]
                        x=previous
                previous.square().mean().backward()
                gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
                input_gradients=[x.grad.clone() if x.grad is not None else None for x in noise]+[text.grad.clone()]
                optimizer.step()
            return dict(output=previous.detach(),gradients=gradients,input_gradients=input_gradients,
                weights={n:p.detach().clone() for n,p in model.named_parameters()},
                named_adam={n:{k:v.clone() if torch.is_tensor(v) else v for k,v in optimizer.state[p].items()}
                    for n,p in model.named_parameters() if p in optimizer.state},rng=torch.get_rng_state(),calls=count[0]),registration
        finally:hook.remove()

    cases=[]
    for exits,first_live,restores in (((0,1,1),2,40),((0,1,0),1,80),((0,0,0),0,120)):
        control,candidate=deepcopy(base),deepcopy(base)
        with patch.object(policy,'native_contract',lambda:deepcopy(spec)):
            torch.manual_seed(13);expected,_=run(control,False,exits)
            torch.manual_seed(13);actual,registration=run(candidate,True,exits)
        assert equal(actual,expected),'actual original tiny 40-block output/grad/Adam/RNG/call mismatch'
        receipt=registration.receipt();stats=receipt['stats']
        assert actual['calls']==sum(exits)+3
        assert stats['checkpoint_calls']==stats['cpu_copy_count']==stats['release_count']==120
        assert stats['restore_count']==stats['restored_release_count']==restores
        assert stats['unused_release_count']==120-restores
        assert stats['restored_bytes']==40*sum(a*b*c*8 for a,b,c in spec['shapes'][first_live:])
        assert all(r['restore_count']==int(r['stage_index']>=first_live) and r['release_count']==1
            and r['shape']==spec['shapes'][r['stage_index']]
            and r['release_outcome']==('restored' if r['stage_index']>=first_live else 'unused') for r in receipt['boundaries'])
        assert stats['live_cpu_bytes']==stats['live_leases']==stats['cancel_count']==0 and not torch.cuda.is_initialized()
        cases.append(dict(exits=exits,stats=stats))
    print(json.dumps(dict(status='PASS_CPU_SYNTHETIC_TEST_ONLY_ACTUAL_ORIGINAL_40_BLOCK_CLASS',torch=torch.__version__,
        cases=cases,production_shapes=production_shapes,native_HSDP='NOT_RUN',pretrained_capacity='NOT_RUN',target_GPU='NOT_RUN',native_GO=False)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
