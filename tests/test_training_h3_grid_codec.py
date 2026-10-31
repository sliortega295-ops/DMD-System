"""Exact original packed-token/codec ABI; no W6 sampler or GPU training."""
from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest


def test_grid_codec_discovery_is_model_free():
    result=subprocess.run([sys.executable,'-B','-c',
        'from distillschedule.training.h3_grid_codec import build_decoder;import sys;'
        'assert not any(n in sys.modules for n in ("torch","diffusers","diffsynth"))'],
        text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_exact_original_packing_inverse_preserves_channel_time_and_spatial_order():
    import torch
    from distillschedule.training.h3_grid_codec import packing_api,unpack_generated
    api,lock=packing_api()
    v=torch.arange(24*7*4*6,dtype=torch.float32).reshape(1,24,7,4,6)/1031
    a=torch.arange(2*32*37,dtype=torch.float32).reshape(2,32,37)/887
    shape=dict(latent_frames=7,latent_height=4,latent_width=6,audio_latents=37,
        video_tokens=(1,42,96),audio_tokens=(1,74,32))
    tokens=dict(video_tokens=api.patchify_video(v).unsqueeze(0),audio_tokens=api.pack_audio(a).unsqueeze(0))
    result=unpack_generated(tokens,shape)
    torch.testing.assert_close(result['video_latents'],v,rtol=0,atol=0)
    torch.testing.assert_close(result['audio_latents'],a,rtol=0,atol=0)
    lock.validate()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault',['dtype','shape','batch','audio_order','nonfinite','grad','mixed','schema'])
def test_invalid_packed_tensor_is_not_silently_cast_or_reshaped(fault):
    import torch
    from distillschedule.training.h3_grid_codec import unpack_generated
    shape=dict(latent_frames=7,latent_height=2,latent_width=2,audio_latents=37,
        video_tokens=(1,7,96),audio_tokens=(1,74,32))
    value=dict(video_tokens=torch.zeros(1,7,96),audio_tokens=torch.zeros(1,74,32))
    if fault=='dtype':value['video_tokens']=value['video_tokens'].bfloat16()
    elif fault=='shape':value['video_tokens']=value['video_tokens'][:,:,:95]
    elif fault=='batch':value['video_tokens']=value['video_tokens'].repeat(2,1,1)
    elif fault=='audio_order':value['audio_tokens']=value['audio_tokens'].permute(0,2,1)
    elif fault=='nonfinite':value['audio_tokens'][0,0,0]=float('nan')
    elif fault=='grad':value['video_tokens'].requires_grad_()
    elif fault=='mixed':value['video_latents']=torch.zeros(1,24,7,2,2)
    else:value['latent_schema']={'video_tokens':{'layout':'BCTHW'}}
    with pytest.raises(ValueError):unpack_generated(value,shape)


def test_optional_codec_does_not_allow_sampler_or_precision_overrides():
    from types import SimpleNamespace
    from distillschedule.training.h3_grid_inference import generate
    for options in ({'num_steps':1},{'precision':'float32'},{'decoder_manifest':{},'sigma':.3}):
        with pytest.raises(ValueError,match='original schedule; no overrides'):
            generate(dict(inference={},student=SimpleNamespace(restored=True,selected=None)),{},options)
