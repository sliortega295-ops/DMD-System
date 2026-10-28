"""Original H3 offline encoders; separate from Student, decoder and CUDA owners.

The actual encode_video/encode_audio methods choose posterior means, apply native
normalization once and own their documented temporal/audio tail padding. No
sampling, crop, resize, resample, text invention or training RNG consumption.
"""
from copy import deepcopy
import math

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from . import h3_av_codec as codec
from .h3_cached_assets import checked_pin, read_json

SCHEMA = 'distillschedule.original_h3_av_encoder.v1'
PRECISION = 'CPU_FP32_ORIGINAL_H3_ENCODE'
VIDEO_NATIVE = dict(in_channels=3, out_ch=3, ch=128, embed_dim=24, z_channels=24,
    num_res_blocks=2, ch_mult=[1,2,2,4,4,8], space_down=[2,2,2,2,1,1],
    time_down=[1,2,2,1,1,1], padding_mode='reflect', use_t_isolated_gn=True,
    causal_encoder=True, clip_length=17, token_drop=3, tile_size=256, tile_overlap_min=64)
AUDIO_NATIVE = dict(encoder_dim=64, encoder_rates=[2,4,4,5,5], decoder_dim=1024,
    decoder_rates=[5,5,2,2,2,2,2], sample_rate=32000, vae_latent_channels=32,
    attn_proj=True, decoder_type='bigvgan')
require = codec.require


def validate_manifest(p):
    require(isinstance(p, dict) and set(p) == {'schema_version','precision','storage_dtype','evidence_kind',
        'source','video_encoder','audio_encoder','weights','budget'}
        and p['schema_version'] == SCHEMA and p['precision'] == PRECISION
        and p['storage_dtype'] in ('float32','bfloat16_to_float32')
        and p['evidence_kind'] in (codec.FIXTURE,codec.LOCAL), 'explicit encoder schema/precision/storage/evidence required')
    fingerprint(p)
    codec.validate_source(p['source'])
    video, audio = p['video_encoder'], p['audio_encoder']
    require(isinstance(video,dict) and set(video) == set(VIDEO_NATIVE)
        and isinstance(audio,dict) and set(audio) == set(AUDIO_NATIVE), 'complete original encoder architecture required')
    expected_v, expected_a = deepcopy(VIDEO_NATIVE), deepcopy(AUDIO_NATIVE)
    if p['evidence_kind'] == codec.FIXTURE:
        require(type(video['ch']) is int and video['ch'] == 32 and type(video['num_res_blocks']) is int
            and video['num_res_blocks'] == 1 and type(audio['encoder_dim']) is int and audio['encoder_dim'] == 8,
            'bounded actual original-class tiny encoder required')
        expected_v.update(ch=32,num_res_blocks=1); expected_a.update(encoder_dim=8)
    for actual, expected in ((video,expected_v),(audio,expected_a)):
        require(fingerprint(actual) == fingerprint(expected),
                'original native encoder architecture changed; reduced production encoder refused')
    limits=dict(max_parameters=3_000_000_000, max_loaded_bytes=256*2**30,
        max_input_elements=4_000_000_000, max_activation_elements=4_000_000_000,
        max_attention_elements=8_000_000_000, host_reserve_bytes=128*2**30)
    b=p['budget']
    require(isinstance(b,dict) and set(b) == set(limits)
        and all(type(b[k]) is int and 0 < b[k] <= limit for k,limit in limits.items()),
        'explicit bounded encoder/input/activation/attention/host capacity required')
    require(isinstance(p['weights'],dict) and set(p['weights']) == {'video','audio'}, 'both original encoder inventories required')
    for pins in p['weights'].values():
        require(isinstance(pins,list) and 0 < len(pins) <= 128, 'bounded encoder safe shard inventory required')
        for pin in pins: codec._pin(pin)
    return deepcopy(p)


def meta_encoders(p, api):
    """Inspect full native safe headers, materialize only encode-owned subtrees."""
    import torch
    from .checkpoint import rng_state, restore_rng
    before=rng_state(device='cpu')
    try:
        with torch.device('meta'):
            models=dict(video=api.video.MiniMaxH3VideoVAE(**p['video_encoder']),
                        audio=api.audio.MiniMaxH3AudioVAE(**p['audio_encoder']))
            original={role:{n:(tuple(t.shape),t.dtype) for n,t in model.state_dict().items()}
                      for role,model in models.items()}
            del models['video'].decoder; del models['video'].post_quant_conv
            for name in ('decoder','dec_in_proj','logs_proj'):delattr(models['audio'],name)
        return models,original
    finally:restore_rng(before)


def build_encoder(parameters):
    import torch
    from .checkpoint import rng_state, restore_rng
    from .single_device import admit_fp32
    from .host_memory import admit_host_memory
    require(isinstance(parameters,dict) and set(parameters) == {'manifest'}, 'explicit pinned encoder manifest required')
    path=checked_pin(parameters['manifest']); p=validate_manifest(read_json(path))
    admit_fp32({'device':'cpu'},require=require)
    require(str(torch.__version__).split('+')[0] == '2.10.0', 'validated original encoder Torch2.10 runtime required')
    before=rng_state(device='cpu')
    try:
        api=codec.source_api(p['source']); models,original=meta_encoders(p,api)
        count=sum(t.numel() for model in models.values() for t in model.state_dict().values())
        require(count <= p['budget']['max_parameters'] and 12*count <= p['budget']['max_loaded_bytes'],
                'encoder parameter/staging capacity exceeded')
        memory=admit_host_memory(12*count,p['budget']['host_reserve_bytes'])
        for role,model in models.items():codec._load_weights(model,p['weights'][role],original[role],p['storage_dtype'])
        checked_pin(parameters['manifest']); api.lock.validate()
        return {**models, 'spec':p, 'api':api, 'manifest':deepcopy(parameters['manifest']),
                'metadata':dict(schema_version=SCHEMA, precision=PRECISION, parameters=count,
                    codec_manifest_sha256=file_sha256(path), source=deepcopy(p['source']), capacity=memory,
                    evidence_kind=p['evidence_kind'], purpose='ENCODE_ONLY_NOT_STUDENT_OR_TRAINING')}
    finally:restore_rng(before)


def admit_geometry(p, shape, samples, fps):
    """Header admission before media decoding or allocating any codec weights."""
    require(len(shape) == 4 and shape[-1] == 3 and all(type(n) is int and n > 0 for n in shape)
        and type(fps) is int and fps == 24 and type(samples) is int and samples > 0,
        'positive THWC/stereo sample geometry and explicit 24FPS required')
    f,h,w,_=shape
    require(5 <= f <= 34821 and (f-5) % 17 == 0 and 32 <= h <= 4096 and 32 <= w <= 4096
        and h % 32 == w % 32 == 0, 'original 17n+5 frames and 32-divisible spatial geometry required; no crop/resize')
    b=p['budget']; a=math.ceil(samples/800)
    input_elements=math.prod(shape)+2*samples
    activation=(12*min(f,17)*min(h,256)*min(w,256)*p['video_encoder']['ch']
                +128*samples*p['audio_encoder']['encoder_dim'])
    attention=16*a*a
    require(input_elements <= b['max_input_elements'] and activation <= b['max_activation_elements']
        and attention <= b['max_attention_elements'], 'encoder input/working-set capacity exceeded')
    return f,h,w,a,activation,attention


def _admit_inputs(p, video, audio, fps):
    import torch
    require(type(video) is torch.Tensor and video.device.type == 'cpu' and video.dtype == torch.uint8
        and video.ndim == 4, 'original encoder requires THWC RGB uint8; no cast/resample')
    require(isinstance(audio,dict) and set(audio) == {'samples','sample_rate'} and type(audio['sample_rate']) is int
        and audio['sample_rate'] == 32000, 'original stereo 32kHz required; no resampling')
    x=audio['samples']
    require(type(x) is torch.Tensor and x.device.type == 'cpu' and x.dtype == torch.float32
        and not x.requires_grad and x.grad_fn is None and x.ndim == 2 and x.shape[1] == 2 and x.shape[0] > 0
        and bool(torch.isfinite(x).all()) and bool((x.abs() <= 1).all()), 'finite detached stereo TC FP32 PCM in [-1,1] required')
    return admit_geometry(p,list(video.shape),len(x),fps)


def latent_shapes(shape, samples):
    """Fixed native 17-frame/5-token and 800-sample mean-encoder geometry."""
    f,h,w,_=shape
    return dict(video_latents=[1,24,math.ceil(f/17)*5-3,h//16,w//16],
                audio_latents=[2,32,math.ceil(samples/800)])


def encode(bundle, video, audio, *, fps):
    import torch
    from .host_memory import admit_host_memory
    p=bundle['spec']; _,_,_,_,activation,attention=_admit_inputs(p,video,audio,fps)
    required=12*bundle['metadata']['parameters']+8*(activation+attention+video.numel()+audio['samples'].numel())
    require(required <= p['budget']['max_loaded_bytes'], 'encoder live staging/working-set capacity exceeded')
    admit_host_memory(required,p['budget']['host_reserve_bytes'])
    checked_pin(bundle['manifest']); bundle['api'].lock.validate()
    with torch.no_grad():
        # Layout/unit conversion only. Native methods own pixel and latent stats.
        pixels=video.permute(3,0,1,2).unsqueeze(0).float()/255
        v=bundle['video'].encode_video(pixels,dtype=torch.float32)
        s=bundle['audio'].encode_audio(audio['samples'].t().contiguous(),dtype=torch.float32)
    shapes=latent_shapes(list(video.shape),len(audio['samples']))
    for t,shape in zip((v,s),shapes.values()):
        require(list(t.shape) == shape and t.device.type == 'cpu' and t.dtype == torch.float32
            and not t.requires_grad and bool(torch.isfinite(t).all()), 'actual original encoder output shape/precision/finite changed')
    checked_pin(bundle['manifest']); bundle['api'].lock.validate()
    return dict(video_latents=v,audio_latents=s,encoding=dict(schema_version=SCHEMA,
        video_input_shape=list(video.shape), audio_input_shape=list(audio['samples'].shape),
        video_latent_shape=list(v.shape),audio_latent_shape=list(s.shape),fps=fps,sample_rate=32000,
        video_native_tail_repeat_frames=(-len(video))%17, audio_right_pad_samples=(-len(audio['samples']))%800,
        normalization='ORIGINAL_ENCODE_VIDEO_AND_ENCODE_AUDIO_MEANS_ONCE', posterior='MEAN_NO_SAMPLING',
        media_alignment='EXPLICIT_INPUT_LENGTHS_NO_CROP_NO_RESAMPLE'))
