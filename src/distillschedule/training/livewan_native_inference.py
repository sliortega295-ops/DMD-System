"""Portable CURRENT full C51 Student, with the original LiveWan block sampler.

Training remains mixed-precision/CUDA; the independent consumer is explicitly
CPU FP32 with original MATH attention. It does not load Teacher/Score/Adam, use
W1's pyramid sampler, replay a training row, or merge any inference weights.
"""
from copy import deepcopy
import json
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256

SCHEMA = 'distillschedule.livewan_native_current_generator.v1'
FACTORY = 'distillschedule.training.livewan_native_inference:build_inference_model'
NATIVE = 'NATIVE_C51_CURRENT_STUDENT_UNVERIFIED_QUALITY'
FIXTURE = 'FULL_ORIGINAL_TINY_CPU_FIXTURE_NOT_PRETRAINED_OR_QUALITY'
PRECISION = 'EXACT_FP32_STUDENT_STORAGE_CPU_FP32_MATH_INFERENCE'
ARCHITECTURE = dict(model_type='t2v', patch_size=[1,2,2], text_len=512, in_dim=16,
    dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16, num_heads=12,
    num_layers=30, window_size=[-1,-1], qk_norm=True, cross_attn_norm=True, eps=1e-6)


def require(ok, message):
    if not ok:
        raise ValueError('C51 native output: ' + message)


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def inference_source():
    from distillschedule.reference import livewan_wan21 as provider
    from . import inference, wan_native_inference, host_memory, adapters, checkpoint
    return dict(original=provider.source_identity(), provider=file_sha256(Path(provider.__file__)),
        consumer=file_sha256(Path(__file__)), helpers={name:file_sha256(Path(api.__file__)) for name,api in (
            ('publisher',inference),
            ('codec',wan_native_inference), ('capacity',host_memory), ('components',adapters), ('checkpoint',checkpoint))})


def runtime_identity():
    from .wan_native_inference import runtime_identity as identity
    return identity()


def budget(value):
    limits = dict(max_parameters=2_000_000_000, max_loaded_bytes=256*2**30,
        max_primary_export_bytes=64*2**30, max_attention_elements=20_000_000_000,
        max_input_elements=64_000_000, host_reserve_bytes=512*2**30)
    require(isinstance(value,dict) and set(value)==set(limits)
        and all(type(value[k]) is int and 0 < value[k] <= v for k,v in limits.items()),
        'complete bounded inference/export capacity policy required')
    return deepcopy(value)


def geometry(value, evidence):
    require(type(value) is dict and set(value)=={'resolution','frames','fps'}
        and type(value['frames']) is int and value['frames']==81
        and type(value['resolution']) is list and all(type(v) is int for v in value['resolution'])
        and type(value['fps']) is int and 1 <= value['fps'] <= 120
        and value['resolution']==([368,640] if evidence==NATIVE else [32,32]),
        'native 368x640/81 frames, or explicitly tiny32 fixture geometry required')
    h,w=value['resolution']
    return dict(**value, noise_shape=[1,16,21,h//8,w//8], video_shape=[81,h,w,3])


def architecture(student):
    # Wan deliberately omits five constructor fields from ConfigMixin.config.
    # Read the ACTUAL retained constructor attributes, never guessed defaults.
    require(all(hasattr(student,k) for k in ARCHITECTURE),'complete original Wan constructor attributes required')
    return plain({k:getattr(student,k) for k in ARCHITECTURE})


def current_view(student):
    from .wan_native_inference import CurrentGenerator
    view=CurrentGenerator(student)
    # Original Wan.freqs is intentionally not registered: Module.to() must not
    # cast its complex precision. Capture the actual positional state on the
    # separate view without mutating/reconstructing the native Student.
    view.register_buffer('rope_frequencies',student.freqs)
    return view


def _admit(parameters):
    p=plain(parameters)
    require(type(p) is dict and set(p)=={'schema_version','source','runtime','architecture','geometry',
        'precision','budget','codec','seed','evidence_kind'} and p['schema_version']==SCHEMA,
        'complete versioned C51 portable inference parameters required')
    require(p['source']==inference_source() and p['runtime']==runtime_identity()
        and p['precision']==PRECISION, 'portable source/runtime/precision changed')
    require(p['evidence_kind'] in (NATIVE,FIXTURE) and type(p['seed']) is int and 0 <= p['seed'] < 2**63,
            'explicit evidence and independent inference seed required')
    shape=geometry(p['geometry'],p['evidence_kind'])
    b=budget(p['budget']); a=p['architecture']
    if p['evidence_kind']==NATIVE:
        require(a==ARCHITECTURE,'current full native 1.3B Student architecture required')
    else:
        from .livewan_public import _architecture
        _architecture(a)  # existing bounded genuine original-model fixture
    from .wan_native_inference import _admit_codec
    from .helios_public_assets import REAL
    _admit_codec(p['codec'])
    require(p['codec']['dtype']=='torch.float32'
        and p['codec']['evidence_kind']==(FIXTURE if p['evidence_kind']==FIXTURE else REAL),
        'independent matching-evidence original codec FP32 storage required')
    s=(shape['resolution'][0]//16)*(shape['resolution'][1]//16)
    attention=a['num_heads']*(9*s)*(22*s+512)
    require(attention<=b['max_attention_elements'] and 22*s*a['dim']<=b['max_input_elements'],
            'original streaming MATH attention/input exceeds capacity')
    return p,shape,attention


def _probe(parameters, *, constructor=None):
    import torch
    from .checkpoint import rng_state,restore_rng
    from .wan_native_inference import _codec_class
    p,shape,attention=_admit(parameters)
    if constructor is None:
        from distillschedule.reference import livewan_wan21 as provider
        constructor=provider.load_cpu(backend='torch-sdpa-cpu',precision='float32').model.WanModel
    before=rng_state()
    try:
        with torch.device('meta'):
            student=constructor(**p['architecture'])
            codec=_codec_class()(**p['codec']['architecture'])
        count=sum(v.numel() for module in (student,codec) for v in module.state_dict().values())
        size=4*count+student.freqs.numel()*student.freqs.element_size()
        s=(shape['resolution'][0]//16)*(shape['resolution'][1]//16)
        kv_bytes=2*p['architecture']['num_layers']*22*s*p['architecture']['dim']*4
        # Strict loading can hold CPU file, model and serialization staging;
        # MATH attention and persistent clean-KV are charged independently.
        working=3*size+16*attention+kv_bytes+16*512*4096
        require(count<=p['budget']['max_parameters'] and working<=p['budget']['max_loaded_bytes']
            and 3*size<=p['budget']['max_primary_export_bytes'],'model/codec/KV/attention capacity exceeded')
        if p['evidence_kind']==FIXTURE:
            require(count<=10_000_000,'tiny original-model fixture cannot claim full native capacity')
        return p,student,codec,dict(parameters=count,state_bytes=size,working_bytes=working)
    finally:
        restore_rng(before)


def metadata(p):
    return dict(schema_version=SCHEMA,selection='CURRENT_FULL_STUDENT_NOT_SHADOW_NOT_EMA',
        precision=PRECISION, sampler='ORIGINAL_FEWSTEPSTREAMER_FOUR_STEP_RENOISE_CLEAN_T0_KV_COMMIT',
        conditioning='NEW_SUPPLIED_CLEAN_WORLD3_6_9_AND_RAW_POSITIVE_TEXT_NO_NEGATIVE',
        training='FULL_C51_MIXED_PRECISION_NATIVE_OWNER_SEPARATE_FROM_CPU_INFERENCE',
        evidence_kind=p['evidence_kind'],quality_validation='NOT_RUN')


def build_inference_model(parameters):
    import torch
    from .adapters import ModelComponents
    from .host_memory import admit_host_memory
    raw=dict(parameters)
    require(raw.pop('device','cpu')=='cpu','portable C51 consumer currently supports CPU FP32 only')
    p,student,codec,sizes=_probe(raw)
    admit_host_memory(sizes['working_bytes'],p['budget']['host_reserve_bytes'])
    # Meta constructors draw no real tensor values. Every concrete value must
    # be restored strictly from the complete export before generation.
    for module in (student,codec):
        module.to_empty(device='cpu')
        module.requires_grad_(False).eval()
    student.freqs=torch.empty_like(student.freqs,device='cpu')
    student._native_output_restored=False
    def restored(module,keys):
        require(not keys.missing_keys and not keys.unexpected_keys,'complete current Student state required')
        module._native_output_restored=True
    student.register_load_state_dict_post_hook(restored)
    state=dict(generator=current_view(student),codec=codec,inference=p)
    def decode_output(generated,options):
        require(not options,'original codec has no decode overrides')
        return decode(state,generated)
    return ModelComponents(state=state,metadata=metadata(p),generate=generate,
        decode=decode_output,admit_training=refuse_training).validate()


def refuse_training(*args,**kwargs):
    raise ValueError('portable C51 output is not a resumable native training checkpoint')


def generate(state,feeds,options):
    import torch
    from .livewan_public import topology
    from .checkpoint import rng_state,restore_rng
    from distillschedule.reference import livewan_wan21 as provider
    topology()
    require(not options,'original C51 sampler/shape have no generation overrides')
    p,shape,_=_admit(state['inference'])
    student=state['generator'].model
    require(student._native_output_restored and not student.training and all(not v.requires_grad
        and v.device.type=='cpu' and v.dtype==torch.float32 and bool(v.isfinite().all())
        for v in student.parameters()),'strict restored finite frozen current Student required')
    require(student.freqs is state['generator'].rope_frequencies and student.freqs.device.type=='cpu'
        and bool(student.freqs.isfinite().all()),'actual exported positional state must be restored, not recomputed')
    require(type(feeds) is dict and set(feeds)=={'world','context'},'new clean-world and positive-text tensors required')
    world,context=feeds['world'],feeds['context']
    require(type(world) is torch.Tensor and world.ndim==5 and world.shape[2] in (3,6,9),
            'supplied clean world3/6/9 required; no inferred or cropped history')
    expected=dict(world=[1,16,world.shape[2],*shape['noise_shape'][-2:]],context=[1,512,4096])
    for name,value in feeds.items():
        require(type(value) is torch.Tensor and list(value.shape)==expected[name] and value.device.type=='cpu'
            and value.dtype==torch.float32 and not value.requires_grad and value.grad_fn is None
            and bool(value.isfinite().all()),'complete detached finite FP32 new input required: '+name)
    api=provider.load_cpu(backend='torch-sdpa-cpu',precision='float32')
    before=rng_state()
    try:
        # Preserve caller/training RNG. Only the explicit NEW inference stream
        # consumes noise; sampler order and clean t=0 KV commits are original.
        generator=torch.Generator(device='cpu').manual_seed(p['seed'])
        with torch.no_grad(), api.attention.portable_inference_capacity(
                max_input_elements=p['budget']['max_input_elements'],
                max_attention_elements=p['budget']['max_attention_elements']):
            streamer=api.stream.FewStepStreamer(student,width=shape['resolution'][1],height=shape['resolution'][0],
                max_frames=278,device='cpu',dtype=torch.float32,window_frames=None,time_scale=1000.,
                cache_frames=22,block_frames=3,num_steps=4,cuda_graphs=False)
            streamer.set_text(student.text_embedding(context))
            streamer.set_world(world)
            blocks=[streamer.generate_block(generator=generator) for _ in range((21-world.shape[2])//3)]
            result=torch.cat([world,*blocks],dim=2)
        require(list(result.shape)==shape['noise_shape'] and result.dtype==torch.float32
            and bool(result.isfinite().all()) and streamer.n_frames==21,'original output shape/finiteness/census changed')
        return dict(latents=result,latent_schema=dict(latents=dict(layout='BCTHW',shape=list(result.shape),dtype='torch.float32')))
    finally:
        restore_rng(before)


def decode(state,generated):
    from .wan_native_inference import decode as wan_decode
    p=state['inference'];_,shape,_=_admit(p)
    return wan_decode({**state,'inference':{**p,'geometry':shape}},generated)
