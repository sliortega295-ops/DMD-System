"""Independent original-H3 Student consumer, preserving native weight dtypes.

No training owner, optimizer, old cache, native extension or model download is
needed. The reference execution is explicitly one-process CPU dense attention;
it is not CUDA-native bitwise or quality qualification.
"""
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

from distillschedule.workloads.best_known.registry import file_sha256
from .adapters import ModelComponents
from .host_memory import admit_host_memory

SCHEMA = 'distillschedule.h3_ar_native_layout_inference.v1'
FACTORY = 'distillschedule.training.h3_ar_portable:build_inference_model'
FIXTURE = 'RANDOM_ORIGINAL_H3_NATIVE_LAYOUT_FIXTURE_NOT_TRAINED'
NATIVE = 'LOCAL_ORIGINAL_H3_CURRENT_STUDENT_UNVERIFIED_QUALITY'
PRECISION = 'EXACT_BF16_OR_FP32_BASE_FP32_UNMERGED_LORA_CPU_BF16_AUTOCAST'
LORA = dict(lora_rank=64,lora_alpha=64,lora_targets='attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2')
SCHEDULE = dict(num_inference_steps=8,video_shift=2.22,audio_shift=2.22)
_Carrier = None


def require(ok, message):
    if not ok:
        raise ValueError('native-layout H3 inference: '+message)


def source_identity():
    from . import h3_cached_inference as original
    return dict(original=original.content_identity(), files={name:file_sha256(Path(path)) for name,path in (
        ('portable',__file__),('original_helpers',original.__file__))})


def runtime_identity():
    from .h3_cached_inference import content_runtime
    return content_runtime()


def original_config():
    return SimpleNamespace(**LORA,**SCHEDULE)


def parameters(*, architecture,geometry,weight_layout,seed,evidence_kind,budget):
    return json.loads(json.dumps(dict(schema_version=SCHEMA,precision=PRECISION,source=source_identity(),
        runtime=runtime_identity(),architecture=architecture,geometry=geometry,weight_layout=weight_layout,
        seed=seed,evidence_kind=evidence_kind,budget=budget,lora=LORA,schedule=SCHEDULE),allow_nan=False))


def _request(value):
    import torch
    from .h3_cached_assets import validate_geometry
    require(isinstance(value,dict) and value.get('device','cpu')=='cpu','explicit CPU inference; no CUDA fallback')
    p=json.loads(json.dumps({k:v for k,v in value.items() if k!='device'},allow_nan=False))
    require(set(p)=={'schema_version','precision','source','runtime','architecture','geometry','weight_layout',
        'seed','evidence_kind','budget','lora','schedule'} and p['schema_version']==SCHEMA
        and p['precision']==PRECISION and p['source']==source_identity() and p['runtime']==runtime_identity()
        and str(torch.__version__).split('+')[0]=='2.10.0','exact source/runtime/schema/precision required')
    require(p['evidence_kind'] in (FIXTURE,NATIVE) and type(p['seed']) is int and 0<=p['seed']<2**63
        and p['lora']==LORA and p['schedule']==SCHEDULE
        and type(p['lora']['lora_rank']) is int and type(p['schedule']['num_inference_steps']) is int,
        'original unmerged LoRA64/eight-step continuous schedule/seed required')
    validate_geometry(p['geometry'])
    limits=dict(max_parameters=40_000_000_000,max_loaded_bytes=512*2**30,max_input_elements=64_000_000,
                max_attention_elements=100_000_000_000,host_reserve_bytes=512*2**30)
    require(isinstance(p['budget'],dict) and set(p['budget'])==set(limits)
        and all(type(p['budget'][k]) is int and 0<p['budget'][k]<=n for k,n in limits.items()),
        'explicit bounded model/working-set/host capacity required')
    require(isinstance(p['weight_layout'],dict) and p['weight_layout']
        and all(isinstance(k,str) and k.startswith('dit.') and '.adapters.fake_score.' not in k
            and isinstance(v,dict) and set(v)=={'shape','dtype'} and isinstance(v['shape'],list)
            and all(type(n) is int and n>0 for n in v['shape'])
            and v['dtype'] in ('torch.float32','torch.bfloat16')
            and ('.adapters.student.' not in k or v['dtype']=='torch.float32')
            for k,v in p['weight_layout'].items()),'exact original base/FP32 Student-only storage layout required')
    return p


def _weights_class():
    global _Carrier
    if _Carrier is None:
        import torch
        class NativeStudentWeights(torch.nn.Module):
            def __init__(self,adapter):
                super().__init__();self.adapter=adapter;self.restored=False
        _Carrier=NativeStudentWeights
    return _Carrier


def _meta(p,model_class,adapter_class):
    import torch
    from .h3_cached_inference import _drop_fake
    require(file_sha256(Path(inspect.getsourcefile(model_class)))=='bd617a4312c70405e916707e1908100b484b2259dfbeba4af7de5bb145206d80'
        and file_sha256(Path(inspect.getsourcefile(adapter_class))) in (
            '77744f5ec391be813d42ebaf6a2692c18ea58f5ca7fb41398fd287d2a177ca62',
            '59393510353191f8b03c8cdb9922f4f071ec1a4dd80f3c520156443d805f9cae'),
        'original native model/adapter source classes required')
    defaults={k:v.default for k,v in inspect.signature(model_class.__init__).parameters.items() if k not in ('self','kwargs')}
    defaults=json.loads(json.dumps(defaults))
    arch=p['architecture']
    require(isinstance(arch,dict) and set(arch)==set(defaults),'complete explicit native architecture required')
    if p['evidence_kind']==NATIVE:
        require(arch==defaults and all(type(arch[k]) is type(v) for k,v in defaults.items()),
                'production requires the full original50block architecture, never a tiny replacement')
    else:
        # Bounds describe the *original* small constructor, not another model.
        integer={'num_layers','token_refiner_num_layers','hidden_size','num_attention_heads','attention_head_dim',
            'ffn_hidden_size','timestep_input_dim','time_embed_hidden_size','time_embed_dim','adaln_out_features',
            'final_adaln_out_features','rope_inv_freq_len'}
        require(all(type(arch[k]) is int and 0<arch[k]<=4608 for k in integer)
            and arch['num_layers']<=2 and arch['token_refiner_num_layers']<=2 and arch['hidden_size']<=256
            and all(arch[k]==defaults[k] and type(arch[k]) is type(defaults[k]) for k in set(defaults)-integer)
            and all(type(n) is int for n in arch['patch_size'])
            and arch['adaln_out_features']==18*arch['hidden_size']
            and arch['final_adaln_out_features']==2*arch['hidden_size']
            and 6*arch['rope_inv_freq_len']<=arch['attention_head_dim']
            and arch['timestep_input_dim']%2==0,'bounded original-class fixture architecture required')
    with torch.device('meta'):
        adapter=adapter_class(model_class(**arch),original_config())
        _drop_fake(adapter)
    values=adapter.state_dict(keep_vars=True)
    require(set(values)==set(p['weight_layout']) and not list(adapter.buffers()),
            'native Student state keys/buffers differ from admitted full layout')
    for name,tensor in values.items():
        spec=p['weight_layout'][name]
        require(list(tensor.shape)==spec['shape'],'native Student state shape changed')
        parent,_,leaf=name.rpartition('.')
        dtype={'torch.float32':torch.float32,'torch.bfloat16':torch.bfloat16}[spec['dtype']]
        adapter.get_submodule(parent)._parameters[leaf]=torch.nn.Parameter(torch.empty(tensor.shape,device='meta',dtype=dtype),requires_grad=False)
    carrier=_weights_class()(adapter).requires_grad_(False).eval()
    g=p['geometry'];b=p['budget']
    count=sum(v.numel() for v in carrier.state_dict().values())
    size=sum(v.numel()*v.element_size() for v in carrier.state_dict().values())
    elements=sum(math.prod(s) for s in g.values())
    tokens=sum(math.prod(g[k][2:])//4 for k in ('video_shape','reference_video_shape'))
    tokens+=sum(2*g[k][-1] for k in ('audio_shape','reference_audio_shape'))+g['prompt_shape'][0]
    tokens=((tokens+63)//64)*64
    attention=arch['num_attention_heads']*tokens*tokens
    activation=max(8*tokens*arch['hidden_size'],4*elements)
    needed=2*size+4*(attention+activation)
    require(count<=b['max_parameters'] and elements<=b['max_input_elements']
        and attention<=b['max_attention_elements'] and needed<=b['max_loaded_bytes'],
        'native model/input/attention/working-set capacity exceeded')
    return carrier,dict(parameters=count,weight_bytes=size,host_bytes=needed,
        attention_elements=attention,activation_elements=activation)


def _metadata(p):
    from .h3_cached_inference import _latent_schema
    return dict(schema_version=SCHEMA,scope='CURRENT_NATIVE_LAYOUT_STUDENT_CPU_REFERENCE_LATENT_OUTPUT',
        precision=PRECISION,parameters=deepcopy(p),selection='FROZEN_BASE_CURRENT_UNMERGED_STUDENT_NOT_EMA',
        evidence_kind=p['evidence_kind'],latent_schema=_latent_schema(p['geometry']),
        media_decoder='EXPLICIT_SEPARATE_DISTILLCTL_DECODE',training_resume='NOT_SUPPORTED_INFERENCE_ONLY',
        quality_validation='NOT_RUN')


def _loaded(module,incompatible):
    import torch
    require(not incompatible.missing_keys and not incompatible.unexpected_keys
        and all(not t.is_meta and t.device.type=='cpu' and bool(torch.isfinite(t).all())
                for t in module.state_dict().values()),'complete finite native inference weights required')
    module.restored=True


def _materialize(carrier):
    carrier.to_empty(device='cpu')
    carrier.register_load_state_dict_post_hook(_loaded)
    return carrier


def build_inference_model(value):
    import torch
    from .h3_cached_assets import validate_topology,BACKEND
    from distillschedule.reference import diffsynth_h3_cached as provider
    from .checkpoint import rng_state,restore_rng
    validate_topology()
    p=_request(value)
    before=rng_state(device='cpu')
    try:
        api=provider.load_cpu(backend=BACKEND)
        carrier,sizes=_meta(p,api.model.MiniMaxH3DiT,api.dmd2.H3Adapter)
        admit_host_memory(sizes['host_bytes'],p['budget']['host_reserve_bytes'])
        _materialize(carrier)
        return ModelComponents(state={'student':carrier,'inference':p},metadata=_metadata(p),
            generate=generate,admit_training=refuse_training).validate()
    finally:
        restore_rng(before)


def export_components(value,*,weights,model_class,adapter_class):
    """Native owner's supplied original classes; meta everywhere, CPU on primary.

    No CPU-provider import/admission is performed inside a CUDA training owner.
    A stable serialization carrier keeps the common artifact class identity
    independent of the original module's import namespace.
    """
    import torch
    from .checkpoint import rng_state,restore_rng
    p=_request(value)
    before=rng_state(device='cpu')
    try:
        carrier,_=_meta(p,model_class,adapter_class)
        if weights is not None:
            require(set(weights)==set(p['weight_layout']) and all(
                isinstance(t,torch.Tensor) and t.device.type=='cpu' and list(t.shape)==p['weight_layout'][k]['shape']
                and str(t.dtype)==p['weight_layout'][k]['dtype'] and bool(torch.isfinite(t).all())
                for k,t in weights.items()),'exact consolidated native Student weights required')
            carrier.adapter.load_state_dict(weights,strict=True,assign=True)
        return ModelComponents(state={'student':carrier},metadata=_metadata(p)).validate()
    finally:
        restore_rng(before)


def generate(state,feeds,options):
    import torch
    from .h3_cached_assets import validate_topology,BACKEND
    from .h3_cached_inference import _latent_schema
    from distillschedule.reference import diffsynth_h3_cached as provider
    from distillschedule.data.h3_cached_inputs import decode_row
    require(not options,'native inference has no unbound sampler/geometry overrides')
    validate_topology()
    p,carrier=state['inference'],state['student']
    require(carrier.restored,'native exported weights must be strictly restored before forward')
    require(p['source']==source_identity() and p['runtime']==runtime_identity()
        and not carrier.training and all(not t.requires_grad and t.device.type=='cpu' for t in carrier.parameters()),
        'separate native inference owner/source/runtime changed')
    require(isinstance(feeds,dict) and isinstance(feeds.get('absolute_step'),torch.Tensor)
        and feeds['absolute_step'].shape==(1,),'actual prepared original previous/current AV input required')
    row=decode_row(feeds,absolute_step=feeds['absolute_step'].item(),geometry=p['geometry'])
    api=provider.load_cpu(backend=BACKEND)
    inputs,negative=api.dmd2.prepare_batch(row['raw'],1.0)
    require(negative is None,'original CFG1 condition required')
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):
        (video,audio),_=api.dmd2.rollout(carrier.adapter,(inputs['input_latents'],inputs['audio_input_latents']),
                                      inputs,api.dmd2.make_schedule(original_config(),device='cpu'))
    require(all(t.dtype==torch.float32 and t.device.type=='cpu' and bool(torch.isfinite(t).all())
        and list(t.shape)==p['geometry'][name] for t,name in ((video,'video_shape'),(audio,'audio_shape'))),
        'actual original native AV output geometry/dtype/finite changed')
    return dict(video_latents=video,audio_latents=audio,latent_schema=_latent_schema(p['geometry']))


def refuse_training(*args,**kwargs):
    raise ValueError('portable native H3 Student is inference-only, not a resumable training checkpoint')
