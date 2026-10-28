"""W5 packed-token to original H3 codec ABI, not W6's sampling algorithm.

Only original packing inverses run here. Codec-owned mean/std, overlap, tiling
and frame planning remain in the separately pinned original H3 decoders.
"""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from distillschedule.workloads.best_known.registry import file_sha256
from .adapters import ModelComponents

SCHEMA='distillschedule.h3_grid_codec_abi.v1'
PACKING_FILE='models/minimax_h3_dit.py'
PACKING_SHA='bd617a4312c70405e916707e1908100b484b2259dfbeba4af7de5bb145206d80'
ABI=dict(schema_version=SCHEMA,video='B_T_HW_PATCHROWS_C_PH_PW_TO_BCTHW_PATCH_1_2_2',
    audio='B_STEREO_CHANNEL_MAJOR_T_ROWS_C32_TO_SCT',
    latent_domain='ORIGINAL_H3_NORMALIZED_VIDEO24_AUDIO32',
    normalization='NO_TRANSFORM_HERE_ORIGINAL_CODEC_MEAN_STD_ONCE',
    timing='ORIGINAL_GRID_FPS24_AUDIO40_LATENTS_PER_SECOND_NO_RESAMPLING')


def require(ok,message):
    if not ok:
        raise ValueError('H3 Grid codec ABI: '+message)


def source_identity():
    from distillschedule.reference import diffsynth_h3_cached as provider
    from . import h3_av_codec,upstream
    return dict(abi=deepcopy(ABI),packing_sha256=PACKING_SHA,packing_bundle=provider.source_identity(),
        consumer_sha256=file_sha256(Path(__file__)),codec_sha256=file_sha256(Path(h3_av_codec.__file__)),
        importer_sha256=file_sha256(Path(upstream.__file__)))


def packing_api():
    """Execute four unchanged pure helpers; no DiT, trainer or W6 sampler."""
    import torch
    from distillschedule.reference import diffsynth_h3_cached as provider
    from .upstream import PinnedSource
    provider.source_identity()
    root=Path(provider.__file__).parent
    path=root/PACKING_FILE
    require(file_sha256(path)==PACKING_SHA,'original packing definitions changed')
    constants=None
    for node in ast.parse(path.read_bytes()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Tuple):
            names=[n.id for n in node.targets[0].elts if isinstance(n,ast.Name)]
            if names==['_PATCH_T','_PATCH_H','_PATCH_W']:
                constants=dict(zip(names,ast.literal_eval(node.value),strict=True))
    require(constants==dict(_PATCH_T=1,_PATCH_H=2,_PATCH_W=2),'original patch ABI changed')
    name=PACKING_FILE[:-3].replace('/','.')
    definitions=['patchify_video','unpatchify_video','pack_audio','unpack_audio']
    lock=PinnedSource(root,{PACKING_FILE:PACKING_SHA},definitions={name:definitions},
        definition_globals={name:dict(torch=torch,**constants)})
    module=lock.import_module(name)
    return SimpleNamespace(**{n:getattr(module,n) for n in definitions}),lock


def admit_binding(manifest,*,source,geometry):
    """Check codec domain/geometry/source before any sampling or model load."""
    from .h3_cached_assets import checked_pin,read_json
    from . import h3_av_codec as codec,h3_grid_inference as grid
    path=checked_pin(manifest)
    p=codec.validate_manifest(read_json(path))
    api=grid.source_api(source)
    shape,_=grid.geometry(api,geometry,text_dim=5120)
    expected=dict(video_latent_shape=[1,24,shape['latent_frames'],shape['latent_height'],shape['latent_width']],
        audio_latent_shape=[2,32,shape['audio_latents']],video_shape=[geometry['frames'],geometry['height'],geometry['width'],3],
        audio_shape=[800*shape['audio_latents'],2],fps=24,sample_rate=32000)
    require(p['geometry']==expected,'original packed/codec frame, stereo, spatial, FPS or normalization domain differs')
    for pins in p['weights'].values():
        for pin in pins:checked_pin(pin)
    codec.source_api(p['source']).lock.validate()
    checked_pin(manifest);api.lock.validate()
    return p,shape


def unpack_generated(generated,shape):
    import torch
    require(type(generated) is dict and {'video_tokens','audio_tokens'} <= set(generated)
        <= {'video_tokens','audio_tokens','latent_schema'},'explicit W5 packed AV tokens required; no mixed W6/native layout')
    if 'latent_schema' in generated:
        expected={name:dict(layout='BLD',shape=list(shape[name]),dtype='torch.float32')
                  for name in ('video_tokens','audio_tokens')}
        require(generated['latent_schema']==expected,'declared packed latent schema/order differs')
    for name in ('video_tokens','audio_tokens'):
        tensor=generated[name]
        require(type(tensor) is torch.Tensor and tensor.device.type=='cpu' and tensor.dtype==torch.float32
            and list(tensor.shape)==list(shape[name]) and not tensor.requires_grad and tensor.grad_fn is None
            and bool(tensor.isfinite().all()),'exact finite detached FP32 packed tensor required: '+name)
    api,lock=packing_api()
    video=api.unpatchify_video(generated['video_tokens'][0],shape['latent_frames'],shape['latent_height'],shape['latent_width'])
    audio=api.unpack_audio(generated['audio_tokens'][0],2,shape['audio_latents'])
    lock.validate()
    return dict(video_latents=video,audio_latents=audio)


def build_decoder(parameters):
    from . import h3_av_codec as codec
    require(type(parameters) is dict and set(parameters)=={'manifest','grid_source','geometry'},
            'explicit original codec manifest and W5 source/geometry required')
    _,shape=admit_binding(parameters['manifest'],source=parameters['grid_source'],geometry=parameters['geometry'])
    comp=codec.build_decoder({'manifest':parameters['manifest']})
    def decode(generated,options):
        require(not options,'packed codec has no shape, scale, crop or precision overrides')
        raw=unpack_generated(generated,shape)
        result=comp.decode(raw,{})
        result['decoder']['packed_input_abi']=deepcopy(ABI)
        result['decoder']['packing_source']=source_identity()
        return result
    return ModelComponents(state=comp.state,metadata={**comp.metadata,'packed_input_abi':deepcopy(ABI),
        'grid_source':deepcopy(parameters['grid_source']),'grid_geometry':deepcopy(parameters['geometry'])},
        decode=decode,close=comp.close).validate()


def decode_generated(generated,*,manifest,source,geometry):
    components=build_decoder(dict(manifest=manifest,grid_source=source,geometry=geometry))
    with components.managed():
        return components.decode(generated,{})
