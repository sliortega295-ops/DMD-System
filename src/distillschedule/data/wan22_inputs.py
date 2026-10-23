"""Native Wan2.2 T2V raw videos/frames, prompt geometry, or genuine safe features.

No resize, frame resample, cache invention, image conditioning or implicit shape.
Prepared payload hashes, source identities, stable IDs/cursor use existing data APIs.
"""
from pathlib import Path

from .manifest import source_identity, validate_source, resolve_path
from .preparation import _publish
from .tensors import tensor_tree
from .flow_image_inputs import _tensor

SCHEMA = 'distillschedule.wan22_t2v_inputs.v1'


def validate_shape(shape, config):
    from distillschedule.training.flow_image import require
    require(isinstance(shape,list) and len(shape) == 5 and all(type(x) is int and x > 0 for x in shape),
            'explicit positive native BCFHW geometry required')
    b,c,f,h,w = shape
    require(b == 1 and c == config['in_channels'] and h % 2 == w % 2 == 0,
            'native T2V requires batch-one matched VAE channels and even H/W latent geometry')
    require(f <= config['rope_max_seq_len'] and h//2 <= config['rope_max_seq_len'] and w//2 <= config['rope_max_seq_len'],
            'video geometry exceeds actual native RoPE sequence bounds')
    return shape


def validate_row(row, *, config, media_config, binding, empty):
    import torch
    from distillschedule.training.flow_image import require
    keys = {'latents','context','context_mask','null_context','null_context_mask','latent_shape','encoder_identity'}
    require(isinstance(row,dict) and set(row) == keys, 'complete T2V fields required; I2V/causal/experts/unknown conditions refused')
    require(_tensor(row['latent_shape'],dtype=torch.int64,shape=(5,)), 'explicit native BCFHW shape tensor required')
    shape = validate_shape(row['latent_shape'].tolist(),config)
    require(isinstance(row['latents'],dict) and set(row['latents']) == {'video'}
            and _tensor(row['latents']['video'],dtype=torch.float32,shape=shape), 'finite detached native FP32 CPU video payload required')
    for ck,mk in (('context','context_mask'),('null_context','null_context_mask')):
        context,mask = row[ck],row[mk]
        require(_tensor(context,dtype=torch.float32,shape=(1,226,config['text_dim']))
                and _tensor(mask,dtype=torch.bool,shape=(1,226)) and bool(mask.any(dim=1).all())
                and bool((mask[:,1:].int() <= mask[:,:-1].int()).all()),
                'actual native226 UMT5 context/right-padded boolean mask required')
        require(not bool(context[~mask].any()), 'native Wan context padded rows must be actual zeros')
    require(torch.equal(row['null_context'],empty[0]) and torch.equal(row['null_context_mask'],empty[1]),
            'unconditional context is not the genuine native empty-prompt encoding')
    require(_tensor(row['encoder_identity'],dtype=torch.uint8,shape=(32,))
            and bytes(row['encoder_identity'].tolist()).hex() == binding, 'native source/codec/dtype binding changed; stale conditions refused')
    return row


def input_identity(base, spec, *, video=False):
    from distillschedule.training.flow_image import require
    require(isinstance(spec,dict) and set(spec) == {'path','sha256'}, 'input needs explicit local path and SHA256')
    path = resolve_path(spec['path'],base)
    identity = source_identity(path,frames=video and path.is_dir())
    require(identity['sha256'] == spec['sha256'], 'native input SHA256 changed')
    return path,identity


def encode_record(record, parameters, *, media=None, config=None):
    import torch
    from distillschedule.training import wan22 as api
    from distillschedule.training.flow_image import require, json_object
    require(isinstance(parameters,dict) and set(parameters) == {'assets','mode','source_dir','binding'}, 'explicit native video encoding source/mode/binding required')
    if media is None:
        media,_,roots,_,_,_,_,_ = api.load_media_bundle(parameters['assets'])
        config = json_object(roots['student']/'config.json')
    require(media.binding == parameters['binding'], 'native encoding source/codec binding changed')
    mode,base = parameters['mode'],Path(parameters['source_dir'])
    require(mode in ('raw','prompts','preencoded'), 'unsupported native video input mode')
    if mode == 'preencoded':
        path,_ = input_identity(base,record['features'])
        try:
            row = torch.load(path,map_location='cpu',weights_only=True)
        except Exception as exc:
            raise ValueError('invalid safe native Wan features') from exc
    else:
        require(isinstance(record.get('prompt'),str), 'real native prompt string required, including explicit empty string')
        context,mask = media.encode_text([record['prompt']])
        null_context,null_mask = media.encode_text([''])
        if mode == 'raw':
            from .decoding import _video
            path,_ = input_identity(base,record['video'],video=True)
            if path.is_dir():
                from PIL import Image
                from .manifest import frame_paths
                for frame in frame_paths(path):
                    with Image.open(frame) as image:
                        require(image.mode == 'RGB', 'raw frames must be RGB; implicit alpha/color conversion refused')
            latent = media.encode_video(_video(path))
            shape = list(latent.shape)
        else:
            shape = validate_shape(record.get('latent_shape'),config)
            # Explicit geometry/noise template only; never an invented condition.
            latent = torch.zeros(shape,dtype=torch.float32)
        row = dict(latents={'video':latent},context=context,context_mask=mask,null_context=null_context,null_context_mask=null_mask,
            latent_shape=torch.tensor(shape,dtype=torch.int64),encoder_identity=torch.tensor(list(bytes.fromhex(media.binding)),dtype=torch.uint8))
    return validate_row(row,config=config,media_config=media.architecture,binding=media.binding,empty=media.encode_text(['']))


def prepare_inputs(source_manifest, output_dir):
    from distillschedule.training import wan22 as api
    from distillschedule.training.flow_image import require, json_object
    from distillschedule.workloads.best_known.registry import fingerprint
    import inspect
    import torch
    path = Path(source_manifest).expanduser().resolve(strict=True)
    manifest = json_object(path)
    require(set(manifest) == {'schema_version','family','assets','mode','records'} and manifest['schema_version'] == SCHEMA
            and manifest['family'] == api.FAMILY and manifest['mode'] in ('raw','prompts','preencoded'),
            'exact native Wan2.2 T2V schema/family and input mode required')
    assets = resolve_path(manifest['assets'],path.parent)
    media,bundle,roots,_,binding,media_config,_,files = api.load_media_bundle(assets)
    config = json_object(roots['student']/'config.json'); api.validate_architecture(config,media_config)
    rows = manifest['records']; require(isinstance(rows,list) and rows, 'nonempty native video records required')
    documents = [source_identity(path),*(source_identity(file) for file in files)]
    documents += [source_identity(Path(__file__)),source_identity(Path(api.__file__))]
    documents += [source_identity(Path(inspect.getfile(cls))) for cls in api.classes()]
    sources,seen = [],set()
    expected = {'raw':{'id','prompt','video'},'prompts':{'id','prompt','latent_shape'},'preencoded':{'id','features'}}
    for row in rows:
        require(isinstance(row,dict) and set(row) == expected[manifest['mode']] and isinstance(row['id'],str)
                and row['id'] and row['id'] not in seen, 'invalid/duplicate native T2V record; image/experts/conditioning fields refused')
        seen.add(row['id'])
        if manifest['mode'] == 'prompts':
            origin = source_identity(path)
        else:
            _,origin = input_identity(path.parent,row['video'] if manifest['mode'] == 'raw' else row['features'],video=manifest['mode'] == 'raw')
            if origin['kind'] == 'file':
                documents.append(origin)
        media_name = {'raw':'video','prompts':'text','preencoded':'features'}[manifest['mode']]
        sources.append({'id':row['id'],'media':{media_name:origin}})
    parameters = dict(assets=str(assets),mode=manifest['mode'],source_dir=str(path.parent),binding=binding)
    code = source_identity(Path(__file__))
    encoder = dict(reference='distillschedule.data.wan22_inputs:encode_record',sha256=code['sha256'],source_path=code['path'],
        parameters=parameters,parameters_sha256=fingerprint(parameters))
    preencoded = manifest['mode'] == 'preencoded'
    with torch.random.fork_rng(devices=[]):
        result = _publish(output_dir,rows=rows,provenance=dict(mode='preencoded' if preencoded else 'raw',documents=documents,
            sources=sources,encoder=None if preencoded else encoder),metadata=dict(schema_version=SCHEMA,family=api.FAMILY,binding=binding,
            evidence_kind=bundle['evidence_kind'],mode=manifest['mode'],prediction='FLOW_EPSILON_MINUS_X0',gpu_execution='NOT_RUN',quality_validation='NOT_RUN'),
            produce=lambda row:tensor_tree(encode_record(row,parameters,media=media,config=config)))
    for identity in documents:
        validate_source(identity)
    return result
