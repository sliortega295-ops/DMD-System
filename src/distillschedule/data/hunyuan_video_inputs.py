"""Native HunyuanVideo1.5 T2V raw video/frames, prompts or safe native features.

No image condition, feature invention, silent resize/truncation/frame resample.
Reuse existing preparation/source/cursor APIs and common flow tensor validation.
"""
from pathlib import Path

from .manifest import source_identity, validate_source, resolve_path
from .preparation import _publish
from .tensors import tensor_tree
from .flow_image_inputs import _tensor, export_record_sources, export_metadata
from .wan22_inputs import input_identity

SCHEMA = 'distillschedule.hunyuan15_t2v_inputs.v1'


def validate_shape(shape, config):
    from distillschedule.training.flow_image import require
    require(isinstance(shape,list) and len(shape) == 5 and all(type(x) is int and x > 0 for x in shape),
            'explicit positive native BCFHW geometry required')
    require(shape[0] == 1 and shape[1] == config['out_channels'],
            'native T2V requires batch-one matched VAE output channels, not concatenated transformer input channels')
    return shape


def validate_row(row, *, config, media_config, binding, empty):
    import torch
    from distillschedule.training.flow_image import require
    fields = ('context','context_mask','context_2','context_mask_2')
    keys = {'latents','latent_shape','encoder_identity',*fields,*('null_'+name for name in fields)}
    require(isinstance(row,dict) and set(row) == keys, 'complete native T2V dual-text fields required; I2V/causal/unknown conditions refused')
    require(_tensor(row['latent_shape'],dtype=torch.int64,shape=(5,)), 'explicit native BCFHW shape tensor required')
    shape = validate_shape(row['latent_shape'].tolist(),config)
    require(isinstance(row['latents'],dict) and set(row['latents']) == {'video'}
            and _tensor(row['latents']['video'],dtype=torch.float32,shape=shape), 'finite detached native FP32 CPU video payload required')
    for prefix in ('','null_'):
        primary,mask = row[prefix+'context'],row[prefix+'context_mask']
        secondary,mask2 = row[prefix+'context_2'],row[prefix+'context_mask_2']
        require(_tensor(primary,dtype=torch.float32,shape=(1,1000,config['text_embed_dim']))
                and _tensor(mask,dtype=torch.bool,shape=(1,1000)) and bool(mask.any(dim=1).all())
                and bool((mask[:,1:].int() <= mask[:,:-1].int()).all()),
                'actual native primary1000/crop108 nonempty right-padded context/mask required')
        require(_tensor(secondary,dtype=torch.float32,shape=(1,256,config['text_embed_2_dim']))
                and _tensor(mask2,dtype=torch.bool,shape=(1,256))
                and bool((mask2[:,1:].int() <= mask2[:,:-1].int()).all()), 'actual native ByT5 glyph256 context/right-padded mask required')
        require(bool(mask2.any()) or not bool(secondary.any()), 'native no-glyph sentinel must be actual zeros plus allfalse mask')
    require(all(torch.equal(row['null_'+name],value) for name,value in zip(fields,empty)),
            'unconditional dual text is not genuine native empty-prompt encoding')
    require(_tensor(row['encoder_identity'],dtype=torch.uint8,shape=(32,))
            and bytes(row['encoder_identity'].tolist()).hex() == binding, 'native source/codec/tokenizer/dtype binding changed; stale conditions refused')
    return row


def encode_record(record, parameters, *, media=None, config=None):
    import torch
    from distillschedule.training import hunyuan_video as api
    from distillschedule.training.flow_image import require, json_object
    keys = {'mode','source_dir','binding'}
    require(isinstance(parameters,dict) and keys <= parameters.keys() and set(parameters) <= keys|{'assets'}
            and (media is not None or 'assets' in parameters), 'explicit native video encoding source/mode/binding required')
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
            raise ValueError('invalid safe native Hunyuan features') from exc
    else:
        require(isinstance(record.get('prompt'),str), 'real native prompt string required, including explicit empty string')
        fields = ('context','context_mask','context_2','context_mask_2')
        conditions = dict(zip(fields,media.encode_text([record['prompt']])))
        conditions.update(zip(('null_'+key for key in fields),media.encode_text([''])))
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
            latent = torch.zeros(shape,dtype=torch.float32)  # explicit geometry/noise template, never text/image features
        row = dict(latents={'video':latent},**conditions,latent_shape=torch.tensor(shape,dtype=torch.int64),
            encoder_identity=torch.tensor(list(bytes.fromhex(media.binding)),dtype=torch.uint8))
    return validate_row(row,config=config,media_config=media.architecture,binding=media.binding,empty=media.encode_text(['']))


def encode_export_input(state, record, parameters):
    from distillschedule.training import hunyuan_video as api
    role, media = state['student'], state['media']
    sources = export_record_sources(record, parameters, media_kind='video')
    row = encode_record(record, dict(mode=parameters['mode'], source_dir=parameters['source_dir'], binding=media.binding),
                        media=media, config=role.architecture)
    role.validate_feeds(row)
    return dict(features=row, metadata=export_metadata(api, media, parameters, schema=SCHEMA), sources=sources)


def prepare_inputs(source_manifest, output_dir):
    from distillschedule.training import hunyuan_video as api
    from distillschedule.training.flow_image import require, json_object
    from distillschedule.workloads.best_known.registry import fingerprint
    import inspect
    import torch
    path = Path(source_manifest).expanduser().resolve(strict=True)
    manifest = json_object(path)
    require(set(manifest) == {'schema_version','family','assets','mode','records'} and manifest['schema_version'] == SCHEMA
            and manifest['family'] == api.FAMILY and manifest['mode'] in ('raw','prompts','preencoded'),
            'exact native HunyuanVideo1.5 T2V schema/family and input mode required')
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
                and row['id'] and row['id'] not in seen, 'invalid/duplicate native T2V record; I2V/conditioning fields refused')
        seen.add(row['id'])
        if manifest['mode'] == 'prompts':
            origin = source_identity(path)
        else:
            _,origin = input_identity(path.parent,row['video'] if manifest['mode'] == 'raw' else row['features'],video=manifest['mode'] == 'raw')
            if origin['kind'] == 'file': documents.append(origin)
        media_name = {'raw':'video','prompts':'text','preencoded':'features'}[manifest['mode']]
        sources.append({'id':row['id'],'media':{media_name:origin}})
    parameters = dict(assets=str(assets),mode=manifest['mode'],source_dir=str(path.parent),binding=binding)
    code = source_identity(Path(__file__))
    encoder = dict(reference='distillschedule.data.hunyuan_video_inputs:encode_record',sha256=code['sha256'],source_path=code['path'],
        parameters=parameters,parameters_sha256=fingerprint(parameters))
    preencoded = manifest['mode'] == 'preencoded'
    with torch.random.fork_rng(devices=[]):
        result = _publish(output_dir,rows=rows,provenance=dict(mode='preencoded' if preencoded else 'raw',documents=documents,
            sources=sources,encoder=None if preencoded else encoder),metadata=dict(schema_version=SCHEMA,family=api.FAMILY,binding=binding,
            evidence_kind=bundle['evidence_kind'],mode=manifest['mode'],prediction='FLOW_EPSILON_MINUS_X0',gpu_execution='NOT_RUN',quality_validation='NOT_RUN'),
            produce=lambda row:tensor_tree(encode_record(row,parameters,media=media,config=config)))
    for identity in documents: validate_source(identity)
    return result
