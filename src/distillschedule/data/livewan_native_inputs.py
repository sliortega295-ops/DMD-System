"""Explicit clean RGB opening + NEW prepared positive text for native LiveWan.

Only the pinned Wan2.1 codec is loaded, on CPU FP32. No Student, training row,
TeacherLatents crop, resize, negative text, new encoder download or RNG draw.
The output is the existing portable consumer's exact {world, context} ABI.
"""
from copy import deepcopy
from pathlib import Path

from .manifest import source_identity, feature_path
from .preparation import _new_output, _publish

SCHEMA='distillschedule.livewan_native_new_inputs.v1'


def prepare_inputs(source_manifest, output_dir):
    import numpy as np
    from PIL import Image
    import torch
    from .dataset import PreparedDataset
    from .tensors import tensor_tree
    from distillschedule.training import livewan_native_inference as native
    from distillschedule.training import wan_native_inference as wan
    from distillschedule.training.checkpoint import rng_state, restore_rng
    from distillschedule.training.helios_public_assets import checked_pin, read_json, REAL
    from distillschedule.training.host_memory import admit_host_memory
    from distillschedule.training.livewan_public import topology

    _new_output(output_dir)
    native.require(not torch.cuda.is_initialized() and not torch.distributed.is_initialized(),
        'new-input preparation is CPU-only without an active training owner')
    topology()
    native.require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('cuda'),
        'clean opening requires exact CPU FP32 without autocast')
    path=Path(source_manifest).expanduser().resolve(strict=True)
    request_source=source_identity(path)
    request=read_json(path)
    native.require(type(request) is dict and set(request)=={'schema_version','evidence_kind','geometry',
        'budget','codec','positive','opening'} and request['schema_version']==SCHEMA,
        'exact versioned native new-input request required')
    evidence=request['evidence_kind']
    native.require(evidence in (native.NATIVE,native.FIXTURE),'explicit native or tiny fixture evidence required')
    shape=native.geometry(request['geometry'],evidence)
    budget=native.budget(request['budget'])
    positive,opening=request['positive'],request['opening']
    native.require(type(positive) is dict and set(positive)=={'manifest','sample_id'}
        and type(positive['sample_id']) is str and positive['sample_id'].strip(),
        'one exact prepared NEW positive sample id required')
    native.require(type(opening) is dict and set(opening)=={'world_frames','frames'}
        and type(opening['world_frames']) is int and opening['world_frames'] in (3,6,9),
        'explicit supplied world3/6/9; no implicit history crop')
    frames=opening['frames'];count=(opening['world_frames']-1)*4+1
    native.require(type(frames) is list and len(frames)==count
        and len({p.get('path') for p in frames if type(p) is dict})==count,
        'exact distinct declared RGB frame files required; no prefix guessing or repetition')
    native.require(all(type(p) is dict and set(p)=={'path','bytes','sha256'}
        and type(p['bytes']) is int and 0<p['bytes']<=8*2**20 for p in frames),
        'bounded RGB file pins required')
    prompt_pin=positive['manifest']
    native.require(type(prompt_pin) is dict and set(prompt_pin)=={'path','bytes','sha256'}
        and type(prompt_pin['bytes']) is int and 0<prompt_pin['bytes']<=4*2**20,
        'bounded prepared positive manifest pin required')
    prompt_path=checked_pin(prompt_pin);dataset=PreparedDataset(prompt_path)
    ids=[dataset.sample_id(i) for i in range(len(dataset))]
    native.require(ids.count(positive['sample_id'])==1,'selected NEW positive sample is absent or ambiguous')
    index=ids.index(positive['sample_id'])
    spec=dataset.manifest['records'][index]['features']
    feature=feature_path(prompt_path.parent,spec['path'])
    native.require(spec['tensors']==dict(kind='dict',items=dict(prompt_embeds=dict(
        kind='tensor',shape=[1,512,4096],dtype='torch.float32')))
        and 0<feature.stat().st_size<=32*2**20,'bounded exact positive feature header required')
    # Admission precedes even the positive tensor read, not just codec loading.
    capacity=admit_host_memory(budget['max_loaded_bytes'],budget['host_reserve_bytes'])
    row=dataset.sample(index)
    native.require(set(row)=={'prompt_embeds'},'positive-only prepared tensor required; no negative or training envelope')
    context=row['prompt_embeds']
    native.require(type(context) is torch.Tensor and context.device.type=='cpu'
        and context.dtype==torch.float32 and list(context.shape)==[1,512,4096]
        and not context.requires_grad and context.grad_fn is None and bool(context.isfinite().all()),
        'complete detached finite raw FP32 positive context required; no casting or padding')
    height,width=shape['resolution']
    frame_paths=[]
    for ref in frames:
        item=checked_pin(ref)
        with Image.open(item) as image:
            native.require(image.format=='PNG' and image.mode=='RGB' and image.size==(width,height),
                'exact RGB PNG opening geometry required; no convert, resize or implicit crop')
        frame_paths.append(item)
    # Conservatively reserve the entire original bounded consumer load cap;
    # preparation does not introduce a smaller unaccounted codec/activation cap.
    before=rng_state(device='cpu')
    try:
        codec,portable=wan._load_codec(request['codec'],budget,REAL if evidence==native.NATIVE else native.FIXTURE)
        pixels=[]
        for ref,item in zip(frames,frame_paths):
            with Image.open(checked_pin(ref)) as image:
                native.require(image.mode=='RGB' and image.size==(width,height),'opening file changed during read')
                pixels.append(np.array(image,copy=True))
            checked_pin(ref)
        video=torch.from_numpy(np.stack(pixels)).permute(3,0,1,2).unsqueeze(0).float()/127.5-1.
        with torch.no_grad():
            encoded=codec.encode(video).latent_dist.mode()
            mean=torch.tensor(codec.config.latents_mean,dtype=torch.float32).view(1,16,1,1,1)
            std=torch.tensor(codec.config.latents_std,dtype=torch.float32).view(1,16,1,1,1)
            world=((encoded-mean)/std).contiguous()
        native.require(world.dtype==torch.float32 and world.device.type=='cpu'
            and list(world.shape)==[1,16,opening['world_frames'],height//8,width//8]
            and not world.requires_grad and bool(world.isfinite().all()),
            'actual deterministic VAE opening shape/precision/finiteness differs')
        dataset.sample(index);checked_pin(prompt_pin)
        # Keep the EXPECTED pins, not a new hash of a file which might have
        # changed after encode. The existing publisher rechecks every one.
        document=lambda pin:dict(kind='file',path=str(checked_pin(pin)),sha256=pin['sha256'])
        documents=[request_source,source_identity(Path(__file__)),document(prompt_pin),
            dict(kind='file',path=str(feature),sha256=spec['sha256']),*(document(p) for p in frames),
            document(request['codec']['config']),*(document(p) for p in request['codec']['weights'])]
        metadata=dict(schema_version=SCHEMA,evidence_kind=evidence,geometry=request['geometry'],
            positive=deepcopy(positive),positive_metadata=dataset.manifest['metadata'],
            opening=deepcopy(opening),codec=portable,capacity=capacity,
            normalization='RGB/127.5-1 -> Wan2.1 VAE posterior mode -> (latent-latents_mean)/latents_std',
            training_inputs='NOT_READ_NO_TEACHERLATENTS_CROP',training_cycles_executed=0,
            quality_validation='NOT_RUN')
        provenance=dict(mode='preencoded',documents=documents,encoder=None,
            sources=[dict(id=positive['sample_id'],media={'features':source_identity(feature)})])
        return _publish(output_dir,rows=[dict(id=positive['sample_id'])],provenance=provenance,
            metadata=metadata,produce=lambda _:tensor_tree(dict(world=world,context=context)))
    finally:
        restore_rng(before)
