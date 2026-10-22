"""Finite public GAN reference inputs from the ORIGINAL native producer.

No rollout, denoising timestep, corrupted history, resize, or synthetic fallback.
Five safe tensor-only native rows form one ordinary PreparedDataset record.
"""
from __future__ import annotations
from pathlib import Path
import pickle
from types import SimpleNamespace

from distillschedule.workloads.best_known.registry import repository_root, file_sha256, fingerprint
from distillschedule.training.helios_public_assets import (
    require, read_json, local_file, checked_pin, read_args, EVIDENCE_KINDS, FIXTURE)

SCHEMA = 'distillschedule.helios_native_inputs.v1'
ROW_SCHEMA = 'distillschedule.helios_native_row.v1'
PRECISION = 'CPU_FP32_FROM_NATIVE'
_PRODUCER_SHA = 'a9dd6b69e49b2c5e11684f2fd2275d158c475fe1ca23f9d670291b8878cfb7b4'


def producer_identity():
    from distillschedule.adapters.helios.reference_model import reference_model_identity
    source = reference_model_identity(backend='torch-sdpa')['source']
    path = repository_root()/'third_party/helios_compat/source/helios/dataset/dataloader_dmd.py'
    require(file_sha256(path)==_PRODUCER_SHA, 'original native input producer changed')
    return dict(path=str(path),sha256=_PRODUCER_SHA,source_tree_sha256=source['content_tree_sha256'],
                callable='BucketedFeatureDataset.prepare_stage1_latent', history_sizes=[16,2,1],return_secondary=True)


def identity_tensor(value):
    import torch
    require(isinstance(value,str) and len(value)==64, 'SHA256 identity required')
    try:
        data = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError('invalid SHA256 identity') from exc
    return torch.tensor(list(data),dtype=torch.uint8)


def tensor_identity(value):
    import torch
    require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.uint8
            and value.shape==(32,), 'native identity must be 32 CPU uint8 bytes')
    return bytes(value.tolist()).hex()


def row_identity(*, source_sha256, content_sha256, absolute_step):
    return fingerprint(dict(schema=ROW_SCHEMA,producer=producer_identity(),source_sha256=source_sha256,
                            content_sha256=content_sha256,absolute_step=absolute_step))


def decode_row(row, *, absolute_step):
    """Validate all metadata/content before handing the owner any native work."""
    import torch
    from distillschedule.training.helios_step_owner import _content
    require(isinstance(row,dict) and set(row)=={'identity','absolute_step','source_sha256','content_sha256','inputs'},
            'malformed native row tensor tree')
    index = row['absolute_step']
    require(isinstance(index,torch.Tensor) and index.device.type=='cpu' and index.dtype==torch.int64
            and index.shape==(1,) and index.item()==absolute_step, 'native row reordered or wrong absolute index')
    content = fingerprint(_content(row['inputs']))
    require(content==tensor_identity(row['content_sha256']), 'native prepared row content changed')
    identifier = tensor_identity(row['identity'])
    source = tensor_identity(row['source_sha256'])
    require(identifier == row_identity(source_sha256=source,content_sha256=content,absolute_step=absolute_step),
            'native row/source identity changed')
    return dict(identity=identifier,**row['inputs'])


def _native_feature(record, geometry, evidence_kind):
    import torch
    path = checked_pin(record['features'])
    try:
        value = torch.load(path,map_location='cpu',weights_only=True)
    except (pickle.UnpicklingError,RuntimeError,TypeError,ValueError,OSError,EOFError) as exc:
        raise ValueError('cannot safely read native feature: ' + str(path)) from exc
    require(isinstance(value,dict) and {'vae_latent','prompt_embed','prompt_raw'} <= set(value)
            and set(value) <= {'vae_latent','prompt_embed','prompt_raw','evidence_kind'},
            'native feature needs original vae_latent/prompt_embed/prompt_raw fields')
    require(isinstance(value['prompt_raw'],str), 'native prompt_raw must be a string')
    if evidence_kind==FIXTURE or 'evidence_kind' in value:
        require(value.get('evidence_kind')==evidence_kind, 'fixture evidence cannot be relabeled as real data')
    vae, prompt = value['vae_latent'],value['prompt_embed']
    noise, condition = geometry['noise_shape'],geometry['prompt_shape']
    require(isinstance(vae,torch.Tensor) and vae.ndim==5 and vae.shape[0]>0
            and list(vae.shape[1:])==noise[1:] and isinstance(prompt,torch.Tensor)
            and list(prompt.shape)==condition[1:], 'native feature geometry differs; no resize/projection')
    require(all(x.device.type=='cpu' and x.dtype in (torch.float32,torch.bfloat16,torch.float16)
            and not x.requires_grad and x.grad_fn is None and x.isfinite().all().item() for x in (vae,prompt)),
            'native feature must be finite detached floating CPU tensors')
    checked_pin(record['features'])
    # This explicit precision contract alone permits the CPU conversion.
    return vae.to(torch.float32), prompt.to(torch.float32)


def prepare_inputs(source_manifest, output):
    import torch
    from .preparation import _new_output, _publish
    from .manifest import source_identity
    from .tensors import tensor_tree
    from distillschedule.training.helios_step_owner import _content
    _new_output(output)
    path = local_file(str(Path(source_manifest).absolute()))
    source_pin = source_identity(path)
    value = read_json(path)
    require(isinstance(value,dict) and set(value)=={'schema_version','evidence_kind','producer','precision',
        'args','geometry','records'} and value['schema_version']==SCHEMA, 'native source manifest fields/schema changed')
    require(value['evidence_kind'] in EVIDENCE_KINDS and value['producer']==producer_identity()
            and value['precision']==PRECISION, 'native producer/evidence/precision contract changed')
    args = read_args(value['args'])
    c = args.training_config
    require(list(c.history_sizes)==[16,2,1] and c.is_use_gt_history, 'original producer requires history [16,2,1]')
    geometry = value['geometry']
    require(isinstance(geometry,dict) and set(geometry)=={'noise_shape','prompt_shape'}
            and len(geometry['noise_shape'])==5 and geometry['noise_shape'][:2]==[1,16]
            and geometry['noise_shape'][2]==c.num_critic_input_frames
            and len(geometry['prompt_shape'])==3 and geometry['prompt_shape'][0]==1,
            'invalid original native input geometry')
    records = value['records']
    require(isinstance(records,list) and records and len(records)%5==0 and len(records)<=c.max_train_steps,
            'finite complete five-native-row inventory within fixed scheduler horizon required')
    # Import the intact original module, not a copied producer or unsafe loader.
    from distillschedule.training.plugins import load_callable
    BucketedFeatureDataset, imported = load_callable(value['producer']['path'] + ':BucketedFeatureDataset')
    require(imported['sha256']==_PRODUCER_SHA
            and Path(BucketedFeatureDataset.prepare_stage1_latent.__code__.co_filename).resolve()
            == Path(value['producer']['path']), 'native producer import binding changed')
    trees, metadata, seen_ids, seen_paths, seen_bytes, seen_content = [],[],set(),set(),set(),set()
    for index,record in enumerate(records):
        require(isinstance(record,dict) and set(record)=={'id','features','seed','epoch','native_index'}
                and isinstance(record['id'],str) and record['id'] and record['id'] not in seen_ids
                and record['native_index']==index and type(record['native_index']) is int
                and all(type(record[k]) is int and 0<=record[k]<2**63 for k in ('seed','epoch')),
                'native record IDs/order/seed/epoch changed')
        source = checked_pin(record['features'])
        vae,prompt = _native_feature(record,geometry,value['evidence_kind'])
        raw_content = fingerprint(_content(dict(vae_latent=vae,prompt_embed=prompt)))
        require(str(source) not in seen_paths and record['features']['sha256'] not in seen_bytes
                and raw_content not in seen_content, 'copied/reused native source is not fresh input')
        seen_ids.add(record['id']); seen_paths.add(str(source)); seen_bytes.add(record['features']['sha256']); seen_content.add(raw_content)
        producer = SimpleNamespace(base_seed=record['seed'],_epoch=record['epoch'])
        primary,secondary = BucketedFeatureDataset.prepare_stage1_latent(producer,vae,index,return_secondary=True)
        def history(tensors):
            x0,h,t = tensors
            return dict(gt_history_latents=h.unsqueeze(0),gt_target_latents=t.unsqueeze(0),gt_x0_latents=x0.unsqueeze(0))
        inputs = dict(prompt_embeds=prompt.unsqueeze(0),gan_prompt_embeds=prompt.unsqueeze(0),
            gan_vae_latents=primary[2].unsqueeze(0),generator_history=history(primary),critic_history=history(secondary))
        content = fingerprint(_content(inputs))
        identifier = row_identity(source_sha256=record['features']['sha256'],content_sha256=content,absolute_step=index)
        row = dict(identity=identity_tensor(identifier),absolute_step=torch.tensor([index],dtype=torch.int64),
            source_sha256=identity_tensor(record['features']['sha256']),content_sha256=identity_tensor(content),inputs=inputs)
        trees.append(row)
        # Section selection matches the producer's local deterministic generators.
        seed = record['seed']+record['epoch']*1000000+index
        selected = [torch.randint(0,vae.shape[0],(1,),generator=torch.Generator().manual_seed(seed+offset)).item()
                    for offset in (0,999999)]
        metadata.append(dict(id=record['id'],absolute_step=index,source=record['features'],
            raw_content_sha256=raw_content,identity=identifier,content_sha256=content,
            primary_section=selected[0],secondary_section=selected[1],seed=record['seed'],epoch=record['epoch']))
    rows = [dict(id=f'native-cycle-{i//5:06d}',metadata=dict(cycle=i//5,native_rows=metadata[i:i+5]))
            for i in range(0,len(trees),5)]
    provenance = dict(mode='preencoded',documents=[source_pin,source_identity(checked_pin(value['args'])),
                        source_identity(Path(value['producer']['path']))] +
                        [source_identity(Path(item['source']['path'])) for item in metadata],
        sources=[dict(id=row['id'],media={'features':source_pin}) for row in rows],encoder=None)
    prepared_metadata = dict(schema_version=SCHEMA,row_schema=ROW_SCHEMA,evidence_kind=value['evidence_kind'],
        producer=value['producer'],precision=PRECISION,args=value['args'],geometry=geometry,native_steps=len(trees),
        quality_validation='NOT_RUN',claimable_speedup=False)
    return _publish(output,rows=rows,provenance=provenance,metadata=prepared_metadata,
        produce=lambda record:tensor_tree({'native_rows':trees[record['metadata']['cycle']*5:record['metadata']['cycle']*5+5]}))
