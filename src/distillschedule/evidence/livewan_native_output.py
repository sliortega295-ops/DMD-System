"""Read-only portable LiveWan output collection, not native-owner acceptance.

Reuses the public export/data/latent/media validators. No model construction,
checkpoint repair, training, CUDA, sampling, or old owner receipt mutation.
"""
from pathlib import Path

SCHEMA = 'distillschedule.livewan_native_output_receipt.v1'


def collect_output(*, weights, prepared, generated, receipt):
    import json
    import os
    import uuid
    import numpy as np
    from PIL import Image
    import torch
    from distillschedule.data import PreparedDataset
    from distillschedule.training import livewan_native_inference as native
    from distillschedule.training.config import digest
    from distillschedule.training.helios_public_assets import read_json
    from distillschedule.training.host_memory import admit_host_memory
    from distillschedule.training.inference import read_export, latent_outputs, decoded_media, _finite_weights
    from distillschedule.training.livewan_public import topology
    from distillschedule.workloads.best_known.registry import file_sha256

    require = native.require
    require(not torch.cuda.is_initialized() and not torch.distributed.is_initialized(),
            'output collection requires an independent CPU process')
    topology()

    def canonical(value):
        path = Path(value).expanduser().absolute()
        require(path == path.resolve() and not any(p.is_symlink() for p in (path, *path.parents)),
                'canonical non-symlink collection paths required')
        return path

    root, target = canonical(generated), canonical(receipt)
    source, data_path = canonical(weights), canonical(prepared)
    require(not target.exists() and not target.is_relative_to(root)
            and not target.is_relative_to(source) and not target.is_relative_to(data_path.parent),
            'fresh receipt outside immutable inputs/output required')
    source, export = read_export(source)
    model, p = export['model'], export['model']['parameters']
    p, shape, _ = native._admit(p)  # original source/runtime/architecture/precision/capacity admission; no model
    budget = p['budget']
    require(model['factory'] == native.FACTORY and model['source'] == dict(
            reference=native.FACTORY, sha256=p['source']['consumer']), 'exact current native factory required')
    dataset = PreparedDataset(data_path)
    require(len(dataset) == 1, 'one explicit new-prompt output per receipt required')
    metadata_path, marker, payload = (root / n for n in ('generation.json', '_SUCCESS', 'generation.pt'))
    files = sorted(path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file())
    expected = ['_SUCCESS', 'generation.json', 'generation.pt'] + [f'frames/frame-{i:06d}.png' for i in range(81)]
    require(files == sorted(expected) and all(not path.is_symlink() for path in root.rglob('*')),
            'exact complete 81-frame generation inventory required')
    require(0 < metadata_path.stat().st_size <= 4*2**20 and 0 < marker.stat().st_size <= 65
            and 0 < payload.stat().st_size <= 512*2**20, 'bounded output files required')
    pins = {name: dict(bytes=(root/name).stat().st_size, sha256=file_sha256(root/name)) for name in files}
    require(marker.read_text().strip() == pins['generation.json']['sha256'], 'original generation marker changed')
    document = read_json(metadata_path)
    require(document['status'] == 'GENERATED' and document['kind'] == 'INFERENCE_FROM_EXPORTED_WEIGHTS'
            and document['output_kind'] == 'video' and document['quality_validation'] == 'NOT_RUN'
            and document['sample_index'] == 0 and document['sample_id'] == dataset.sample_id(0)
            and document['input_identity'] == dataset.identity
            and document['generation_parameters'] == {} and document['inference_seed'] == p['seed']
            and document['export_manifest_sha256'] == file_sha256(source/'export.json')
            and document['checkpoint_cycle'] == export['checkpoint_cycle']
            and document['loaded_factory'] == model['source'] and document['model'] == model
            and document['artifact'] == str(payload), 'generation/export/input provenance disagrees')
    charge = 4*payload.stat().st_size
    require(charge <= budget['max_loaded_bytes'], 'bounded collector payload exceeds original load budget')
    capacity = admit_host_memory(charge, budget['host_reserve_bytes'])
    dataset.validate(sources=True)
    feeds = dataset.sample(0)
    require(set(feeds) == {'world', 'context'} and feeds['world'].dtype == feeds['context'].dtype == torch.float32
            and list(feeds['context'].shape) == [1,512,4096] and feeds['world'].shape[2] in (3,6,9),
            'native world/context inputs required')
    values = torch.load(payload, map_location='cpu', weights_only=True, mmap=True)
    require(type(values) is dict and set(values) == {'generated', 'decoded'}, 'exact generation payload required')
    latent_outputs(values['generated']); _finite_weights(values)
    latent = values['generated'].get('latents')
    height, width = shape['resolution']
    require(type(latent) is torch.Tensor and latent.dtype == torch.float32
            and list(latent.shape) == [1,16,21,height//8,width//8]
            and torch.equal(latent[:,:,:feeds['world'].shape[2]], feeds['world']),
            'complete FP32 latent output or original clean opening differs')
    video = decoded_media(values['decoded'], required_kind='video')['video']
    require(list(video.shape) == [81,height,width,3] and document['video']['shape'] == list(video.shape)
            and document['video']['frames'] == str(root/'frames'), 'full native decoded RGB geometry differs')
    for i, frame in enumerate(video):
        with Image.open(root/'frames'/f'frame-{i:06d}.png') as image:
            require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (width,height)
                    and np.array_equal(np.array(image), frame.numpy()), 'PNG differs from actual decoded tensor')
    require(pins == {name:dict(bytes=(root/name).stat().st_size, sha256=file_sha256(root/name)) for name in files}
            and digest(read_export(source)[1]) == digest(export), 'output/export changed during read-only collection')
    dataset.validate(sources=True)
    result = dict(schema_version=SCHEMA, status='PASS_COMPLETE_PORTABLE_OUTPUT_COMPONENT_ONLY',
        evidence_kind=p['evidence_kind'], geometry=p['geometry'], precision=p['precision'],
        export_manifest_sha256=file_sha256(source/'export.json'), prepared_identity=dataset.identity,
        sample_id=dataset.sample_id(0), checkpoint_cycle=export['checkpoint_cycle'],
        weights_initialization=export['identity'].get('weights_initialization'),
        generated_root=str(root), files=pins, capacity=capacity,
        collector_sha256=file_sha256(Path(__file__)), training_cycles_executed=0,
        original_owner_acceptance='NOT_ASSESSED_NOT_PROMOTED_BY_OUTPUT_COLLECTION',
        numerical_validation='NOT_RUN', quality_validation='NOT_RUN', claimable_speedup=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('.'+target.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, target)  # exclusive publication; no existing receipt can be overwritten
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('weights', 'prepared', 'generated', 'receipt'):
        parser.add_argument('--'+name, required=True)
    print(collect_output(**vars(parser.parse_args()))['status'])


if __name__ == '__main__':
    main()
