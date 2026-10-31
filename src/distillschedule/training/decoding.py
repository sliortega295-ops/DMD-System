"""Independent prepared-latent decoding using existing adapters/data/artifacts."""
import json
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256
from .plugins import load_callable, resolve_reference

SCHEMA = 'distillschedule.decode_prepared.v1'


def _request(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate decode config field: ' + key)
            value[key] = item
        return value
    p = Path(path).expanduser().resolve()
    raw = p.read_bytes()
    value = json.loads(raw, object_pairs_hook=unique)
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'decoder', 'data', 'output_kind'}
            or value['schema_version'] != SCHEMA or value['output_kind'] not in ('image', 'video', 'audio', 'audio_video')):
        raise ValueError('explicit supported prepared decode config required')
    decoder, data = value['decoder'], value['data']
    if (not isinstance(decoder, dict) or set(decoder) != {'factory', 'parameters'}
            or not isinstance(decoder['parameters'], dict) or not isinstance(data, dict)
            or set(data) != {'prepared', 'sample_index'} or not isinstance(data['prepared'], str)
            or type(data['sample_index']) is not int or data['sample_index'] < 0):
        raise ValueError('explicit decoder factory/parameters and prepared sample required')
    # The existing digest refuses nonfinite/noncanonical JSON before imports.
    from .config import digest
    digest(value)
    return p, value, file_sha256(p)


def decode_prepared(config, output):
    from distillschedule.data import PreparedDataset
    from .adapters import ModelComponents
    from .checkpoint import cpu_tree, rng_state, restore_rng
    from .inference import decoded_media, write_generation
    from .distributed import rank_world
    import os
    import torch
    path, value, config_hash = _request(config)
    if rank_world()[1] != 1 or any(os.environ.get(k, v) != v for k, v in (
            ('WORLD_SIZE', '1'), ('LOCAL_WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0'))):
        raise ValueError('standalone decoding requires one process; no distributed fallback')
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('decode output already exists: ' + str(output))
    dataset = PreparedDataset(path.parent / value['data']['prepared'])
    index = value['data']['sample_index']
    if index >= len(dataset):
        raise ValueError('decode sample index out of range')
    feeds, sample_id = dataset.sample(index), dataset.sample_id(index)
    before = rng_state(device='cpu')
    try:
        factory, source = load_callable(resolve_reference(value['decoder']['factory'], path.parent))
        components = factory(value['decoder']['parameters'])
        if not isinstance(components, ModelComponents):
            raise ValueError('decoder factory must return ModelComponents')
        with components.managed():
            components.validate()
            if components.decode is None:
                raise ValueError('decoder factory has no decode callback')
            with torch.no_grad():
                decoded = cpu_tree(components.decode(feeds, {}))
            decoded_media(decoded, required_kind=value['output_kind'])
            # Refuse changed inputs/config before publishing success. A failed
            # later write retains its partial directory without _SUCCESS.
            if file_sha256(path) != config_hash or dataset.sample_id(index) != sample_id:
                raise ValueError('decode config/input changed during execution')
            dataset._feature(index)  # Recheck only this already admitted sample, not the whole dataset.
            output.mkdir(parents=True, exist_ok=False)
            result = write_generation(output, dict(generated=feeds, decoded=decoded), metadata=dict(
                status='DECODED', kind='DECODE_ONLY_NOT_TRAINING_OR_SAMPLING', sample_id=sample_id,
                sample_index=index, input_identity=dataset.identity, decoder_source=source,
                decoder_metadata=dict(components.metadata), config_sha256=config_hash), output_kind=value['output_kind'])
            (output / '_SUCCESS').write_text(file_sha256(output / 'generation.json') + '\n')
            return result
    finally:
        restore_rng(before)
