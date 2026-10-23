import json
import wave

import pytest
import torch
from PIL import Image

from distillschedule.data import prepare
from distillschedule.data.dataset import PreparedDataset
from distillschedule.training.inference import write_generation


def test_raw_image_is_encoded_without_resize_and_provenance_is_preserved(tmp_path):
    Image.new('RGB', (12, 8), (20, 30, 40)).save(tmp_path / 'input.png')
    (tmp_path / 'encoder.py').write_text('def encode(record, parameters):\n    return {"pixels": record["image"]}\n')
    (tmp_path / 'raw.json').write_text(json.dumps({
        'schema_version': 'distillschedule.data.raw.v1',
        'encoder': {'factory': 'encoder.py:encode'},
        'records': [{'id': 'image1', 'image': 'input.png'}]}))
    path = prepare(tmp_path / 'raw.json', tmp_path / 'prepared')
    data = PreparedDataset(path)
    assert data.sample(0)['pixels'].shape == (8, 12, 3)
    assert data.manifest['provenance']['sources'][0]['media']['image']['kind'] == 'file'


def test_image_and_audio_generation_produce_real_media(tmp_path):
    image = torch.full((8, 12, 3), 100, dtype=torch.uint8)
    samples = torch.linspace(-.5, .5, 100).reshape(-1, 1)
    values = {'generated': {'latent': torch.zeros(1)},
              'decoded': {'image': image, 'audio': {'samples': samples, 'sample_rate': 16000}}}
    result = write_generation(tmp_path, values, metadata={'status': 'GENERATED'}, output_kind='image')
    assert Image.open(result['image']['path']).size == (12, 8)
    with wave.open(result['audio']['path']) as stream:
        assert stream.getnframes() == 100
        assert stream.getframerate() == 16000
    assert result['output_kind'] == 'image'


def test_media_output_rejects_missing_modality_and_invalid_audio(tmp_path):
    from distillschedule.training.inference import decoded_media
    with pytest.raises(ValueError, match='image'):
        decoded_media({'video': torch.zeros(1, 8, 8, 3, dtype=torch.uint8)}, required_kind='image')
    with pytest.raises(ValueError, match='audio'):
        decoded_media({'audio': {'samples': torch.tensor([[float('nan')]]), 'sample_rate': 1}}, required_kind='audio')


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_low_precision_audio_is_converted_to_pcm_without_overflow(tmp_path, dtype):
    import numpy as np
    samples = torch.tensor([[-1.], [0.], [1.]], dtype=dtype)
    values = {'generated': torch.zeros(1), 'decoded': {
        'audio': {'samples': samples, 'sample_rate': 16000}}}
    result = write_generation(tmp_path, values, metadata={}, output_kind='audio')
    with wave.open(result['audio']['path']) as stream:
        pcm = np.frombuffer(stream.readframes(3), dtype='<i2')
    assert pcm.tolist() == [-32767, 0, 32767]
