from dataclasses import replace
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
from distillschedule.strategies.wan_inputs import WanInputSources, load_feature, preflight_sources
from distillschedule.strategies.wan_task import OriginalWanTask


def sources(tmp_path, task):
    files = []
    for rank in range(16):
        path = tmp_path / f'{rank}.pt'
        torch.save({'vae_latent': torch.ones((2, *task.latent_shape[1:])) * (rank+1),
                    'prompt_embed': torch.ones(1, 2, 4096),
                    'attention_mask': torch.tensor([[1, 0]])}, path)
        files.append(str(path))
    return WanInputSources(tuple(files))


def test_preflight_uses_new_real_feature_bytes(tmp_path):
    task = OriginalWanTask(video_height=64, video_width=64, video_frames=5, fake_updates=1, first_step=2)
    src = sources(tmp_path, task)
    report = preflight_sources(task, src)
    assert report['status'] == 'PASS_SOURCE_FEATURES'
    assert len(report['files']) == 16
    assert WanInputSources.from_dict(src.to_dict()) == src
    with pytest.raises(ValueError, match='re-encode'):
        preflight_sources(replace(task, video_height=128), src)


def test_no_missing_mask_or_mutated_feature_fallback(tmp_path):
    task = OriginalWanTask(video_height=64, video_width=64, video_frames=5)
    src = sources(tmp_path, task); row = src.inventory()['files'][0]
    path = Path(row['path']); data = torch.load(path, weights_only=True)
    del data['attention_mask']; torch.save(data, path)
    with pytest.raises(ValueError, match='changed'):
        load_feature(task, row)
    with pytest.raises(ValueError, match='attention_mask'):
        load_feature(task, src.inventory()['files'][0])


def test_no_duplicate_sources_and_no_unknown_contract_fields():
    with pytest.raises(ValueError, match='distinct'):
        WanInputSources(('same.pt',)*16)
    with pytest.raises(ValueError, match='unknown'):
        WanInputSources.from_dict({'features': [f'{i}.pt' for i in range(16)], 'resize': True})
