"""Actual source-preparation chain with tiny asset files, never full model weights.

The native dataset/history/scheduler, trace writer/loader, bootstrap and negative
asset validators, campaign builder and resume code are real. The safetensors
weight payload is an explicit file-format fixture; no model is constructed.
"""
import hashlib
import json
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('safetensors')
from test_strategy_wan_capture import native, make_sources, config_file
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.strategies.wan_prepare import prepare
from distillschedule.workloads.best_known.registry import file_sha256


def test_real_feature_capture_to_original_strategy_campaign(tmp_path, native):
    from safetensors.torch import save_file
    from distillschedule.runtime.wan13b_upper_bound import _canonical
    from runtime.gst_pool_contract import write_negative_prompt_asset
    task = OriginalWanTask(video_height=64, video_width=64, video_frames=5,
                           fake_updates=1, first_step=2, lora_rank=1, stage2_inference_steps=(1,2,3))
    sources = make_sources(tmp_path, task)
    model = tmp_path/'model'; (model/'transformer').mkdir(parents=True)
    (model/'transformer'/'config.json').write_text(json.dumps({'fixture': 'FILE_FORMAT_ONLY_NOT_A_TRAINABLE_CHECKPOINT'}))
    save_file({'fixture_weight': torch.ones(1)}, str(model/'transformer'/'fixture.safetensors'))
    bootstrap = tmp_path/'bootstrap'; bootstrap.mkdir()
    components = {}
    for role in ('generator','critic'):
        payload = bootstrap/f'{role}.safetensors'
        save_file({'fixture_adapter':torch.ones(1)},str(payload))
        components[role] = {'trainable_payload':payload.name,'sha256':file_sha256(payload)}
    manifest = {'schema_version':'distillschedule.wan13b_bootstrap.v1', 'block_count':30,
                'lora_rank':1,'model_root':str(model),'components':components,
                'fixture':'ASSET_VALIDATOR_ONLY_NO_MODEL_CONSTRUCTION'}
    manifest['manifest_hash'] = hashlib.sha256(_canonical(manifest)).hexdigest()
    (bootstrap/'manifest.json').write_text(json.dumps(manifest))
    negative = tmp_path/'negative'
    write_negative_prompt_asset(negative,embedding=torch.zeros(1,512,4096,dtype=torch.bfloat16),model_root=model)
    assets = {'model_root':str(model),'bootstrap':str(bootstrap),'negative_prompt':str(negative)}
    training = config_file(tmp_path)
    result = prepare(task,sources,assets,training,tmp_path/'prepared',execute=True)
    assert result['gpu_execution']=='NOT_RUN' and not result['claimable_speedup']
    campaign = json.loads(Path(result['campaign']).read_text())
    specs = [json.loads(Path(job['implementation']).read_text()) for job in campaign['jobs']]
    assert len(specs)==7 and any(s['placement']=='fsdp' for s in specs)
    assert any(s['placement']=='resident' and s['chunk_tokens']==2048 and s['lora_epilogue'] for s in specs)
    assert prepare(task,sources,assets,training,tmp_path/'prepared',execute=True,resume=True)==result
    from dataclasses import replace
    with pytest.raises(ValueError,match='LoRA rank changed'):
        prepare(replace(task,lora_rank=2),sources,assets,training,tmp_path/'wrong',execute=True)
    assert not (tmp_path/'wrong').exists()
