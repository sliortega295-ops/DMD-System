"""New source cohort -> validated old-backend assets -> existing search campaign.

No models or GPUs are constructed here. A real, rank-matching bootstrap is
required: changing LoRA rank must not silently reset a trained adapter.
"""
from __future__ import annotations

import json
from pathlib import Path

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, write_new_json
from .wan_inputs import WanInputSources, preflight_sources
from .wan_task import OriginalWanTask


def validate_model_assets(task, value):
    from distillschedule.runtime.wan13b_upper_bound import Wan13BBootstrap
    from runtime.gst_pool_contract import NegativePromptAsset
    require(isinstance(value, dict) and set(value) == {'model_root', 'bootstrap', 'negative_prompt'},
            'model assets need exactly model_root/bootstrap/negative_prompt; trace hashes are generated', path='wan.prepare')
    require(all(isinstance(p, str) and p for p in value.values()), 'non-empty local asset paths required', path='wan.prepare')
    paths = {k: str(Path(v).expanduser().resolve()) for k, v in value.items()}
    bootstrap = Wan13BBootstrap.load(paths['bootstrap'], expected_lora_rank=task.lora_rank, verify_payloads=True)
    require(Path(bootstrap.manifest['model_root']).resolve() == Path(paths['model_root']),
            'bootstrap model root mismatch', path='wan.prepare')
    model_dir = Path(paths['model_root'])/'transformer'
    files = [model_dir/'config.json', *sorted(model_dir.glob('*.safetensors')), *sorted(model_dir.glob('*.index.json'))]
    require(len(files) > 1 and all(p.is_file() for p in files) and list(model_dir.glob('*.safetensors')),
            'local native transformer config and weights required', path='wan.prepare')
    negative = NegativePromptAsset.load(paths['negative_prompt'])
    return {**paths, 'expected_bootstrap_hash': bootstrap.manifest_hash,
            'model_files': {p.name: file_sha256(p) for p in files},
            'negative_manifest_hash': negative.manifest['manifest_hash']}


def prepare(task, sources, model_assets, training_config, output, *,
            include_fusion=True, max_launches=8, execute=False, resume=False):
    from .wan_capture import capture, read_training_config
    from .wan_campaign import prepare as prepare_campaign, verify as verify_campaign
    from .wan_planning import source_fingerprint
    from .wan_worker import OriginalWanAssets
    from .wan_trace import load_original_trace
    require(isinstance(task, OriginalWanTask) and isinstance(sources, WanInputSources),
            'typed task and source contract required', path='wan.prepare')
    require(type(include_fusion) is bool and type(max_launches) is int and 2 <= max_launches <= 8,
            'invalid explicit campaign budget/capabilities', path='wan.prepare')
    from .wan_strategy import implementations
    require(len(implementations(task, triton_available=include_fusion))+1 <= max_launches,
            'budget must include each admitted arm and confirmation', path='wan.prepare')
    read_training_config(training_config)
    model = validate_model_assets(task, model_assets)
    inventory = preflight_sources(task, sources)
    identity = {'task':task.to_dict(), 'source_inventory':inventory, 'model_assets':model,
                'training_config_sha256':file_sha256(Path(training_config)), 'source_fingerprint':source_fingerprint(),
                'include_fusion':include_fusion, 'max_launches':max_launches}
    key = fingerprint(identity)
    root = Path(output).resolve()
    if not execute:
        return {'status':'VALIDATED_INPUT_PREPARATION_NOT_EXECUTED', 'identity':identity,
                'gpu_execution':'NOT_RUN', 'preparation_sha256':key}
    if root.exists():
        require(resume, 'preparation output exists; explicit resume required', path='wan.prepare')
        saved = json.loads((root/'preparation.json').read_text())
        require(saved == {'identity':identity, 'preparation_sha256':key},
                'preparation changed; old campaign budget must not be reset', path='wan.prepare')
    else:
        root.mkdir(parents=True)
        write_new_json(root/'preparation.json', {'identity':identity, 'preparation_sha256':key})
    # Exclusive lock avoids two callers racing to capture or create a campaign.
    lock = root/'prepare.lock'
    with lock.open('x') as handle:
        handle.write(key+'\n')
    try:
        capture_root = root/'trace'
        if (capture_root/'capture.json').is_file():
            record = json.loads((capture_root/'capture.json').read_text())
            require(record['status']=='PASS_CAPTURED_ORIGINAL_INPUTS' and record['task']==task.to_dict()
                    and record['input_inventory']==inventory
                    and record['training_config_sha256']==identity['training_config_sha256']
                    and record['source_fingerprint']==identity['source_fingerprint'],
                    'capture differs from prepared task/source', path='wan.prepare')
            load_original_trace(task, capture_root, expected_hash=record['expected_trace_hash'])
        else:
            require(not capture_root.exists(), 'incomplete capture preserved; do not overwrite or resample', path='wan.prepare')
            record = capture(task, sources, training_config, capture_root)
        assets = OriginalWanAssets.from_dict({**{k:model[k] for k in ('model_root','bootstrap','negative_prompt','expected_bootstrap_hash')},
                    'trace_bundle':record['trace_bundle'], 'expected_trace_hash':record['expected_trace_hash'],
                    'schema_version':'distillschedule.original_wan_assets.v1'})
        campaign_path = root/'campaign'/'campaign.json'
        if campaign_path.is_file():
            campaign = verify_campaign(campaign_path)
            require(campaign['task_sha256']==task.digest
                    and json.loads(Path(campaign['assets']).read_text())==assets.__dict__
                    and campaign['max_native_launches']==max_launches,
                    'prepared campaign changed', path='wan.prepare')
        else:
            require(not campaign_path.parent.exists(), 'incomplete campaign preserved; inspect before recovery', path='wan.prepare')
            campaign = prepare_campaign(task, assets, campaign_path.parent,
                                        include_fusion=include_fusion, max_launches=max_launches)
        # Recheck all source bytes after preparation, not only filenames.
        require(validate_model_assets(task, model_assets)==model and sources.inventory()['digest']==inventory['digest']
                and file_sha256(Path(training_config))==identity['training_config_sha256']
                and source_fingerprint()==identity['source_fingerprint'], 'source changed during preparation', path='wan.prepare')
        ready = {'status':'READY_FOR_ORIGINAL_WAN_CAMPAIGN', 'preparation_sha256':key,
                 'campaign':str(campaign_path), 'campaign_sha256':campaign['campaign_sha256'],
                 'task':campaign['task'], 'assets':campaign['assets'],
                 'trace_hash':record['expected_trace_hash'], 'gpu_execution':'NOT_RUN', 'claimable_speedup':False}
        if (root/'ready.json').exists():
            require(json.loads((root/'ready.json').read_text())==ready, 'ready receipt changed', path='wan.prepare')
        else:
            write_new_json(root/'ready.json', ready)
        return ready
    finally:
        lock.unlink()
