"""Lightweight profile discovery: provenance is not a new timing claim."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_latest_official_profiles_keep_distinct_contracts_and_sources():
    from distillschedule.training.official_profiles import describe, resolve
    rows = describe()
    assert set(rows) == {'dmd2-sd15-resident-tree-v1', 'self-forcing-wan-fsdp2-r6-v1'}
    image = resolve('dmd2-sd15-resident-tree-v1')
    video = resolve('self-forcing-wan-fsdp2-r6-v1')
    assert image['origin']['run_id'] == 'official_dmd2_min16_tree_r10'
    assert video['origin']['run_id'] == 'official_sf_teacher_pair_r6'
    assert image['contract']['updates'] == {'generator': 1, 'critic': 10, 'ema': 0}
    assert video['contract']['updates'] == {'generator': 1, 'critic': 5, 'ema': 0}
    assert image['origin']['numerical_status'] == video['origin']['numerical_status'] == 'REVIEW'
    assert all(row['integration_gpu_validation'] == 'NOT_RUN' for row in rows.values())
    assert all(row['performance_guarantee'] is False for row in rows.values())
    assert 'merged_lora' not in json.dumps(rows).lower()
    image['contract']['updates']['critic'] = 99
    assert resolve('dmd2-sd15-resident-tree-v1')['contract']['updates']['critic'] == 10


@pytest.mark.parametrize('profile,launch_sha256,source_sha256', [
    ('dmd2-sd15-resident-tree-v1',
     '8b88a2b518448534219e46ca567044a92697e879ff2319051b8b046734342acc',
     '772fa2ebff6a18cf4db3cd1ff8bc36e0c9856adb03434cc862c8931fcd47ac80'),
    ('self-forcing-wan-fsdp2-r6-v1',
     '11d46be1942299bb903ee57a874087d94779c61c3d324217a1c299893fcf9c0c',
     'baf7d9331b64bcf31cd14b0f6e7d2921d471cbbff9dd529e1fb6fce5138061bc'),
])
def test_profile_origin_distinguishes_launch_and_source_manifests(profile, launch_sha256, source_sha256):
    from distillschedule.training.official_profiles import resolve
    origin = resolve(profile)['origin']
    assert origin['launch_manifest_sha256'] == launch_sha256
    assert origin['source_manifest_sha256'] == source_sha256
    assert launch_sha256 != source_sha256


def test_official_profiles_keep_workload_specific_launch_policies():
    from distillschedule.training.official_profiles import resolve
    assert resolve('dmd2-sd15-resident-tree-v1')['environment'] == {
        'NCCL_ALGO': 'allreduce:Tree', 'NCCL_MIN_CTAS': '16'}
    assert resolve('self-forcing-wan-fsdp2-r6-v1')['environment'] == {'NCCL_CROSS_NIC': '1'}


def test_unknown_or_other_workload_profile_is_rejected():
    from distillschedule.training.official_profiles import resolve
    with pytest.raises(ValueError, match='unknown'):
        resolve('official_dmd2_script_overlap_prefetch_r25')
    with pytest.raises(ValueError, match='model'):
        resolve('dmd2-sd15-resident-tree-v1', model='sdxl')
    with pytest.raises(ValueError, match='model'):
        resolve('self-forcing-wan-fsdp2-r6-v1', model='livewan')


def test_discovery_never_imports_optional_training_or_initializes_cuda():
    script = '''
import sys, json
from distillschedule.training.official_profiles import describe
from distillschedule.training.strategy_catalog import describe as strategies
assert len(describe()) == 2
assert 'official-dmd2-resident' in strategies()['capabilities']
assert 'torch' not in sys.modules and 'triton' not in sys.modules
assert 'diffusers' not in sys.modules and 'transformers' not in sys.modules
print(json.dumps({'profiles': list(describe()), 'optional_modules_loaded': False}))
'''
    result = subprocess.run([sys.executable, '-B', '-c', script],
                            text=True, capture_output=True, timeout=30,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stdout + result.stderr


def test_source_lock_contains_real_helpers_not_experiment_launchers():
    from distillschedule.training.official_profiles import verify_sources
    rows = verify_sources('dmd2-sd15-resident-tree-v1')
    assert rows and all(row['definitions_match_origin'] for row in rows)
    assert {'PackedClassifier', 'UpdateCastScope', 'ActivationRetention'} <= {
        name for row in rows for name in row['definitions']}
    assert all(not Path(row['target']).is_absolute() for row in rows)
    assert all('native_fsdp2_pair' not in row['target'] for row in rows)
