"""Public codec CI must prove its narrow contract, not silently skip private AV."""
from pathlib import Path
import shlex

import yaml

ROOT=Path(__file__).resolve().parents[1]


def job():
    value=yaml.load((ROOT/'.github/workflows/h3-cached-reference-cpu.yml').read_text(),Loader=yaml.BaseLoader)
    return value,value['jobs']['original-codec-contracts']


def test_codec_ci_is_separate_from_cached_math_and_requires_cpu_only_runtime():
    workflow,config=job()
    assert 'original-cached-av' in workflow['jobs']
    assert config['strategy']['matrix']['python']==['3.10','3.12']
    assert config['env']['CUDA_VISIBLE_DEVICES']==''
    assert config['env']['HF_HUB_OFFLINE']=='1'
    assert config['permissions']=={'contents':'read'}
    paths=workflow['on']['pull_request']['paths']
    assert 'tests/test_h3_av_codec.py' in paths and 'tests/test_h3_raw_encoding.py' in paths
    commands='\n'.join(s.get('run','') for s in config['steps'])
    assert 'torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu' in commands
    assert 'git clone' not in commands and 'wget' not in commands and 'huggingface-cli' not in commands


def test_codec_ci_runs_explicit_source_free_proofs_with_existing_no_skip_gate():
    _,config=job()
    command=next(s['run'] for s in config['steps'] if s.get('name')=='Public codec contracts, no private source or weight fixtures')
    tokens=shlex.split(command)
    expected={'tests/test_h3_av_codec.py','tests/test_framework_codec_ci.py',
        'tests/test_h3_raw_encoding.py::test_encoder_contract_is_available_without_importing_original_or_cuda',
        'tests/test_h3_raw_encoding.py::test_encoder_invalid_manifest_fails_before_source_import',
        'tests/test_h3_raw_encoding.py::test_encoder_discovery_imports_no_model_or_torch'}
    assert {t for t in tokens if t.startswith('tests/')}==expected
    assert '--junitxml=h3-codec-contracts.xml' in tokens
    gate=next(s['run'] for s in config['steps'] if s.get('name')=='Require source-free proofs without failures or skips')
    args=shlex.split(gate)
    assert args[:3]==['python','scripts/check_cpu_receipt.py','h3-codec-contracts.xml']
    required={args[i+1] for i,t in enumerate(args) if t=='--require'}
    assert required=={'test_encoder_contract_is_available_without_importing_original_or_cuda',
        'test_encoder_invalid_manifest_fails_before_source_import[nested_bool]',
        'test_encoder_discovery_imports_no_model_or_torch',
        'test_explicit_bf16_storage_loads_only_selected_decoder_keys_exactly',
        'test_fl2va_factory_is_explicit_and_lazy',
        'test_fl2va_wrapper_never_changes_canonical_v1_acceptance',
        'test_fl2va_header_relabels_preserve_tensors_and_only_official_exclusion'}
    upload=next(s for s in config['steps'] if s.get('uses')=='actions/upload-artifact@v4')
    assert upload['if']=='always()' and upload['with']['path']=='h3-codec-contracts.xml'
    assert upload['with']['if-no-files-found']=='error'
