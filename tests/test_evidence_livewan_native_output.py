"""Actual tiny original producer/consumer outputs; never native GPU evidence."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch

from test_data_livewan_native_inputs import case, prepare
from test_training_livewan_generator_import import same


@pytest.fixture(scope='module')
def published(case, tmp_path_factory):
    from distillschedule.training import livewan_generator_import as importer, session
    from distillschedule.training.config import load_training_config
    root=tmp_path_factory.mktemp('output-collection-SYNTHETIC_ONLY')
    prepared=prepare(case['path'],root/'prepared')
    importer.import_generator_weights(case['base']/'origin/request.json',root/'weights')
    session.generate_from_export(load_training_config(root/'weights.inference.json'),root/'weights',
        root/'generated',prepared=prepared,output_kind='video')
    return dict(weights=root/'weights',prepared=prepared,generated=root/'generated')


def collect(published, receipt):
    if importlib.util.find_spec('distillschedule.evidence.livewan_native_output') is None:
        # Characterization of the existing marker's EXACT scope: JSON only.
        from distillschedule.workloads.best_known.registry import file_sha256
        root=published['generated']
        assert (root/'_SUCCESS').read_text().strip()==file_sha256(root/'generation.json')
        return None
    from distillschedule.evidence.livewan_native_output import collect_output
    return collect_output(**published,receipt=receipt)


def test_full_producer_consumer_collector_retains_origin_failure(published,tmp_path):
    from distillschedule.training.checkpoint import rng_state
    before=rng_state()
    target=tmp_path/'component.json'
    result=collect(published,target)
    assert same(before,rng_state())
    assert result['status']=='PASS_COMPLETE_PORTABLE_OUTPUT_COMPONENT_ONLY'
    assert len(result['files'])==84 and result['training_cycles_executed']==0
    assert result['weights_initialization']['origin_cycle']==100
    assert result['weights_initialization']['origin_numerical_status']=='FAIL_EXTREME_NUMERICAL_DRIFT'
    assert result['original_owner_acceptance']=='NOT_ASSESSED_NOT_PROMOTED_BY_OUTPUT_COLLECTION'
    assert result['quality_validation']==result['numerical_validation']=='NOT_RUN'
    assert json.loads(target.read_text())==result
    with pytest.raises(ValueError):collect(published,target)


@pytest.mark.parametrize('fault',['png','payload_nonfinite','missing_frame','extra_file','marker','input_identity'])
def test_incomplete_or_changed_output_refused_without_receipt(published,tmp_path,fault):
    from PIL import Image
    import numpy as np
    root=published['generated']; restore=None
    if fault=='png':
        path=root/'frames/frame-000080.png'; old=path.read_bytes()
        image=np.array(Image.open(path),copy=True); image[0,0,0]^=1;Image.fromarray(image).save(path)
        restore=lambda:path.write_bytes(old)
    elif fault=='payload_nonfinite':
        path=root/'generation.pt';old=path.read_bytes()
        values=torch.load(path,weights_only=True); values['generated']['latents'][0,0,-1,0,0]=float('nan')
        torch.save(values,path);restore=lambda:path.write_bytes(old)
    elif fault=='missing_frame':
        path=root/'frames/frame-000080.png';old=path.read_bytes();path.unlink();restore=lambda:path.write_bytes(old)
    elif fault=='extra_file':
        path=root/'unrecorded.txt';path.write_text('NOT_ADMITTED');restore=lambda:path.unlink()
    elif fault=='input_identity':
        from distillschedule.workloads.best_known.registry import file_sha256
        path=root/'generation.json';marker=root/'_SUCCESS';old=path.read_bytes();old_marker=marker.read_bytes()
        value=json.loads(old);value['input_identity']='0'*64;path.write_text(json.dumps(value))
        marker.write_text(file_sha256(path)+'\n')
        def restore():path.write_bytes(old);marker.write_bytes(old_marker)
    else:
        path=root/'_SUCCESS';old=path.read_bytes();path.write_text('0'*64+'\n');restore=lambda:path.write_bytes(old)
    try:
        with pytest.raises(ValueError):collect(published,tmp_path/'must-not-exist.json')
        assert not (tmp_path/'must-not-exist.json').exists()
    finally:
        restore()


def test_symlink_and_input_namespace_receipt_refused(published,tmp_path):
    alias=tmp_path/'alias';alias.symlink_to(published['generated'],target_is_directory=True)
    with pytest.raises(ValueError):collect({**published,'generated':alias},tmp_path/'bad.json')
    with pytest.raises(ValueError):collect(published,published['generated']/'fake-acceptance.json')
    with pytest.raises(ValueError):collect(published,published['prepared'].parent/'fake-acceptance.json')


def test_exact_cpu_collector_cli_in_fresh_process(published,tmp_path):
    target=tmp_path/'cli-component.json'
    command=[sys.executable,'-B','-m','distillschedule.evidence.livewan_native_output','--receipt',str(target)]
    for key,value in published.items():command+=['--'+key,str(value)]
    result=subprocess.run(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},
        capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(target.read_text())['status']=='PASS_COMPLETE_PORTABLE_OUTPUT_COMPONENT_ONLY'
