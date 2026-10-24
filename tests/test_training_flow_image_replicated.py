"""Actual native tiny FLUX/Qwen Gloo2 acceptance; SYNTHETIC_TEST_ONLY.

Pinned optional native dependencies are required; no substitute architecture or
version is accepted when unavailable. GPU evidence lives in the GPU test file.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def native_replica_case(tmp_path, family, device):
    from native_flow_replicated import request, lifecycle
    repo = Path(__file__).resolve().parents[1]
    recipe = repo/'recipes/flow_image/create_cpu_fixture.py'
    result = subprocess.run([sys.executable,str(recipe),str(tmp_path/'fixture'),'--family',family,'--samples','6'],
        capture_output=True,text=True,timeout=300,
        env={**os.environ,'OMP_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
    assert result.returncode == 0, result.stdout+result.stderr
    files = json.loads(result.stdout)
    config = request(files['config'],tmp_path/'replicated.json',device)
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node','2',
        str(repo/'tests/native_flow_replicated.py'),'oracle','--config',str(config),'--output',str(tmp_path/'oracle')]
    result = subprocess.run(command,capture_output=True,text=True,timeout=600,
        env={**os.environ,'OMP_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    (tmp_path/'oracle.log').write_text(result.stdout+'\n'+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
    proof = json.loads((tmp_path/'oracle/proof.json').read_text())
    assert len(proof['ranks']) == 2 and all(row['average_before_clip_adam'] for row in proof['ranks'])
    assert all(row['oracle_cycles'] == 3 and row['later_student_gradient_nonzero'] for row in proof['ranks'])
    assert all(len(record['cycles']) == 3 for row in proof['ranks'] for record in row['records'])
    lifecycle(config,tmp_path/'lifecycle',ranks=2)


@pytest.mark.skipif(os.environ.get('DMD_MODERN_FLOW_CPU') != '1' or os.environ.get('DMD_NATIVE_FLOW_REPLICATED') != '1',
    reason='native flow Gloo2 NOT_RUN: opt in with pinned stack and DMD_MODERN_FLOW_CPU=1 DMD_NATIVE_FLOW_REPLICATED=1')
@pytest.mark.parametrize('family',['flux2-klein-base','qwen-image'])
def test_actual_native_flow_gloo2_independent_oracle_and_fresh_cli(tmp_path,family):
    native_replica_case(tmp_path,family,'cpu')
