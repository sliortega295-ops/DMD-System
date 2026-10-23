"""End-user unknown role factory goes through common measured pool search, not recipes.

This is an actual torchrun Gloo / P2P / optimizer test. It neither allocates GPUs
nor claims CUDA/NCCL correctness or performance.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from distillschedule.open_workload.role_cli import load_factory, main
from distillschedule.workloads.best_known.registry import repository_root


ROOT = repository_root()
FACTORY = ROOT / 'examples/open_workloads/unseen_score_pipeline.py'
PARAMETERS = ROOT / 'examples/open_workloads/unseen_score_pipeline.json'


def test_unseen_factory_changes_real_graph_not_best_plan(capsys):
    original=json.loads(PARAMETERS.read_text())
    t,a,receipt=load_factory(str(FACTORY)+':build',original)
    alternative={**original,'frames':original['frames']+1,'microbatches':original['microbatches']+1}
    t2,b,receipt2=load_factory(str(FACTORY)+':build',alternative)
    assert t.program().digest!=t2.program().digest
    assert t.bytes_per_latent!=t2.bytes_per_latent
    assert receipt['factory_sha256']==receipt2['factory_sha256']
    assert receipt['parameters_sha256']!=receipt2['parameters_sha256']
    assert a.role_storage_estimate_bytes==b.role_storage_estimate_bytes
    assert not any('placement' in key or 'pool' in key or 'window' in key for key in original)
    assert main(['role-describe','--factory',str(FACTORY)+':build','--parameters',str(PARAMETERS),'--execute'])==0
    doc=json.loads(capsys.readouterr().out)
    assert doc['task']['microbatches']==original['microbatches']
    assert doc['task']['microbatches']==t.microbatches
    assert len(doc['program']['calls']) == len(t.program().calls)


def test_explicit_execution_required_no_allocations(tmp_path,capsys):
    out=tmp_path/'not-started'
    assert main(['role-auto','--factory',str(FACTORY)+':build','--parameters',str(PARAMETERS),
                 '--output',str(out),'--distributed','gloo'])==0
    assert json.loads(capsys.readouterr().out)['status']=='NOT_EXECUTED'
    assert not out.exists()


def invoke_torchrun(tmp_path, args, timeout=100):
    env={**os.environ,'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT),
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    command=[sys.executable,'-m','torch.distributed.run','--standalone',
             '--nproc-per-node=2',str(ROOT/'scripts/distillctl.py'),
             'workloads','native','program',*args]
    return subprocess.run(command,cwd=ROOT,env=env,text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)


def test_unknown_role_factory_real_multirank_auto_then_rejects_changed_task(tmp_path):
    output=tmp_path/'role-campaign'
    standard=['--factory',str(FACTORY)+':build','--parameters',str(PARAMETERS),
              '--distributed','gloo','--memory-limit-bytes','10000000',
              '--host-memory-limit-bytes','10000000','--execute']
    run=invoke_torchrun(tmp_path,['role-auto',*standard,'--output',str(output),
                                  '--max-candidates','3','--samples','1','--max-native-runs','4'])
    assert run.returncode==0,run.stdout[-7000:]
    result=json.loads((output/'result.json').read_text())
    calibration=json.loads((output/'calibration/calibration.json').read_text())
    plan=json.loads((output/'plan.json').read_text())
    receipt=json.loads((output/'selected/execution.json').read_text())
    assert result['status']=='EXECUTED_ROLE_POOL_NEEDS_INDEPENDENT_REFERENCE'
    assert result['gpu_execution']=='NOT_RUN' and result['claimable_speedup'] is False
    assert len(calibration['candidates'])==3 and calibration['budget']['calibration_runs']==3
    assert plan['selected']['implementation']==result['selected_implementation']
    assert plan['selected']['implementation']['window']>=1
    assert {r['role'] for r in receipt['ranks']}=={'generator','score'}
    assert all(r['host_leases_closed'] for r in receipt['ranks'])
    # Public saved-plan execution, not a private executor call: selected model
    # and optimizer survive three explicit fresh-cohort cycles.
    continuing=tmp_path/'continuing'
    again=invoke_torchrun(tmp_path,['role-run',*standard,'--plan',str(output/'plan.json'),
                                  '--cycles','3','--output',str(continuing)])
    assert again.returncode==0,again.stdout[-7000:]
    final=json.loads((continuing/'execution.json').read_text())
    assert final['cycles_executed']==3 and final['final_versions']['generator']==3
    assert final['training_state']=='PERSISTED_IN_PROCESS_NOT_CHECKPOINT_RESTART'
    assert [r['initial_versions']['generator'] for r in final['cycle_summaries']]==[0,1,2]
    assert len(list(continuing.glob('cycle-*.json')))==3
    # Changing model/shape parameters invalidates the measured task, not a
    # reason to reuse another workload's observed fastest layout.
    new={**json.loads(PARAMETERS.read_text()),'frames':4}
    new_path=tmp_path/'different-parameters.json';new_path.write_text(json.dumps(new))
    wrong=invoke_torchrun(tmp_path,['role-run','--factory',str(FACTORY)+':build',
        '--parameters',str(new_path),'--distributed','gloo',
        '--memory-limit-bytes','10000000','--host-memory-limit-bytes','10000000',
        '--plan',str(output/'plan.json'),'--output',str(tmp_path/'replay'), '--execute'])
    assert wrong.returncode!=0 and ('changed' in wrong.stdout or 'mismatch' in wrong.stdout),wrong.stdout[-7000:]
    assert not (tmp_path/'replay/execution.json').exists()
