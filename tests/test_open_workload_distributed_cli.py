"""Real torchrun of a user factory; no task-name or winner entries are added."""
import json
import os
from pathlib import Path
import subprocess
import sys
from distillschedule.open_workload.cli import main
from distillschedule.workloads.best_known.registry import repository_root


def test_no_execute_does_not_construct_factory_or_initialize_group(tmp_path,capsys):
    assert main(['describe','--factory','/missing.py:build','--parameters','/missing.json','--distributed','gloo'])==0
    assert 'NOT_EXECUTED' in capsys.readouterr().out


def test_real_torchrun_factory_auto_pipeline(tmp_path):
    root=repository_root();params=tmp_path/'params.json'
    params.write_text(json.dumps({'width':2,'frames':3,'updates':2,'device':'cpu'}))
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(root/'src')}
    result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
        str(root/'scripts/distillctl.py'),'workloads','native','program','auto',
        '--factory',str(root/'examples/open_workloads/replicated_input_gradient.py')+':build',
        '--parameters',str(params),'--output',str(tmp_path/'auto'),
        '--distributed','gloo','--group-timeout','40','--samples','1','--max-candidates','6',
        '--max-program-runs','9','--memory-limit-bytes','1000000','--execute'],
        cwd=root,env=env,text=True,capture_output=True,timeout=75)
    assert result.returncode==0,result.stdout[-5000:]+result.stderr[-10000:]
    summary=json.loads((tmp_path/'auto/result.json').read_text())
    assert summary['status']=='PASS_SELECTED_PROGRAM' and summary['replicas']==2
    assert summary['program_runs']==9
    assert summary['reference_gate']['status']=='PASS'
    assert summary['gpu_execution']=='NOT_RUN'
    calibration=json.loads((tmp_path/'auto/calibration/calibration.json').read_text())
    assert any(v['implementation']['reduction']=='ddp_overlap' for v in calibration['variants'])
    # Planning is metadata-only: it can run WITHOUT torchrun/model construction.
    planned=subprocess.run([sys.executable,str(root/'scripts/distillctl.py'),'workloads','native','program','plan',
        '--calibration',str(tmp_path/'auto/calibration/calibration.json'),'--output',str(tmp_path/'replanned.json'),
        '--memory-limit-bytes','1000000'],cwd=root,env=env,text=True,capture_output=True,timeout=20)
    assert planned.returncode==0,planned.stderr
    assert json.loads((tmp_path/'replanned.json').read_text())['replicas']==2
