"""Hardware microprofiling has a standalone entry with no user model or GPU claims."""
from pathlib import Path
import json
from distillschedule.open_workload.cli import main


def test_hardware_entry_without_factory_does_not_execute(tmp_path,capsys):
    out=tmp_path/'never-created.json'
    assert main(['hardware','--output',str(out)])==0
    assert not out.exists()
    assert 'NOT_EXECUTED' in capsys.readouterr().out


def test_cpu_hardware_probe_is_labeled_not_gpu(tmp_path):
    path=tmp_path/'hardware.json'
    assert main(['hardware','--output',str(path),'--probe-samples','1','--execute'])==0
    doc=json.loads(path.read_text())
    assert doc['status']=='NOT_GPU_PROFILE' and doc['gpu_execution']=='NOT_RUN'
    assert doc['rank_profiles'][0]['hardware']['gpu_family']=='CPU_NOT_GPU_EVIDENCE'
    assert doc['rank_profiles'][0]['benchmarks']==[]


def test_invalid_sample_count_fails_without_creating_output(tmp_path):
    path=tmp_path/'hardware.json'
    assert main(['hardware','--output',str(path),'--probe-samples','0','--execute'])==2
    assert not path.exists()
