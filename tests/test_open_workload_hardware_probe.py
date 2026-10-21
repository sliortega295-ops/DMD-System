"""Hardware recognition, offline-safe probing and provenance checks."""
import pytest
import torch
from distillschedule.open_workload.hardware_probe import (gpu_family, hardware_facts,
    probe, ProbeConfig, SCHEMA)
from distillschedule.open_workload.planning import hardware_identity


@pytest.mark.parametrize('name,sm,expected', [
    ('NVIDIA A100-SXM4-40GB', (8,0), 'NVIDIA_A100_SM80'),
    ('NVIDIA A100-PCIE-80GB', (8,0), 'NVIDIA_A100_SM80'),
    ('NVIDIA H100 80GB HBM3', (9,0), 'NVIDIA_H100_SM90'),
    ('NVIDIA H100 PCIe', (9,0), 'NVIDIA_H100_SM90'),
    ('NVIDIA A100-SXM4-40GB', (9,0), 'OTHER_CUDA_UNVALIDATED'),
    ('NVIDIA H200', (9,0), 'OTHER_CUDA_UNVALIDATED'),
])
def test_hardware_classification_never_invents_a100_h100(name,sm,expected):
    assert gpu_family(name, sm)==expected


def test_cpu_probing_cannot_be_mislabeled_gpu():
    info=hardware_facts(torch.device('cpu'))
    r=probe(torch.device('cpu'), distributed=False, config=ProbeConfig(samples=2))
    assert info['gpu_family']=='CPU_NOT_GPU_EVIDENCE'
    assert r['schema_version']==SCHEMA and r['status']=='NOT_GPU_PROFILE'
    assert r['benchmarks']==[] and r['gpu_execution']=='NOT_RUN'
    assert hardware_identity(torch.device('cpu'))['accelerator']==info


@pytest.mark.parametrize('kwargs', [{'samples':0},{'samples':17},{'warmup':9},
                                     {'transfer_mib':65},{'allreduce_mib':(65,)}])
def test_bounded_probes_fail_closed(kwargs):
    with pytest.raises(ValueError):ProbeConfig(**kwargs)


def test_calibration_probes_are_explicitly_not_gpu_on_cpu(tmp_path):
    from distillschedule.open_workload.planning import calibrate,load_calibration
    from test_open_workload_execution import build
    workload = build(n=1)
    manifest = calibrate(workload, tmp_path/'cpu', max_candidates=1,samples=1,
                         max_program_runs=2,profile_hardware=True)
    assert manifest['hardware_probe']['status']=='NOT_GPU_PROFILE'
    assert manifest['hardware_probe']['ranks'][0]['benchmarks']==[]
    cost_files=list((tmp_path/'cpu').glob('candidate-*/op-*.json'))
    assert cost_files
    import json
    row=json.loads(cost_files[0].read_text())
    assert row['timing_source']=='synchronized_host_wall_ns'
    assert row['gpu_event_diagnostic_ns']==[[None]]
    assert len(load_calibration(tmp_path/'cpu/calibration.json')[1])==1


def test_default_cpu_calibration_does_not_run_gpu_probe(tmp_path,monkeypatch):
    from distillschedule.open_workload.planning import calibrate
    import distillschedule.open_workload.hardware_probe as hp
    from test_open_workload_execution import build
    def forbidden(*args,**kwargs):raise AssertionError('unexpected microprofiling')
    monkeypatch.setattr(hp,'probe',forbidden)
    manifest=calibrate(build(n=1),tmp_path/'no-probe',max_candidates=1,samples=1,max_program_runs=2)
    assert manifest['hardware_probe']=={'status':'NOT_REQUESTED','ranks':[]}
