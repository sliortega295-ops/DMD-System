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
