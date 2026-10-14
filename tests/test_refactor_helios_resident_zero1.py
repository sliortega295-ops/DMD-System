"""CPU mechanics only; actual H100 capacity remains a separate gate."""
import importlib.util
from pathlib import Path

import pytest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def runtime():
    path = ROOT / 'src/distillschedule/runtime/helios_resident_zero1.py'
    spec = importlib.util.spec_from_file_location('helios_resident_zero1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_has_full_parameters_and_gpu_only_state():
    c = runtime().resident_zero1_config()
    assert c['train_batch_size'] == 16
    assert c['train_micro_batch_size_per_gpu'] == 2
    assert c['gradient_accumulation_steps'] == 1
    assert c['bf16']['enabled'] is True
    assert c['communication_data_type'] == 'fp32'
    assert c['data_types']['grad_accum_dtype'] == 'fp32'
    assert c['gradient_clipping'] == 1.0
    assert c['zero_optimization']['stage'] == 1
    assert 'offload_optimizer' not in c['zero_optimization']
    assert 'offload_param' not in c['zero_optimization']
    assert 'stage3_prefetch_bucket_size' not in c['zero_optimization']


def test_flat_ema_partition_handles_cross_tensor_boundary_and_padding():
    torch = pytest.importorskip('torch')
    m = runtime()
    tensors = [torch.arange(3, dtype=torch.float32), torch.arange(5, dtype=torch.float32) + 10]
    expected = torch.cat(tensors)
    shards = [m.local_ema_shadow(tensors, partition_rank=r, partition_numel=3,
                              device='cpu') for r in range(3)]
    assert torch.equal(torch.cat(shards)[:8], expected)
    assert shards[-1][-1] == 0
    from distillschedule.runtime.fsdp2_static import ema_update_reference
    for i in range(5):
        source = expected + i
        ema_update_reference(expected, source, 0.5)
        for r, shadow in enumerate(shards):
            local = torch.zeros_like(shadow)
            count = min(3, max(0, 8-r*3))
            local[:count].copy_(source[r*3:r*3+count])
            ema_update_reference(shadow, local, 0.5)
        assert torch.equal(torch.cat(shards)[:8], expected)


def test_runtime_does_not_call_parameter_sharding_or_cpu_offload():
    source = (ROOT / 'src/distillschedule/runtime/helios_resident_zero1.py').read_text()
    assert 'fully_shard(' not in source
    assert 'parameter_streaming' in source
    assert 'restore_zero1_optimizer_partition_from_logical(' in source
    assert 'average' in source


def test_native_bf16_optimizer_api_keeps_real_master_identity():
    m = runtime()
    master = object()
    native = type('BF16_Optimizer', (), {})()
    native.fp32_groups_flat_partition = [master]
    assert m.zero1_master_partitions(SimpleNamespace(optimizer=native))[0] is master
    assert not hasattr(native, 'single_partition_of_fp32_groups')
    with pytest.raises(TypeError):
        m.zero1_master_partitions(SimpleNamespace(optimizer=object()))
