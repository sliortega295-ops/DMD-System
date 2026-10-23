"""Synthetic CPU counters; no claimed CUDA or measured GPU capacity."""
from types import SimpleNamespace
import pytest
from distillschedule.strategies.wan_memory import CudaHighwater


class Device:
    index = 0
    def __str__(self): return 'cuda:0'


class Cuda:
    def __init__(self): self.allocated = 10; self.reserved = 20; self.calls = []
    def max_memory_allocated(self, device): return self.allocated
    def max_memory_reserved(self, device): return self.reserved
    def reset_peak_memory_stats(self, device=None):
        self.calls.append(device); self.allocated = 0; self.reserved = 0
        return 'ORIGINAL_RETURN'


def test_all_native_resets_preserve_original_effects_and_return():
    cuda = Cuda(); original = cuda.reset_peak_memory_stats
    tracker = CudaHighwater(cuda, Device()); tracker.start()
    cuda.allocated = 80; cuda.reserved = 100
    assert cuda.reset_peak_memory_stats() == 'ORIGINAL_RETURN'
    cuda.allocated = 40; cuda.reserved = 120
    assert cuda.reset_peak_memory_stats(0) == 'ORIGINAL_RETURN'
    cuda.allocated = 30; cuda.reserved = 50
    result = tracker.close()
    assert result['peak_allocated_bytes'] == 80 and result['peak_reserved_bytes'] == 120
    assert result['native_resets_observed'] == 2 and cuda.calls == [None, 0]
    assert cuda.reset_peak_memory_stats == original
    assert tracker.close() == result


def test_unexpected_device_rejected_without_reset_and_scope_restored():
    cuda = Cuda(); original = cuda.reset_peak_memory_stats
    tracker = CudaHighwater(cuda, Device()); tracker.start()
    try:
        with pytest.raises(RuntimeError, match='unexpected device'):
            cuda.reset_peak_memory_stats(1)
        assert not cuda.calls
        with pytest.raises(RuntimeError, match='already active'): tracker.start()
    finally:
        tracker.close()
    assert cuda.reset_peak_memory_stats == original


def test_worker_default_off_and_recorder_closes_before_state_export():
    import inspect
    from distillschedule.strategies.wan_worker import run_worker
    assert inspect.signature(run_worker).parameters['record_memory_highwater'].default is False
    source = inspect.getsource(run_worker)
    assert source.index('highwater.close()') < source.index('save_final_state(')
    assert 'finally:\n        if highwater is not None:' in source
