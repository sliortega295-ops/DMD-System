from contextlib import nullcontext
from types import SimpleNamespace
import pytest

from distillschedule.search.async_range_calibration import AsyncRangeCollector


def fake_cuda(complete=True):
    counter = iter(range(100))
    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}
        def record(self, stream):
            assert stream == "existing"
            self.t = next(counter)
        def query(self):
            return complete
        def elapsed_time(self, other):
            return other.t - self.t
    return SimpleNamespace(Event=Event, current_stream=lambda device: "existing",
                           memory_allocated=lambda device: 17,
                           max_memory_allocated=lambda device: 31,
                           max_memory_reserved=lambda device: 35)


def collector(complete=True):
    return AsyncRangeCollector(torch=SimpleNamespace(cuda=fake_cuda(complete)),
        device="cuda:0", rank=0, source_head="a" * 40, implementation="fixture")


def test_nested_occurrences_kept_and_no_sync_api_needed():
    c = collector()
    c.cycle = 1
    with c.range("cycle"):
        for _ in range(5):
            with c.range("critic_update"):
                pass
    result = c.finish()
    assert len(result["rows"]) == 6
    assert [r["parent"] for r in result["rows"]] == [None, 0, 0, 0, 0, 0]
    assert all(r["cycle"] == 1 for r in result["rows"])
    assert all(r["duration_ns"] > 0 for r in result["rows"])
    assert result["extra_synchronizations"] == result["extra_collectives"] == 0
    assert result["formal_timing_eligible"] is False
    assert result["cross_rank_clock_alignment"].startswith("NOT_ESTABLISHED")
    with pytest.raises(ValueError):
        c.finish()


def test_uncompleted_events_fail_closed_no_wait():
    c = collector(False)
    with c.range("work"):
        pass
    with pytest.raises(ValueError, match="terminal synchronization"):
        c.finish()


def test_failed_work_not_promoted():
    c = collector()
    with pytest.raises(RuntimeError):
        with c.range("work"):
            raise RuntimeError("original failure")
    assert c.finish()["status"] == "FAIL"


def test_commit_boundaries_are_timestamp_only_and_preserve_inventory():
    c = collector()
    for cycle in (1, 2):
        c.cycle = cycle
        for step in range(5):
            with c.range('C_update'):
                pass
            c.boundary(cycle=cycle, step=step)
    r = c.finish()
    assert [(b['cycle'], b['step']) for b in r['commit_boundaries']] == [(c,s) for c in (1,2) for s in range(5)]
    assert all(b['peak_allocated_bytes'] == 31 and b['peak_reserved_bytes'] == 35 for b in r['commit_boundaries'])
    assert r['extra_synchronizations'] == r['extra_collectives'] == r['tensor_value_reads'] == 0
