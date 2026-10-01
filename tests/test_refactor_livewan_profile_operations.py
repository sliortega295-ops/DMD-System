from types import SimpleNamespace

import pytest

from distillschedule.runtime.livewan_profile_operations import OperationMarkers, FLAG, RANKS


class Context:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        self.calls.append("enter")

    def __exit__(self, *args):
        self.calls.append("exit")


def test_default_off_never_calls_torch():
    for kind in ("trajectory", "two-cycle-timing", "correctness", "profile"):
        markers = OperationMarkers(run_kind=kind, rank=8, environ={})
        markers.start(None)
        markers.stop()
        assert markers.receipt()["execution_status"] == "DISABLED"
        assert markers.receipt()["entries"] == 0


@pytest.mark.parametrize("kind", ["trajectory", "two-cycle-timing", "timing", "correctness"])
def test_cannot_pollute_native_or_correctness(kind):
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        OperationMarkers(run_kind=kind, rank=8, environ={FLAG: "1"})


@pytest.mark.parametrize("ranks", ["", "8,8", "-1", "16", "a"])
def test_invalid_rank_selection_fails_closed(ranks):
    with pytest.raises(RuntimeError, match="ranks"):
        OperationMarkers(run_kind="profile", rank=8, environ={FLAG: "1", RANKS: ranks})


def test_real_api_lifecycle_selected_and_unselected():
    calls = []

    def emit_nvtx(**kwargs):
        assert kwargs == {"record_shapes": True}
        calls.append("create")
        return Context(calls)

    torch = SimpleNamespace(__version__="2.10.0+cu128",
        autograd=SimpleNamespace(profiler=SimpleNamespace(emit_nvtx=emit_nvtx)))
    marker = OperationMarkers(run_kind="profile", rank=8, environ={FLAG: "1"})
    marker.start(torch)
    with pytest.raises(RuntimeError, match="re-enter"):
        marker.start(torch)
    with pytest.raises(RuntimeError, match="not closed"):
        marker.receipt()
    marker.stop()
    assert calls == ["create", "enter", "exit"]
    assert marker.receipt()["execution_status"] == "PASS"
    assert marker.receipt()["tensor_values_read"] is False
    with pytest.raises(RuntimeError, match="incomplete"):
        marker.stop()
    other = OperationMarkers(run_kind="profile", rank=9, environ={FLAG: "1"})
    other.start(None)
    other.stop()
    assert other.receipt()["execution_status"] == "NOT_APPLICABLE"


def test_version_mismatch_and_invalid_flag():
    marker = OperationMarkers(run_kind="profile", rank=8, environ={FLAG: "1"})
    with pytest.raises(RuntimeError, match="Torch"):
        marker.start(SimpleNamespace(__version__="2.9"))
    with pytest.raises(RuntimeError, match="flag"):
        OperationMarkers(run_kind="profile", rank=8, environ={FLAG: "true"})
