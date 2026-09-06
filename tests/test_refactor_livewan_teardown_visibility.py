import importlib.util
from pathlib import Path
import pytest


def module():
    p = Path(__file__).resolve().parents[1] / "scripts/distillschedule/summarize_livewan_two_cycle_timing_world16.py"
    s = importlib.util.spec_from_file_location("livewan_teardown_summary", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_partial_publication_waits_without_changing_content(monkeypatch):
    m = module()
    final = {"status": "PASS", "gpu_process_count": "0", "owner_run_id": "run"}
    sequence = iter([{}, {"status": "PASS"}, final])
    monkeypatch.setattr(m, "_manifest", lambda p: next(sequence))
    monkeypatch.setattr(m.time, "sleep", lambda t: None)
    assert m._status(Path("unused"), owner_run_id="run") == final


@pytest.mark.parametrize("observed", [
    {"status": "FAIL"}, {"gpu_process_count": "1"}, {"owner_run_id": "other"},
])
def test_explicit_failure_is_not_retried(monkeypatch, observed):
    m = module()
    monkeypatch.setattr(m, "_manifest", lambda p: observed)
    monkeypatch.setattr(m.time, "sleep", lambda t: pytest.fail("must not retry a known failure"))
    assert m._status(Path("unused"), owner_run_id="run") == observed


def test_missing_receipt_is_not_fabricated(tmp_path):
    m = module()
    assert m._status(tmp_path / "absent", owner_run_id="run", wait_seconds=0) == {}
    assert not (tmp_path / "absent").exists()
