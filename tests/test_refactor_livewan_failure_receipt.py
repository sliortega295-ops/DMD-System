import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def test_failure_receipt_retains_original_exception_and_frame(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/distillschedule/livewan_crossscale_entrypoint.py'
    spec = importlib.util.spec_from_file_location('livewan_failure_entry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('RANK', '8')
    monkeypatch.setenv('LIVEWAN_EXPECTED_HEAD', 'a' * 40)
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'evidence': {'output_dir': str(tmp_path)}}))
    args = SimpleNamespace(plan=plan, run_kind='trajectory')
    try:
        raise RuntimeError('test cublasCreate allocation failure')
    except RuntimeError as error:
        module.write_failure_receipt(args, error)
    value = json.loads((tmp_path / 'failures/rank_008.json').read_text())
    assert value['status'] == 'FAIL'
    assert value['rank'] == 8 and value['head'] == 'a' * 40
    assert 'cublasCreate' in value['error']
    assert value['frames'] and 'traceback' in value
