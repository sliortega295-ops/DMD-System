"""The dedicated optional provider lanes require real tests, never skipped stubs."""
import importlib.util
from pathlib import Path
import pytest


def load():
    spec = importlib.util.spec_from_file_location('check_cpu_receipt', Path(__file__).resolve().parents[1]/'scripts/check_cpu_receipt.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_real_cpu_receipt_requires_cases_and_named_proofs(tmp_path):
    path = tmp_path/'tests.xml'
    path.write_text('<testsuites><testsuite><testcase name="native"/><testcase name="restart"/></testsuite></testsuites>')
    assert load().check(path, ['native', 'restart'])['tests'] == 2
    with pytest.raises(ValueError, match='missing'):
        load().check(path, ['absent'])


@pytest.mark.parametrize('body', ['', '<testcase name="native"><failure/></testcase>',
    '<testcase name="native"><error/></testcase>', '<testcase name="native"><skipped/></testcase>'])
def test_empty_failed_or_skipped_receipt_is_not_cpu_qualification(tmp_path, body):
    path = tmp_path/'tests.xml'; path.write_text('<testsuite>'+body+'</testsuite>')
    with pytest.raises(ValueError):
        load().check(path)
