"""Installed qualification must not silently swap a validated numerical stack."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def tool():
    spec = importlib.util.spec_from_file_location('check_framework_install', ROOT/'scripts/check_framework_install.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_roots_preserve_overlay_order_without_project_leaks(tmp_path):
    checkout = tmp_path/'checkout'; (checkout/'src').mkdir(parents=True)
    overlay = tmp_path/'torch210'; (overlay/'torch-2.10.dist-info').mkdir(parents=True)
    base = tmp_path/'base'/'site-packages'; base.mkdir(parents=True)
    foreign = tmp_path/'foreign'/'site-packages'; (foreign/'distillschedule').mkdir(parents=True)
    compatibility = tmp_path/'compat'/'site-packages'; (compatibility/'runtime').mkdir(parents=True)
    roots = tool().dependency_roots(checkout, paths=[str(checkout/'src'),str(overlay),str(base),
        str(overlay),str(foreign),str(compatibility),'','import evil',str(base)+'\nimport evil'])
    assert roots == [str(overlay.resolve()), str(base.resolve())]


def test_actual_dependency_origins_must_be_in_the_admitted_roots():
    import pytest
    with pytest.raises(RuntimeError, match='dependency origin'):
        tool().qualified_dependencies([])
