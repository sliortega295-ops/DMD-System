"""CPU provenance checks; the real CUDA roundtrip/update gate is separate."""
import ast
import hashlib
from pathlib import Path

from distillschedule.runtime.helios_official_role_swap import OfficialRoleSwap

ROOT = Path(__file__).resolve().parents[1]


def test_official_manager_class_is_the_exact_audited_fragment():
    source = (ROOT / "runtime/helios_official_low_vram.py").read_text()
    node = next(x for x in ast.parse(source).body if isinstance(x, ast.ClassDef))
    fragment = ast.get_source_segment(source, node)
    assert hashlib.sha256(fragment.encode()).hexdigest() == (
        "a47f363141b284977e4c7424f9a53970c2df829010d80f397fecc74e3a87cc5b")


def test_unregistered_roles_fail_before_any_model_copy():
    # Constructor-independent: importing the frozen Helios runtime is not
    # needed to reject an invalid operation and does not create a CUDA context.
    swap = OfficialRoleSwap.__new__(OfficialRoleSwap)
    swap.engines = {}
    import pytest
    with pytest.raises(ValueError, match="unregistered"):
        swap._registered(object())
