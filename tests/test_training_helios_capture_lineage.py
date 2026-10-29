"""Source-lineage proof only; not upstream execution or numerical qualification."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import helios_capture_source_review as review


def _current_source():
    path = Path(__file__).resolve().parents[1] / review.SOURCE_PATH
    text = path.read_text()
    function = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == "capture_rank")
    return "\n".join(text.splitlines()[function.lineno - 1:function.end_lineno])


def test_versioned_review_preserves_original_and_reviewed_source_identities():
    contract = review.load_review()
    parent = "\n".join(contract["parent_source_lines"])
    assert contract["parent_commit"] == "80d4f16baffccef28cdb9cba8d267645ed34bc14"
    assert hashlib.sha256(parent.encode()).hexdigest() == (
        "536933ea1409fae1893327e216c290bc9cf960178edc70c001f192551e672ea5")
    assert hashlib.sha256(_current_source().encode()).hexdigest() == (
        "ee448184fe2b31e58f3e1d7e73ea831d3bb103c1330027cd1025e11d26a993f5")
    review.assert_nogan_capture_lineage(_current_source())
    review.assert_native_history_ast(parent, _current_source())


_OPTIONS = '''            options = ({"all_history_zero": all(bool(tensors[name].sum() == 0)
                        for name in ("history_short", "history_mid", "history_long")), "rank": rank}
                       if task.native_policy is not None else {})\n'''


@pytest.mark.parametrize("old,new", [
    ("torch.randn(task.latent_shape", "torch.zeros(task.latent_shape"),
    ("stable_block_noise(16 *", "stable_block_noise(32 *"),
    ("tensors[name].sum() == 0", "tensors[name].sum() != 0"),
    ("all(bool(tensors[name]", "any(bool(tensors[name]"),
    ('("history_short", "history_mid", "history_long")',
     '("history_long", "history_mid", "history_short")'),
    ('"rank": rank}', '"rank": 0}'),
    ("task.native_policy is not None else {}", "task.native_policy is None else {}"),
    ("task.native_policy is not None else {}", 'task.native_policy is not None else {"rank": rank}'),
    ("*, rank, output, provenance", "*, rank=0, output, provenance"),
    ('(("generator", "critic") if offset == 0', '(("critic", "generator") if offset == 0'),
    ("step, phase, **options)", "step, phase)"),
    ("step, phase, **options)", "step, phase, **options, extra=True)"),
    (_OPTIONS, ""),
    ('            keys = list(_keys(step, phase).values())\n' + _OPTIONS,
     _OPTIONS + '            keys = list(_keys(step, phase).values())\n'),
    ('            keys = list(_keys(step, phase).values())',
     '            tensors["entry_noise"] *= 2\n            keys = list(_keys(step, phase).values())'),
], ids=["entry-noise-math", "block-noise-math", "zero-predicate", "all-to-any",
        "history-order", "rank-scope", "opt-in-condition", "default-dispatch",
        "default-signature", "phase-order", "missing-dispatch", "extra-dispatch",
        "missing-options", "options-order", "extra-math"])
def test_unreviewed_math_order_zero_or_default_tampering_is_rejected(old, new):
    original = _current_source()
    assert original.count(old) == 1
    changed = original.replace(old, new, 1)
    ast.parse(changed)  # Rejection must not be a syntax-error coincidence.
    parent = "\n".join(review.load_review()["parent_source_lines"])
    # Independently exercise the bounded AST proof, not just the raw SHA gate.
    with pytest.raises(AssertionError):
        review.assert_native_history_ast(parent, changed)
    with pytest.raises(AssertionError):
        review.assert_nogan_capture_lineage(changed)


def test_raw_reviewed_source_hash_remains_required_even_when_ast_is_equal():
    original = _current_source()
    changed = original.replace('    import torch', '    import torch  # not reviewed bytes', 1)
    parent = "\n".join(review.load_review()["parent_source_lines"])
    review.assert_native_history_ast(parent, changed)
    with pytest.raises(AssertionError, match="reviewed capture source SHA"):
        review.assert_nogan_capture_lineage(changed)


@pytest.mark.parametrize("field,value", [
    ("parent_commit", "0" * 40),
    ("parent_source_sha256", "0" * 64),
    ("reviewed_source_sha256", "0" * 64),
    ("parent_source_lines", ["def capture_rank():", "    return None"]),
])
def test_contract_cannot_relabel_the_parent_or_reviewed_source(tmp_path, monkeypatch, field, value):
    contract = deepcopy(review.load_review())
    contract[field] = value
    path = tmp_path / "unreviewed.json"
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(review, "REVIEW_PATH", path)
    with pytest.raises(AssertionError):
        review.assert_nogan_capture_lineage(_current_source())
