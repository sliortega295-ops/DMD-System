"""Source-lineage proof only; not upstream execution or numerical qualification."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import helios_capture_source_review as review


def _current_source(source_path=review.SOURCE_PATH):
    path = Path(__file__).resolve().parents[1] / source_path
    text = path.read_text()
    function = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == "capture_rank")
    return "\n".join(text.splitlines()[function.lineno - 1:function.end_lineno])


def _v1_source(*, gan=False):
    # Actual pre-aligned source, not a reconstruction or current-source alias.
    return "\n".join(review.load_aligned_review(gan=gan)["parent_source_lines"])


def test_versioned_review_preserves_original_and_reviewed_source_identities():
    contract = review.load_review()
    parent = "\n".join(contract["parent_source_lines"])
    assert contract["parent_commit"] == "80d4f16baffccef28cdb9cba8d267645ed34bc14"
    assert hashlib.sha256(parent.encode()).hexdigest() == (
        "536933ea1409fae1893327e216c290bc9cf960178edc70c001f192551e672ea5")
    previous = _v1_source()
    assert hashlib.sha256(previous.encode()).hexdigest() == (
        "ee448184fe2b31e58f3e1d7e73ea831d3bb103c1330027cd1025e11d26a993f5")
    assert hashlib.sha256(_current_source().encode()).hexdigest() == review.ALIGNED_SHA256
    assert _current_source() != previous
    review.assert_nogan_capture_lineage(previous)
    review.assert_native_history_ast(parent, previous)
    with pytest.raises(AssertionError, match="exactly one reviewed native-policy options assignment"):
        review.assert_native_history_ast(parent, _current_source())
    review.assert_nogan_capture_lineage(_current_source())


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
    original = _v1_source()
    assert original.count(old) == 1
    changed = original.replace(old, new, 1)
    ast.parse(changed)  # Rejection must not be a syntax-error coincidence.
    parent = "\n".join(review.load_review()["parent_source_lines"])
    # Independently exercise the bounded AST proof, not just the raw SHA gate.
    review.assert_native_history_ast(parent, original)
    with pytest.raises(AssertionError):
        review.assert_native_history_ast(parent, changed)
    with pytest.raises(AssertionError):
        review.assert_nogan_capture_lineage(changed)


def test_raw_reviewed_source_hash_remains_required_even_when_ast_is_equal():
    original = _v1_source()
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


def test_gan_review_retains_the_actual_parent_golden_and_only_the_two_opt_in_nodes():
    contract = review.load_gan_review()
    parent = "\n".join(contract["parent_source_lines"])
    assert contract["parent_commit"] == "f85dfda02b7958a383a3391205f4df588ab1dd5b"
    assert hashlib.sha256(parent.encode()).hexdigest() == review.GAN_ORIGINAL_SHA256
    source = _v1_source(gan=True)
    assert hashlib.sha256(source.encode()).hexdigest() == review.GAN_REVIEWED_SHA256
    review.assert_gan_capture_lineage(source)
    review.assert_native_history_ast(parent, source, control_name="controls")
    current = _current_source(review.GAN_SOURCE_PATH)
    assert hashlib.sha256(current.encode()).hexdigest() == review.GAN_ALIGNED_SHA256
    assert current != source
    with pytest.raises(AssertionError, match="exactly one reviewed native-policy options assignment"):
        review.assert_native_history_ast(parent, current, control_name="controls")
    review.assert_gan_capture_lineage(current)


@pytest.mark.parametrize("old,new", [
    ("torch.randn(task.latent_shape", "torch.zeros(task.latent_shape"),
    ("tensors[name].sum() == 0", "tensors[name].sum() != 0"),
    ("all(bool(tensors[name]", "any(bool(tensors[name]"),
    ('"rank": rank}', '"rank": 0}'),
    ("task.native_policy is not None else {}", 'task.native_policy is not None else {"rank": rank}'),
    ("*, rank, output, provenance", "*, rank=0, output, provenance"),
    ("step, phase, **options)", "step, phase)"),
    ("torch.randint(0, 25", "torch.randint(0, 24"),
    (_OPTIONS, ""),
])
def test_gan_unreviewed_sampling_crop_scope_and_default_changes_are_rejected(old, new):
    source = _v1_source(gan=True)
    assert source.count(old) == 1
    changed = source.replace(old, new, 1)
    assert changed != source
    ast.parse(changed)
    parent = "\n".join(review.load_gan_review()["parent_source_lines"])
    review.assert_native_history_ast(parent, source, control_name="controls")
    with pytest.raises(AssertionError):
        review.assert_native_history_ast(parent, changed, control_name="controls")
    with pytest.raises(AssertionError):
        review.assert_gan_capture_lineage(changed)


def test_gan_review_cannot_relabel_the_historical_parent(tmp_path, monkeypatch):
    contract = deepcopy(review.load_gan_review())
    contract["parent_source_sha256"] = contract["reviewed_source_sha256"]
    path = tmp_path / "unreviewed-gan.json"
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(review, "GAN_REVIEW_PATH", path)
    with pytest.raises(AssertionError, match="GAN source review mapping changed"):
        review.assert_gan_capture_lineage(_current_source(review.GAN_SOURCE_PATH))


_ALIGNED_OPTIONS = '''            options = history_control_options(task, zero=all(bool(tensors[name].sum() == 0)
                for name in ("history_short", "history_mid", "history_long")), rank=rank)
'''


def _aligned_parts(gan):
    contract = review.load_aligned_review(gan=gan)
    previous, helper = ("\n".join(contract[key]) for key in ("parent_source_lines", "helper_source_lines"))
    source = _current_source(review.GAN_SOURCE_PATH if gan else review.SOURCE_PATH)
    control = "controls" if gan else "control"
    check = review.assert_gan_capture_lineage if gan else review.assert_nogan_capture_lineage
    # A mutation can earn a rejection only after its unchanged base passes.
    review.assert_aligned_capture_ast(previous, source, helper, control_name=control)
    check(source)
    return previous, source, helper, control, check


@pytest.mark.parametrize("gan", [False, True], ids=["W2", "W3"])
@pytest.mark.parametrize("old,new", [
    ("torch.randn(task.latent_shape", "torch.zeros(task.latent_shape"),
    ("stable_block_noise(16 *", "stable_block_noise(32 *"),
    ("tensors[name].sum() == 0", "tensors[name].sum() != 0"),
    ("all(bool(tensors[name]", "any(bool(tensors[name]"),
    ('("history_short", "history_mid", "history_long")',
     '("history_long", "history_mid", "history_short")'),
    (", rank=rank)", ", rank=0)"),
    ("*, rank, output, provenance", "*, rank=0, output, provenance"),
    ('(("generator", "critic") if offset == 0', '(("critic", "generator") if offset == 0'),
    ("step, phase, **options)", "step, phase)"),
    ("step, phase, **options)", "step, phase, **options, extra=True)"),
    (_ALIGNED_OPTIONS, ""),
    ('            for name, tensor in tensors.items():',
     '            tensors["entry_noise"] *= 2\n            for name, tensor in tensors.items():'),
], ids=["entry-noise", "block-noise", "zero-predicate", "all-to-any", "history-order",
        "rank", "signature", "phase-order", "missing-dispatch", "extra-dispatch",
        "missing-options", "extra-math"])
def test_v2_whole_capture_ast_rejects_unreviewed_changes_after_valid_base(gan, old, new):
    previous, source, helper, control, check = _aligned_parts(gan)
    assert source.count(old) == 1
    changed = source.replace(old, new, 1)
    ast.parse(changed)
    with pytest.raises(AssertionError):
        review.assert_aligned_capture_ast(previous, changed, helper, control_name=control)
    with pytest.raises(AssertionError, match="reviewed capture source SHA"):
        check(changed)


@pytest.mark.parametrize("gan", [False, True], ids=["W2", "W3"])
def test_v2_options_must_be_unique_and_immediately_precede_original_dispatch(gan):
    previous, source, helper, control, _ = _aligned_parts(gan)
    dispatch = f"            {control} = stage_control(native, task, sources.seed, task.index, step, phase, **options)\n"
    original = _ALIGNED_OPTIONS + dispatch
    assert source.count(original) == 1
    for replacement in (dispatch + _ALIGNED_OPTIONS,
                        _ALIGNED_OPTIONS + _ALIGNED_OPTIONS + dispatch,
                        _ALIGNED_OPTIONS + "            pass\n" + dispatch):
        changed = source.replace(original, replacement, 1)
        ast.parse(changed)
        with pytest.raises(AssertionError):
            review.assert_aligned_capture_ast(previous, changed, helper, control_name=control)


def test_v2_gan_crop_distribution_still_requires_original_draw_bounds():
    previous, source, helper, control, check = _aligned_parts(True)
    assert source.count("torch.randint(0, 25") == 1
    changed = source.replace("torch.randint(0, 25", "torch.randint(0, 24", 1)
    ast.parse(changed)
    with pytest.raises(AssertionError, match="actual V1 AST"):
        review.assert_aligned_capture_ast(previous, changed, helper, control_name=control)
    with pytest.raises(AssertionError):
        check(changed)


@pytest.mark.parametrize("gan", [False, True], ids=["W2", "W3"])
@pytest.mark.parametrize("old,new", [
    ("type(zero) is bool", "True"),
    ("task.native_policy is not None", "task.native_policy is None"),
    (" is not None or ", " is not None and "),
    (" == ALIGNED_SCHEDULE_POLICY", ' == "source_normal_only"'),
    ('"all_history_zero":zero', '"all_history_zero":not zero'),
    ('"rank":rank', '"rank":0'),
    ("require(not zero,", "require(True,"),
    ("return {}", 'return {"rank":rank}'),
])
def test_v2_complete_helper_ast_rejects_scope_zero_rank_and_default_mutations(gan, old, new):
    previous, source, helper, control, _ = _aligned_parts(gan)
    assert helper.count(old) == 1
    changed = helper.replace(old, new, 1)
    ast.parse(changed)
    with pytest.raises(AssertionError, match="exact reviewed helper AST"):
        review.assert_aligned_capture_ast(previous, source, changed, control_name=control)
    with pytest.raises(AssertionError, match="helper source SHA"):
        review.assert_history_control_helper(changed)


@pytest.mark.parametrize("gan", [False, True], ids=["W2", "W3"])
def test_v2_raw_function_and_helper_bytes_remain_required_when_ast_is_equal(gan):
    previous, source, helper, control, check = _aligned_parts(gan)
    assert source.count("    import torch") == 1
    changed = source.replace("    import torch", "    import torch  # unreviewed bytes", 1)
    review.assert_aligned_capture_ast(previous, changed, helper, control_name=control)
    with pytest.raises(AssertionError, match="reviewed capture source SHA"):
        check(changed)
    changed_helper = helper.replace("    return {}", "    return {}  # unreviewed bytes", 1)
    review.assert_aligned_capture_ast(previous, source, changed_helper, control_name=control)
    with pytest.raises(AssertionError, match="helper source SHA"):
        review.assert_history_control_helper(changed_helper)


@pytest.mark.parametrize("gan", [False, True], ids=["W2", "W3"])
@pytest.mark.parametrize("field,value", [
    ("parent_commit", "0" * 40),
    ("transition_commit", "0" * 40),
    ("reviewed_source_sha256", "0" * 64),
    ("parent_review_sha256", "0" * 64),
    ("helper_source_sha256", "0" * 64),
    ("reviewed_module_sha256", "0" * 64),
    ("parent_source_lines", ["def capture_rank():", "    return None"]),
    ("helper_source_lines", ["def history_control_options(task, *, zero, rank):", "    return {}"]),
])
def test_v2_review_cannot_relabel_v1_transition_helper_or_current_source(tmp_path, monkeypatch, gan, field, value):
    _, source, _, _, check = _aligned_parts(gan)
    contract = deepcopy(review.load_aligned_review(gan=gan))
    contract[field] = value
    path = tmp_path / "unreviewed-v2.json"
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(review, "GAN_ALIGNED_REVIEW_PATH" if gan else "ALIGNED_REVIEW_PATH", path)
    with pytest.raises(AssertionError):
        check(source)


@pytest.mark.parametrize("gan,source_path,old,new", [
    (False, review.SOURCE_PATH, 'ALIGNED_SCHEDULE_POLICY = "benchmark_v4_strict_global"',
     'ALIGNED_SCHEDULE_POLICY = "source_normal_only"'),
    (False, review.SOURCE_PATH, '    return {}\n', '    return {}  # unreviewed helper bytes\n'),
    (True, review.GAN_SOURCE_PATH, 'history_control_options, ALIGNED_SCHEDULE_POLICY)',
     'history_control_options as unreviewed_options, ALIGNED_SCHEDULE_POLICY)'),
])
def test_v2_source_review_pins_helper_module_constant_and_import_binding(tmp_path, monkeypatch, gan, source_path, old, new):
    _, source, _, _, check = _aligned_parts(gan)
    for relative in (review.SOURCE_PATH, review.GAN_SOURCE_PATH,
                     "docs/contracts/helios_nogan_capture_source_review_v1.json",
                     "docs/contracts/helios_gan_capture_source_review_v1.json"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((review._ROOT / relative).read_bytes())
    monkeypatch.setattr(review, "_ROOT", tmp_path)
    check(source)
    path = tmp_path / source_path
    text = path.read_text()
    assert text.count(old) == 1
    changed = text.replace(old, new, 1)
    ast.parse(changed)
    path.write_text(changed)
    with pytest.raises(AssertionError, match="module source SHA"):
        check(source)
