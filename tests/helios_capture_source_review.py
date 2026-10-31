"""Versioned native-history source reviews; not numerical-equivalence gates.

The parent snapshot is the actual 80d4f16 capture source, not a reconstructed
"expected" function. Its original golden remains pinned independently of the
reviewed opt-in SHA. No production source is executed or rewritten here.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

SOURCE_PATH = "src/distillschedule/training/helios_inputs.py"
REVIEW_PATH = (Path(__file__).resolve().parents[1] / "docs/contracts/"
               "helios_nogan_capture_source_review_v1.json")
ORIGINAL_SHA256 = "536933ea1409fae1893327e216c290bc9cf960178edc70c001f192551e672ea5"
REVIEWED_SHA256 = "ee448184fe2b31e58f3e1d7e73ea831d3bb103c1330027cd1025e11d26a993f5"
_MAPPING = {
    "schema_version": "distillschedule.helios_nogan_capture_source_review.v1",
    "source_path": SOURCE_PATH,
    "function": "capture_rank",
    "parent_commit": "80d4f16baffccef28cdb9cba8d267645ed34bc14",
    "parent_git_blob": "69f612d713c91d0d7d04b741edfe7627579bf75f",
    "parent_module_sha256": "51965914dfde3e926faba13da0dc62f54e2b5c41021c01895446a92373b58c0e",
    "parent_source_sha256": ORIGINAL_SHA256,
    "reviewed_source_sha256": REVIEWED_SHA256,
    "hash_scope": 'inspect.getsource(function).rstrip("\\n").encode("utf-8")',
}
# Exact reviewed AST nodes, not arbitrary "options" assignments/keyword removals.
_OPTIONS = ast.parse('''options = ({"all_history_zero": all(bool(tensors[name].sum() == 0)
    for name in ("history_short", "history_mid", "history_long")), "rank": rank}
    if task.native_policy is not None else {})''').body[0]
_DEFAULT_DISPATCH = ast.parse(
    "control = stage_control(native, task, sources.seed, task.index, step, phase)").body[0]
_NATIVE_DISPATCH = ast.parse(
    "control = stage_control(native, task, sources.seed, task.index, step, phase, **options)").body[0]


def _sha(source):
    return hashlib.sha256(source.rstrip("\n").encode("utf-8")).hexdigest()


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _capture_ast(source):
    tree = ast.parse(source)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert tree.body[0].name == "capture_rank"
    return tree


def load_review():
    review = json.loads(REVIEW_PATH.read_text())
    assert {key: review.get(key) for key in _MAPPING} == _MAPPING, "source review mapping changed"
    lines = review["parent_source_lines"]
    assert isinstance(lines, list) and all(isinstance(line, str) for line in lines)
    assert _sha("\n".join(lines)) == ORIGINAL_SHA256, "actual parent capture source SHA changed"
    return review


def assert_native_history_ast(parent_source, reviewed_source, *, control_name="control"):
    """Remove exactly the two reviewed, adjacent opt-in nodes, nothing else.

    The remaining whole function (signature, math, loops, RNG ordering, default
    dispatch and contract included) must equal the pinned actual parent AST.
    Exact matching also pins the all/zero predicate, rank and empty default dict.
    """
    assert control_name in ("control", "controls")
    default_dispatch, native_dispatch = deepcopy(_DEFAULT_DISPATCH), deepcopy(_NATIVE_DISPATCH)
    default_dispatch.targets[0].id = native_dispatch.targets[0].id = control_name
    parent, actual = _capture_ast(parent_source), _capture_ast(reviewed_source)
    matches = []
    for owner in ast.walk(actual):
        for _, value in ast.iter_fields(owner):
            if isinstance(value, list):
                matches.extend((value, index) for index, item in enumerate(value)
                               if isinstance(item, ast.Assign) and _dump(item) == _dump(_OPTIONS))
    assert len(matches) == 1, "exactly one reviewed native-policy options assignment required"
    body, index = matches[0]
    assert index + 1 < len(body) and _dump(body[index + 1]) == _dump(native_dispatch), (
        "reviewed options must immediately precede the exact **options stage_control dispatch")
    del body[index]
    body[index] = default_dispatch
    assert _dump(actual) == _dump(parent), "capture default/math/order differs from actual parent AST"


def assert_nogan_capture_lineage(source):
    review = load_review()
    assert _sha(source) == REVIEWED_SHA256, "reviewed capture source SHA changed; new review required"
    assert_native_history_ast("\n".join(review["parent_source_lines"]), source)


GAN_SOURCE_PATH = "src/distillschedule/training/helios_gan_inputs.py"
GAN_REVIEW_PATH = (Path(__file__).resolve().parents[1] / "docs/contracts/"
                   "helios_gan_capture_source_review_v1.json")
GAN_ORIGINAL_SHA256 = "8439fe7e9692ccc71297826a40b812af866a3e96ab14caa9ee36aa797fe3d24f"
GAN_REVIEWED_SHA256 = "9536e56bf06ffa8945e9e9fd675aad14e818d70af2bd2251afa1b1d862af1a0a"
_GAN_MAPPING = {
    "schema_version": "distillschedule.helios_gan_capture_source_review.v1",
    "source_path": GAN_SOURCE_PATH,
    "function": "capture_rank",
    "parent_commit": "f85dfda02b7958a383a3391205f4df588ab1dd5b",
    "parent_git_blob": "c62e48446eb4683cfb5e62c29da12d261300b2c6",
    "parent_module_sha256": "e2a61ec3d4e222588fbb24943964ca862d9685988a64718fce3efe486de9936a",
    "parent_source_sha256": GAN_ORIGINAL_SHA256,
    "reviewed_source_sha256": GAN_REVIEWED_SHA256,
    "hash_scope": 'inspect.getsource(function).rstrip("\\n").encode("utf-8")',
}


def load_gan_review():
    review = json.loads(GAN_REVIEW_PATH.read_text())
    assert {key: review.get(key) for key in _GAN_MAPPING} == _GAN_MAPPING, "GAN source review mapping changed"
    lines = review["parent_source_lines"]
    assert isinstance(lines, list) and all(isinstance(line, str) for line in lines)
    assert _sha("\n".join(lines)) == GAN_ORIGINAL_SHA256, "actual parent GAN capture source SHA changed"
    return review


def assert_gan_capture_lineage(source):
    review = load_gan_review()
    assert _sha(source) == GAN_REVIEWED_SHA256, "reviewed GAN capture source SHA changed; new review required"
    assert_native_history_ast("\n".join(review["parent_source_lines"]), source, control_name="controls")
