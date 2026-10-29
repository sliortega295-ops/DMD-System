"""One versioned No-GAN source exception; never a numerical-equivalence gate.

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


def assert_native_history_ast(parent_source, reviewed_source):
    """Remove exactly the two reviewed, adjacent opt-in nodes, nothing else.

    The remaining whole function (signature, math, loops, RNG ordering, default
    dispatch and contract included) must equal the pinned actual parent AST.
    Exact matching also pins the all/zero predicate, rank and empty default dict.
    """
    parent, actual = _capture_ast(parent_source), _capture_ast(reviewed_source)
    matches = []
    for owner in ast.walk(actual):
        for _, value in ast.iter_fields(owner):
            if isinstance(value, list):
                matches.extend((value, index) for index, item in enumerate(value)
                               if isinstance(item, ast.Assign) and _dump(item) == _dump(_OPTIONS))
    assert len(matches) == 1, "exactly one reviewed native-policy options assignment required"
    body, index = matches[0]
    assert index + 1 < len(body) and _dump(body[index + 1]) == _dump(_NATIVE_DISPATCH), (
        "reviewed options must immediately precede the exact **options stage_control dispatch")
    del body[index]
    body[index] = deepcopy(_DEFAULT_DISPATCH)
    assert _dump(actual) == _dump(parent), "capture default/math/order differs from actual parent AST"


def assert_nogan_capture_lineage(source):
    review = load_review()
    assert _sha(source) == REVIEWED_SHA256, "reviewed capture source SHA changed; new review required"
    assert_native_history_ast("\n".join(review["parent_source_lines"]), source)
