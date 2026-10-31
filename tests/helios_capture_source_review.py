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
    if _sha(source) != REVIEWED_SHA256:
        return _assert_aligned_capture_lineage(source, gan=False)
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
    if _sha(source) != GAN_REVIEWED_SHA256:
        return _assert_aligned_capture_lineage(source, gan=True)
    assert _sha(source) == GAN_REVIEWED_SHA256, "reviewed GAN capture source SHA changed; new review required"
    assert_native_history_ast("\n".join(review["parent_source_lines"]), source, control_name="controls")


# V1's maps, historical goldens and exact two-node inverse above stay intact.
# V2 reviews the later, explicit aligned-scope integration as ONE replacement
# assignment plus its complete new helper, not as byte identity with V1.
_ROOT = Path(__file__).resolve().parents[1]
ALIGNED_REVIEW_PATH = _ROOT / "docs/contracts/helios_nogan_capture_source_review_v2.json"
GAN_ALIGNED_REVIEW_PATH = _ROOT / "docs/contracts/helios_gan_capture_source_review_v2.json"
ALIGNED_SHA256 = "d9ee346257c26547e10ba7aa6e1a228b40c2527bbc2dc484a7e4eb92d8f1dc6f"
GAN_ALIGNED_SHA256 = "ab2f40246eb04dfa15303f1b375d398321677b5fa778182c5e4fe2a5973ffbef"
HELPER_SHA256 = "9314c9ca9e3b566dfdc5d67d823eaec91c430f4c477706d99130bc3851548678"
_ALIGNED_COMMON = {
    "function": "capture_rank",
    "parent_commit": "48b2c4ed7d8b0b1a0ec3e9f6043cde1eaad6f5e4",
    "transition_commit": "e6fb2e1b9653cdd4a8cfc8d7f9b8fa212a506e27",
    "reviewed_commit": "555d3250721601489f94e92eb765bfcce5efc60f",
    "helper_source_path": SOURCE_PATH,
    "helper_function": "history_control_options",
    "helper_source_sha256": HELPER_SHA256,
    "helper_module_sha256": "f4d9f524ee205a5993e5d867795ddc32259ca9970ebfe1c7ec46f5f0d9b045e2",
    "hash_scope": _MAPPING["hash_scope"],
}
_ALIGNED_MAPPING = {
    **_ALIGNED_COMMON,
    "schema_version": "distillschedule.helios_nogan_capture_source_review.v2",
    "source_path": SOURCE_PATH,
    "parent_git_blob": "5396b8d1ce74b38955cf9d1a22f71263c1ceab95",
    "parent_module_sha256": "dd7183aa89db88b054186b7c1e88f5b31a0eaad084e84f958ad3df504852fb17",
    "parent_source_sha256": REVIEWED_SHA256,
    "parent_review_path": "docs/contracts/helios_nogan_capture_source_review_v1.json",
    "parent_review_sha256": "5052bffdf6b3af45216f29138309a3ece9687adb6aa55323980259d676cbd72d",
    "reviewed_git_blob": "9805411d1294a88120eaf9d8a8d83d7b72921b4e",
    "reviewed_module_sha256": _ALIGNED_COMMON["helper_module_sha256"],
    "reviewed_source_sha256": ALIGNED_SHA256,
}
_GAN_ALIGNED_MAPPING = {
    **_ALIGNED_COMMON,
    "schema_version": "distillschedule.helios_gan_capture_source_review.v2",
    "source_path": GAN_SOURCE_PATH,
    "parent_git_blob": "502d0dd9aecba705198a376a7e894a2ffdc32b9c",
    "parent_module_sha256": "0f8057b12b10c9a39096a2ad8815311d29c7aa7c2abff683eb1f7657362437f4",
    "parent_source_sha256": GAN_REVIEWED_SHA256,
    "parent_review_path": "docs/contracts/helios_gan_capture_source_review_v1.json",
    "parent_review_sha256": "4dab090b8918bf40481421b7b01efffe3b9db1878b68dc74e7b5dee4879a6405",
    "reviewed_git_blob": "942b712555ec92d5f4083f4f6295a653f7a01c7a",
    "reviewed_module_sha256": "752973c24f354b6ed6b1faf000c4d9a16c583c0311ab4d7d8ae8926ea6f686b0",
    "reviewed_source_sha256": GAN_ALIGNED_SHA256,
}
_ALIGNED_OPTIONS = ast.parse('''options = history_control_options(task,
    zero=all(bool(tensors[name].sum() == 0)
        for name in ("history_short", "history_mid", "history_long")), rank=rank)''').body[0]
_HISTORY_CONTROL_OPTIONS = ast.parse('''def history_control_options(task, *, zero, rank):
    """Keep sampled history and its native grid; only the explicit scope owns exits."""
    require(type(zero) is bool, "explicit original history predicate required", path="helios.inputs")
    if task.native_policy is not None or task.stage_schedule_policy == ALIGNED_SCHEDULE_POLICY:
        return {"all_history_zero":zero, "rank":rank}
    require(not zero, "normal reference forbids rank-local amplified zero history", path="helios.inputs")
    return {}
''')


def assert_history_control_helper(source):
    assert _sha(source) == HELPER_SHA256, "reviewed history-control helper source SHA changed"
    assert _dump(ast.parse(source)) == _dump(_HISTORY_CONTROL_OPTIONS), "exact reviewed helper AST required"


def assert_aligned_capture_ast(previous_source, source, helper_source, *, control_name="control"):
    """Invert only the exact options assignment; compare the entire V1 function.

    The whole helper AST (including the empty default and normal-zero refusal)
    is independently exact. This routine deliberately does not use raw hashes,
    so mutation tests prove semantic rejection rather than a failing base gate.
    """
    assert control_name in ("control", "controls")
    assert _dump(ast.parse(helper_source)) == _dump(_HISTORY_CONTROL_OPTIONS), "exact reviewed helper AST required"
    previous, actual = _capture_ast(previous_source), _capture_ast(source)
    dispatch = deepcopy(_NATIVE_DISPATCH)
    dispatch.targets[0].id = control_name
    matches = []
    for owner in ast.walk(actual):
        for _, value in ast.iter_fields(owner):
            if isinstance(value, list):
                matches.extend((value, index) for index, item in enumerate(value)
                               if isinstance(item, ast.Assign) and _dump(item) == _dump(_ALIGNED_OPTIONS))
    assert len(matches) == 1, "exactly one reviewed aligned options assignment required"
    body, index = matches[0]
    assert index + 1 < len(body) and _dump(body[index + 1]) == _dump(dispatch), (
        "aligned options must immediately precede the exact **options stage_control dispatch")
    body[index] = deepcopy(_OPTIONS)
    assert _dump(actual) == _dump(previous), "capture default/math/order differs from actual V1 AST"


def load_aligned_review(*, gan=False):
    old = load_gan_review() if gan else load_review()
    mapping = _GAN_ALIGNED_MAPPING if gan else _ALIGNED_MAPPING
    review = json.loads((GAN_ALIGNED_REVIEW_PATH if gan else ALIGNED_REVIEW_PATH).read_text())
    assert {key: review.get(key) for key in mapping} == mapping, "aligned source review mapping changed"
    assert hashlib.sha256((_ROOT / mapping["parent_review_path"]).read_bytes()).hexdigest() == (
        mapping["parent_review_sha256"]), "original V1 source review bytes changed"
    lines = review["parent_source_lines"]
    assert isinstance(lines, list) and all(isinstance(line, str) for line in lines)
    previous = "\n".join(lines)
    assert _sha(previous) == mapping["parent_source_sha256"], "actual V1 capture source SHA changed"
    assert_native_history_ast("\n".join(old["parent_source_lines"]), previous,
                              control_name="controls" if gan else "control")
    helper = review["helper_source_lines"]
    assert isinstance(helper, list) and all(isinstance(line, str) for line in helper)
    assert_history_control_helper("\n".join(helper))
    return review


def _assert_aligned_capture_lineage(source, *, gan):
    review = load_aligned_review(gan=gan)
    assert _sha(source) == review["reviewed_source_sha256"], (
        "reviewed capture source SHA changed; explicit new review required")
    # Pin the actual helper binding/import/ALIGNED constant as well as function
    # bodies. This is source review, not a Python runtime rebinding audit.
    for path_key, hash_key in (("source_path", "reviewed_module_sha256"),
                               ("helper_source_path", "helper_module_sha256")):
        assert hashlib.sha256((_ROOT / review[path_key]).read_bytes()).hexdigest() == review[hash_key], (
            "reviewed capture/helper module source SHA changed")
    assert_aligned_capture_ast("\n".join(review["parent_source_lines"]), source,
                               "\n".join(review["helper_source_lines"]),
                               control_name="controls" if gan else "control")
