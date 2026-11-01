"""Source navigation and scoped reuse regressions; never execute training code."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
REPO = SKILL.parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
import dmd_knowledge as knowledge
import dmd_navigation as nav
import dmd_retrieval as retrieval


class RetrievalRegressionTests(unittest.TestCase):
    def test_labeled_queries(self):
        data = knowledge.validate_pack()
        cases = json.loads((SKILL / "tests/retrieval-cases.json").read_text())["cases"]
        for case in cases:
            with self.subTest(query=case["query"]):
                rows, decision = retrieval.rank(data, case["query"])
                if case["top"]:
                    self.assertTrue(rows)
                    self.assertIn(rows[0]["pattern"]["id"], case["top"])
                else:
                    self.assertFalse(rows)
                if "status" in case:
                    self.assertEqual(decision["status"], case["status"])

    def test_query_never_supplies_gradient_hardware_or_cache_facts(self):
        result = knowledge.recall("H100 cross-KV no-grad B1", {})
        self.assertEqual(result["normalized_context"], {})
        case = next(c for p in result["matches"] for group in p["cases"].values() for c in group
                    if c["id"] == "helios-crosskv-gan-positive")
        self.assertIn("grad_mode", case["correctness_comparison"]["unknown"])
        self.assertEqual(case["next_action"], "CHECK_UNKNOWN_PRECONDITIONS")

    def test_case_conflict_is_returned_even_when_generic_pattern_matches(self):
        context = {"cache_key_complete": True, "invalidation_validated": True,
                   "grad_mode": "with_grad", "microbatch": 2}
        result = knowledge.recall("cross-KV", context, limit=1)
        case = next(c for c in result["matches"][0]["cases"]["positive"]
                    if c["id"] == "helios-crosskv-gan-positive")
        self.assertEqual(case["next_action"], "DO_NOT_APPLY_AS_IS")
        self.assertEqual(set(case["correctness_comparison"]["conflicting"]), {"grad_mode", "microbatch"})
        self.assertTrue(case["code_navigation"]["implementations"][0]["entrypoints"])

    def test_external_inference_case_conflicts_with_training(self):
        result = knowledge.locate_case("rb-inference-graph-shape-miss", {"training_mode": "training"})
        self.assertEqual(result["next_action"], "DO_NOT_APPLY_AS_IS")
        self.assertEqual(result["implementations"][0]["relationship"], "adaptation_example")
        self.assertEqual(result["evidence_status"], "EXTERNAL_REPORTED_NOT_REPRODUCED")

    def test_all_cases_expose_specific_retry_and_navigation(self):
        cases = knowledge.validate_pack()["cases"]
        self.assertEqual(len({c["retry"]["minimum_probe"] for c in cases}), len(cases))
        for case in cases:
            self.assertNotIn("相关前提、容量或实现改变后", case["reentry"])
            self.assertIn(case["navigation"]["status"], nav.STATUSES)

    def test_missing_code_is_explicit_not_an_empty_success(self):
        method = knowledge.locate_case("rb-judge-negative-controls", {})
        unresolved = knowledge.locate_case("helios-h3-blockmajor-lead", {})
        self.assertEqual(method["navigation"]["status"], "method_only")
        self.assertEqual(unresolved["navigation"]["status"], "unresolved")
        self.assertEqual(method["implementations"], [])
        self.assertEqual(unresolved["implementations"], [])

    def test_broken_case_code_relation_is_rejected(self):
        case = copy.deepcopy(knowledge.validate_pack()["cases"][0])
        case["navigation"]["implementations"][0]["id"] = "nonexistent"
        with self.assertRaises(knowledge.InputError):
            nav.validate_case(case, nav.validate_catalog())

    def test_missing_route_pattern_is_rejected(self):
        with self.assertRaises(knowledge.InputError):
            retrieval.validate_routes(set())

    def test_brief_retains_all_case_ids_groups_and_conflicts_with_less_output(self):
        result = knowledge.recall("相同输入的投影反复计算，而且有的 rank 长时间等待",
                                  {"grad_mode": "grad_enabled"}, limit=4)
        brief = knowledge.brief_recall(result, all_cases=True)
        for full, short in zip(result["matches"], brief["matches"]):
            for group in full["cases"]:
                self.assertEqual([c["id"] for c in full["cases"][group]], [c["id"] for c in short["cases"][group]])
                for original, indexed in zip(full["cases"][group], short["cases"][group]):
                    self.assertEqual(indexed["correctness_conflicts"], original["correctness_comparison"]["conflicting"])
            self.assertEqual(short["detail_level"], "index_only")
        self.assertLess(len(json.dumps(brief, ensure_ascii=False).encode()), 20000)
        case = knowledge.read_case("helios-crosskv-nogan-negligible", {"grad_mode": "grad_enabled"})
        self.assertEqual(case["outcome"], "inconclusive")
        self.assertTrue(case["observation"])
        self.assertTrue(case["retry"]["minimum_probe"])
        self.assertEqual(case["next_action"], "DO_NOT_APPLY_AS_IS")

    def test_compound_request_exposes_routes_omitted_by_result_limit(self):
        result = knowledge.recall("相同输入的投影反复计算，而且有的 rank 长时间等待，请检索成功和失败经验", {}, limit=1)
        self.assertNotIn("budget", result["retrieval"]["matched_symptoms"])
        self.assertIn("role-scheduling", {p["id"] for p in result["retrieval"]["omitted_routed_patterns"]})
        self.assertIn("Split compound", result["retrieval"]["coverage_notice"])


class CodeNavigationTests(unittest.TestCase):
    def test_all_bundled_sources_and_symbols_match_without_imports(self):
        checked = set()
        for entry in nav.validate_catalog().values():
            for ref in nav.references(entry):
                key = (ref["repository"], ref["revision"], ref["path"], ref["symbol"], ref.get("focus_call"))
                if key in checked:
                    continue
                checked.add(key)
                result = nav.inspect_reference(ref, REPO)
                self.assertEqual(result["symbol_status"], "SYMBOL_AND_SPAN_CHECKED", key)
                self.assertEqual(result["artifact_verification"], "HASH_CHECKED_NOT_TRUTH_PROVEN", key)
        self.assertNotIn("torch", sys.modules)

    def test_local_code_excerpt_is_bounded_and_names_actual_wrapper(self):
        result = nav.locate("wan-lora-epilogue", {"grad_mode": "grad_enabled"}, REPO, show_code=True)
        self.assertEqual(result["next_action"], "DO_NOT_APPLY_AS_IS")
        self.assertIn("install_generator_nograd_lora_fused_epilogue", result["integration"][0]["symbol"])
        self.assertLessEqual(len(result["entrypoints"][0]["excerpt"].splitlines()), 30)
        self.assertTrue(result["test_gap"])
        self.assertEqual(result["gpu_validation"], "NOT_RUN")

    def test_benefit_conflict_does_not_become_correctness_rejection(self):
        entry = nav.validate_catalog()["fsdp-phase-retention"]
        result = nav.locate(entry["id"], {**entry["correctness_conditions"], "exposed_gather": False})
        self.assertEqual(result["next_action"], "ASSESS_CURRENT_BENEFIT")
        self.assertIn("exposed_gather", result["performance_comparison"]["conflicting"])

    def test_wrong_symbol_or_line_span_cannot_be_called_checked(self):
        ref = copy.deepcopy(nav.validate_catalog()["helios-cross-kv"]["entrypoints"][0])
        for change in ({"symbol": "nonexistent"}, {"line_end": ref["line_end"] + 1}):
            with self.subTest(change=change):
                result = nav.inspect_reference({**ref, **change}, REPO, show_code=True)
                self.assertEqual(result["symbol_status"], "SYMBOL_OR_SPAN_MISMATCH")
                self.assertEqual(result["artifact_verification"], "STALE_OR_UNAVAILABLE")
                self.assertNotIn("excerpt", result)

    def test_unavailable_checkout_preserves_pinned_anchors_and_test_gaps(self):
        result = nav.locate("helios-cross-kv", {}, show_code=True)
        ref = result["entrypoints"][0]
        self.assertEqual(ref["symbol_status"], "NOT_LOCALLY_CHECKED")
        self.assertIn("#L", ref["anchor_url"])
        self.assertNotIn("excerpt", ref)
        self.assertEqual(result["tests"], [])
        self.assertTrue(result["test_gap"])

    def test_ast_reader_does_not_execute_hostile_module(self):
        self.assertEqual(nav.symbol_span('raise RuntimeError("do not execute")\nclass C:\n def f(self):\n  pass\n', "C.f"), (3, 4))

    def test_actual_integration_calls_are_verified_and_shown(self):
        result = nav.locate("helios-cross-kv", {}, REPO, show_code=True)
        calls = [r for r in result["integration"] if "focus_call" in r]
        self.assertEqual(len(calls), 2)
        for ref in calls:
            self.assertIn(ref["focus_call"] + "(", ref["excerpt"])
            bad = {**ref, "focus_call": "nonexistent_call"}
            inspected = nav.inspect_reference(bad, REPO, show_code=True)
            self.assertEqual(inspected["symbol_status"], "CALLSITE_MISMATCH")
            self.assertNotIn("excerpt", inspected)

    def test_no_grad_and_grad_lora_entries_keep_separate_scope_and_tests(self):
        nograd = nav.locate("wan-lora-epilogue", {"grad_mode": "grad_enabled"})
        grad = nav.locate(nograd["related_implementations"][0]["id"], {"grad_mode": "grad_enabled"})
        self.assertEqual(nograd["next_action"], "DO_NOT_APPLY_AS_IS")
        self.assertIn("grad_mode", grad["correctness_comparison"]["matched"])
        self.assertEqual(grad["next_action"], "CHECK_UNKNOWN_PRECONDITIONS")
        self.assertTrue(nograd["tests"])
        self.assertIn("mock", nograd["test_gap"])
        self.assertIn("checkpoint replay", grad["test_gap"])

    def test_changed_working_file_reads_and_labels_pinned_blob(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def git(*args):
                return subprocess.check_output(["git", "-C", d, *args], stderr=subprocess.DEVNULL).decode().strip()
            git("init", "-q"); git("config", "user.name", "Test"); git("config", "user.email", "test@example.invalid")
            code = b"def old():\n    return 1\n"
            (root/"code.py").write_bytes(code)
            git("add", "code.py"); git("commit", "-qm", "fixture")
            revision = git("rev-parse", "HEAD")
            url = f"https://github.com/{knowledge.DEFAULT_REPOSITORY}/blob/{revision}/code.py"
            ref = {"repository": knowledge.DEFAULT_REPOSITORY, "revision": revision, "path": "code.py",
                   "sha256": hashlib.sha256(code).hexdigest(), "url": url, "locator": "old",
                   "symbol": "old", "line_start": 1, "line_end": 2, "anchor_url": url+"#L1-L2"}
            (root/"code.py").write_text("def new():\n    return 2\n")
            result = nav.inspect_reference(ref, root, show_code=True)
            self.assertEqual(result["read_revision"], revision)
            self.assertFalse(result["evidence_checks"][0]["current_tree_matches"])
            self.assertEqual(result["symbol_status"], "SYMBOL_AND_SPAN_CHECKED")
            self.assertIn("def old", result["excerpt"])
            self.assertIn("def new", (root/"code.py").read_text())

    def test_installed_locator_works_outside_repository(self):
        spec = importlib.util.spec_from_file_location("installer_nav", REPO/"tools/install_optimizer_skill.py")
        installer = importlib.util.module_from_spec(spec); spec.loader.exec_module(installer)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); dest = root/"installed"
            installer.install(SKILL, dest)
            result = subprocess.run([sys.executable, str(dest/"scripts/dmd_knowledge.py"), "locate",
                                     "--case", "helios-crosskv-gan-positive", "--show-code"],
                                    cwd=root, check=True, capture_output=True, text=True)
            data = json.loads(result.stdout)
            self.assertEqual(data["navigation"]["status"], "linked")
            self.assertEqual(data["implementations"][0]["entrypoints"][0]["symbol_status"], "NOT_LOCALLY_CHECKED")


if __name__ == "__main__":
    unittest.main()
