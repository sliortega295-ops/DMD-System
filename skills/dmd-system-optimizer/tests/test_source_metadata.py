"""Compact source storage must not change historical experience or public output."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
import dmd_knowledge as knowledge
import dmd_navigation as navigation
import dmd_sources as sources
import render_knowledge


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class SourceMetadataCharacterizationTests(unittest.TestCase):
    # Captured before normalization, 2026-11-06. Changing a case/source intentionally
    # requires reviewing the expanded historical data, not just the compact file.
    def test_all_expanded_experience_and_implementation_fields_are_preserved(self):
        self.assertEqual(digest(knowledge.validate_pack()),
                         "93d0a535404451153a8d36177af3bd346c8272ef66e89794d6ceedabafe8a53a")
        self.assertEqual(digest(navigation.validate_catalog()),
                         "eb56fc07609816410399697124d92dd537111798325cf47bab9b3275bc84f5af")

    def test_recall_and_locate_outputs_are_preserved(self):
        queries = [("参数保留 residency", {}),
                   ("cross-KV", {"grad_mode": "with_grad", "microbatch": 2}),
                   ("graph capture", {"training_mode": "training"}),
                   ("FFN-up replay", {"workload": "W6", "ranks": 16}),
                   ("H100 pinned offload", {"training_mode": "training"})]
        # The current-tree navigation snapshot has a separate owner; this digest
        # covers historical recall without binding future runtime source hashes.
        self.assertEqual(digest([knowledge._recall(q, c, limit=20) for q, c in queries]),
                         "0c021b86f6904d212c5e91257269b890508d04337b1270931a2186be142e0679")
        catalog = navigation.validate_catalog()
        self.assertEqual(digest([navigation.locate(i, {"grad_mode": "grad_enabled"}, catalog=catalog)
                                 for i in catalog]),
                         "5cfe729948ccc06b82eca4f869bba2eb49ea39a702b9df92c549cff9781bde2d")

    def test_sources_api_is_preserved_without_adding_code_only_candidates(self):
        self.assertEqual(digest([knowledge.source_search(q, limit=100)
                                 for q in ["H100", "FSDP", "graph", "DMD_LIVEWAN_POOL_DAG"]]),
                         "b3e840372920e701e5146a1ff23e333dbde9ca1bc98fbbcbcc34a810373b3dc7")

    def test_storage_declares_every_exact_pinned_source_once(self):
        index = knowledge.load(knowledge.SOURCE_INDEX)
        self.assertIn("source_files", index)
        identities = index["source_files"]
        self.assertEqual(len(identities), 161)
        seen = []
        def visit(value):
            if isinstance(value, dict):
                if "source_id" in value:
                    self.assertFalse(sources.SOURCE_FIELDS & value.keys())
                    self.assertNotIn("url", value)
                    self.assertNotIn("anchor_url", value)
                    seen.append(value["source_id"])
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        for path in (knowledge.PACK, navigation.CATALOG, knowledge.SOURCE_INDEX):
            visit(knowledge.load(path))
        self.assertEqual(len(seen), 382)
        self.assertEqual(set(seen), set(identities))
        self.assertEqual(len({tuple(row[k] for k in sorted(sources.SOURCE_FIELDS))
                              for row in identities.values()}), len(identities))

    def test_generated_cards_remain_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "skill"
            shutil.copytree(SKILL, root, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            self.assertEqual(render_knowledge.render(root), {"patterns": 21, "cases": 109})
            for original in (SKILL / "references/knowledge").glob("*.md"):
                with self.subTest(card=original.name):
                    self.assertEqual((root / original.relative_to(SKILL)).read_bytes(), original.read_bytes())


class CompactSourceMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = {"repository": "Owner/Repo", "revision": "a" * 40,
                         "path": "code.py", "sha256": "b" * 64}
        self.index = {"schema_version": 1, "source_files": {"src-one": self.identity}, "sources": []}
        self.path = self.root / "implementation-map.json"
        self.write(self.root / "experience-source-index.json", self.index)

    def write(self, path, data):
        path.write_text(json.dumps(data))

    def load(self, data):
        self.write(self.path, data)
        return sources.load_public_data(self.path)

    def test_legacy_inline_files_do_not_require_a_source_index(self):
        (self.root / "experience-source-index.json").unlink()
        inline = {"refs": [{**self.identity, "url": "unchanged", "locator": "body"}]}
        self.assertEqual(self.load(inline), inline)

    def test_compact_refs_expand_identity_urls_focus_and_metadata(self):
        ref = {"source_id": "src-one", "locator": "C.f", "symbol": "C.f", "line_start": 2,
               "line_end": 20, "focus_call": "run", "focus_start": 5, "focus_end": 7,
               "content_corrections": [{"note": "original wording remains qualified"}]}
        result = self.load({"refs": [ref, {"source_id": "src-one", "locator": "body", "anchor": "#L4"}]})
        first, second = result["refs"]
        url = "https://github.com/Owner/Repo/blob/" + "a" * 40 + "/code.py"
        self.assertEqual(first, {**self.identity, **{k: v for k, v in ref.items() if k != "source_id"},
                                 "url": url, "anchor_url": url + "#L5-L7"})
        self.assertEqual(second, {**self.identity, "locator": "body", "url": url, "anchor_url": url + "#L4"})
        first["content_corrections"].append({"note": "local mutation"})
        self.assertEqual(len(self.load({"refs": [ref]})["refs"][0]["content_corrections"]), 1)

    def test_missing_source_and_identity_or_url_overrides_fail_closed(self):
        for ref in [{"source_id": "missing", "locator": "body"},
                    *[{"source_id": "src-one", "locator": "body", key: value}
                      for key, value in {**self.identity, "url": "https://example.invalid",
                                         "anchor_url": "https://example.invalid"}.items()]]:
            with self.subTest(ref=ref), self.assertRaises(knowledge.InputError):
                self.load({"refs": [ref]})

    def test_invalid_canonical_identities_are_rejected(self):
        for change in [{"path": "../escape"}, {"revision": "main"}, {"sha256": "bad"}, {"sha256": 123}]:
            index = copy.deepcopy(self.index)
            index["source_files"]["src-one"].update(change)
            self.write(self.root / "experience-source-index.json", index)
            with self.subTest(change=change), self.assertRaises(knowledge.InputError):
                self.load({"refs": [{"source_id": "src-one", "locator": "body"}]})

    def test_duplicate_or_conflicting_pinned_identity_is_rejected(self):
        for change in [{}, {"sha256": "c" * 64}]:
            index = copy.deepcopy(self.index)
            index["source_files"]["src-two"] = {**self.identity, **change}
            self.write(self.root / "experience-source-index.json", index)
            with self.subTest(change=change), self.assertRaises(knowledge.InputError):
                self.load({"refs": [{"source_id": "src-one", "locator": "body"}]})

    def test_distinct_revisions_remain_distinct_and_index_is_reread(self):
        index = copy.deepcopy(self.index)
        index["source_files"]["src-two"] = {**self.identity, "revision": "c" * 40}
        self.write(self.root / "experience-source-index.json", index)
        rows = self.load({"refs": [{"source_id": name, "locator": "body"}
                                   for name in ("src-one", "src-two")]})["refs"]
        self.assertNotEqual(rows[0]["url"], rows[1]["url"])
        index["source_files"]["src-one"]["sha256"] = "d" * 64
        self.write(self.root / "experience-source-index.json", index)
        self.assertEqual(self.load({"refs": [{"source_id": "src-one", "locator": "body"}]})["refs"][0]["sha256"],
                         "d" * 64)

    def test_invalid_explicit_anchors_are_rejected(self):
        for anchor in ["https://example.invalid", "#heading", "#L0", "#L10-L2", "#L4\n"]:
            with self.subTest(anchor=anchor), self.assertRaises(knowledge.InputError):
                self.load({"refs": [{"source_id": "src-one", "locator": "body", "anchor": anchor}]})


if __name__ == "__main__":
    unittest.main()
