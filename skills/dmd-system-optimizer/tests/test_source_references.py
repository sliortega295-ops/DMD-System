"""Historical and multi-repository provenance behavior with small local Git fixtures."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dmd_knowledge as m


class PinnedSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        (self.root / "evidence.md").write_text("historical observation\n")
        self.git("add", "evidence.md")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
        rev = self.git("rev-parse", "HEAD").strip()
        self.ref = {"repository": m.DEFAULT_REPOSITORY, "revision": rev, "path": "evidence.md",
                    "sha256": hashlib.sha256((self.root / "evidence.md").read_bytes()).hexdigest(),
                    "url": f"https://github.com/{m.DEFAULT_REPOSITORY}/blob/{rev}/evidence.md", "locator": "body"}

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True,
                              check=True, text=True).stdout

    def test_changed_working_tree_uses_pinned_blob_without_modifying_it(self):
        (self.root / "evidence.md").write_text("new unrelated observation\n")
        result = m.verify_references([self.ref], self.root)
        self.assertEqual(result["artifact_verification"], "HASH_CHECKED_NOT_TRUTH_PROVEN")
        self.assertEqual(result["evidence_checks"][0]["checked_from"], "pinned_git_blob")
        self.assertFalse(result["evidence_checks"][0]["current_tree_matches"])
        self.assertEqual((self.root / "evidence.md").read_text(), "new unrelated observation\n")

    def test_deleted_report_still_resolves_at_historical_revision(self):
        (self.root / "evidence.md").unlink()
        result = m.verify_references([self.ref], self.root)
        self.assertEqual(result["evidence_checks"][0]["checked_from"], "pinned_git_blob")
        self.assertFalse((self.root / "evidence.md").exists())

    def test_corrupt_expected_hash_is_not_upgraded_by_git_fallback(self):
        ref = {**self.ref, "sha256": "0" * 64}
        result = m.verify_references([ref], self.root)
        self.assertEqual(result["artifact_verification"], "STALE_OR_UNAVAILABLE")
        self.assertEqual(result["evidence_checks"][0]["status"], "MISMATCH")

    def external_ref(self):
        repository = "External/Research"
        return {**self.ref, "repository": repository,
                "url": self.ref["url"].replace(m.DEFAULT_REPOSITORY, repository)}

    def test_external_path_is_never_checked_in_dmd_root(self):
        result = m.verify_references([self.external_ref()], self.root)
        self.assertEqual(result["artifact_verification"], "PUBLIC_REFERENCES_NOT_LOCALLY_CHECKED")
        self.assertEqual(result["evidence_checks"][0]["status"], "NOT_LOCALLY_CHECKED")

    def test_explicit_external_checkout_checks_external_bytes(self):
        result = m.verify_references([self.external_ref()], source_repos={"External/Research": self.root})
        self.assertEqual(result["artifact_verification"], "HASH_CHECKED_NOT_TRUTH_PROVEN")

    def test_mixed_checked_and_unchecked_is_partial(self):
        result = m.verify_references([self.ref, self.external_ref()], self.root)
        self.assertEqual(result["artifact_verification"], "PARTIALLY_CHECKED")

    def test_url_cannot_point_to_another_repository_or_revision(self):
        for change in ({"repository": "Other/Repo"}, {"revision": "1" * 40}, {"path": "another.md"}):
            with self.subTest(change=change), self.assertRaises(m.InputError):
                m.validate_reference({**self.ref, **change})

    def test_unsafe_or_unpinned_reference_rejected(self):
        for change in ({"path": "../escape"}, {"path": "/absolute"}, {"revision": "main"}, {"revision": "HEAD:bad"}):
            with self.subTest(change=change), self.assertRaises(m.InputError):
                m.validate_reference({**self.ref, **change})

    def test_local_symlink_escape_not_followed(self):
        (self.root / "evidence.md").unlink()
        (self.root / "evidence.md").symlink_to(self.root.parent / "outside.md")
        result = m.verify_references([self.ref], self.root)
        self.assertEqual(result["evidence_checks"][0]["status"], "UNSAFE_LOCAL_PATH")


if __name__ == "__main__":
    unittest.main()
