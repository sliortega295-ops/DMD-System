"""Instruction regressions only; these do not measure GPU warmup or speed."""
from pathlib import Path
import re
import unittest

SKILL = Path(__file__).resolve().parents[1]


def normalized(path):
    return " ".join(path.read_text(encoding="utf-8").lower().split())


class CycleTimingPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry = normalized(SKILL / "SKILL.md")
        cls.policy = normalized(SKILL / "references/experiment-protocol.md")

    def test_main_entry_links_qualification_before_quick_timing(self):
        self.assertIn("references/experiment-protocol.md#cycle-2-timing", self.entry)
        self.assertIn("cycle 2 alone does not establish steady state", self.entry)
        self.assertIn("matching evidence", self.entry)
        self.assertIn("cycle 1 for warmup", self.entry)
        self.assertIn("matched a/b repeats", self.entry)

    def test_new_workload_requires_later_cycle_evidence_not_self_agreement(self):
        for term in ("3–4 consecutive complete cycles", "cycle 3", "cycle 4",
                     "neither cycle 2 alone nor agreement between cycles 1 and 2",
                     "initial probe", "unseen phases or shapes", "report it unresolved"):
            with self.subTest(term=term):
                self.assertIn(term, self.policy)

    def test_screening_keeps_training_state_and_scope_honest(self):
        for term in ("same process", "cycle 1 performs real updates", "initial checkpoints",
                     "seeds", "input sequence", "update schedule", "including warmup",
                     "cycle 2 screening, not a steady-state speedup"):
            with self.subTest(term=term):
                self.assertIn(term, self.policy)

    def test_changed_warmup_and_native_limits_need_explicit_handling(self):
        for term in ("graph signatures", "cache construction/lifetime", "state retention",
                     "recheck later cycles", "supports only two cycles", "keep its contract",
                     "never silently extend its cycle limit"):
            with self.subTest(term=term):
                self.assertIn(term, self.policy)

    def test_small_gains_use_paired_evidence_without_fixed_quota(self):
        for term in ("observed noise", "matched a/b repeats", "paired differences",
                     "retain all valid samples", "repeated qualified cycle 2",
                     "fixed sample quota", "unresolved small gain may remain a candidate"):
            with self.subTest(term=term):
                self.assertIn(term, self.policy)

    def test_cold_hot_and_distributed_scope_remain_separate(self):
        for term in ("cold setup/compilation", "full-job wall time",
                     "cycle 1 is not itself a pure startup measurement", "recurring transfers",
                     "unprofiled", "all ranks", "window boundaries", "slowest-rank elapsed time",
                     "avoid per-stage synchronization", "stronger project measurement protocol"):
            with self.subTest(term=term):
                self.assertIn(term, self.policy)
        links = re.findall(r"\]\((references/[^)#]+)(?:#[^)]+)?\)",
                           (SKILL / "SKILL.md").read_text(encoding="utf-8"))
        for link in links:
            self.assertTrue((SKILL / link).is_file(), link)


if __name__ == "__main__":
    unittest.main()
