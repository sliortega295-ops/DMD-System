"""Exercise the worker's pure admission rule without importing the GPU stack.

Actual one-cycle runtime census/version validation still requires an eight-rank
canary; these tests do not stand in for that gate.
"""
import ast
from pathlib import Path
import unittest


def load_rule():
    path = Path(__file__).resolve().parents[1] / 'scripts/distillschedule/probe_h3_diffsynth_h100_world8.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'controlled_cycle_count')
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['controlled_cycle_count']


class CorrectnessCycleScopeTests(unittest.TestCase):
    def test_default_two_cycles_unchanged(self):
        rule = load_rule()
        self.assertEqual(rule('correctness', False), 2)
        self.assertEqual(rule('timing', False), 2)

    def test_single_cycle_is_explicit_correctness_only(self):
        rule = load_rule()
        self.assertEqual(rule('correctness', True), 1)
        with self.assertRaisesRegex(ValueError, 'CORRECTNESS_ONLY_NOT_FORMAL_TIMING'):
            rule('timing', True)
