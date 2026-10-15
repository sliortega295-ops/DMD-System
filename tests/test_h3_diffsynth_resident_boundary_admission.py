"""Execute actual CLI guards, without constructing or admitting a GPU model."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


WORKER = Path(__file__).resolve().parents[1] / 'scripts/distillschedule/probe_h3_diffsynth_h100_world8.py'


class ResidentBoundaryAdmissionTests(unittest.TestCase):
    def run_guards(self, *, placement='resident', boundary=False, cast=False,
                   unit=False, backend='native'):
        tree = ast.parse(WORKER.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        messages = {'BOUNDARY_CPU_IS_NAMED_NATIVE_RESIDENT_FALLBACK',
                    'LORA_INPUT_CAST_IS_SEPARATE_NATIVE_FA2_RUNTIME_ARM',
                    'LORA_UNIT_SUM_IS_SEPARATE_NATIVE_FA2_RUNTIME_ARM'}
        selected = []
        for node in main.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                    t.id == 'lora_placement_supported' for t in node.targets):
                selected.append(node)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and any(
                    isinstance(v, ast.Constant) and v.value in messages for v in node.value.args):
                selected.append(node)
        self.assertEqual(len(selected), 4)
        args = SimpleNamespace(placement=placement, checkpoint_boundary_cpu=boundary,
            lora_input_cast_elision=cast, lora_unit_inplace=unit, attention_backend=backend,
            nograd_host_bounds=False, nograd_retain_main_blocks=0, attention_cache_gpu_tail_blocks=0)

        def require(value, message):
            if not value:
                raise ValueError(message)

        exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                     str(WORKER), 'exec'), dict(args=args, require=require))

    def test_defaults_and_existing_candidate_remain_admitted(self):
        self.run_guards()
        self.run_guards(placement='fsdp')
        self.run_guards(placement='candidate', cast=True, unit=True)

    def test_named_native_resident_fallback_can_compose_exact_lora_operations(self):
        self.run_guards(boundary=True, cast=True, unit=True)

    def test_new_flags_do_not_silently_modify_pure_resident(self):
        for cast, unit in ((True, False), (False, True), (True, True)):
            with self.subTest(cast=cast, unit=unit), self.assertRaises(ValueError):
                self.run_guards(cast=cast, unit=unit)

    def test_boundary_is_not_allowed_on_fsdp_or_fa3(self):
        for placement, backend in (('fsdp', 'native'), ('candidate', 'native'), ('resident', 'fa3')):
            with self.subTest(placement=placement, backend=backend), self.assertRaises(ValueError):
                self.run_guards(placement=placement, boundary=True, backend=backend)


if __name__ == '__main__':
    unittest.main()
