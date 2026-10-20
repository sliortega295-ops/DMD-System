"""CPU-only synthetic acquisition fixtures, never native/GPU evidence."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from distillschedule.search import helios_model_adapter as model, workflow_cli
from distillschedule.search import component_calibration
from distillschedule.search.plan_ir import PlanningProblem
from distillschedule.workloads.best_known import cli, registry


class AllocatorMarginTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        scratch = Path(tempfile.gettempdir()) / 'lyy-experiments/DMD-System/helios-allocator-margin'
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix='unit-', dir=scratch)))
        self.layout = self.tmp / 'CPU_ONLY_layout.json'
        self.data = dict(schema_version=model.SCHEMA, fragments=[{'index': 0}, {'index': 1}])
        self.pair = model.anchors(model.NO_GAN)
        steps = registry.load_profile(model.NO_GAN)['contract']['logical_steps']
        memory = dict(allocated_bytes=100, reserved_bytes=200,
                      driver_total_bytes=40*2**30, driver_free_bytes=40*2**30-1200)
        self.rows = [dict(rank=i, post_setup_memory=memory,
            cycles=[dict(cycle=c, actions=[dict(memory_before=memory, memory_after=memory)],
                         retained_lifetimes=[dict(source_step=s, bytes=10) for s in steps])
                    for c in (1, 2)]) for i in range(16)]
        self.identity = {key: 'a'*(40 if key == 'source_head' else 64) for key in (
            'source_head', 'hardware_fingerprint', 'environment_fingerprint',
            'asset_fingerprint', 'semantic_fingerprint', 'precision_fingerprint')}
        work = {op.operation_id: {} for op in model.action_graph(self.pair[0]).operations}
        self.stack.enter_context(patch.object(model, '_read_fragment', side_effect=lambda ref, root:
            ({'binding': self.identity}, self.pair[ref['index']], deepcopy(self.rows))))
        self.stack.enter_context(patch.object(model, 'observed_work', return_value=work))
        self.stack.enter_context(patch.object(model, 'component_contract', return_value=NS(digest='b'*64)))

    def build(self):
        self.layout.write_text(json.dumps(self.data))
        return model.build_problem(self.layout)

    def test_default_exact_two_gib_and_explicit_margin(self):
        default = self.build()
        self.assertNotIn('allocator_margin_bytes', json.loads(self.layout.read_text()))
        self.assertEqual(default.memory_limits[0], 40*2**30-1000-2*2**30)
        self.assertIn('; TWO_GIB_MARGIN;', default.lowering['memory_scope'])
        self.assertEqual(model.build_problem(self.layout).digest, default.digest)
        for margin in (0, 512*2**20, 2*2**30):
            self.data['allocator_margin_bytes'] = margin
            problem = self.build()
            self.assertEqual(problem.memory_limits, {i: 40*2**30-1000-margin for i in range(16)})
            model.validate_native_problem(problem)

    def test_invalid_layout_margin_before_acquisition(self):
        for value in (True, False, -1, 1.5, '1024', None, []):
            self.data['allocator_margin_bytes'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                self.build()
        for value in (True, -1, 1.5, '1024', []):
            with self.subTest(api_value=value), self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                model.assemble_calibrations([], output=self.tmp/'unused', expected_head='a'*40,
                                            allocator_margin_bytes=value)

    def test_assembly_persists_only_explicit_margin(self):
        costs = NS(verify_receipts=lambda: None, bind=lambda *a, **kw: None,
                   to_dict=lambda: {'scope': 'CPU_FIXTURE_ONLY'})
        for index, margin in enumerate((None, 0, 512*2**20)):
            out = self.tmp / f'assembly-{index}'
            rows = [dict(status='PASS_NATIVE_COMPONENT_CALIBRATION', source_head='a'*40,
                         layout_fragment={'index': i}, operation_costs={}) for i in range(2)]
            read_object = model._object
            with patch.object(model, 'git_head', return_value='a'*40), \
                    patch.object(model, '_object', side_effect=lambda path:
                                 rows[path] if type(path) is int else read_object(path)), \
                    patch.object(model, 'bound', return_value={}), \
                    patch.object(model.CostModel, 'from_dict', return_value=costs), \
                    patch.object(component_calibration, 'pool_component_models', return_value=costs):
                problem, _ = model.assemble_calibrations([0, 1], output=out,
                    expected_head='a'*40, allocator_margin_bytes=margin)
            layout = json.loads((out/'layout.json').read_text())
            if margin is None:
                self.assertEqual(layout, self.data)
            else:
                self.assertEqual(layout['allocator_margin_bytes'], margin)
            self.assertEqual(problem.memory_limits[0], 40*2**30-1000-(2*2**30 if margin is None else margin))

    def test_margin_exhausting_observed_capacity_rejected(self):
        self.data['allocator_margin_bytes'] = 40 * 2**30
        with self.assertRaisesRegex(ValueError, 'invalid observed capacity'):
            self.build()

    def test_tampered_problem_plan_and_layout_rejected(self):
        self.data['allocator_margin_bytes'] = 0
        problem = self.build()
        raw = problem.to_dict(); raw['memory_limits']['0'] += 1
        tampered = PlanningProblem.from_dict(raw)
        with self.assertRaisesRegex(ValueError, 'graph/lowering changed'):
            model.validate_native_problem(tampered)
        with self.assertRaisesRegex(ValueError, 'graph/lowering changed'):
            model.validate_lowered_plan({'model_plan': {'problem': raw}})
        self.data['allocator_margin_bytes'] = 1
        self.layout.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, 'layout hash changed'):
            model.validate_native_problem(problem)

    def test_public_assembly_routes_explicit_mib_only_for_helios(self):
        for margin, expected in ((None, {}), (0, {'allocator_margin_bytes': 0}),
                                 (512, {'allocator_margin_bytes': 512*2**20})):
            argv = ['assemble-calibration', '--workload', model.NO_GAN, '--result',
                    str(self.tmp/'result.json'), '--expected-head', 'a'*40,
                    '--output', str(self.tmp/'out')]
            if margin is not None: argv += ['--helios-allocator-margin-mib', str(margin)]
            row = {'implementation_id': self.pair[0].id}
            with patch.object(workflow_cli, '_object', return_value=row), \
                    patch.object(model, 'assemble_calibrations') as assemble:
                self.assertEqual(cli.main(argv), 0)
            kwargs = assemble.call_args.kwargs
            self.assertEqual({k:v for k,v in kwargs.items() if k == 'allocator_margin_bytes'}, expected)

    def test_nonhelios_and_bad_mib_reject_before_reading_receipts(self):
        for workload, margin in ((model.NO_GAN, True), (model.NO_GAN, -1),
                                 (model.NO_GAN, 1.5), ('wan13b_nogan_best', 0)):
            args = NS(workload=workload, helios_allocator_margin_mib=margin, result=[])
            with self.subTest(workload=workload, margin=margin), self.assertRaises(ValueError):
                workflow_cli.assemble(args)


if __name__ == '__main__':
    unittest.main()
