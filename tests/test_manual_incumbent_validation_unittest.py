"""CPU-only protocol tests. No fixture is GPU, timing or numerical evidence.

Only external acquisition/dispatch boundaries are replaced; graph construction,
calibration-file hashes, search, lowering and plan serialization are real code.
"""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from distillschedule.search import execution, wan_model_adapter as adapter
from distillschedule.search.contracts import semantic_contract
from distillschedule.search.cost_model import CostModel, OperationCost
from distillschedule.search.model_search import search_plans
from distillschedule.search.model_validation import ModelValidationSession
from distillschedule.search import manual_incumbent_validation as match
from distillschedule.search.wan_implementation import PRECISION, WAN, WanImplementation
from distillschedule.workloads.best_known import registry

ROOT = Path(__file__).resolve().parents[1]
HEAD = 'd56963419db9e7d07fa257b246b45bd3a1096a76'


class StopBeforeGPU(RuntimeError):
    pass


class SavedNativePlanTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        scratch = Path(tempfile.gettempdir()) / 'lyy-experiments/DMD-System/matched-verifier-r1'
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory(
            prefix='unit-', dir=scratch)))
        # The delivered isolated archive deliberately has no .git directory.
        for module in (registry, execution, adapter):
            self.stack.enter_context(patch.object(module, 'git_head', return_value=HEAD))
        self.assets = json.loads((ROOT / 'configs/distillschedule/best_known/assets_examples'
                                 / (WAN + '.json')).read_text())
        self.assets['run_root'] = str(self.tmp / 'run-output')
        self.assets['correctness'] = {'path': str(self.tmp / 'fixture-correctness.json'), 'sha256': 'a'*64}
        self.assets['trajectory_comparison'] = {'path': str(self.tmp / 'fixture-comparison.json'), 'sha256': 'b'*64}
        self.assets_path = self.tmp / 'assets.json'
        self.assets_path.write_text(json.dumps(self.assets))
        self.problem, self.costs = self._calibration()
        self.prediction = search_plans(self.problem, self.costs, exhaustive=True, max_proposals=12)
        self.session = ModelValidationSession(problem=self.problem, costs=self.costs,
            prediction=self.prediction, output=self.tmp / 'session', max_trials=3)
        self.cid = WanImplementation('resident', 1024, True).id
        self.candidate = self.session.candidates[self.cid]
        self.runner = execution.NativeTrialRunner(workload=WAN, identity=self.session.identity,
            assets=self.assets, output=self.tmp / 'runner', run_prefix='CPU_FIXTURE_NO_GPU',
            reference_id=self.session.seeds[0], bootstrap_peer=self.session.seeds[1],
            admitted_candidate_ids=set(self.session.candidates))
        self.launched = []
        self.stack.enter_context(patch.object(self.runner, '_prepare', return_value={'assets': self.assets}))
        self.stack.enter_context(patch.object(self.runner, '_observe_model_runtime', return_value=[]))
        self.stack.enter_context(patch.object(self.runner, '_rpc', side_effect=self._resolve_rpc))
        self.stack.enter_context(patch.object(execution, 'run_plan', side_effect=self._stop_before_gpu))

    def _calibration(self):
        binding = {'source_head': HEAD, 'precision_fingerprint': PRECISION.digest,
            'semantic_fingerprint': registry.fingerprint(semantic_contract(registry.load_profile(WAN))),
            'asset_fingerprint': registry.fingerprint(registry.asset_identity(self.assets)),
            'environment_fingerprint': 'a'*64, 'hardware_fingerprint': 'b'*64}
        layout = {'schema_version': 'distillschedule.wan_stage_layout.v1', 'status': 'PASS',
            'profiled': False, 'binding': binding, 'adapter': adapter.ADAPTER,
            'memory_limits': {str(i): 1000 for i in range(16)},
            'memory_definition': 'POST_SETUP_STATIC; STAGE_COST_INCLUDES_ALL_ADDITIONAL_LIVE_MEMORY',
            'implementations': {impl.id: {'status': 'PASS', 'ranks': [dict(rank=i,
                status='PASS', source_head=HEAD, precision_fingerprint=PRECISION.digest,
                post_setup_static_bytes=100) for i in range(16)]} for impl in adapter.implementations()}}
        path = self.tmp / 'layout.json'; path.write_text(json.dumps(layout))
        problem = adapter.build_problem(path)
        records = []
        for impl in adapter.implementations():
            for op in problem.graph.operations:
                key = adapter.cost_key(op, impl)
                evidence = dict(status='PASS', profiled=False, kind='native_operation',
                    operation_contract_sha256=registry.fingerprint(adapter.operation_contract(op, impl)))
                duration = 1 if impl.chunk_tokens == 1024 and impl.lora_epilogue else 5
                raw = dict(schema_version='distillschedule.operation_calibration.v1', binding=binding,
                    key=key, samples_ns=[duration], workspace_bytes_per_rank=10, **evidence)
                record = self.tmp / (key + '.json'); record.write_text(json.dumps(raw))
                evidence.update(source_receipt=str(record), source_receipt_sha256=registry.file_sha256(record))
                records.append(OperationCost(key, (duration,), 10, evidence))
        return problem, CostModel(binding, tuple(records))

    def _resolve_rpc(self, action, **kwargs):
        self.assertEqual(action, 'resolve')
        self.assertEqual(kwargs['assets'], self.assets)
        return execution.resolve_candidate(kwargs['workload'], kwargs['candidate_id'],
            assets_path=self.assets_path, run_id=kwargs['run_id'], mode=kwargs['mode'])

    def _stop_before_gpu(self, receipt, **kwargs):
        self.launched.append(deepcopy(receipt))
        raise StopBeforeGPU('external GPU dispatcher deliberately not called')

    def _saved_lowering(self):
        plan = adapter.lower_prediction(self.prediction, self.problem, self.costs, run_id='CPU_LOWERING_ONLY')
        path = self.tmp / 'model-lowering.json'; path.write_text(json.dumps(plan))
        return path, {'path': str(path), 'sha256': registry.file_sha256(path)}

    def test_real_lower_to_runner_to_saved_execution_plan_keeps_model_plan(self):
        with self.assertRaises(StopBeforeGPU):
            self.session._trial(self.cid, 'candidate', self.runner)
        directory = next((self.tmp / 'session/trials').iterdir())
        lowering = json.loads((directory / 'model-lowering.json').read_text())
        actual = json.loads((directory / 'execution-plan.json').read_text())
        self.assertIsNotNone(lowering['asset_fingerprint'])
        self.assertEqual(lowering['asset_fingerprint'], self.problem.binding['asset_fingerprint'])
        self.assertEqual(lowering['mode'], 'correctness')
        self.assertEqual(actual['mode'], 'diagnostic-timing')
        asset_path = directory / 'assets-config.json'
        self.assertEqual(json.loads(asset_path.read_text()), {key: value
            for key, value in self.assets.items()
            if key not in ('correctness', 'trajectory_comparison')})
        self.assertEqual(actual['asset_config'], self.assets)
        with self.assertRaises(FileExistsError):
            registry.write_new_json(asset_path, self.assets)
        for key in ('integration_head', 'workload_semantics_hash', 'asset_fingerprint',
                    'profile_sha256', 'search_candidate', 'execution_fingerprint'):
            self.assertEqual(actual[key], lowering[key])
        self.assertEqual(lowering['model_plan']['problem']['binding'], self.problem.binding)
        self.assertTrue('model_plan' in actual, 'actual serialized execution-plan lost model_plan')
        self.assertEqual(actual['model_plan'], lowering['model_plan'])
        self.assertEqual(actual['model_plan_source'], dict(path=str((directory / 'model-lowering.json').resolve()),
            sha256=registry.file_sha256(directory / 'model-lowering.json')))
        self.assertEqual(actual, self.launched[0])
        execution.validated_profile(actual, root=ROOT)

    def _raw_assets_with_cached_preparation(self):
        raw = deepcopy(self.assets)
        del raw['correctness'], raw['trajectory_comparison']
        prepared = deepcopy(self.assets)
        self.runner.assets = raw
        self.runner.prepared[self.cid] = dict(source_head=HEAD, candidate_id=self.cid,
                                              assets=prepared)
        return raw, prepared

    def test_raw_assets_lowering_keeps_selection_and_prepares_after_runtime_preflight(self):
        raw, prepared = self._raw_assets_with_cached_preparation()
        order = []

        def observe(candidate, *, receipt=None):
            self.assertEqual(candidate, self.candidate)
            if receipt is None:
                self.assertEqual(order, [])
                order.append('runtime-preflight')
            else:
                self.assertEqual(order[-1], 'timing-resolve')
                self.assertEqual(receipt['mode'], 'diagnostic-timing')
                order.append('timing-runtime-identity')
            return []  # External runtime acquisition only; no GPU evidence.

        def cached_prepare(candidate):
            self.assertEqual(order, ['runtime-preflight'])
            order.append('cached-preparation')
            ready = execution.NativeTrialRunner._prepare(self.runner, candidate)
            self.assertIs(ready, self.runner.prepared[self.cid])
            self.assertIs(ready['assets'], prepared)
            self.assertIsNot(ready['assets'], raw)
            return ready

        def resolve(action, **kwargs):
            self.assertEqual(order[-1], 'cached-preparation')
            self.assertIs(kwargs['assets'], prepared)
            self.assertEqual(kwargs['mode'], 'diagnostic-timing')
            order.append('timing-resolve')
            return self._resolve_rpc(action, **kwargs)

        def stop(receipt, **kwargs):
            self.assertEqual(order[-1], 'timing-runtime-identity')
            order.append('dispatch-boundary')
            self._stop_before_gpu(receipt, **kwargs)

        with patch.object(self.runner, '_observe_model_runtime', side_effect=observe), \
                patch.object(self.runner, '_prepare', side_effect=cached_prepare), \
                patch.object(self.runner, '_rpc', side_effect=resolve), \
                patch.object(execution, 'run_plan', side_effect=stop):
            with self.assertRaises(StopBeforeGPU):
                self.session._trial(self.cid, 'candidate', self.runner)

        directory = next((self.tmp / 'session/trials').iterdir())
        path = directory / 'model-lowering.json'
        saved = json.loads(path.read_text())
        actual = json.loads((directory / 'execution-plan.json').read_text())
        self.assertEqual(saved['mode'], 'correctness')
        self.assertEqual(saved['asset_config'], raw)
        self.assertEqual(actual['mode'], 'diagnostic-timing')
        self.assertEqual(actual['asset_config'], prepared)
        for key in ('model_plan', 'execution_fingerprint', 'search_candidate',
                    'integration_head', 'profile_sha256', 'workload_semantics_hash',
                    'asset_fingerprint'):
            self.assertEqual(actual[key], saved[key])
        self.assertEqual(actual['model_plan_source'], dict(path=str(path.resolve()),
            sha256=registry.file_sha256(path)))
        self.assertEqual(actual, self.launched[0])
        self.assertEqual(order, ['runtime-preflight', 'cached-preparation',
            'timing-resolve', 'timing-runtime-identity', 'dispatch-boundary'])
        self.assertNotIn('correctness', raw)
        self.assertNotIn('trajectory_comparison', raw)
        execution.validated_profile(saved, root=ROOT)
        execution.validated_profile(actual, root=ROOT)

    def test_raw_assets_failed_runtime_admission_never_prepares_or_dispatches(self):
        self._raw_assets_with_cached_preparation()
        with patch.object(self.runner, '_observe_model_runtime',
                          side_effect=ValueError('runtime admission rejected')) as observed, \
                patch.object(self.runner, '_prepare') as prepare, \
                patch.object(execution, 'run_plan') as dispatch:
            with self.assertRaisesRegex(ValueError, 'runtime admission rejected'):
                self.session._trial(self.cid, 'candidate', self.runner)
        observed.assert_called_once_with(self.candidate)
        prepare.assert_not_called()
        dispatch.assert_not_called()
        self.assertFalse(self.launched)

    def test_raw_assets_missing_prepared_timing_gate_never_dispatches(self):
        _, prepared = self._raw_assets_with_cached_preparation()
        del prepared['correctness']
        path = self.tmp / 'missing-prepared-correctness.json'
        path.write_text(json.dumps(prepared))

        def resolve(action, **kwargs):
            self.assertEqual(action, 'resolve')
            self.assertIs(kwargs['assets'], prepared)
            return execution.resolve_candidate(kwargs['workload'], kwargs['candidate_id'],
                assets_path=path, run_id=kwargs['run_id'], mode=kwargs['mode'])

        def cached_prepare(candidate):
            return execution.NativeTrialRunner._prepare(self.runner, candidate)

        with patch.object(self.runner, '_prepare', side_effect=cached_prepare) as prepare, \
                patch.object(self.runner, '_rpc', side_effect=resolve), \
                patch.object(execution, 'run_plan') as dispatch:
            with self.assertRaisesRegex(ValueError, 'timing requires an explicit correctness'):
                self.session._trial(self.cid, 'candidate', self.runner)
        prepare.assert_called_once_with(self.candidate)
        dispatch.assert_not_called()
        self.assertFalse(self.launched)
        directory = next((self.tmp / 'session/trials').iterdir())
        self.assertTrue((directory / 'model-lowering.json').is_file())
        self.assertFalse((directory / 'execution-plan.json').exists())

    def test_legacy_runner_without_lowering_does_not_invent_model_plan(self):
        directory = self.tmp / 'legacy'; directory.mkdir()
        with self.assertRaises(StopBeforeGPU):
            self.runner(self.candidate, 'candidate', directory)
        self.assertNotIn('model_plan', self.launched[0])
        self.assertNotIn('model_plan_source', self.launched[0])

    def test_changed_saved_lowering_hash_rejected_before_prepare_or_launch(self):
        path, ref = self._saved_lowering()
        path.write_text(path.read_text() + '\n')
        with patch.object(self.runner, '_prepare', side_effect=AssertionError('must not prepare')):
            with self.assertRaisesRegex(ValueError, 'hash'):
                self.runner(self.candidate, 'candidate', self.tmp, lowered_plan=ref)
        self.assertFalse(self.launched)

    def test_rehashed_tampered_saved_prediction_is_not_a_valid_plan(self):
        path, ref = self._saved_lowering()
        value = json.loads(path.read_text())
        value['model_plan']['search_result']['predicted_best']['assignment']['chunk_tokens'] = '2048'
        path.write_text(json.dumps(value)); ref['sha256'] = registry.file_sha256(path)
        with self.assertRaisesRegex(ValueError, 'serialized prediction|decision'):
            self.runner(self.candidate, 'candidate', self.tmp, lowered_plan=ref)
        self.assertFalse(self.launched)

    def test_lowering_cannot_be_used_for_a_different_candidate(self):
        _, ref = self._saved_lowering()
        other = self.session.candidates[self.session.seeds[0]]
        with self.assertRaisesRegex(ValueError, 'candidate'):
            self.runner(other, 'candidate', self.tmp, lowered_plan=ref)
        self.assertFalse(self.launched)

    def test_lowering_must_match_all_runtime_identity_fields(self):
        _, ref = self._saved_lowering()
        for field in ('source_head', 'asset_fingerprint', 'semantic_fingerprint',
                      'hardware_fingerprint', 'environment_fingerprint', 'precision_fingerprint'):
            with self.subTest(field=field):
                old = self.runner.identity[field]; self.runner.identity[field] = 'c'*len(old)
                try:
                    with self.assertRaises(ValueError):
                        self.runner(self.candidate, 'candidate', self.tmp, lowered_plan=ref)
                finally:
                    self.runner.identity[field] = old
        self.assertFalse(self.launched)

    def test_resolved_candidate_drift_rejected_before_dispatch(self):
        _, ref = self._saved_lowering()
        def changed_resolve(action, **kwargs):
            kwargs['candidate_id'] = self.session.seeds[0]
            return self._resolve_rpc(action, **kwargs)
        with patch.object(self.runner, '_rpc', side_effect=changed_resolve):
            with self.assertRaises(ValueError):
                self.runner(self.candidate, 'candidate', self.tmp, lowered_plan=ref)
        self.assertFalse(self.launched)

    def test_legacy_cpu_executor_still_receives_three_arguments(self):
        called = []
        def legacy(candidate, phase, directory):
            called.append((candidate.id, phase, directory))
            raise StopBeforeGPU('CPU callback only')
        with self.assertRaises(StopBeforeGPU):
            self.session._trial(self.cid, 'candidate', legacy)
        self.assertEqual(len(called), 1)
        directory = called[0][2]
        lowering = json.loads((directory / 'model-lowering.json').read_text())
        self.assertIsNone(lowering['asset_fingerprint'])
        self.assertEqual(lowering['mode'], 'correctness')
        self.assertFalse((directory / 'assets-config.json').exists())

    def test_replaced_runner_factory_remains_a_three_argument_cpu_callback(self):
        called = []
        class CPUReceiptProducer:
            def __call__(producer, candidate, phase, directory):
                called.append((candidate.id, phase, directory))
                raise StopBeforeGPU('legacy CPU callback factory, not a native runner')
        with patch.object(execution, 'NativeTrialRunner', CPUReceiptProducer):
            with self.assertRaises(StopBeforeGPU):
                self.session._trial(self.cid, 'candidate', CPUReceiptProducer())
        self.assertEqual(len(called), 1)
        self.assertFalse((called[0][2] / 'assets-config.json').exists())

    def test_native_session_asset_drift_is_rejected_by_real_lowering_before_prepare(self):
        self.runner.assets = deepcopy(self.assets)
        self.runner.assets['assets']['model']['sha256'] = 'f'*64
        with patch.object(self.runner, '_prepare', side_effect=AssertionError('must not prepare')):
            with self.assertRaisesRegex(ValueError, 'assets differ'):
                self.session._trial(self.cid, 'candidate', self.runner)
        self.assertFalse(self.launched)


class MatchAuditMissingEvidenceTests(unittest.TestCase):
    def test_historical_seconds_are_not_an_execution_receipt(self):
        report = match.validate_manual_incumbent_match({'manual_seconds': 85.109693955,
                                                       'selected_seconds': 84.529197954})
        self.assertEqual(report['status'], 'NOT_ESTABLISHED')
        self.assertEqual(set(report['missing_evidence']), set(match.INPUTS))
        self.assertNotIn('selected_over_manual', report)
        self.assertFalse(report['formal_timing_eligible'])

    def test_missing_manual_arm_or_budget_cannot_be_reported_as_a_match(self):
        for key in match.INPUTS:
            evidence = {k: {'path': '/nonexistent/evidence.json', 'sha256': '0'*64}
                        for k in match.INPUTS if k != key}
            with self.subTest(key=key):
                report = match.validate_manual_incumbent_match(evidence)
                self.assertEqual(report['status'], 'NOT_ESTABLISHED')
                self.assertIn(key, report['missing_evidence'])

    def test_unreadable_or_unhashed_native_evidence_fails_closed(self):
        evidence = {k: {'path': '/nonexistent/evidence.json', 'sha256': '0'*64}
                    for k in match.INPUTS}
        report = match.validate_manual_incumbent_match(evidence)
        self.assertEqual(report['status'], 'NOT_ESTABLISHED')
        self.assertNotIn('selected_over_manual', report)

    def test_capacity_requires_both_node_budgets_and_explicit_reserves(self):
        policy = dict(gpu_metric='peak_allocated_bytes', memory_limits={str(i): 100 for i in range(16)},
                      host_limits={'0': 100, '1': 100}, gpu_reserve_bytes=0, host_reserve_bytes=0)
        match._capacity(policy)
        for key in policy:
            with self.subTest(key=key), self.assertRaises(ValueError):
                match._capacity({k: v for k, v in policy.items() if k != key})

    def test_n70_metadata_or_enablement_cannot_be_silently_accepted(self):
        witness = dict(merge_count=0, PEFT_merge_count=0, N70_window_construction_count=0,
            enabled_install_count=0, enabled_scope_count=0, guard_errors=[],
            execution_shadow=False, once_refresh=False, rollout_merged_shadow=False,
            model_metadata_samples=[dict(point=p, role=r, N70_model_state_present=False, PEFT_merged_layers=0)
                for p in ('AFTER_SETUP_BEFORE_BOTH_TIMERS', 'AFTER_BOTH_TIMERS') for r in ('generator', 'critic')])
        match._nomerge([{'no_merge_audit': witness}])
        for key in ('merge_count', 'PEFT_merge_count', 'N70_window_construction_count'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                match._nomerge([{'no_merge_audit': {**witness, key: 1}}])
        with self.assertRaises(ValueError):
            match._nomerge([{}])


class GANBundleNomergeWitnessTests(unittest.TestCase):
    """Witness protocol only; fixture counters do not constitute GPU evidence."""
    def setUp(self):
        self.cid = 'joint-helios1-gan-fsdp-c2048-p1-w5-kgan-fa3-down40-nomerge-v1'
        try:
            self.impl = match.HeliosImplementation.parse(self.cid)
        except ValueError:
            self.skipTest('requires the separately owned GAN bundle integration')
        self.witness = dict(installed=True, restored=True,
            effective_attention_backend='pinned-sm80-fa3', unsupported_request_count=0,
            merge_attempt_count=0, guarded_merge_methods=815, lora_layer_count=815,
            lora_inventory=deepcopy(match.LORA_INVENTORY),
            merged_layer_count=0,
            model_metadata_samples=[dict(point=p, role=r, LoRA_count=409 if r == 'generator' else 406,
                LoRA_linear_count=406, LoRA_conv3d_names=match.LORA_INVENTORY[r]['conv3d_names'],
                N70_model_state_present=False, PEFT_merged_layers=0)
                for p in ('AFTER_SETUP_BEFORE_BOTH_TIMERS', 'AFTER_BOTH_TIMERS') for r in ('generator', 'critic')])
        self.row = dict(candidate_id=self.cid,
            planner_binding=dict(candidate_bundle=self.impl.candidate_bundle.cost_identity()),
            build=dict(gan_manual_runtime=self.witness))

    def test_new_witness_uses_real_bundle_source_without_fabricated_legacy_counters(self):
        self.assertNotIn('no_merge_audit', self.row)
        self.assertNotIn('N70_window_construction_count', self.witness)
        match._nomerge([self.row], candidate_id=self.cid)

    def test_new_witness_rejects_candidate_and_actual_source_identity_drift(self):
        for cid in (None, 'joint-helios1-gan-fsdp-c2048-p1-w5'):
            with self.subTest(candidate_id=cid), self.assertRaises(ValueError):
                match._nomerge([self.row], candidate_id=cid)
        changed = deepcopy(self.row); changed['candidate_id'] += '-other'
        with self.assertRaises(ValueError):
            match._nomerge([changed], candidate_id=self.cid)
        changed = deepcopy(self.row)
        source = changed['planner_binding']['candidate_bundle']['source']['files']
        source[next(iter(source))] = '0' * 64
        with self.assertRaises(ValueError):
            match._nomerge([changed], candidate_id=self.cid)

    def test_new_witness_missing_observations_or_enabled_merging_fail_closed(self):
        for key in self.witness:
            changed = deepcopy(self.row)
            del changed['build']['gan_manual_runtime'][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                match._nomerge([changed], candidate_id=self.cid)
        for key, value in (('restored', False), ('merge_attempt_count', 1),
                           ('guarded_merge_methods', 406), ('lora_layer_count', 406), ('merged_layer_count', 1),
                           ('effective_attention_backend', 'pinned-local-fa2')):
            changed = deepcopy(self.row); changed['build']['gan_manual_runtime'][key] = value
            with self.subTest(changed=key), self.assertRaises(ValueError):
                match._nomerge([changed], candidate_id=self.cid)
        for key, value in (('LoRA_linear_count', None), ('LoRA_linear_count', 409),
                           ('LoRA_conv3d_names', None), ('LoRA_conv3d_names', [])):
            changed = deepcopy(self.row)
            sample = changed['build']['gan_manual_runtime']['model_metadata_samples'][0]
            if value is None:
                del sample[key]
            else:
                sample[key] = value
            with self.subTest(sample_field=key, value=value), self.assertRaises(ValueError):
                match._nomerge([changed], candidate_id=self.cid)
        for count in (None, 405, 812):
            changed = deepcopy(self.row)
            sample = changed['build']['gan_manual_runtime']['model_metadata_samples'][0]
            if count is None:
                del sample['LoRA_count']
            else:
                sample['LoRA_count'] = count
            with self.subTest(per_role_count=count), self.assertRaises(ValueError):
                match._nomerge([changed], candidate_id=self.cid)


if __name__ == '__main__':
    unittest.main()
