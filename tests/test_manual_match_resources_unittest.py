"""CPU protocol/telemetry fixtures only; never invoke a GPU or native trainer."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from distillschedule.search import resources
from distillschedule.workloads.best_known import dispatch


class ManualMatchResourceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        scratch = Path(tempfile.gettempdir()) / 'lyy-experiments/DMD-System/matched-verifier-r1'
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        # This disposable miniature Git fixture is not a commit to project source.
        self.repo = self.tmp / 'fixture-repo'; self.repo.mkdir()
        self.git('init', '-q')
        (self.repo / 'tracked').write_text('original\n')
        self.git('add', 'tracked')
        self.git('-c', 'user.name=CPU fixture', '-c', 'user.email=fixture@invalid',
                 '-c', 'commit.gpgsign=false', 'commit', '-qm', 'CPU fixture')
        self.head = self.git('rev-parse', 'HEAD').strip()
        self.policy = dict(gpu_metric='peak_allocated_bytes',
            memory_limits={str(i): 100 * 2**20 for i in range(16)},
            host_limits={'0': 2048 * 2**20, '1': 2048 * 2**20},
            gpu_reserve_bytes=10 * 2**20, host_reserve_bytes=20 * 2**20)
        self.binding = dict(source_head=self.head, hardware_fingerprint='a'*64,
            environment_fingerprint='b'*64, asset_fingerprint='c'*64,
            semantic_fingerprint='d'*64, precision_fingerprint='e'*64)
        self.protocol = dict(identity=self.binding, capacity_policy=self.policy,
            scope='MANUAL_NATIVE_VS_SAVED_SELECTED_PLAN_NATIVE',
            registered_unix_ns=1, max_planner_overhead=0.03,
            manual_candidate={'id': 'fixture-manual'})
        self.path = self.tmp / 'protocol.json'
        self.receipt = dict(integration_head=self.head, asset_fingerprint='c'*64,
            workload_semantics_hash='d'*64, mode='diagnostic-timing',
            search_candidate={'id': 'fixture-selected'},
            model_plan=dict(problem=dict(binding=self.binding,
                memory_limits=self.policy['memory_limits'], lowering={})))
        self.bind_protocol()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True,
                                       stderr=subprocess.DEVNULL)

    def bind_protocol(self):
        self.path.write_text(json.dumps(self.protocol))
        self.ref = dict(path=str(self.path), sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.receipt['manual_match_protocol'] = dict(self.ref)

    def telemetry(self):
        self.cgroup = self.tmp / 'cgroup'; self.cgroup.mkdir(exist_ok=True)
        self.child = self.cgroup / 'job'; self.child.mkdir(exist_ok=True)
        for root, limit in ((self.cgroup, 'max'), (self.child, str(3072 * 2**20))):
            (root / 'memory.max').write_text(limit)
        for name, value in {'memory.current': '1000', 'memory.peak': '2000',
                            'memory.events': 'oom 0\noom_kill 0\n'}.items():
            (self.child / name).write_text(value)
        original_read = Path.read_text
        def read(path, *args, **kwargs):
            if str(path) == '/proc/self/cgroup': return '0::/job\n'
            if str(path) == '/proc/self/mountinfo':
                return f'1 0 0:1 / {self.cgroup} rw - cgroup2 cgroup rw\n'
            if str(path) == '/proc/meminfo': return 'MemTotal:       4194304 kB\n'
            return original_read(path, *args, **kwargs)
        original_query = subprocess.check_output
        def query(argv, **kwargs):
            if argv[0] == 'nvidia-smi':
                return '\n'.join(f'{i}, GPU-{i}, 1000, 900' for i in range(8))
            return original_query(argv, **kwargs)
        self.stack.enter_context(patch.object(Path, 'read_text', read))
        self.stack.enter_context(patch.object(resources.subprocess, 'check_output', query))

    def observer(self, *, capacity=True):
        return resources.ResourceObserver(self.tmp / 'resources.json', 'CPU_NO_GPU', 0,
            source_root=self.repo, capacity_policy=self.policy if capacity else None)

    def test_actual_git_clean_dirty_untracked_and_missing_are_not_echoed(self):
        clean = resources.observe_source_identity(self.repo)
        self.assertEqual(clean['identity'], dict(head=self.head, dirty=False,
            diff_sha256=hashlib.sha256(b'').hexdigest()))
        (self.repo / 'tracked').write_text('different\n')
        dirty = resources.observe_source_identity(self.repo)
        self.assertTrue(dirty['identity']['dirty'])
        self.assertNotEqual(dirty['identity']['diff_sha256'], clean['identity']['diff_sha256'])
        (self.repo / 'tracked').write_text('original\n'); (self.repo / 'untracked.py').write_text('x=1')
        self.assertTrue(resources.observe_source_identity(self.repo)['identity']['dirty'])
        missing = resources.observe_source_identity(self.tmp)
        self.assertEqual(missing['status'], 'UNKNOWN')
        self.assertIsNone(missing['identity']['dirty'])

    def test_absent_protocol_is_a_noop_even_without_git(self):
        self.assertIsNone(dispatch.read_manual_match_protocol({}, root=self.tmp))

    def test_actual_protocol_hash_nested_binding_and_memory_match(self):
        self.receipt['model_plan']['problem']['lowering']['manual_match_protocol'] = dict(self.ref)
        self.assertEqual(dispatch.read_manual_match_protocol(self.receipt, root=self.repo), self.protocol)
        del self.receipt['manual_match_protocol']
        self.assertEqual(dispatch.read_manual_match_protocol(self.receipt, root=self.repo), self.protocol)
        self.receipt['manual_match_protocol'] = {**self.ref, 'sha256': '0'*64}
        with self.assertRaises(ValueError):
            dispatch.read_manual_match_protocol(self.receipt, root=self.repo)

    def test_protocol_rejects_hash_scope_identity_budget_source_and_manual_drift(self):
        cases = []
        changed=deepcopy(self.receipt); changed['manual_match_protocol']['sha256']='0'*64; cases.append(changed)
        changed=deepcopy(self.receipt); changed['model_plan']['problem']['binding']['precision_fingerprint']='0'*64; cases.append(changed)
        changed=deepcopy(self.receipt); changed['model_plan']['problem']['memory_limits']['0']+=1; cases.append(changed)
        changed=deepcopy(self.receipt); del changed['model_plan']; cases.append(changed)
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                dispatch.read_manual_match_protocol(changed, root=self.repo)
        for key, value in (('scope', 'UNRESTRICTED'), ('max_planner_overhead', 0.04),
                           ('registered_unix_ns', None), ('capacity_policy', {})):
            original=deepcopy(self.protocol); self.protocol[key]=value; self.bind_protocol()
            with self.subTest(field=key), self.assertRaises(ValueError):
                dispatch.read_manual_match_protocol(self.receipt, root=self.repo)
            self.protocol=original
        self.bind_protocol(); (self.repo/'tracked').write_text('dirty')
        with self.assertRaises(ValueError):
            dispatch.read_manual_match_protocol(self.receipt, root=self.repo)

    def test_manual_protocol_without_model_plan_requires_exact_declared_candidate(self):
        del self.receipt['model_plan']; self.receipt['search_candidate']=self.protocol['manual_candidate']
        self.assertEqual(dispatch.read_manual_match_protocol(self.receipt, root=self.repo), self.protocol)

    def test_optin_records_observed_not_declared_capacity_and_source(self):
        self.telemetry()
        with self.observer() as observer:
            self.policy['memory_limits']['0'] = 1  # Declaration is immutable within observer.
        record=json.loads(observer.path.read_text())
        self.assertEqual(record['status'], 'PASS')
        self.assertEqual(record['source_identity']['head'], self.head)
        self.assertEqual(record['capacity_policy']['memory_limits']['0'], 100*2**20)
        actual=record['samples'][0]['capacity_observation']
        self.assertEqual(actual['effective_host_limit_bytes'], 3072*2**20)
        self.assertEqual(actual['physical_host_bytes'], 4096*2**20)
        self.assertEqual(record['samples'][0]['gpus'][0]['total_bytes'], 1000*2**20)
        self.assertEqual(record['samples'][0]['cgroup_peak_bytes'], 2000)
        self.assertNotIn('peak_allocated_bytes', record['samples'][0])

    def test_unknown_limits_or_insufficient_physical_budget_fail_before_body(self):
        for broken in ('unknown', 'too-small', 'unreadable-peak'):
            with self.subTest(broken=broken):
                self.telemetry()
                if broken == 'unknown': (self.child/'memory.max').write_text('unknown')
                elif broken == 'too-small': (self.child/'memory.max').write_text('1')
                else: (self.child/'memory.peak').unlink()
                observer=self.observer()
                with self.assertRaises(ValueError):
                    with observer:
                        self.fail('unknown resources must not launch a trainer')
                self.assertEqual(json.loads(observer.path.read_text())['status'], 'FAIL')
                observer.path.unlink()

    def test_source_change_during_observation_fails_closed(self):
        self.telemetry(); observer=self.observer()
        with self.assertRaises(ValueError):
            with observer:
                (self.repo/'tracked').write_text('changed during interval')
        record=json.loads(observer.path.read_text())
        self.assertEqual(record['status'], 'FAIL')
        self.assertTrue(record['source_identity']['dirty'])

    def test_unavailable_peak_allows_only_exact_finite_hard_cap_and_zero_reserve(self):
        self.telemetry(); (self.child/'memory.peak').unlink()
        self.policy['host_reserve_bytes']=0
        self.policy['host_limits']['0']=3072*2**20
        with self.observer() as observer:
            pass
        record=json.loads(observer.path.read_text())
        self.assertEqual(record['status'], 'PASS')
        for sample in record['samples']:
            self.assertIsNone(sample['cgroup_peak_bytes'])
            self.assertEqual(sample['cgroup_peak_status'], 'UNAVAILABLE')
            self.assertEqual(sample['host_capacity_evidence'],
                             'FINITE_CGROUP_HARD_LIMIT_AND_UNCHANGED_OOM_NOT_OBSERVED_PEAK')
            self.assertEqual(sample['capacity_observation']['cgroup_hard_limit_bytes'], 3072*2**20)

    def test_unavailable_peak_does_not_accept_lower_budget_unlimited_cap_or_read_errors(self):
        self.telemetry(); self.policy['host_reserve_bytes']=0
        (self.child/'memory.peak').unlink()
        with self.assertRaises(ValueError):
            with self.observer(): self.fail('smaller declaration is not an enforced hard cap')
        (self.tmp/'resources.json').unlink()
        self.policy['host_limits']['0']=4096*2**20; (self.child/'memory.max').write_text('max')
        with self.assertRaises(ValueError):
            with self.observer(): self.fail('physical MemTotal is not a finite cgroup hard cap')
        (self.tmp/'resources.json').unlink()
        self.policy['host_limits']['0']=3072*2**20; (self.child/'memory.max').write_text(str(3072*2**20))
        original=Path.read_text
        def denied(path, *args, **kwargs):
            if path.name=='memory.peak': raise PermissionError('denied CPU fixture')
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',denied), self.assertRaises(ValueError):
            with self.observer(): self.fail('non-ENOENT read errors must fail')

    def test_unavailable_peak_hard_cap_proof_rejects_oom_delta(self):
        self.telemetry(); (self.child/'memory.peak').unlink()
        self.policy['host_reserve_bytes']=0; self.policy['host_limits']['0']=3072*2**20
        observer=self.observer()
        with self.assertRaisesRegex(ValueError,'OOM'):
            with observer:
                (self.child/'memory.events').write_text('oom 1\noom_kill 0\n')
        self.assertEqual(json.loads(observer.path.read_text())['status'], 'FAIL')

    def test_dispatch_passes_hash_checked_protocol_without_changing_training_command(self):
        self.telemetry()
        node=dict(repo_root=str(self.repo), python='/fixture/env/bin/python',
                  environment={'BOUND_FIXTURE': 'unchanged'}, command=['bash', '/fixture/native.sh', 'arg'])
        self.receipt.update(nodes=[node, deepcopy(node)], run_id='CPU_DISPATCH_NO_GPU',
            profile_id='helios14b_gan_best', asset_config={'run_root': str(self.tmp/'runs')})
        def validate(receipt, *, root, node_rank):
            # Only unrelated asset/training admission is outside this CPU fixture.
            return {'manual_match_protocol': dispatch.read_manual_match_protocol(receipt, root=root)}
        expected_env=dispatch.clean_environment(node)
        with patch.object(dispatch, 'validate_local', side_effect=validate), \
                patch.object(dispatch.subprocess, 'call', return_value=0) as launch:
            self.assertEqual(dispatch.execute_node({'receipt': self.receipt, 'node_rank': 0}), 0)
        out=self.tmp/'runs'/self.receipt['run_id']
        launch.assert_called_once_with(['bash', str(self.repo/'scripts/distillschedule/remote_stage_wrapper.sh'),
            str(out/'owned-node0.pid'), *node['command']], cwd=self.repo, env=expected_env)
        resource=json.loads((out/'search-resources.node0.json').read_text())
        self.assertEqual(resource['capacity_policy'], self.protocol['capacity_policy'])
        effective=json.loads((out/'effective-contract.node0.json').read_text())
        self.assertEqual(effective['local_validation']['manual_match_protocol'], self.protocol)

    def test_missing_git_and_driver_timeout_cannot_enter_optin_scope(self):
        self.telemetry()
        observer=resources.ResourceObserver(self.tmp/'resources.json', 'CPU_NO_GPU', 0,
            source_root=self.tmp, capacity_policy=self.policy)
        with self.assertRaises(ValueError):
            with observer: self.fail('unknown source must not enter scope')
        result=json.loads(observer.path.read_text())
        self.assertEqual(result['status'], 'FAIL')
        self.assertIsNone(result['source_identity']['dirty'])
        observer.path.unlink()
        original=subprocess.check_output
        def timeout(argv, **kwargs):
            if argv[0] == 'nvidia-smi': raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
            return original(argv, **kwargs)
        with patch.object(resources.subprocess, 'check_output', timeout):
            observer=self.observer()
            with self.assertRaises(ValueError):
                with observer: self.fail('unknown driver must not enter scope')
        self.assertEqual(json.loads(observer.path.read_text())['status'], 'FAIL')

    def test_legacy_sample_layout_and_timeout_behavior_are_unchanged(self):
        self.telemetry()
        original=Path.read_text
        def read(path, *args, **kwargs):
            if str(path).startswith('/sys/fs/cgroup/'):
                return (self.child/path.name).read_text()
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', read):
            with self.observer(capacity=False) as observer:
                pass
        record=json.loads(observer.path.read_text())
        self.assertEqual(record['status'], 'PASS')
        self.assertNotIn('capacity_policy', record)
        self.assertNotIn('capacity_observation', record['samples'][0])
        self.assertNotIn('cgroup_peak_bytes', record['samples'][0])


if __name__ == '__main__':
    unittest.main()
