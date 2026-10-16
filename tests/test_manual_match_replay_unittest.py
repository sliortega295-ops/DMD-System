"""Real resolver/replay/transport CPU tests; never launch SSH or a trainer."""
from copy import deepcopy
import io
import json
import shlex
import unittest
from unittest.mock import patch

from distillschedule.search import execution
from distillschedule.workloads.best_known import dispatch
import test_manual_incumbent_validation_unittest as saved_plan_tests


class ManualMatchReplayTests(unittest.TestCase):
    def setUp(self):
        self.fixture = saved_plan_tests.SavedNativePlanTests(methodName='runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        path, _ = self.fixture._saved_lowering()
        lowering = json.loads(path.read_text())
        self.plan = execution.resolve_candidate(self.fixture.candidate.workload, self.fixture.cid,
            assets_path=self.fixture.assets_path, run_id='CPU_SOURCE_PLAN', mode='diagnostic-timing')
        self.plan['model_plan'] = lowering['model_plan']
        self.reference = dict(path=str(self.fixture.tmp/'protocol-not-executed.json'), sha256='a'*64)

    def replay(self, plan, name):
        with patch.object(execution.NativeTrialRunner, '_rpc', side_effect=self.fixture._resolve_rpc):
            return execution.rebind_plan(plan, run_id='CPU_REPLAY_'+name,
                coordinator_output=self.fixture.tmp/name)

    def test_manual_top_level_protocol_is_deepcopied_through_real_rebind(self):
        del self.plan['model_plan']
        self.plan['manual_match_protocol'] = self.reference
        rebound = self.replay(self.plan, 'manual')
        self.assertEqual(rebound.get('manual_match_protocol'), self.reference)
        self.assertIsNot(rebound['manual_match_protocol'], self.reference)
        self.reference['sha256'] = 'b'*64
        self.assertEqual(rebound['manual_match_protocol']['sha256'], 'a'*64)

    def test_model_plan_and_protocol_survive_both_native_worker_transports(self):
        self.plan['manual_match_protocol'] = self.reference
        rebound = self.replay(self.plan, 'selected')
        self.assertEqual(rebound.get('manual_match_protocol'), self.reference)
        self.assertEqual(rebound['model_plan'], self.plan['model_plan'])
        for transport in ('stdin', 'argv-json-tty-v1'):
            for action in ('_node', '_collect'):
                payload = {'receipt': rebound, 'node_rank': 0} if action == '_node' else rebound
                command, stdin = dispatch._worker_request(rebound['nodes'][0], action, payload,
                    transport=transport)
                if stdin is not None:
                    actual = json.loads(stdin)
                else:
                    # Run the actual encoding boot, replacing only external worker entry.
                    tokens = shlex.split(command[-1][:-len('; exit')])
                    boot = tokens[tokens.index('-c')+1]; observed=[]
                    def entry(*args, **kwargs):
                        import sys
                        observed.append(json.load(sys.stdin))
                    with patch('runpy.run_path', side_effect=entry), patch('sys.stdout', io.StringIO()), \
                            patch('sys.stdin', io.StringIO()), patch('sys.argv', []):
                        exec(boot, {})
                    self.assertEqual(len(observed), 1); actual=observed[0]
                with self.subTest(transport=transport, action=action):
                    self.assertEqual(actual, payload)

    def test_replayed_bad_reference_still_fails_actual_worker_protocol_reader(self):
        del self.plan['model_plan']
        self.plan['manual_match_protocol'] = self.reference
        rebound = self.replay(self.plan, 'invalid')
        with self.assertRaisesRegex(ValueError, 'hash changed/missing'):
            dispatch.read_manual_match_protocol(rebound, root=self.fixture.tmp)

    def test_legacy_rebind_does_not_invent_protocol_or_change_environment(self):
        before=deepcopy(self.plan['nodes'])
        rebound=self.replay(self.plan, 'legacy')
        self.assertNotIn('manual_match_protocol', rebound)
        for old,new in zip(before,rebound['nodes']):
            # Only existing run-scoped fields may differ; protocol injects no env flag.
            self.assertEqual(set(old['environment']), set(new['environment']))
            self.assertFalse(any('MANUAL_MATCH' in key for key in new['environment']))


if __name__ == '__main__':
    unittest.main()
