"""Retirement must cover recall, fallback sources and implementation navigation."""
import json
from pathlib import Path
import re
import sys
import unittest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / 'scripts'))
import dmd_knowledge as knowledge
import dmd_navigation as navigation


class RetiredRouteTests(unittest.TestCase):
    def test_early_route_evidence_is_absent_from_all_public_indexes(self):
        early = re.compile(r'202608(?:0[1-9]|1[0-4])')
        cases = knowledge.validate_pack()['cases']
        self.assertFalse(any(early.search(r['path']) for c in cases for r in c['evidence']))
        sources = json.loads((SKILL / 'assets/experience-source-index.json').read_text())['sources']
        self.assertFalse(any(early.search(s['path']) for s in sources))
        entries = navigation.validate_catalog()
        self.assertFalse(any(i.startswith('legacy-') for i in entries))
        self.assertNotIn('rolephase-transport', entries)

    def test_offload_recall_cannot_reintroduce_retired_block_policies(self):
        for query in ('offload 尾部 block', '参数驻留 FSDP2', 'Route-A g12 Score4', 'RouteB 流水'):
            with self.subTest(query=query):
                result = knowledge.recall(query, {'parallel_backend': 'fsdp2'}, limit=20)
                for match in result['matches']:
                    for group in match['cases'].values():
                        for case in group:
                            self.assertNotIn(case['id'], {
                                'helios-score4-streaming-admission', 'helios-full5-pr8-oom',
                                'helios-phase-shift-after-g12', 'helios-lp-flat-release',
                                'helios-isolated-offload-search', 'helios-matched-pipeline-positive'})
                for query_source in ('ra16_f5', 'rb16_step10a', 'gan_systems_experiment_20260811'):
                    self.assertEqual(knowledge.source_search(query_source, limit=100)['matches'], [])

    def test_retired_identity_fails_once_instead_of_returning_reuse_guidance(self):
        for operation in (knowledge.read_case, knowledge.locate_case):
            with self.assertRaisesRegex(knowledge.InputError, 'unknown case'):
                operation('helios-score4-streaming-admission', {})
        with self.assertRaises(knowledge.InputError):
            navigation.locate('legacy-frozen-streaming', {})

    def test_current_fsdp_and_h100_navigation_still_works(self):
        result = knowledge.locate_case('helios-w2-retention-r36', {})
        self.assertTrue(result['implementations'])
        self.assertIn('fsdp-phase-retention', {x['id'] for x in result['implementations']})
        cases = knowledge.validate_pack()['cases']
        self.assertTrue(any(c['provenance']['collection'] == 'Helios H100 history' for c in cases))
        self.assertTrue(any(c['provenance']['collection'] == 'ResearchBoost external' for c in cases))


if __name__ == '__main__':
    unittest.main()
