"""Navigation, relevance and bounded display regressions; no model/GPU execution."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
REPO = SKILL.parents[1]
sys.path.insert(0, str(SKILL / 'scripts'))
import dmd_current as current
import dmd_knowledge as k
from dmd_perf import sha_file


class CurrentNavigationTests(unittest.TestCase):
    def test_three_requested_queries(self):
        for query, ids in (
            ('SD1.5 DMD2 resident tree fused Adam', ['dmd2_sd15']),
            ('Official Self-Forcing FSDP2 R6', ['self_forcing_official']),
            ('FLUX Qwen Image frozen base sharing', ['flux_dmd', 'qwen_image_dmd']),
        ):
            out = current.lookup(query)
            self.assertEqual({r['id'] for r in out['matches'][:len(ids)]}, set(ids))
            self.assertFalse(out['automatic_application'])
            for row in out['matches'][:len(ids)]:
                self.assertTrue(row['implementations'])
                self.assertTrue(row['checks'])
                self.assertFalse(row['tests_run'])
        first = current.lookup('SD1.5 DMD2')['matches'][0]
        self.assertIn('recipes/dmd2_sd15/training.optimized.json', first['recipes']['optimized'])
        self.assertIn('src/distillschedule/training/official_optimizations/dmd2.py', first['implementations'])

    def test_new_families_not_limited_to_six(self):
        data = current.build_index(REPO)
        self.assertEqual(len(data['workflows']), 14)
        for query, name in [('LTX', 'ltx25'), ('HunyuanVideo1.5', 'hunyuan_video_dmd'),
                            ('Wan2.2', 'wan22_dmd'), ('Qwen Image', 'qwen_image_dmd')]:
            self.assertEqual(current.lookup(query)['matches'][0]['id'], name)
        row = next(r for r in data['workflows'] if r['id'] == 'qwen_image_dmd')
        self.assertFalse(row['has_optimized_recipe'])  # Do not invent an optimized release.
        self.assertEqual(row['recipes']['optimized'], [])

    def test_bundled_snapshot_matches_existing_registry_and_recipe_bytes(self):
        saved = json.loads(current.INDEX.read_text())
        rebuilt = current.build_index(REPO, saved['source_revision'])
        self.assertEqual(saved, rebuilt)
        for path, digest in saved['source_hashes'].items():
            self.assertEqual(sha_file(REPO / path), digest)
        for row in saved['workflows']:
            for path in row['implementations'] + row['checks'] + [row['guide']]:
                self.assertTrue((REPO / path).is_file(), path)

    def test_explicit_checkout_wins_without_importing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in [current.REGISTRY, current.PROFILES, current.WORKLOADS, current.STRATEGIES]:
                dst = root / path; dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / path, dst)
            shutil.copytree(REPO / 'recipes/dmd2_sd15', root / 'recipes/dmd2_sd15')
            (root / 'tests').mkdir()
            sentinel = root / 'MUST_NOT_EXIST'
            reg = root / current.REGISTRY
            reg.write_text(reg.read_text() + '\nraise RuntimeError("not safe to import")\n')
            path = root / 'recipes/dmd2_sd15/training.optimized.json'
            path.unlink()
            out = current.lookup('SD1.5 DMD2', root)
            self.assertEqual(out['status'], 'CHECKOUT_METADATA_NOT_RUNTIME_VALIDATED')
            self.assertFalse(out['matches'][0]['has_optimized_recipe'])
            self.assertFalse(sentinel.exists())

    def test_corrupt_explicit_checkout_does_not_silently_use_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root / current.REGISTRY
            p.parent.mkdir(parents=True); p.write_text('invalid Python !!!')
            out = current.lookup('DMD2', root)
            self.assertEqual(out['status'], 'NAVIGATION_UNAVAILABLE')
            self.assertEqual(out['matches'], [])
            self.assertIn('warning', out)

    def test_installed_standalone_navigation_stdlib_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'skill'; shutil.copytree(SKILL, dest)
            run = subprocess.run([sys.executable, '-S', str(dest / 'scripts/dmd_knowledge.py'),
                'navigate', '--query', 'Official Self-Forcing FSDP2 R6'], cwd=tmp,
                capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            data = json.loads(run.stdout)
            self.assertEqual(data['status'], 'BUNDLED_SNAPSHOT_NOT_CURRENT_TREE')
            self.assertEqual(data['matches'][0]['id'], 'self_forcing_official')

    def test_unknown_query_and_external_project_are_explicit(self):
        self.assertEqual(current.lookup('zzzzNO_MATCH')['matches'], [])
        with tempfile.TemporaryDirectory() as tmp:
            out = current.lookup('DMD2', Path(tmp))
            self.assertEqual(out['status'], 'NOT_A_DMD_CHECKOUT')
        out = k.recall('SD1.5 DMD2 resident tree fused Adam', {})
        self.assertEqual(out['current_implementations']['matches'][0]['id'], 'dmd2_sd15')
        self.assertEqual(out['normalized_context'], {})


class RankingAndBriefTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.store = self.root / 'optional-existing-store'
        self.report = self.root / 'report.txt'; self.report.write_text('CPU synthetic fixture only')

    def record(self, ident, text='ordinary caching', outcome='inconclusive', **extra):
        return k.put(self.store, {'id': ident, 'kind': 'experience', 'summary': text,
            'pattern': 'state-caching', 'conditions': {}, 'correctness_conditions': {},
            'performance_conditions': {}, 'strength': 'observation', 'outcome': outcome,
            'provenance': {'measurement_kind': 'synthetic'},
            'evidence': [{'path': str(self.report), 'sha256': sha_file(self.report), 'locator': 'fixture'}], **extra}, self.root)

    def test_twelve_generic_records_cannot_hide_unique_failure(self):
        for i in range(12): self.record(f'a-{i:02}')
        self.record('z-critical', 'cache uniquebadtoken backward failed', 'incorrect')
        out = k.search(self.store, 'uniquebadtoken', {}, limit=10)
        self.assertEqual(out['matches'][0]['id'], 'z-critical')
        self.assertEqual(out['matches'][0]['outcome'], 'correctness_failure')
        self.assertGreater(out['matches'][0]['score'], out['matches'][1]['score'])

    def test_chinese_unique_failure_sorts_first(self):
        for i in range(12): self.record(f'a-{i:02}', '缓存常规记录')
        self.record('z-deadlock', '异步缓存导致锁死故障，梯度未完成', 'incorrect')
        self.assertEqual(k.search(self.store, '锁死故障', {}, limit=1)['matches'][0]['id'], 'z-deadlock')

    def test_current_correction_inherits_old_query_relevance(self):
        self.record('a-old', 'needlecorrectunique caching is faster', 'success', provenance={'kind': 'fixture-for-correction-logic'})
        new = self.record('z-new', 'Measurement conclusion corrected', 'no_benefit',
            provenance={'kind': 'fixture-for-correction-logic'},
            supersedes=[{'id': 'a-old', 'kind': 'correction', 'conditions': {}, 'reason': 'bad denominator'}])
        out = k.search(self.store, 'needlecorrectunique', {}, limit=1)
        self.assertEqual(out['matches'][0]['qualified_id'], new['qualified_id'])
        # This is a metadata fixture, not real performance evidence.

    def test_brief_is_bounded_by_outcome_and_can_page_without_loss(self):
        for outcome in ('success', 'no_benefit', 'inconclusive'):
            for i in range(71): self.record(f'{outcome}-{i:02}', 'stressneedle cache ' + ('long note ' * 60), outcome)
        full = k.recall('stressneedle', {}, stores=[self.store], limit=1)
        before = {p.name: p.read_bytes() for p in self.store.glob('*.json')}
        brief = k.brief_recall(full)
        packet = brief['matches'][0]
        self.assertEqual(sum(packet['case_counts'].values()), sum(len(v) for v in full['matches'][0]['cases'].values()))
        self.assertTrue(all(len(g) <= 3 for g in packet['cases'].values()))
        self.assertFalse(packet['all_linked_case_ids_included'])
        self.assertTrue(packet['has_more'])
        self.assertLess(len(json.dumps(brief, ensure_ascii=False).encode()), 18000)
        collected = {g: [] for g in packet['cases']}
        offset = 0
        while True:
            page = k.brief_recall(full, case_offset=offset)['matches'][0]
            for group, cases in page['cases'].items(): collected[group].extend(c['id'] for c in cases)
            if not page['has_more']: break
            offset = page['next_case_offset']
        for group, cases in full['matches'][0]['cases'].items():
            self.assertEqual(collected[group], [c['id'] for c in cases])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.store.glob('*.json')})

    def test_explicit_all_case_ids_keeps_original_completeness(self):
        for i in range(15): self.record(f'n-{i}', 'allneedle cache', 'incorrect')
        full = k.recall('allneedle', {}, stores=[self.store], limit=1)
        brief = k.brief_recall(full, all_cases=True)
        for p, b in zip(full['matches'], brief['matches']):
            self.assertTrue(b['all_linked_case_ids_included'])
            for group, rows in p['cases'].items():
                self.assertEqual([r['id'] for r in rows], [r['id'] for r in b['cases'][group]])

    def test_hidden_corrections_keep_aggregate_warning_and_expandable_ids(self):
        self.record('a', 'correctionneedle cache', 'success')
        self.record('b', 'correctionneedle cache', 'success')
        self.record('z', 'correctionneedle cache', 'incorrect', supersedes=[
            {'id': 'a', 'kind': 'correction', 'conditions': {}, 'reason': 'synthetic needs review'}])
        out = k.brief_recall(k.recall('correctionneedle', {}, stores=[self.store], limit=1), case_limit=1)
        packet = out['matches'][0]
        self.assertGreaterEqual(packet['correction_counts'].get('CORRECTION_NEEDS_REVIEW', 0), 1)
        self.assertIn('omitted_cases', packet)
        self.assertTrue(any(c['id'].endswith(':z') for c in packet['cases']['negative']))

    def test_bad_limits_do_not_start_or_modify_anything(self):
        full = k.recall('LoRA', {})
        for kwargs in ({'case_limit': 0}, {'case_limit': 21}, {'case_offset': -1}):
            with self.assertRaises(ValueError): k.brief_recall(full, **kwargs)
        self.assertFalse(self.store.exists())

    def test_cli_defaults_and_expansion_options(self):
        self.record('specific', 'needlex case', 'incorrect')
        result = subprocess.run([sys.executable, str(SKILL / 'scripts/dmd_knowledge.py'),
            'recall', '--query', 'needlex', '--store', str(self.store), '--brief', '--case-limit', '1'],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['brief_policy']['per_outcome_limit'], 1)


if __name__ == '__main__':
    unittest.main()
